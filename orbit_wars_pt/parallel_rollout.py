"""Lockstep batched rollout: all envs synchronize at the env-step boundary.

Finished episodes reset independently per env so batch mixes early/mid/late games.

Phases 1-5 of the JAX-exploitation rework. The whole rollout pipeline lives
on device:

* Batched ``OrbitWarsState`` lives on device throughout an episode.
* Per-microstep observation, must-halt, and ``virt`` mutation run as JIT'd
  vmap'd JAX kernels; per-sampled (origin, fraction) all-target geometry is
  ``selected_origin_fraction_targets_batched`` (shared with PPO replay).
* Sampling stays on PyTorch, batched across all active envs (no Python loop
  over envs in the hot path).
* Legacy closed-form ``compute_pair_geom_and_etas`` remains for diagnostics;
  the live policy gates targets with the sweep, not the old toward-dest ray.
* Per-microstep transitions land in PyTorch-backed ``TorchTransitionBuffer``
  records; only selected minibatch rows are handed back to JAX for prefix
  replay during PPO.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from time import perf_counter
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

import jax
import jax.numpy as jnp

import jax_orbit_wars as jow

from jax_orbit_wars import DEFAULT_MAX_ACTIONS, FLEET_ETA, FLEET_TARGET_PLANET, OrbitWarsState

from orbit_wars_pt.reset_prefetch import RolloutResetPrefetch

from orbit_wars_pt.batched_env import (
    obs_jax_to_torch,
    reset_env_at_index,
    stack_initial_states,
    step_env_with_scores_batched,
)
from orbit_wars_pt.constants import BOARD_SIZE, CENTER, FRACTIONS, MAX_PLANETS, SUN_RADIUS
from orbit_wars_pt.env_wrapper import OrbitWarsEnvConfig
from orbit_wars_pt.gpu_mem import log_cuda_mem
from orbit_wars_pt.micro_jax import (
    apply_micro_step_batched,
    must_halt_no_owned_ships_batched,
    selected_origin_fraction_targets_batched,
)
from orbit_wars_pt.model import OrbitWarsPolicy
from orbit_wars_pt.observation_jax import build_observation_batched_jax
from orbit_wars_pt.transition_buffer import (
    TorchTransitionBuffer,
    append_active_to_torch_buffer,
    append_to_torch_buffer,
    init_torch_transition_buffer,
)

_MAX_FLEET_EXPAND_RETRIES = 48

_INITIAL_TURN_CACHE_ROWS = 128


def _bucket_size(count: int, capacity: int) -> int:
    count = max(1, int(count))
    b = 1
    while b < count:
        b *= 2
    return min(b, int(capacity))


def _unique_padded_indices(active_idx: np.ndarray, capacity: int) -> tuple[np.ndarray, np.ndarray]:
    active_idx = np.asarray(active_idx, dtype=np.int32)
    bucket = _bucket_size(int(active_idx.size), capacity)
    if active_idx.size == bucket:
        return active_idx, np.ones((bucket,), dtype=np.bool_)
    used = np.zeros((capacity,), dtype=np.bool_)
    used[active_idx] = True
    pad = np.flatnonzero(~used).astype(np.int32)
    need = bucket - int(active_idx.size)
    idx = np.concatenate([active_idx, pad[:need]]).astype(np.int32)
    mask = np.zeros((bucket,), dtype=np.bool_)
    mask[: active_idx.size] = True
    return idx, mask


def _validate_numpy_index_range(name: str, idx: np.ndarray, size: int, *, mask: Optional[np.ndarray] = None) -> None:
    idx = np.asarray(idx)
    if mask is not None:
        idx = idx[np.asarray(mask, dtype=np.bool_)]
    if idx.size == 0:
        return
    lo = int(np.min(idx))
    hi = int(np.max(idx))
    if lo < 0 or hi >= int(size):
        raise RuntimeError(f"{name} out of range: min={lo} max={hi} size={int(size)}")


@jax.jit
def _scatter_state_bucket(
    state_b: OrbitWarsState,
    idx: jnp.ndarray,
    new_bucket: OrbitWarsState,
    apply_mask: jnp.ndarray,
) -> OrbitWarsState:
    def one(old_leaf: jnp.ndarray, new_leaf: jnp.ndarray) -> jnp.ndarray:
        old_sel = old_leaf[idx]
        m = apply_mask
        for _ in range(old_sel.ndim - 1):
            m = m[..., None]
        merged = jnp.where(m, new_leaf, old_sel)
        return old_leaf.at[idx].set(merged)

    return jax.tree.map(one, state_b, new_bucket)


@jax.jit
def _copy_state_slices(
    dst_b: OrbitWarsState,
    src_b: OrbitWarsState,
    idx: jnp.ndarray,
) -> OrbitWarsState:
    return jax.tree.map(lambda dst, src: dst.at[idx].set(src[idx]), dst_b, src_b)


def _jax_state_to_torch_cache(state_b: OrbitWarsState, rows: int, device: torch.device) -> OrbitWarsState:
    """Allocate a PyTorch turn-state cache with leading ``[rows]`` axis."""

    def one(leaf: jnp.ndarray) -> torch.Tensor:
        src = torch.from_dlpack(leaf)
        # PyTorch can represent uint16 tensors but does not implement indexed
        # assignment for them. Keep the live JAX env compact; widen only this
        # cache leaf so turn-state scatter/replay stays on the Torch path.
        dtype = torch.int32 if src.dtype is torch.uint16 else src.dtype
        return torch.zeros((rows,) + tuple(src.shape), dtype=dtype, device=device)

    return jax.tree.map(one, state_b)


def _expand_turn_state_cache_if_needed(
    turn_state_cache: OrbitWarsState,
    required_turn_index: int,
) -> OrbitWarsState:
    """Ensure row ``required_turn_index`` exists on the leading axis."""

    cur = int(turn_state_cache.planets.shape[0])
    need = int(required_turn_index) + 1
    if cur >= need:
        return turn_state_cache
    new_rows = max(cur * 2, need)
    if new_rows == cur * 2:
        print(f"[orbit_wars_pt] turn state cache doubled rows {cur} -> {new_rows}", flush=True)
    pad_rows = new_rows - cur

    def _pad_leaf(leaf: torch.Tensor) -> torch.Tensor:
        pad = torch.zeros((pad_rows,) + tuple(leaf.shape[1:]), dtype=leaf.dtype, device=leaf.device)
        return torch.cat([leaf, pad], dim=0)

    return jax.tree.map(_pad_leaf, turn_state_cache)


def _expand_turn_state_cache_fleets(turn_state_cache: OrbitWarsState, new_F: int) -> OrbitWarsState:
    """Pad cached turn-start fleet leaves along the F axis.

    ``turn_state_cache`` is ``state_b`` with a leading turn-row axis, so
    ``fleets`` is ``[T, N, F, W]`` and ``fleet_active`` is ``[T, N, F]`` (see
    ``expand_fleet_buffers_batched`` for the live ``[N, F, ...]`` layout).
    """

    cur_F = int(turn_state_cache.fleets.shape[2])
    if cur_F >= new_F:
        return turn_state_cache
    pad = new_F - cur_F
    fleets = torch.nn.functional.pad(turn_state_cache.fleets, (0, 0, 0, pad, 0, 0, 0, 0))
    fleet_active = torch.nn.functional.pad(turn_state_cache.fleet_active, (0, pad, 0, 0, 0, 0))
    return turn_state_cache._replace(fleets=fleets, fleet_active=fleet_active)


def _scatter_turn_start_state(
    turn_state_cache: OrbitWarsState,
    state_b: OrbitWarsState,
    turn_slot_per_env: np.ndarray,
) -> OrbitWarsState:
    """Write full ``state_b[n]`` into ``turn_state_cache[turn_slot_per_env[n], n]``."""

    _validate_numpy_index_range("turn_slot_per_env", turn_slot_per_env, int(turn_state_cache.planets.shape[0]))
    device = turn_state_cache.planets.device
    slots = torch.as_tensor(turn_slot_per_env, dtype=torch.long, device=device)
    n_idx = torch.arange(slots.shape[0], dtype=torch.long, device=device)

    def _scatter_leaf(cache_leaf: torch.Tensor, live_leaf: jnp.ndarray) -> torch.Tensor:
        cache_leaf[slots, n_idx] = torch.from_dlpack(live_leaf).to(device=cache_leaf.device, dtype=cache_leaf.dtype)
        return cache_leaf

    return jax.tree.map(_scatter_leaf, turn_state_cache, state_b)


@dataclass
class RolloutCarry:
    """Batched env state carried across PPO iterations (continues unfinished games)."""

    state_b: OrbitWarsState
    cfg: OrbitWarsEnvConfig
    #: Env-turn counter for the current episode per env (for stats across rollout segments).
    episode_turns: List[int]


@dataclass
class RolloutGameStats:
    """Aggregates over episodes that finished during this rollout segment."""

    n_completed: int = 0
    n_step_limit: int = 0
    n_decisive: int = 0
    #: Sum of terminal absolute ship counts (planets + fleets) per player.
    sum_final_ships_p0: float = 0.0
    sum_final_ships_p1: float = 0.0
    sum_episode_turns: float = 0.0
    n_p0_positive_reward: int = 0
    n_p1_positive_reward: int = 0

    def record_completion(
        self,
        *,
        step_limit: bool,
        ships_p0: float,
        ships_p1: float,
        episode_turns: int,
        reward0: float,
        reward1: float,
    ) -> None:
        self.n_completed += 1
        if step_limit:
            self.n_step_limit += 1
        else:
            self.n_decisive += 1
        self.sum_final_ships_p0 += float(ships_p0)
        self.sum_final_ships_p1 += float(ships_p1)
        self.sum_episode_turns += float(episode_turns)
        if reward0 > 0.0:
            self.n_p0_positive_reward += 1
        if reward1 > 0.0:
            self.n_p1_positive_reward += 1


@dataclass
class RolloutSegment:
    """One rollout segment's data, mostly on device.

    ``buf0`` / ``buf1`` store per-row prefix stacks of in-phase micro deltas
    (halt, stored ``send`` / ``slot``, ``pair_flat``, ``frac_idx``) with length
    ``max_micro_steps_per_player``, plus scalar policy fields and
    ``target_planet_reachable`` (rollout-time reachable-destination mask per planet).
    Full canonical ``OrbitWarsState`` at each env-turn start lives in ``turn_state_cache``.
    Per-row ``turn_tag_p0`` / ``turn_tag_p1`` tie buffer rows to a turn slot;
    PPO gather applies a prefix with ``apply_prefix_micro_deltas_batched``.

    Host arrays hold GAE inputs (rewards, dones, valid mask, old logprobs /
    values / bootstrap), all ``[H_buf, N]``.

    ``write_idx_pX[n]`` is the count of valid transitions env ``n`` has
    for player ``X`` in this segment. ``valid_pX[t, n]`` is ``True`` for
    ``t < write_idx_pX[n]`` and ``False`` elsewhere.
    """

    buf0: TorchTransitionBuffer
    buf1: TorchTransitionBuffer
    turn_state_cache: OrbitWarsState
    turn_tag_p0: torch.Tensor
    turn_tag_p1: torch.Tensor
    write_idx_p0: np.ndarray
    write_idx_p1: np.ndarray
    valid_p0: np.ndarray
    valid_p1: np.ndarray
    old_logprob_p0: np.ndarray
    old_logprob_p1: np.ndarray
    old_value_p0: np.ndarray
    old_value_p1: np.ndarray
    reward_p0: np.ndarray
    reward_p1: np.ndarray
    done_p0: np.ndarray
    done_p1: np.ndarray
    bootstrap_p0: np.ndarray
    bootstrap_p1: np.ndarray
    bootstrap_valid_p0: np.ndarray
    bootstrap_valid_p1: np.ndarray
    env_steps_per_env: np.ndarray


@dataclass
class RolloutTiming:
    """Wall and per-phase times inside `collect_parallel_micro_rollouts` (perf_counter, host-side)."""

    init_s: float = 0.0
    env_step_s: float = 0.0
    env_prep_s: float = 0.0
    env_step_core_s: float = 0.0
    env_reset_s: float = 0.0
    env_bookkeeping_s: float = 0.0
    env_python_s: float = 0.0
    micro_cap_s: float = 0.0
    obs_build_s: float = 0.0
    policy_batch_s: float = 0.0
    policy_forward_s: float = 0.0
    policy_model_s: float = 0.0
    policy_sample_origin_s: float = 0.0
    policy_raycast_s: float = 0.0
    policy_target_s: float = 0.0
    policy_scatter_s: float = 0.0
    micro_apply_s: float = 0.0
    # Sub-phases of the ``micro_apply`` window (``_accum_micro_apply_breakdown``).
    micro_apply_dlpack_in_s: float = 0.0
    micro_apply_jax_s: float = 0.0
    micro_apply_dlpack_out_s: float = 0.0
    micro_apply_torch_prep_s: float = 0.0
    micro_prep_active_s: float = 0.0
    micro_prep_wr_mk_s: float = 0.0
    micro_prep_validate_s: float = 0.0
    micro_apply_buf_append_s: float = 0.0
    micro_apply_turn_tag_s: float = 0.0
    micro_apply_numpy_s: float = 0.0
    state_unstack_s: float = 0.0
    loop_s: float = 0.0
    wall_s: float = 0.0
    outer_iters: int = 0

    def accounted_loop_s(self) -> float:
        return (
            self.env_step_s
            + self.micro_cap_s
            + self.obs_build_s
            + self.policy_batch_s
            + self.policy_forward_s
            + self.micro_apply_s
            + self.state_unstack_s
        )


def _accum_micro_apply_breakdown(
    timing: RolloutTiming,
    t0: float,
    t1: float,
    t2: float,
    t3: float,
    t3a: float,
    t3b: float,
    t4: float,
    t5: float,
    t6: float,
    t7: float,
) -> None:
    """``t0``..``t7`` partition one micro-apply block (``perf_counter`` marks).

    ``t3`` end of JAX→torch dlpack. ``t3a`` end of active row / micro index
    materialization. ``t3b`` end of ``micro_k_t`` / ``write_idx_t`` (and related
    host mask arrays). ``t4`` end of ``_validate_numpy_index_range``. ``t5`` end
    of ``append_to_torch_buffer``. ``t6`` end of ``turn_tag`` write. ``t7`` end
    of small CPU numpy extracts.
    """

    timing.micro_apply_s += t7 - t0
    timing.micro_apply_dlpack_in_s += t1 - t0
    timing.micro_apply_jax_s += t2 - t1
    timing.micro_apply_dlpack_out_s += t3 - t2
    timing.micro_prep_active_s += t3a - t3
    timing.micro_prep_wr_mk_s += t3b - t3a
    timing.micro_prep_validate_s += t4 - t3b
    timing.micro_apply_torch_prep_s += t4 - t3
    timing.micro_apply_buf_append_s += t5 - t4
    timing.micro_apply_turn_tag_s += t6 - t5
    timing.micro_apply_numpy_s += t7 - t6


def _build_batched_actions(
    actions0: List[List[Tuple[float, float, float]]],
    actions1: List[List[Tuple[float, float, float]]],
    max_actions: int = DEFAULT_MAX_ACTIONS,
) -> np.ndarray:
    """Pack per-env action lists into ``[num_envs, 2, max_actions, 3]`` (zero-padded)."""

    num_envs = len(actions0)
    arr = np.zeros((num_envs, 2, max_actions, 3), dtype=np.float32)
    for i in range(num_envs):
        for j, tup in enumerate(actions0[i][:max_actions]):
            arr[i, 0, j, 0] = tup[0]
            arr[i, 0, j, 1] = tup[1]
            arr[i, 0, j, 2] = tup[2]
        for j, tup in enumerate(actions1[i][:max_actions]):
            arr[i, 1, j, 0] = tup[0]
            arr[i, 1, j, 1] = tup[1]
            arr[i, 1, j, 2] = tup[2]
    return arr


def _build_batched_action_metadata(
    meta0: List[List[Tuple[float, float]]],
    meta1: List[List[Tuple[float, float]]],
    max_actions: int = DEFAULT_MAX_ACTIONS,
) -> np.ndarray:
    """Pack per-action ``(target_planet, hit_tick)`` metadata beside actions."""

    num_envs = len(meta0)
    arr = np.zeros((num_envs, 2, max_actions, 2), dtype=np.float32)
    arr[..., 0] = -1.0
    arr[..., 1] = 500.0
    for i in range(num_envs):
        for j, tup in enumerate(meta0[i][:max_actions]):
            arr[i, 0, j, 0] = tup[0]
            arr[i, 0, j, 1] = tup[1]
        for j, tup in enumerate(meta1[i][:max_actions]):
            arr[i, 1, j, 0] = tup[0]
            arr[i, 1, j, 1] = tup[1]
    return arr


def _point_to_segment_distance_np(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    delta = end - start
    l2 = float(np.dot(delta, delta))
    if l2 <= 0.0:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, delta) / l2, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * delta)))


def _swept_pair_hit_np(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray, radius: float) -> bool:
    d0 = a0 - b0
    dv = (a1 - a0) - (b1 - b0)
    qa = float(np.dot(dv, dv))
    qb = 2.0 * float(np.dot(d0, dv))
    qc = float(np.dot(d0, d0)) - float(radius) * float(radius)
    if qa < 1e-12:
        return qc <= 0.0
    disc = qb * qb - 4.0 * qa * qc
    if disc < 0.0:
        return False
    sqrt_disc = float(np.sqrt(disc))
    t1 = (-qb - sqrt_disc) / (2.0 * qa)
    t2 = (-qb + sqrt_disc) / (2.0 * qa)
    return (t2 >= 0.0) and (t1 <= 1.0)


def _warn_lost_fleets_if_any(
    state_before_b: OrbitWarsState,
    next_state_b: OrbitWarsState,
    *,
    max_examples: int = 8,
) -> None:
    """Warn when a previously active fleet vanishes by sun/board rather than planet hit."""

    before_fleets, before_active, before_planets, before_planet_active = jax.device_get(
        (
            state_before_b.fleets,
            state_before_b.fleet_active,
            state_before_b.planets,
            state_before_b.planet_active,
        )
    )
    after_fleets, after_active, after_planets = jax.device_get(
        (next_state_b.fleets, next_state_b.fleet_active, next_state_b.planets)
    )
    before_fleets = np.asarray(before_fleets)
    after_fleets = np.asarray(after_fleets)
    before_active = np.asarray(before_active, dtype=np.bool_)
    after_active = np.asarray(after_active, dtype=np.bool_)
    before_planets = np.asarray(before_planets)
    after_planets = np.asarray(after_planets)
    before_planet_active = np.asarray(before_planet_active, dtype=np.bool_)

    sun_xy = np.asarray([CENTER, CENTER], dtype=np.float64)
    examples: list[str] = []
    lost_count = 0
    candidate_count = 0

    for env_i in range(before_active.shape[0]):
        vanished = np.where(before_active[env_i] & ~after_active[env_i])[0]
        for fleet_i in vanished:
            old_pos = before_fleets[env_i, fleet_i, jow.FLEET_X : jow.FLEET_Y + 1].astype(np.float64)
            new_pos = after_fleets[env_i, fleet_i, jow.FLEET_X : jow.FLEET_Y + 1].astype(np.float64)
            oob = (
                (new_pos[0] < 0.0)
                or (new_pos[0] > BOARD_SIZE)
                or (new_pos[1] < 0.0)
                or (new_pos[1] > BOARD_SIZE)
            )
            sun_hit = _point_to_segment_distance_np(sun_xy, old_pos, new_pos) < SUN_RADIUS
            if not (oob or sun_hit):
                continue
            candidate_count += 1
            planet_hit = False
            for planet_i in np.where(before_planet_active[env_i])[0]:
                planet_hit = _swept_pair_hit_np(
                    old_pos,
                    new_pos,
                    before_planets[env_i, planet_i, jow.PLANET_X : jow.PLANET_Y + 1].astype(np.float64),
                    after_planets[env_i, planet_i, jow.PLANET_X : jow.PLANET_Y + 1].astype(np.float64),
                    float(before_planets[env_i, planet_i, jow.PLANET_RADIUS]),
                )
                if planet_hit:
                    break
            if planet_hit:
                continue
            lost_count += 1
            if len(examples) < max_examples:
                row = before_fleets[env_i, fleet_i]
                examples.append(
                    "env={} slot={} fleet_id={:.0f} owner={:.0f} target={:.0f} eta={:.1f} "
                    "sun_hit={} oob={} old=({:.3f},{:.3f}) new=({:.3f},{:.3f})".format(
                        env_i,
                        fleet_i,
                        float(row[jow.FLEET_ID]),
                        float(row[jow.FLEET_OWNER]),
                        float(row[FLEET_TARGET_PLANET]),
                        float(row[FLEET_ETA]),
                        bool(sun_hit),
                        bool(oob),
                        old_pos[0],
                        old_pos[1],
                        new_pos[0],
                        new_pos[1],
                    )
                )

    if lost_count:
        print(
            f"[orbit_wars_pt] WARNING {lost_count} fleet(s) disappeared via sun/board "
            f"without a swept planet hit ({candidate_count} sun/board candidate removals).",
            flush=True,
        )
        for ex in examples:
            print(f"[orbit_wars_pt]   lost fleet: {ex}", flush=True)


def _run_micro_phase(
    *,
    ego: int,
    state_b: OrbitWarsState,
    buf: TorchTransitionBuffer,
    write_idx_per_env: np.ndarray,
    valid: np.ndarray,
    old_logprob: np.ndarray,
    old_value: np.ndarray,
    policy: OrbitWarsPolicy,
    device: torch.device,
    rng: Optional[torch.Generator],
    greedy: bool,
    ship_speed: float,
    max_micro_steps: int,
    timing: RolloutTiming,
    turn_tag_j: torch.Tensor,
    turn_slot_np: np.ndarray,
    first_hit_n_rays: int = 2048,
    first_hit_ray_chunk_size: int = 0,
) -> Tuple[
    TorchTransitionBuffer,
    np.ndarray,
    List[List[Tuple[float, float, float]]],
    List[List[Tuple[float, float]]],
    List[int],
    torch.Tensor,
]:
    """Run one player's micro-loop in lockstep across all envs (Phase 5).

    All transitions are written into the device-resident ``buf`` with
    per-env row indexing; only GAE-relevant scalars cross PCIe.

    ``write_idx_per_env[n]`` is env ``n``'s next free row in ``buf``. It
    advances per microstep only for envs still playing this ego's micro-loop;
    halted envs scatter to their stale row (overwriting safely with
    ``valid=False`` filtering).

    ``turn_slot_np[n]`` is env ``n``'s current in-segment turn index; each
    append writes that index into ``turn_tag_j`` at ``(row, n)`` so replay
    can reconstruct full state from turn starts.

    Returns ``(buf, new_write_idx_per_env, action_lists, action_meta_lists,
    reward_idx, turn_tag_j)``.
    ``reward_idx[n]`` is the row index of env ``n``'s last valid write in
    this phase (``-1`` if env ``n`` produced no rows).
    """

    num_envs = int(state_b.planets.shape[0])
    halted: List[bool] = [False] * num_envs
    action_lists: List[List[Tuple[float, float, float]]] = [[] for _ in range(num_envs)]
    action_meta_lists: List[List[Tuple[float, float]]] = [[] for _ in range(num_envs)]
    reward_idx: List[int] = [-1] * num_envs
    write_idx = write_idx_per_env.copy()
    n_idx_full = np.arange(num_envs)

    virt_b = state_b
    ego_jax = jnp.int32(ego)

    for k in range(max_micro_steps):
        if all(halted):
            break

        # ---- JAX-side compute: obs + must_halt (geometry runs per sampled origin/fraction). ----
        t0 = perf_counter()
        obs_jax = build_observation_batched_jax(virt_b, ego, ship_speed)
        must_halt_j = must_halt_no_owned_ships_batched(virt_b, ego_jax)
        timing.obs_build_s += perf_counter() - t0

        # ---- Hand to PyTorch (zero-copy via dlpack). ----
        t0 = perf_counter()
        obs_torch = obs_jax_to_torch(obs_jax)
        must_halt_t = torch.from_dlpack(must_halt_j)
        timing.policy_batch_s += perf_counter() - t0

        active_idx_list: List[int] = [i for i in range(num_envs) if not halted[i]]
        if not active_idx_list:
            break
        active_idx_t = torch.as_tensor(active_idx_list, device=device, dtype=torch.long)
        n_active = len(active_idx_list)

        active_obs = {key: v.index_select(0, active_idx_t) for key, v in obs_torch.items()}
        must_halt_a = must_halt_t.index_select(0, active_idx_t)

        # ---- Policy forward (active subset). ----
        t0 = perf_counter()
        out = policy.forward_dense_rollout(**active_obs)
        timing.policy_forward_s += perf_counter() - t0

        # ---- Batched sampling (PyTorch). ----
        # Manual log-prob (no ``torch.distributions.Categorical``) keeps the
        # graph compile-friendly: Categorical's Python validation + lazy
        # ``logits``/``probs`` cache trigger graph breaks under torch.compile.
        t0 = perf_counter()
        halt_logits = out["halt_logits"]
        halt_lp = torch.log_softmax(halt_logits, dim=-1)
        if greedy:
            halt_sampled = halt_logits.argmax(dim=-1)
        else:
            halt_probs = halt_lp.exp()
            halt_sampled = torch.multinomial(halt_probs, 1, generator=rng).squeeze(-1)
        halt_action = torch.where(must_halt_a, torch.ones_like(halt_sampled), halt_sampled)
        halt_logp = halt_lp.gather(1, halt_action[:, None]).squeeze(-1)

        origin_frac_mask = out["origin_frac_mask"]
        flat_mask = origin_frac_mask.flatten(start_dim=1)
        any_valid_origin_frac = flat_mask.any(dim=-1)
        flat_logits = out["origin_frac_logits"].flatten(start_dim=1)
        masked_origin_frac = flat_logits.masked_fill(~flat_mask, -1e4)
        origin_frac_lp = torch.log_softmax(masked_origin_frac, dim=-1)
        safe_origin_frac = torch.where(
            any_valid_origin_frac[:, None],
            masked_origin_frac,
            torch.zeros_like(masked_origin_frac),
        )
        if greedy:
            origin_frac_flat = safe_origin_frac.argmax(dim=-1)
        else:
            origin_frac_probs = torch.softmax(safe_origin_frac, dim=-1)
            origin_frac_flat = torch.multinomial(origin_frac_probs, 1, generator=rng).squeeze(-1)
        origin_frac_logp = origin_frac_lp.gather(1, origin_frac_flat[:, None]).squeeze(-1)

        P = MAX_PLANETS
        o_idx = origin_frac_flat // len(FRACTIONS)
        frac_idx = origin_frac_flat % len(FRACTIONS)
        origin_frac_used = (halt_action == 0) & any_valid_origin_frac

        # Keep the JAX geometry batch at fixed ``num_envs`` shape. Gathering
        # ``virt_b[active_idx]`` with a changing active count forces a new XLA
        # compile for each count/capacity during rollout.
        o_idx_all = torch.zeros(num_envs, dtype=torch.int32, device=device)
        frac_idx_geom_all = torch.zeros(num_envs, dtype=torch.int32, device=device)
        o_idx_all.index_copy_(0, active_idx_t, o_idx.to(torch.int32))
        frac_idx_geom_all.index_copy_(0, active_idx_t, frac_idx.to(torch.int32))
        origin_idx_j = jax.dlpack.from_dlpack(o_idx_all.contiguous().detach())
        frac_idx_geom_j = jax.dlpack.from_dlpack(frac_idx_geom_all.contiguous().detach())
        target_angle_j, _target_width_j, target_valid_j, target_overflow_j, target_hit_tick_j = selected_origin_fraction_targets_batched(
            virt_b,
            origin_idx_j,
            frac_idx_geom_j,
            horizon=24,
            ship_speed=ship_speed,
            samples_per_span=17,
            n_rays=first_hit_n_rays,
            ray_chunk_size=first_hit_ray_chunk_size,
        )
        target_angle_t = torch.from_dlpack(target_angle_j).index_select(0, active_idx_t)
        target_valid_t = torch.from_dlpack(target_valid_j).index_select(0, active_idx_t)
        target_overflow_t = torch.from_dlpack(target_overflow_j).index_select(0, active_idx_t)
        target_hit_tick_t = torch.from_dlpack(target_hit_tick_j).index_select(0, active_idx_t)

        n_a_idx = torch.arange(n_active, device=device)
        ph = out["planet_hidden"]
        target_logits = policy.target_logits_for_origin_fraction(ph, o_idx, frac_idx)
        target_mask = out["pair_mask"][n_a_idx, o_idx, :] & target_valid_t & ~target_overflow_t[:, None]
        any_valid_target = target_mask.any(dim=-1)
        masked_target = target_logits.masked_fill(~target_mask, -1e4)
        target_lp = torch.log_softmax(masked_target, dim=-1)
        safe_target = torch.where(any_valid_target[:, None], masked_target, torch.zeros_like(masked_target))
        if greedy:
            d_idx = safe_target.argmax(dim=-1)
        else:
            target_probs = torch.softmax(safe_target, dim=-1)
            d_idx = torch.multinomial(target_probs, 1, generator=rng).squeeze(-1)
        target_logp = target_lp.gather(1, d_idx[:, None]).squeeze(-1)
        pair_flat = o_idx * P + d_idx
        angle = target_angle_t[n_a_idx, d_idx]
        fleet_eta = target_hit_tick_t[n_a_idx, d_idx]

        dispatch_used = origin_frac_used & any_valid_target
        total_logp = halt_logp + origin_frac_used.float() * origin_frac_logp + dispatch_used.float() * target_logp
        # Cast the value head's output back to fp32 at the autocast boundary so
        # the per-env fp32 bookkeeping buffers (``values_all``, ``total_logp_all``)
        # stay consistent. ``log_softmax`` already promotes ``total_logp`` to fp32.
        values_active = out["value"].float()

        halt_now = ~dispatch_used  # active env halts iff it did not dispatch

        # ---- Build full-N tensors for buffer append + JAX apply. ----
        halt_now_all = torch.ones(num_envs, dtype=torch.bool, device=device)
        pair_flat_all = torch.zeros(num_envs, dtype=torch.int32, device=device)
        frac_idx_all = torch.zeros(num_envs, dtype=torch.int32, device=device)
        angle_all = torch.zeros(num_envs, dtype=torch.float32, device=device)
        fleet_eta_all = torch.zeros(num_envs, dtype=torch.float32, device=device)
        halt_action_all = torch.ones(num_envs, dtype=torch.int32, device=device)
        no_valid_pairs_all = torch.zeros(num_envs, dtype=torch.bool, device=device)
        no_valid_fracs_all = torch.zeros(num_envs, dtype=torch.bool, device=device)
        total_logp_all = torch.zeros(num_envs, dtype=torch.float32, device=device)
        values_all = torch.zeros(num_envs, dtype=torch.float32, device=device)
        halt_now_all.index_copy_(0, active_idx_t, halt_now)
        pair_flat_all.index_copy_(0, active_idx_t, pair_flat.to(torch.int32))
        frac_idx_all.index_copy_(0, active_idx_t, frac_idx.to(torch.int32))
        halt_action_all.index_copy_(0, active_idx_t, halt_action.to(torch.int32))
        angle_all.index_copy_(0, active_idx_t, angle.to(torch.float32))
        fleet_eta_all.index_copy_(0, active_idx_t, fleet_eta.to(torch.float32))
        no_valid_pairs_all.index_copy_(0, active_idx_t, origin_frac_used & ~any_valid_target)
        no_valid_fracs_all.index_copy_(0, active_idx_t, ~any_valid_origin_frac & (halt_action == 0))
        must_halt_no_ships_all = torch.zeros(num_envs, dtype=torch.bool, device=device)
        must_halt_no_ships_all.index_copy_(0, active_idx_t, must_halt_a.to(torch.bool))
        total_logp_all.index_copy_(0, active_idx_t, total_logp)
        values_all.index_copy_(0, active_idx_t, values_active)
        timing.policy_forward_s += perf_counter() - t0

        # ---- Apply microstep on device, then append closed deltas + prefix. ----
        t0 = perf_counter()
        halt_now_j = jax.dlpack.from_dlpack(halt_now_all.contiguous().detach())
        pair_flat_j = jax.dlpack.from_dlpack(pair_flat_all.contiguous().detach())
        frac_idx_j = jax.dlpack.from_dlpack(frac_idx_all.contiguous().detach())
        angle_j = jax.dlpack.from_dlpack(angle_all.contiguous().detach())
        fleet_eta_j = jax.dlpack.from_dlpack(fleet_eta_all.contiguous().detach())
        halt_action_j = jax.dlpack.from_dlpack(halt_action_all.contiguous().detach())
        no_valid_pairs_j = jax.dlpack.from_dlpack(no_valid_pairs_all.contiguous().detach())
        no_valid_fracs_j = jax.dlpack.from_dlpack(no_valid_fracs_all.contiguous().detach())
        must_halt_no_ships_j = jax.dlpack.from_dlpack(must_halt_no_ships_all.contiguous().detach())
        target_pr_all = torch.zeros((num_envs, P), dtype=torch.bool, device=device)
        target_pr_all.index_copy_(
            0, active_idx_t, (target_valid_t & ~target_overflow_t[:, None]).to(torch.bool)
        )
        target_planet_reachable_j = jax.dlpack.from_dlpack(target_pr_all.contiguous().detach())
        t1 = perf_counter()

        virt_b, oid_j, angle_j, send_j, dispatched_j, slot_j = apply_micro_step_batched(
            virt_b, ego_jax, halt_now_j, pair_flat_j, frac_idx_j, angle_j, fleet_eta_j
        )
        t2 = perf_counter()

        oid_t = torch.from_dlpack(oid_j)
        angle_applied_t = torch.from_dlpack(angle_j)
        send_t = torch.from_dlpack(send_j)
        dispatched_t = torch.from_dlpack(dispatched_j)
        slot_t = torch.from_dlpack(slot_j)
        t3 = perf_counter()

        active_t = torch.as_tensor(np.array([not h for h in halted], dtype=np.bool_), device=device)
        t3a = perf_counter()
        micro_k_t = torch.full((num_envs,), k, dtype=torch.int32, device=device)
        write_idx_t = torch.as_tensor(write_idx, dtype=torch.int32, device=device)
        active_mask_np = ~np.array(halted, dtype=np.bool_)
        t3b = perf_counter()
        _validate_numpy_index_range("rollout write_idx", write_idx, int(buf.micro_halt_now.shape[0]), mask=active_mask_np)
        _validate_numpy_index_range("rollout micro_k", np.full((num_envs,), k, dtype=np.int32), max_micro_steps, mask=active_mask_np)
        t4 = perf_counter()
        buf = append_to_torch_buffer(
            buf,
            halt_now_all,
            send_t,
            angle_applied_t,
            fleet_eta_all,
            slot_t,
            halt_action_all,
            pair_flat_all,
            frac_idx_all,
            no_valid_pairs_all,
            no_valid_fracs_all,
            must_halt_no_ships_all,
            target_pr_all,
            write_idx_t,
            micro_k_t,
            active_t,
            max_micro_steps,
        )
        t5 = perf_counter()

        active_env_t = torch.as_tensor(np.flatnonzero(active_mask_np).astype(np.int32), dtype=torch.long, device=device)
        rows_t = torch.as_tensor(write_idx[active_mask_np], dtype=torch.long, device=device)
        turn_tag_j[rows_t, active_env_t] = torch.as_tensor(
            turn_slot_np[active_mask_np], dtype=torch.int32, device=device
        )
        t6 = perf_counter()

        oid_np = oid_t.detach().cpu().numpy()
        angle_np = angle_applied_t.detach().cpu().numpy()
        send_np = send_t.detach().cpu().numpy()
        dispatched_np = dispatched_t.detach().cpu().numpy()
        t7 = perf_counter()
        _accum_micro_apply_breakdown(timing, t0, t1, t2, t3, t3a, t3b, t4, t5, t6, t7)

        # ---- Pull tiny scalar metadata for GAE / action_lists. ----
        total_logp_np = total_logp_all.detach().cpu().numpy()
        values_np = values_all.detach().cpu().numpy()
        halt_now_np = halt_now_all.detach().cpu().numpy()
        d_idx_np = d_idx.detach().cpu().numpy()
        fleet_eta_np = fleet_eta.detach().cpu().numpy()

        active_mask = ~np.array(halted, dtype=np.bool_)
        active_envs = np.where(active_mask)[0]
        if active_envs.size:
            rows = write_idx[active_envs]
            valid[rows, active_envs] = True
            old_logprob[rows, active_envs] = total_logp_np[active_envs]
            old_value[rows, active_envs] = values_np[active_envs]
            for j_arr, i in enumerate(active_envs):
                reward_idx[int(i)] = int(rows[j_arr])
                if bool(dispatched_np[i]):
                    action_lists[int(i)].append(
                        (float(oid_np[i]), float(angle_np[i]), float(send_np[i]))
                    )
                    action_meta_lists[int(i)].append(
                        (float(d_idx_np[j_arr]), float(fleet_eta_np[j_arr]))
                    )
                if bool(halt_now_np[i]):
                    halted[int(i)] = True
            write_idx[active_envs] += 1

    return buf, write_idx, action_lists, action_meta_lists, reward_idx, turn_tag_j


def _run_async_micro_step(
    *,
    ego: int,
    active_envs: np.ndarray,
    virt_b: OrbitWarsState,
    buf: TorchTransitionBuffer,
    write_idx: np.ndarray,
    write_idx_dev: torch.Tensor,
    micro_k: np.ndarray,
    micro_k_dev: torch.Tensor,
    valid: np.ndarray,
    old_logprob: np.ndarray,
    old_value: np.ndarray,
    pending_actions: np.ndarray,
    pending_action_count: np.ndarray,
    reward_idx: np.ndarray,
    halted: np.ndarray,
    policy: OrbitWarsPolicy,
    device: torch.device,
    rng: Optional[torch.Generator],
    greedy: bool,
    ship_speed: float,
    max_micro_steps: int,
    timing: RolloutTiming,
    turn_tag_j: torch.Tensor,
    turn_slot_dev: torch.Tensor,
    first_hit_n_rays: int,
    first_hit_ray_chunk_size: int = 0,
) -> tuple[OrbitWarsState, TorchTransitionBuffer, torch.Tensor]:
    """Run one micro decision for ``ego`` on a masked subset of envs."""

    num_envs = int(virt_b.planets.shape[0])
    if active_envs.size == 0:
        return virt_b, buf, turn_tag_j

    active_idx_t = torch.as_tensor(active_envs, device=device, dtype=torch.long)
    n_active = int(active_envs.size)
    ego_jax = jnp.int32(ego)

    t0 = perf_counter()
    obs_jax = build_observation_batched_jax(virt_b, ego, ship_speed)
    must_halt_j = must_halt_no_owned_ships_batched(virt_b, ego_jax)
    timing.obs_build_s += perf_counter() - t0

    t0 = perf_counter()
    obs_torch = obs_jax_to_torch(obs_jax)
    must_halt_t = torch.from_dlpack(must_halt_j)
    active_obs = {key: v.index_select(0, active_idx_t) for key, v in obs_torch.items()}
    must_halt_a = must_halt_t.index_select(0, active_idx_t)
    timing.policy_batch_s += perf_counter() - t0

    t0 = perf_counter()
    out = policy.forward_dense_rollout(**active_obs)
    t_model = perf_counter()
    timing.policy_model_s += t_model - t0

    halt_logits = out["halt_logits"]
    halt_lp = torch.log_softmax(halt_logits, dim=-1)
    if greedy:
        halt_sampled = halt_logits.argmax(dim=-1)
    else:
        halt_sampled = torch.multinomial(halt_lp.exp(), 1, generator=rng).squeeze(-1)
    halt_action = torch.where(must_halt_a, torch.ones_like(halt_sampled), halt_sampled)
    halt_logp = halt_lp.gather(1, halt_action[:, None]).squeeze(-1)

    origin_frac_mask = out["origin_frac_mask"]
    flat_mask = origin_frac_mask.flatten(start_dim=1)
    any_valid_origin_frac = flat_mask.any(dim=-1)
    flat_logits = out["origin_frac_logits"].flatten(start_dim=1)
    masked_origin_frac = flat_logits.masked_fill(~flat_mask, -1e4)
    origin_frac_lp = torch.log_softmax(masked_origin_frac, dim=-1)
    safe_origin_frac = torch.where(
        any_valid_origin_frac[:, None],
        masked_origin_frac,
        torch.zeros_like(masked_origin_frac),
    )
    if greedy:
        origin_frac_flat = safe_origin_frac.argmax(dim=-1)
    else:
        origin_frac_flat = torch.multinomial(torch.softmax(safe_origin_frac, dim=-1), 1, generator=rng).squeeze(-1)
    origin_frac_logp = origin_frac_lp.gather(1, origin_frac_flat[:, None]).squeeze(-1)

    P = MAX_PLANETS
    o_idx = origin_frac_flat // len(FRACTIONS)
    frac_idx = origin_frac_flat % len(FRACTIONS)
    origin_frac_used = (halt_action == 0) & any_valid_origin_frac
    t_origin = perf_counter()
    timing.policy_sample_origin_s += t_origin - t_model

    o_idx_all = torch.zeros(num_envs, dtype=torch.int32, device=device)
    frac_idx_geom_all = torch.zeros(num_envs, dtype=torch.int32, device=device)
    o_idx_all.index_copy_(0, active_idx_t, o_idx.to(torch.int32))
    frac_idx_geom_all.index_copy_(0, active_idx_t, frac_idx.to(torch.int32))
    origin_idx_j = jax.dlpack.from_dlpack(o_idx_all.contiguous().detach())
    frac_idx_geom_j = jax.dlpack.from_dlpack(frac_idx_geom_all.contiguous().detach())
    target_angle_j, _target_width_j, target_valid_j, target_overflow_j, target_hit_tick_j = selected_origin_fraction_targets_batched(
        virt_b,
        origin_idx_j,
        frac_idx_geom_j,
        horizon=24,
        ship_speed=ship_speed,
        samples_per_span=17,
        n_rays=first_hit_n_rays,
        ray_chunk_size=first_hit_ray_chunk_size,
    )
    target_angle_t = torch.from_dlpack(target_angle_j).index_select(0, active_idx_t)
    target_valid_t = torch.from_dlpack(target_valid_j).index_select(0, active_idx_t)
    target_overflow_t = torch.from_dlpack(target_overflow_j).index_select(0, active_idx_t)
    target_hit_tick_t = torch.from_dlpack(target_hit_tick_j).index_select(0, active_idx_t)
    t_raycast = perf_counter()
    timing.policy_raycast_s += t_raycast - t_origin

    n_a_idx = torch.arange(n_active, device=device)
    target_logits = policy.target_logits_for_origin_fraction(out["planet_hidden"], o_idx, frac_idx)
    target_mask = out["pair_mask"][n_a_idx, o_idx, :] & target_valid_t & ~target_overflow_t[:, None]
    any_valid_target = target_mask.any(dim=-1)
    masked_target = target_logits.masked_fill(~target_mask, -1e4)
    target_lp = torch.log_softmax(masked_target, dim=-1)
    safe_target = torch.where(any_valid_target[:, None], masked_target, torch.zeros_like(masked_target))
    if greedy:
        d_idx = safe_target.argmax(dim=-1)
    else:
        d_idx = torch.multinomial(torch.softmax(safe_target, dim=-1), 1, generator=rng).squeeze(-1)
    target_logp = target_lp.gather(1, d_idx[:, None]).squeeze(-1)
    pair_flat = o_idx * P + d_idx
    angle = target_angle_t[n_a_idx, d_idx]
    fleet_eta = target_hit_tick_t[n_a_idx, d_idx]

    dispatch_used = origin_frac_used & any_valid_target
    total_logp = halt_logp + origin_frac_used.float() * origin_frac_logp + dispatch_used.float() * target_logp
    values_active = out["value"].float()
    halt_now = ~dispatch_used
    t_target = perf_counter()
    timing.policy_target_s += t_target - t_raycast

    halt_now_all = torch.ones(num_envs, dtype=torch.bool, device=device)
    pair_flat_all = torch.zeros(num_envs, dtype=torch.int32, device=device)
    frac_idx_all = torch.zeros(num_envs, dtype=torch.int32, device=device)
    angle_all = torch.zeros(num_envs, dtype=torch.float32, device=device)
    fleet_eta_all = torch.zeros(num_envs, dtype=torch.float32, device=device)
    halt_action_all = torch.ones(num_envs, dtype=torch.int32, device=device)
    no_valid_pairs_all = torch.zeros(num_envs, dtype=torch.bool, device=device)
    no_valid_fracs_all = torch.zeros(num_envs, dtype=torch.bool, device=device)
    total_logp_all = torch.zeros(num_envs, dtype=torch.float32, device=device)
    values_all = torch.zeros(num_envs, dtype=torch.float32, device=device)
    must_halt_no_ships_all = torch.zeros(num_envs, dtype=torch.bool, device=device)

    halt_now_all.index_copy_(0, active_idx_t, halt_now)
    pair_flat_all.index_copy_(0, active_idx_t, pair_flat.to(torch.int32))
    frac_idx_all.index_copy_(0, active_idx_t, frac_idx.to(torch.int32))
    halt_action_all.index_copy_(0, active_idx_t, halt_action.to(torch.int32))
    angle_all.index_copy_(0, active_idx_t, angle.to(torch.float32))
    fleet_eta_all.index_copy_(0, active_idx_t, fleet_eta.to(torch.float32))
    no_valid_pairs_all.index_copy_(0, active_idx_t, origin_frac_used & ~any_valid_target)
    no_valid_fracs_all.index_copy_(0, active_idx_t, ~any_valid_origin_frac & (halt_action == 0))
    must_halt_no_ships_all.index_copy_(0, active_idx_t, must_halt_a.to(torch.bool))
    total_logp_all.index_copy_(0, active_idx_t, total_logp)
    values_all.index_copy_(0, active_idx_t, values_active)
    t_scatter = perf_counter()
    timing.policy_scatter_s += t_scatter - t_target
    timing.policy_forward_s += t_scatter - t0

    t0 = perf_counter()
    halt_now_j = jax.dlpack.from_dlpack(halt_now_all.contiguous().detach())
    pair_flat_j = jax.dlpack.from_dlpack(pair_flat_all.contiguous().detach())
    frac_idx_j = jax.dlpack.from_dlpack(frac_idx_all.contiguous().detach())
    angle_j = jax.dlpack.from_dlpack(angle_all.contiguous().detach())
    fleet_eta_j = jax.dlpack.from_dlpack(fleet_eta_all.contiguous().detach())
    halt_action_j = jax.dlpack.from_dlpack(halt_action_all.contiguous().detach())
    no_valid_pairs_j = jax.dlpack.from_dlpack(no_valid_pairs_all.contiguous().detach())
    no_valid_fracs_j = jax.dlpack.from_dlpack(no_valid_fracs_all.contiguous().detach())
    must_halt_no_ships_j = jax.dlpack.from_dlpack(must_halt_no_ships_all.contiguous().detach())
    target_pr_all = torch.zeros((num_envs, P), dtype=torch.bool, device=device)
    target_pr_all.index_copy_(0, active_idx_t, (target_valid_t & ~target_overflow_t[:, None]).to(torch.bool))
    target_planet_reachable_j = jax.dlpack.from_dlpack(target_pr_all.contiguous().detach())
    t1 = perf_counter()

    virt_b, oid_j, angle_j, send_j, dispatched_j, slot_j = apply_micro_step_batched(
        virt_b, ego_jax, halt_now_j, pair_flat_j, frac_idx_j, angle_j, fleet_eta_j
    )
    t2 = perf_counter()

    oid_t = torch.from_dlpack(oid_j)
    angle_applied_t = torch.from_dlpack(angle_j)
    send_t = torch.from_dlpack(send_j)
    dispatched_t = torch.from_dlpack(dispatched_j)
    slot_t = torch.from_dlpack(slot_j)
    t3 = perf_counter()

    rows_active_np = write_idx[active_envs]
    micro_k_active_np = micro_k[active_envs]
    t3a = perf_counter()
    micro_k_t = micro_k_dev.index_select(0, active_idx_t)
    write_idx_t = write_idx_dev.index_select(0, active_idx_t)
    t3b = perf_counter()
    _validate_numpy_index_range("rollout write_idx", rows_active_np, int(buf.micro_halt_now.shape[0]))
    _validate_numpy_index_range("rollout micro_k", micro_k_active_np, max_micro_steps)
    t4 = perf_counter()
    buf = append_active_to_torch_buffer(
        buf,
        active_idx_t,
        halt_now,
        send_t.index_select(0, active_idx_t),
        angle_applied_t.index_select(0, active_idx_t),
        fleet_eta,
        slot_t.index_select(0, active_idx_t),
        halt_action.to(torch.int32),
        pair_flat.to(torch.int32),
        frac_idx.to(torch.int32),
        origin_frac_used & ~any_valid_target,
        ~any_valid_origin_frac & (halt_action == 0),
        must_halt_a.to(torch.bool),
        (target_valid_t & ~target_overflow_t[:, None]).to(torch.bool),
        write_idx_t,
        micro_k_t,
        max_micro_steps,
    )
    t5 = perf_counter()

    rows_t = write_idx_t.to(torch.long)
    turn_tag_j[rows_t, active_idx_t] = turn_slot_dev.index_select(0, active_idx_t)
    t6 = perf_counter()

    oid_np = oid_t.index_select(0, active_idx_t).detach().cpu().numpy()
    angle_np = angle_applied_t.index_select(0, active_idx_t).detach().cpu().numpy()
    send_np = send_t.index_select(0, active_idx_t).detach().cpu().numpy()
    dispatched_np = dispatched_t.index_select(0, active_idx_t).detach().cpu().numpy()
    t7 = perf_counter()
    _accum_micro_apply_breakdown(timing, t0, t1, t2, t3, t3a, t3b, t4, t5, t6, t7)

    total_logp_np = total_logp.detach().cpu().numpy()
    values_np = values_active.detach().cpu().numpy()
    halt_now_np = halt_now.detach().cpu().numpy()
    d_idx_np = d_idx.detach().cpu().numpy()
    fleet_eta_np = fleet_eta.detach().cpu().numpy()

    rows = rows_active_np
    valid[rows, active_envs] = True
    old_logprob[rows, active_envs] = total_logp_np
    old_value[rows, active_envs] = values_np

    for j_arr, env_i in enumerate(active_envs):
        env_i = int(env_i)
        reward_idx[env_i] = int(write_idx[env_i])
        if bool(dispatched_np[j_arr]):
            ac = int(pending_action_count[ego, env_i])
            if ac < pending_actions.shape[2]:
                pending_actions[env_i, ego, ac, 0] = float(oid_np[j_arr])
                pending_actions[env_i, ego, ac, 1] = float(angle_np[j_arr])
                pending_actions[env_i, ego, ac, 2] = float(send_np[j_arr])
                pending_actions[env_i, ego, ac, 3] = float(d_idx_np[j_arr])
                pending_actions[env_i, ego, ac, 4] = float(fleet_eta_np[j_arr])
            pending_action_count[ego, env_i] = ac + 1
        if bool(halt_now_np[j_arr]):
            halted[ego, env_i] = True
        write_idx[env_i] += 1
        micro_k[env_i] += 1
        if micro_k[env_i] >= max_micro_steps:
            halted[ego, env_i] = True

    write_idx_dev[active_idx_t] = write_idx_t + 1
    micro_k_dev[active_idx_t] = micro_k_t + 1

    return virt_b, buf, turn_tag_j


def _reset_prefetch_resync(
    reset_prefetch: Optional[RolloutResetPrefetch],
    seed_base: int,
    seeds_consumed: int,
    cfg: OrbitWarsEnvConfig,
) -> None:
    if reset_prefetch is None:
        return
    reset_prefetch.notify_max_fleets(int(cfg.max_fleets))
    reset_prefetch.prefetch_ahead(
        int(seed_base + seeds_consumed), int(cfg.num_agents), int(cfg.max_fleets)
    )


def collect_parallel_micro_rollouts(
    policy: OrbitWarsPolicy,
    cfg_template: OrbitWarsEnvConfig,
    num_envs: int,
    device: torch.device,
    *,
    seed_base: int,
    rng: Optional[torch.Generator] = None,
    greedy: bool = False,
    ship_speed: float = 6.0,
    max_micro_steps_per_player: int = 64,
    rollout_micro_horizon: int = 256,
    carry_in: Optional[RolloutCarry] = None,
    max_outer_iters: int = 5_000_000,
    mem_debug: int = 0,
    train_iter: int = -1,
    amp_dtype: Optional[torch.dtype] = None,
    min_max_fleets: int = 1,
    reset_prefetch: Optional[RolloutResetPrefetch] = None,
    first_hit_n_rays: int = 2048,
    first_hit_ray_chunk_size: int = 0,
) -> Tuple[RolloutSegment, RolloutTiming, RolloutCarry, int, RolloutGameStats]:
    """Collect one rollout segment using device-resident transition buffers.

    Stops after a full env-step when any env's player-0 or player-1 segment
    micro-step count reaches ``rollout_micro_horizon`` (end-of-turn cut).
    Episodes that end mid-segment reset only that env's slice so the batch
    stays phase-mixed. When the horizon fires, attaches per-env bootstrap
    values for GAE on non-terminal last transitions.

    Returns ``(segment, timing, next_carry, seeds_consumed, game_stats)``.

    If the segment finishes with fewer than half of the fleet slots in use
    (peak concurrent active fleets, any env) and the upper tail of the table
    is empty, ``next_carry``'s batched state is shrunk to ``max_fleets // 2``
    (not below ``min_max_fleets``). The returned ``segment`` still reflects the
    pre-shrink buffer width so PPO replay stays consistent for this iteration.
    """

    t_wall0 = perf_counter()
    timing = RolloutTiming()

    profile_rollout = mem_debug >= 1 and train_iter == 0
    if profile_rollout:
        log_cuda_mem("rollout enter (before env work)", device)

    seeds_consumed = 0
    t_init0 = perf_counter()
    if reset_prefetch is not None:
        mf0 = int(carry_in.cfg.max_fleets) if carry_in is not None else int(cfg_template.max_fleets)
        na0 = int(cfg_template.num_agents)
        reset_prefetch.notify_max_fleets(mf0)
        reset_prefetch.prefetch_ahead(int(seed_base + seeds_consumed), na0, mf0)
    if carry_in is None:
        state_b, cfg = stack_initial_states(
            cfg_template, num_envs, seed_base, reset_prefetch=reset_prefetch
        )
        seeds_consumed += num_envs
        episode_turns = [0] * num_envs
    else:
        state_b, cfg = carry_in.state_b, carry_in.cfg
        episode_turns = list(carry_in.episode_turns)
        if len(episode_turns) != num_envs:
            episode_turns = [0] * num_envs

    _reset_prefetch_resync(reset_prefetch, seed_base, seeds_consumed, cfg)

    episode_lim = int(np.asarray(jax.device_get(jow.OrbitWarsConfig().episode_steps)))
    episode_timeout_step_count = episode_lim - 1
    game_stats = RolloutGameStats()

    # Allocate per-segment device buffers. Once any player's write index hits
    # the horizon, async collection drains every env to its next real env-step
    # boundary. In the worst phase mix, either player's buffer may still need
    # one full local turn of micro rows before that boundary.
    H_buf = rollout_micro_horizon + max_micro_steps_per_player + 2
    buf0 = init_torch_transition_buffer(num_envs, H_buf, max_micro_steps_per_player, device=device)
    buf1 = init_torch_transition_buffer(num_envs, H_buf, max_micro_steps_per_player, device=device)

    turn_slot_np = np.zeros((num_envs,), dtype=np.int32)
    turn_slot_dev = torch.zeros((num_envs,), dtype=torch.int32, device=device)
    # Start near the usual per-segment turn count so early rollout does not
    # recompile through 1/2/4/8/16/32 cache shapes, while keeping memory modest.
    turn_state_cache = _jax_state_to_torch_cache(state_b, _INITIAL_TURN_CACHE_ROWS, device)
    turn_state_cache = _scatter_turn_start_state(turn_state_cache, state_b, turn_slot_np)

    turn_tag_p0 = torch.full((H_buf, num_envs), -1, dtype=torch.int32, device=device)
    turn_tag_p1 = torch.full((H_buf, num_envs), -1, dtype=torch.int32, device=device)

    valid_p0 = np.zeros((H_buf, num_envs), dtype=np.bool_)
    valid_p1 = np.zeros((H_buf, num_envs), dtype=np.bool_)
    old_logprob_p0 = np.zeros((H_buf, num_envs), dtype=np.float32)
    old_logprob_p1 = np.zeros((H_buf, num_envs), dtype=np.float32)
    old_value_p0 = np.zeros((H_buf, num_envs), dtype=np.float32)
    old_value_p1 = np.zeros((H_buf, num_envs), dtype=np.float32)
    reward_p0 = np.zeros((H_buf, num_envs), dtype=np.float32)
    reward_p1 = np.zeros((H_buf, num_envs), dtype=np.float32)
    done_p0 = np.zeros((H_buf, num_envs), dtype=np.bool_)
    done_p1 = np.zeros((H_buf, num_envs), dtype=np.bool_)
    write_idx_p0 = np.zeros((num_envs,), dtype=np.int32)
    write_idx_p1 = np.zeros((num_envs,), dtype=np.int32)
    write_idx_p0_dev = torch.zeros((num_envs,), dtype=torch.int32, device=device)
    write_idx_p1_dev = torch.zeros((num_envs,), dtype=torch.int32, device=device)

    env_steps_per_env = np.zeros((num_envs,), dtype=np.int32)
    horizon_fired = False
    timing.init_s = perf_counter() - t_init0

    logged_first_policy_fwd = False
    outer = 0
    t_loop0 = perf_counter()
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if amp_dtype is not None
        else nullcontext()
    )

    virt0_b = state_b
    virt1_b = state_b
    halted = np.zeros((2, num_envs), dtype=np.bool_)
    micro_k = np.zeros((2, num_envs), dtype=np.int32)
    micro_k_dev = torch.zeros((2, num_envs), dtype=torch.int32, device=device)
    pending_actions = np.zeros((num_envs, 2, DEFAULT_MAX_ACTIONS, 5), dtype=np.float32)
    pending_actions[..., 3] = -1.0
    pending_actions[..., 4] = 500.0
    pending_action_count = np.zeros((2, num_envs), dtype=np.int32)
    reward_idx = np.full((2, num_envs), -1, dtype=np.int32)
    segment_done = np.zeros((num_envs,), dtype=np.bool_)

    with torch.inference_mode(), amp_ctx:
        while outer < max_outer_iters:
            outer += 1

            if horizon_fired and np.all(segment_done):
                break

            ready_mask = (~segment_done) & halted[0] & halted[1]
            pending0 = (~segment_done) & (~halted[0])
            pending1 = (~segment_done) & (~halted[1])
            ready_count = int(np.sum(ready_mask))
            pending0_count = int(np.sum(pending0))
            pending1_count = int(np.sum(pending1))
            pending_total = pending0_count + pending1_count

            # Env stepping is cheap relative to policy/geometry work, but
            # tiny step buckets still pay host/JAX dispatch overhead. Step
            # once roughly a quarter-batch is ready, or force the step when
            # every remaining player is already halted.
            ready_step_threshold = max(2, num_envs // 4)
            should_step = ready_count >= ready_step_threshold or (ready_count > 0 and pending_total == 0)
            if should_step:
                t0 = perf_counter()
                t_prep0 = perf_counter()
                step_envs = np.flatnonzero(ready_mask).astype(np.int32)
                bucket_idx, bucket_mask = _unique_padded_indices(step_envs, num_envs)
                idx_j = jnp.asarray(bucket_idx, dtype=jnp.int32)
                mask_j = jnp.asarray(bucket_mask, dtype=jnp.bool_)
                actions_bucket = pending_actions[bucket_idx].copy()
                actions_bucket[~bucket_mask] = 0.0
                actions_bucket[~bucket_mask, :, :, 3] = -1.0
                actions_bucket[~bucket_mask, :, :, 4] = 500.0
                timing.env_prep_s += perf_counter() - t_prep0

                t_core0 = perf_counter()
                state_bucket = jax.tree.map(lambda leaf: leaf[idx_j], state_b)
                next_bucket, dr0_jax, dr1_jax, s0_post, s1_post = step_env_with_scores_batched(
                    state_bucket, actions_bucket
                )
                dr0_np, dr1_np, done_np, step_count_np, rewards_np, s0_fin_np, s1_fin_np = jax.device_get(
                    (
                        dr0_jax,
                        dr1_jax,
                        next_bucket.done,
                        next_bucket.step_count,
                        next_bucket.rewards,
                        s0_post,
                        s1_post,
                    )
                )
                dr0_np = np.asarray(dr0_np)
                dr1_np = np.asarray(dr1_np)
                done_np = np.asarray(done_np)
                step_count_np = np.asarray(step_count_np)
                rewards_np = np.asarray(rewards_np)
                s0_fin_np = np.asarray(s0_fin_np)
                s1_fin_np = np.asarray(s1_fin_np)
                timing.env_step_core_s += perf_counter() - t_core0

                t_book0 = perf_counter()
                state_b = _scatter_state_bucket(state_b, idx_j, next_bucket, mask_j)
                timing.env_bookkeeping_s += perf_counter() - t_book0

                t_py0 = perf_counter()
                for local_i, env_i in enumerate(step_envs):
                    env_i = int(env_i)
                    env_steps_per_env[env_i] += 1
                    episode_turns[env_i] += 1
                    if reward_idx[0, env_i] >= 0:
                        reward_p0[reward_idx[0, env_i], env_i] += float(dr0_np[local_i])
                        done_p0[reward_idx[0, env_i], env_i] = bool(done_np[local_i]) or bool(done_p0[reward_idx[0, env_i], env_i])
                    if reward_idx[1, env_i] >= 0:
                        reward_p1[reward_idx[1, env_i], env_i] += float(dr1_np[local_i])
                        done_p1[reward_idx[1, env_i], env_i] = bool(done_np[local_i]) or bool(done_p1[reward_idx[1, env_i], env_i])
                    if bool(done_np[local_i]):
                        sc_i = int(step_count_np[local_i])
                        game_stats.record_completion(
                            step_limit=sc_i >= episode_timeout_step_count,
                            ships_p0=float(s0_fin_np[local_i]),
                            ships_p1=float(s1_fin_np[local_i]),
                            episode_turns=int(episode_turns[env_i]),
                            reward0=float(rewards_np[local_i, 0]),
                            reward1=float(rewards_np[local_i, 1]),
                        )
                        episode_turns[env_i] = 0
                        sid = int(seed_base + seeds_consumed)
                        t_reset0 = perf_counter()
                        if reset_prefetch is not None:
                            fresh_np = reset_prefetch.pop_state(sid, int(cfg.num_agents), int(cfg.max_fleets))
                            state_b = reset_env_at_index(state_b, env_i, sid, cfg, fresh_np=fresh_np)
                        else:
                            state_b = reset_env_at_index(state_b, env_i, sid, cfg)
                        reset_dt = perf_counter() - t_reset0
                        timing.env_reset_s += reset_dt
                        t_py0 += reset_dt
                        seeds_consumed += 1

                    halted[:, env_i] = False
                    micro_k[:, env_i] = 0
                    pending_actions[env_i] = 0.0
                    pending_actions[env_i, :, :, 3] = -1.0
                    pending_actions[env_i, :, :, 4] = 500.0
                    pending_action_count[:, env_i] = 0
                    reward_idx[:, env_i] = -1
                    if horizon_fired:
                        segment_done[env_i] = True
                    turn_slot_np[env_i] += 1
                timing.env_python_s += perf_counter() - t_py0

                t_book0 = perf_counter()
                step_env_t = torch.as_tensor(step_envs, dtype=torch.long, device=device)
                micro_k_dev[:, step_env_t] = 0
                turn_slot_dev[step_env_t] += 1
                _reset_prefetch_resync(reset_prefetch, seed_base, seeds_consumed, cfg)
                max_turn = int(np.max(turn_slot_np))
                turn_state_cache = _expand_turn_state_cache_if_needed(turn_state_cache, max_turn)
                turn_state_cache = _scatter_turn_start_state(turn_state_cache, state_b, turn_slot_np)
                actual_idx_j = jnp.asarray(step_envs, dtype=jnp.int32)
                virt0_b = _copy_state_slices(virt0_b, state_b, actual_idx_j)
                virt1_b = _copy_state_slices(virt1_b, state_b, actual_idx_j)
                timing.env_bookkeeping_s += perf_counter() - t_book0
                timing.env_step_s += perf_counter() - t0
                continue

            if pending_total == 0:
                break

            ego = 0 if pending0_count >= pending1_count else 1
            active_envs = np.flatnonzero(pending0 if ego == 0 else pending1).astype(np.int32)
            if ego == 0:
                virt0_b, buf0, turn_tag_p0 = _run_async_micro_step(
                    ego=0,
                    active_envs=active_envs,
                    virt_b=virt0_b,
                    buf=buf0,
                    write_idx=write_idx_p0,
                    write_idx_dev=write_idx_p0_dev,
                    micro_k=micro_k[0],
                    micro_k_dev=micro_k_dev[0],
                    valid=valid_p0,
                    old_logprob=old_logprob_p0,
                    old_value=old_value_p0,
                    pending_actions=pending_actions,
                    pending_action_count=pending_action_count,
                    reward_idx=reward_idx[0],
                    halted=halted,
                    policy=policy,
                    device=device,
                    rng=rng,
                    greedy=greedy,
                    ship_speed=ship_speed,
                    max_micro_steps=max_micro_steps_per_player,
                    timing=timing,
                    turn_tag_j=turn_tag_p0,
                    turn_slot_dev=turn_slot_dev,
                    first_hit_n_rays=first_hit_n_rays,
                    first_hit_ray_chunk_size=first_hit_ray_chunk_size,
                )
                if profile_rollout and device.type == "cuda" and not logged_first_policy_fwd:
                    log_cuda_mem("rollout after first batched policy forward", device)
                    logged_first_policy_fwd = True
            else:
                virt1_b, buf1, turn_tag_p1 = _run_async_micro_step(
                    ego=1,
                    active_envs=active_envs,
                    virt_b=virt1_b,
                    buf=buf1,
                    write_idx=write_idx_p1,
                    write_idx_dev=write_idx_p1_dev,
                    micro_k=micro_k[1],
                    micro_k_dev=micro_k_dev[1],
                    valid=valid_p1,
                    old_logprob=old_logprob_p1,
                    old_value=old_value_p1,
                    pending_actions=pending_actions,
                    pending_action_count=pending_action_count,
                    reward_idx=reward_idx[1],
                    halted=halted,
                    policy=policy,
                    device=device,
                    rng=rng,
                    greedy=greedy,
                    ship_speed=ship_speed,
                    max_micro_steps=max_micro_steps_per_player,
                    timing=timing,
                    turn_tag_j=turn_tag_p1,
                    turn_slot_dev=turn_slot_dev,
                    first_hit_n_rays=first_hit_n_rays,
                    first_hit_ray_chunk_size=first_hit_ray_chunk_size,
                )

            if np.any(write_idx_p0 >= rollout_micro_horizon) or np.any(write_idx_p1 >= rollout_micro_horizon):
                horizon_fired = True

    timing.loop_s = perf_counter() - t_loop0
    timing.outer_iters = outer
    timing.wall_s = perf_counter() - t_wall0

    if profile_rollout:
        log_cuda_mem("rollout exit (segment finished)", device)

    bootstrap_p0 = np.zeros((num_envs,), dtype=np.float32)
    bootstrap_p1 = np.zeros((num_envs,), dtype=np.float32)
    bootstrap_valid_p0 = np.zeros((num_envs,), dtype=np.bool_)
    bootstrap_valid_p1 = np.zeros((num_envs,), dtype=np.bool_)
    if horizon_fired:
        t0 = perf_counter()
        obs0_j = build_observation_batched_jax(state_b, 0, ship_speed)
        timing.obs_build_s += perf_counter() - t0
        tb0 = perf_counter()
        with amp_ctx:
            out0 = policy.forward_dense_rollout(**obs_jax_to_torch(obs0_j))
        timing.policy_batch_s += perf_counter() - tb0
        tf0 = perf_counter()
        v0_np = out0["value"].float().detach().cpu().numpy()
        timing.policy_forward_s += perf_counter() - tf0

        t0 = perf_counter()
        obs1_j = build_observation_batched_jax(state_b, 1, ship_speed)
        timing.obs_build_s += perf_counter() - t0
        tb1 = perf_counter()
        with amp_ctx:
            out1 = policy.forward_dense_rollout(**obs_jax_to_torch(obs1_j))
        timing.policy_batch_s += perf_counter() - tb1
        tf1 = perf_counter()
        v1_np = out1["value"].float().detach().cpu().numpy()
        timing.policy_forward_s += perf_counter() - tf1

        for i in range(num_envs):
            # Bootstrap only if env i has at least one valid step in the
            # segment AND its last step's done flag is False. With per-env
            # row indexing the last valid row is just ``write_idx[i] - 1``.
            last_t0 = int(write_idx_p0[i]) - 1
            if last_t0 >= 0 and not bool(done_p0[last_t0, i]):
                bootstrap_p0[i] = float(v0_np[i])
                bootstrap_valid_p0[i] = True
            last_t1 = int(write_idx_p1[i]) - 1
            if last_t1 >= 0 and not bool(done_p1[last_t1, i]):
                bootstrap_p1[i] = float(v1_np[i])
                bootstrap_valid_p1[i] = True

    segment = RolloutSegment(
        buf0=buf0,
        buf1=buf1,
        turn_state_cache=turn_state_cache,
        turn_tag_p0=turn_tag_p0,
        turn_tag_p1=turn_tag_p1,
        write_idx_p0=write_idx_p0,
        write_idx_p1=write_idx_p1,
        valid_p0=valid_p0,
        valid_p1=valid_p1,
        old_logprob_p0=old_logprob_p0,
        old_logprob_p1=old_logprob_p1,
        old_value_p0=old_value_p0,
        old_value_p1=old_value_p1,
        reward_p0=reward_p0,
        reward_p1=reward_p1,
        done_p0=done_p0,
        done_p1=done_p1,
        bootstrap_p0=bootstrap_p0,
        bootstrap_p1=bootstrap_p1,
        bootstrap_valid_p0=bootstrap_valid_p0,
        bootstrap_valid_p1=bootstrap_valid_p1,
        env_steps_per_env=env_steps_per_env,
    )

    next_carry = RolloutCarry(
        state_b=state_b,
        cfg=cfg,
        episode_turns=episode_turns,
    )
    return segment, timing, next_carry, seeds_consumed, game_stats

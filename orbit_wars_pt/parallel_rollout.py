"""Lockstep batched rollout: all envs synchronize at the env-step boundary.

Finished episodes reset independently per env so batch mixes early/mid/late games.

Phases 1-5 of the JAX-exploitation rework. The whole rollout pipeline lives
on device:

* Batched ``OrbitWarsState`` lives on device throughout an episode.
* Per-microstep observation, pair geometry, ETAs, must-halt and the
  ``virt`` mutation all run as JIT'd vmap'd JAX kernels.
* Sampling stays on PyTorch, batched across all active envs (no Python loop
  over envs in the hot path).
* Closed-form ``compute_pair_geom_and_etas`` removes the legacy 32-retry
  pair resample and is bit-identical between rollout and PPO replay.
* Per-microstep transitions land in a device-resident ``TransitionBuffer``
  (``orbit_wars_pt.transition_buffer``); only tiny dispatch / scalar
  metadata cross PCIe. PPO replay gathers minibatches directly from the
  buffer via JIT'd advanced indexing.
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

from jax_orbit_wars import DEFAULT_MAX_ACTIONS, OrbitWarsState

from orbit_wars_pt.reset_prefetch import RolloutResetPrefetch

from orbit_wars_pt.batched_env import (
    expand_fleet_buffers_batched,
    inactive_fleet_count_batched,
    max_concurrent_fleets_any_env,
    obs_jax_to_torch,
    reset_env_at_index,
    ship_ratio_scores_batched,
    shrink_fleet_buffers_batched,
    ship_totals_batched,
    stack_initial_states,
    upper_bound_fleet_writes_per_env,
    vmap_step_env,
)
from orbit_wars_pt.constants import FRACTIONS, MAX_PLANETS
from orbit_wars_pt.env_wrapper import OrbitWarsEnvConfig
from orbit_wars_pt.gpu_mem import log_cuda_mem
from orbit_wars_pt.micro_jax import (
    apply_micro_step_batched,
    compute_pair_geom_and_etas,
    must_halt_no_owned_ships_batched,
)
from orbit_wars_pt.model import OrbitWarsPolicy
from orbit_wars_pt.observation_jax import build_observation_batched_jax
from orbit_wars_pt.transition_buffer import (
    TransitionBuffer,
    append_to_buffer,
    init_transition_buffer,
    scatter_turn_tags,
)

_MAX_FLEET_EXPAND_RETRIES = 48

_INITIAL_TURN_CACHE_ROWS = 1


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

    def _pad_leaf(leaf: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate(
            [leaf, jnp.zeros((pad_rows,) + leaf.shape[1:], dtype=leaf.dtype)],
            axis=0,
        )

    return jax.tree.map(_pad_leaf, turn_state_cache)


def _expand_turn_state_cache_fleets(turn_state_cache: OrbitWarsState, new_F: int) -> OrbitWarsState:
    """Pad cached turn-start fleet leaves along the F axis.

    ``turn_state_cache`` is ``state_b`` with a leading turn-row axis, so
    ``fleets`` is ``[T, N, F, 7]`` and ``fleet_active`` is ``[T, N, F]`` (see
    ``expand_fleet_buffers_batched`` for the live ``[N, F, ...]`` layout).
    """

    cur_F = int(turn_state_cache.fleets.shape[2])
    if cur_F >= new_F:
        return turn_state_cache
    pad = new_F - cur_F
    fleets = jnp.pad(turn_state_cache.fleets, ((0, 0), (0, 0), (0, pad), (0, 0)))
    fleet_active = jnp.pad(turn_state_cache.fleet_active, ((0, 0), (0, 0), (0, pad)))
    return turn_state_cache._replace(fleets=fleets, fleet_active=fleet_active)


@jax.jit
def _scatter_turn_start_state(
    turn_state_cache: OrbitWarsState,
    state_b: OrbitWarsState,
    turn_slot_per_env: jnp.ndarray,
) -> OrbitWarsState:
    """Write full ``state_b[n]`` into ``turn_state_cache[turn_slot_per_env[n], n]``."""

    n_idx = jnp.arange(turn_slot_per_env.shape[0], dtype=jnp.int32)

    def _scatter_leaf(cache_leaf: jnp.ndarray, live_leaf: jnp.ndarray) -> jnp.ndarray:
        return cache_leaf.at[turn_slot_per_env, n_idx].set(live_leaf)

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
    ``max_micro_steps_per_player``, plus scalar policy fields.  Full canonical
    ``OrbitWarsState`` at each env-turn start lives in ``turn_state_cache``.
    Per-row ``turn_tag_p0`` / ``turn_tag_p1`` tie buffer rows to a turn slot;
    PPO gather applies a prefix with ``apply_prefix_micro_deltas_batched``.

    Host arrays hold GAE inputs (rewards, dones, valid mask, old logprobs /
    values / bootstrap), all ``[H_buf, N]``.

    ``write_idx_pX[n]`` is the count of valid transitions env ``n`` has
    for player ``X`` in this segment. ``valid_pX[t, n]`` is ``True`` for
    ``t < write_idx_pX[n]`` and ``False`` elsewhere.
    """

    buf0: TransitionBuffer
    buf1: TransitionBuffer
    turn_state_cache: OrbitWarsState
    turn_tag_p0: jnp.ndarray
    turn_tag_p1: jnp.ndarray
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
    micro_cap_s: float = 0.0
    obs_build_s: float = 0.0
    policy_batch_s: float = 0.0
    policy_forward_s: float = 0.0
    micro_apply_s: float = 0.0
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


def _run_micro_phase(
    *,
    ego: int,
    state_b: OrbitWarsState,
    buf: TransitionBuffer,
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
    turn_tag_j: jnp.ndarray,
    turn_slot_np: np.ndarray,
) -> Tuple[TransitionBuffer, np.ndarray, List[List[Tuple[float, float, float]]], List[int], jnp.ndarray]:
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

    Returns ``(buf, new_write_idx_per_env, action_lists, reward_idx, turn_tag_j)``.
    ``reward_idx[n]`` is the row index of env ``n``'s last valid write in
    this phase (``-1`` if env ``n`` produced no rows).
    """

    num_envs = int(state_b.planets.shape[0])
    halted: List[bool] = [False] * num_envs
    action_lists: List[List[Tuple[float, float, float]]] = [[] for _ in range(num_envs)]
    reward_idx: List[int] = [-1] * num_envs
    write_idx = write_idx_per_env.copy()
    n_idx_full = np.arange(num_envs)

    virt_b = state_b
    ego_jax = jnp.int32(ego)
    fracs_t = torch.tensor(FRACTIONS, device=device, dtype=torch.float32)

    for k in range(max_micro_steps):
        if all(halted):
            break

        # ---- JAX-side compute: obs + pair geometry + must_halt. ----
        t0 = perf_counter()
        obs_jax = build_observation_batched_jax(virt_b, ego, ship_speed)
        pair_geom_j, etas_j = compute_pair_geom_and_etas(virt_b, ship_speed)
        must_halt_j = must_halt_no_owned_ships_batched(virt_b, ego_jax)
        timing.obs_build_s += perf_counter() - t0

        # ---- Hand to PyTorch (zero-copy via dlpack). ----
        t0 = perf_counter()
        obs_torch = obs_jax_to_torch(obs_jax)
        pair_geom_t = torch.from_dlpack(pair_geom_j)
        etas_t = torch.from_dlpack(etas_j)
        must_halt_t = torch.from_dlpack(must_halt_j)
        timing.policy_batch_s += perf_counter() - t0

        active_idx_list: List[int] = [i for i in range(num_envs) if not halted[i]]
        if not active_idx_list:
            break
        active_idx_t = torch.as_tensor(active_idx_list, device=device, dtype=torch.long)
        n_active = len(active_idx_list)

        active_obs = {key: v.index_select(0, active_idx_t) for key, v in obs_torch.items()}
        pair_geom_a = pair_geom_t.index_select(0, active_idx_t)
        etas_a = etas_t.index_select(0, active_idx_t)
        must_halt_a = must_halt_t.index_select(0, active_idx_t)

        # ---- Policy forward (active subset). ----
        t0 = perf_counter()
        out = policy(**active_obs)
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

        pair_mask_combined = out["pair_mask"] & pair_geom_a
        flat_mask = pair_mask_combined.flatten(start_dim=1)
        any_valid_pair = flat_mask.any(dim=-1)
        flat_logits = out["pair_logits"].flatten(start_dim=1)
        masked_pair = flat_logits.masked_fill(~flat_mask, -1e4)
        flat_lp = torch.log_softmax(masked_pair, dim=-1)
        safe_pair = torch.where(any_valid_pair[:, None], masked_pair, torch.zeros_like(masked_pair))
        if greedy:
            pair_flat = safe_pair.argmax(dim=-1)
        else:
            pair_probs = torch.softmax(safe_pair, dim=-1)
            pair_flat = torch.multinomial(pair_probs, 1, generator=rng).squeeze(-1)
        pair_logp = flat_lp.gather(1, pair_flat[:, None]).squeeze(-1)

        pair_used = (halt_action == 0) & any_valid_pair
        P = MAX_PLANETS
        o_idx = pair_flat // P
        d_idx = pair_flat % P

        planet_ships = active_obs["features"][:, 1:1 + P, 1] * 1000.0
        ships_avail = planet_ships.gather(1, o_idx[:, None]).squeeze(-1)
        sends = torch.floor(fracs_t[None, :] * ships_avail[:, None])
        frac_mask = sends >= 1.0
        any_valid_frac = frac_mask.any(dim=-1)

        n_a_idx = torch.arange(n_active, device=device)
        eta_chosen = etas_a[n_a_idx, o_idx, d_idx]
        times_norm = eta_chosen / 500.0

        ph = out["planet_hidden"]
        frac_logits = policy.fraction_logits(ph, o_idx, d_idx, times_norm)
        masked_frac = frac_logits.masked_fill(~frac_mask, -1e4)
        frac_lp = torch.log_softmax(masked_frac, dim=-1)
        safe_frac = torch.where(any_valid_frac[:, None], masked_frac, torch.zeros_like(masked_frac))
        if greedy:
            frac_idx = safe_frac.argmax(dim=-1)
        else:
            frac_probs = torch.softmax(safe_frac, dim=-1)
            frac_idx = torch.multinomial(frac_probs, 1, generator=rng).squeeze(-1)
        frac_logp = frac_lp.gather(1, frac_idx[:, None]).squeeze(-1)

        frac_used = pair_used & any_valid_frac
        total_logp = halt_logp + pair_used.float() * pair_logp + frac_used.float() * frac_logp
        # Cast the value head's output back to fp32 at the autocast boundary so
        # the per-env fp32 bookkeeping buffers (``values_all``, ``total_logp_all``)
        # stay consistent. ``log_softmax`` already promotes ``total_logp`` to fp32.
        values_active = out["value"].float()

        halt_now = ~frac_used  # active env halts iff it did not dispatch

        # ---- Build full-N tensors for buffer append + JAX apply. ----
        halt_now_all = torch.ones(num_envs, dtype=torch.bool, device=device)
        pair_flat_all = torch.zeros(num_envs, dtype=torch.int32, device=device)
        frac_idx_all = torch.zeros(num_envs, dtype=torch.int32, device=device)
        halt_action_all = torch.ones(num_envs, dtype=torch.int32, device=device)
        no_valid_pairs_all = torch.zeros(num_envs, dtype=torch.bool, device=device)
        no_valid_fracs_all = torch.zeros(num_envs, dtype=torch.bool, device=device)
        total_logp_all = torch.zeros(num_envs, dtype=torch.float32, device=device)
        values_all = torch.zeros(num_envs, dtype=torch.float32, device=device)
        halt_now_all.index_copy_(0, active_idx_t, halt_now)
        pair_flat_all.index_copy_(0, active_idx_t, pair_flat.to(torch.int32))
        frac_idx_all.index_copy_(0, active_idx_t, frac_idx.to(torch.int32))
        halt_action_all.index_copy_(0, active_idx_t, halt_action.to(torch.int32))
        no_valid_pairs_all.index_copy_(0, active_idx_t, ~any_valid_pair & (halt_action == 0))
        no_valid_fracs_all.index_copy_(0, active_idx_t, pair_used & ~any_valid_frac)
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
        halt_action_j = jax.dlpack.from_dlpack(halt_action_all.contiguous().detach())
        no_valid_pairs_j = jax.dlpack.from_dlpack(no_valid_pairs_all.contiguous().detach())
        no_valid_fracs_j = jax.dlpack.from_dlpack(no_valid_fracs_all.contiguous().detach())
        must_halt_no_ships_j = jax.dlpack.from_dlpack(must_halt_no_ships_all.contiguous().detach())

        virt_b, oid_j, angle_j, send_j, dispatched_j, slot_j = apply_micro_step_batched(
            virt_b, ego_jax, halt_now_j, pair_flat_j, frac_idx_j
        )

        active_j = jnp.asarray(np.array([not h for h in halted], dtype=np.bool_), dtype=jnp.bool_)
        micro_k_vec = jnp.full((num_envs,), k, dtype=jnp.int32)
        write_idx_j = jnp.asarray(write_idx, dtype=jnp.int32)
        buf = append_to_buffer(
            buf,
            halt_now_j,
            send_j,
            slot_j,
            halt_action_j,
            pair_flat_j,
            frac_idx_j,
            no_valid_pairs_j,
            no_valid_fracs_j,
            must_halt_no_ships_j,
            write_idx_j,
            micro_k_vec,
            active_j,
            max_micro_steps,
        )
        turn_slot_j = jnp.asarray(turn_slot_np, dtype=jnp.int32)
        turn_tag_j = scatter_turn_tags(turn_tag_j, write_idx_j, turn_slot_j)
        oid_np, angle_np, send_np, dispatched_np = jax.device_get(
            (oid_j, angle_j, send_j, dispatched_j)
        )
        timing.micro_apply_s += perf_counter() - t0

        # ---- Pull tiny scalar metadata for GAE / action_lists. ----
        total_logp_np = total_logp_all.detach().cpu().numpy()
        values_np = values_all.detach().cpu().numpy()
        halt_now_np = halt_now_all.detach().cpu().numpy()

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
                if bool(halt_now_np[i]):
                    halted[int(i)] = True
            write_idx[active_envs] += 1

    return buf, write_idx, action_lists, reward_idx, turn_tag_j


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

    # Allocate per-segment device buffers. ``H_buf`` covers the worst case
    # where the horizon trigger fires at the start of a turn but the final
    # turn still adds up to ``max_micro_steps_per_player`` more steps before
    # the segment cuts.
    H_buf = rollout_micro_horizon + max_micro_steps_per_player + 1
    buf0 = init_transition_buffer(num_envs, H_buf, max_micro_steps_per_player)
    buf1 = init_transition_buffer(num_envs, H_buf, max_micro_steps_per_player)

    turn_slot_np = np.zeros((num_envs,), dtype=np.int32)
    turn_state_cache = jax.tree.map(
        lambda leaf: jnp.zeros((_INITIAL_TURN_CACHE_ROWS,) + leaf.shape, dtype=leaf.dtype),
        state_b,
    )
    turn_state_cache = _scatter_turn_start_state(
        turn_state_cache, state_b, jnp.asarray(turn_slot_np, dtype=jnp.int32)
    )

    turn_tag_p0 = jnp.full((H_buf, num_envs), -1, dtype=jnp.int32)
    turn_tag_p1 = jnp.full((H_buf, num_envs), -1, dtype=jnp.int32)

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

    env_steps_per_env = np.zeros((num_envs,), dtype=np.int32)
    horizon_fired = False
    timing.init_s = perf_counter() - t_init0

    logged_first_policy_fwd = False
    outer = 0
    t_loop0 = perf_counter()
    peak_max_active = int(jax.device_get(max_concurrent_fleets_any_env(state_b)))

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if amp_dtype is not None
        else nullcontext()
    )

    with torch.inference_mode(), amp_ctx:
        while outer < max_outer_iters:
            outer += 1

            # ===== Phase A: ego=0 micro-loop =====
            buf0, write_idx_p0, actions0_per_env, p0_reward_idx, turn_tag_p0 = _run_micro_phase(
                ego=0,
                state_b=state_b,
                buf=buf0,
                write_idx_per_env=write_idx_p0,
                valid=valid_p0,
                old_logprob=old_logprob_p0,
                old_value=old_value_p0,
                policy=policy,
                device=device,
                rng=rng,
                greedy=greedy,
                ship_speed=ship_speed,
                max_micro_steps=max_micro_steps_per_player,
                timing=timing,
                turn_tag_j=turn_tag_p0,
                turn_slot_np=turn_slot_np,
            )
            if profile_rollout and device.type == "cuda" and not logged_first_policy_fwd:
                log_cuda_mem("rollout after first batched policy forward", device)
                logged_first_policy_fwd = True

            # ===== Phase B: ego=1 micro-loop =====
            buf1, write_idx_p1, actions1_per_env, p1_reward_idx, turn_tag_p1 = _run_micro_phase(
                ego=1,
                state_b=state_b,
                buf=buf1,
                write_idx_per_env=write_idx_p1,
                valid=valid_p1,
                old_logprob=old_logprob_p1,
                old_value=old_value_p1,
                policy=policy,
                device=device,
                rng=rng,
                greedy=greedy,
                ship_speed=ship_speed,
                max_micro_steps=max_micro_steps_per_player,
                timing=timing,
                turn_tag_j=turn_tag_p1,
                turn_slot_np=turn_slot_np,
            )

            # ===== Phase C: batched env step (vmap) =====
            t0 = perf_counter()
            actions_np = _build_batched_actions(actions0_per_env, actions1_per_env)

            need_host = upper_bound_fleet_writes_per_env(actions_np)
            inactive_host = np.asarray(jax.device_get(inactive_fleet_count_batched(state_b)))
            expansions = 0
            while np.any(need_host > inactive_host):
                expansions += 1
                if expansions > _MAX_FLEET_EXPAND_RETRIES:
                    raise RuntimeError(
                        "Orbit Wars fleet buffer expansions exhausted during pre-expand; "
                        "check action tensor shape or raise _MAX_FLEET_EXPAND_RETRIES."
                    )
                old_cap = int(state_b.fleets.shape[1])
                new_cap = old_cap * 2
                cfg.max_fleets = new_cap
                worst_env = int(np.argmax(need_host - inactive_host))
                print(
                    f"[orbit_wars_pt] max_fleets pre-expand {old_cap} -> {new_cap} "
                    f"(worst env {worst_env}: inactive {inactive_host[worst_env]} "
                    f"< launch upper bound {need_host[worst_env]}, expansion {expansions})",
                    flush=True,
                )
                state_b = expand_fleet_buffers_batched(state_b, new_cap)
                turn_state_cache = _expand_turn_state_cache_fleets(turn_state_cache, new_cap)
                inactive_host = np.asarray(jax.device_get(inactive_fleet_count_batched(state_b)))
                _reset_prefetch_resync(reset_prefetch, seed_base, seeds_consumed, cfg)

            r0_pre, r1_pre = ship_ratio_scores_batched(state_b)
            next_state_b = vmap_step_env(state_b, actions_np)

            overflow_any = bool(np.asarray(jax.device_get(jnp.any(next_state_b.overflow))))
            while overflow_any:
                expansions += 1
                if expansions > _MAX_FLEET_EXPAND_RETRIES:
                    raise RuntimeError(
                        "Orbit Wars overflow after expanding fleet buffers; "
                        "this may indicate planet-slot overflow rather than fleets."
                    )
                old_cap = int(state_b.fleets.shape[1])
                new_cap = old_cap * 2
                cfg.max_fleets = new_cap
                print(
                    f"[orbit_wars_pt] max_fleets replay (non-fleet overflow?): "
                    f"{old_cap} -> {new_cap} (expansion {expansions})",
                    flush=True,
                )
                state_b = expand_fleet_buffers_batched(state_b, new_cap)
                turn_state_cache = _expand_turn_state_cache_fleets(turn_state_cache, new_cap)
                next_state_b = vmap_step_env(state_b, actions_np)
                overflow_any = bool(np.asarray(jax.device_get(jnp.any(next_state_b.overflow))))
                _reset_prefetch_resync(reset_prefetch, seed_base, seeds_consumed, cfg)

            peak_max_active = max(
                peak_max_active,
                int(jax.device_get(max_concurrent_fleets_any_env(next_state_b))),
            )

            r0_post, r1_post = ship_ratio_scores_batched(next_state_b)
            s0_post, s1_post = ship_totals_batched(next_state_b)
            dr0_jax = r0_post - r0_pre
            dr1_jax = r1_post - r1_pre
            done_jax = next_state_b.done

            (
                dr0_np,
                dr1_np,
                done_np,
                step_count_np,
                rewards_np,
                s0_fin_np,
                s1_fin_np,
            ) = jax.device_get(
                (
                    dr0_jax,
                    dr1_jax,
                    done_jax,
                    next_state_b.step_count,
                    next_state_b.rewards,
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

            for i in range(num_envs):
                env_steps_per_env[i] += 1
                episode_turns[i] += 1
                if p0_reward_idx[i] >= 0:
                    reward_p0[p0_reward_idx[i], i] += float(dr0_np[i])
                    done_p0[p0_reward_idx[i], i] = bool(done_np[i]) or bool(done_p0[p0_reward_idx[i], i])
                if p1_reward_idx[i] >= 0:
                    reward_p1[p1_reward_idx[i], i] += float(dr1_np[i])
                    done_p1[p1_reward_idx[i], i] = bool(done_np[i]) or bool(done_p1[p1_reward_idx[i], i])
                if bool(done_np[i]):
                    sc_i = int(step_count_np[i])
                    step_limit_end = sc_i >= episode_timeout_step_count
                    rwi = rewards_np[i]
                    rw0 = float(rwi[0])
                    rw1 = float(rwi[1])
                    game_stats.record_completion(
                        step_limit=step_limit_end,
                        ships_p0=float(s0_fin_np[i]),
                        ships_p1=float(s1_fin_np[i]),
                        episode_turns=int(episode_turns[i]),
                        reward0=rw0,
                        reward1=rw1,
                    )
                    episode_turns[i] = 0

            state_b = next_state_b
            for i in range(num_envs):
                if not bool(done_np[i]):
                    continue
                sid = int(seed_base + seeds_consumed)
                if reset_prefetch is not None:
                    fresh_np = reset_prefetch.pop_state(
                        sid, int(cfg.num_agents), int(cfg.max_fleets)
                    )
                    state_b = reset_env_at_index(state_b, i, sid, cfg, fresh_np=fresh_np)
                else:
                    state_b = reset_env_at_index(state_b, i, sid, cfg)
                seeds_consumed += 1
            _reset_prefetch_resync(reset_prefetch, seed_base, seeds_consumed, cfg)
            turn_slot_np += 1
            max_turn = int(np.max(turn_slot_np))
            turn_state_cache = _expand_turn_state_cache_if_needed(turn_state_cache, max_turn)
            turn_state_cache = _scatter_turn_start_state(
                turn_state_cache, state_b, jnp.asarray(turn_slot_np, dtype=jnp.int32)
            )
            timing.env_step_s += perf_counter() - t0

            # Horizon triggers when any env's player-0 or player-1 count
            # crosses ``rollout_micro_horizon``. ``write_idx_p{0,1}`` is the
            # per-env transition count (number of valid rows written).
            if np.any(write_idx_p0 >= rollout_micro_horizon) or np.any(
                write_idx_p1 >= rollout_micro_horizon
            ):
                horizon_fired = True
                break

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
            out0 = policy(**obs_jax_to_torch(obs0_j))
        timing.policy_batch_s += perf_counter() - tb0
        tf0 = perf_counter()
        v0_np = out0["value"].float().detach().cpu().numpy()
        timing.policy_forward_s += perf_counter() - tf0

        t0 = perf_counter()
        obs1_j = build_observation_batched_jax(state_b, 1, ship_speed)
        timing.obs_build_s += perf_counter() - t0
        tb1 = perf_counter()
        with amp_ctx:
            out1 = policy(**obs_jax_to_torch(obs1_j))
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

    cap = int(state_b.fleets.shape[1])
    new_cap = cap // 2
    if (
        new_cap >= int(min_max_fleets)
        and peak_max_active * 2 < cap
        and new_cap < cap
    ):
        tail_inactive = bool(
            jax.device_get(jnp.logical_not(jnp.any(state_b.fleet_active[:, new_cap:])))
        )
        if tail_inactive:
            print(
                f"[orbit_wars_pt] max_fleets shrink {cap} -> {new_cap} "
                f"(peak concurrent fleets any env {peak_max_active} < cap/2, "
                f"floor min_max_fleets {int(min_max_fleets)})",
                flush=True,
            )
            state_b = shrink_fleet_buffers_batched(state_b, new_cap)
            cfg.max_fleets = new_cap
            _reset_prefetch_resync(reset_prefetch, seed_base, seeds_consumed, cfg)

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

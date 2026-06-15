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
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

import jax
import jax.numpy as jnp

from jax_orbit_wars import DEFAULT_MAX_ACTIONS, FLEET_ETA, FLEET_TARGET_PLANET, OrbitWarsState

from orbit_wars_pt.reset_prefetch import PrefetchPopMeta, RolloutResetPrefetch

from orbit_wars_pt.batched_env import (
    post_step_stats_batched,
    reset_env_at_index,
    reset_envs_from_bank_tail,
    reset_envs_at_indices,
    reset_envs_at_indices_batched,
    reward_delta_from_state_pair_batched,
    reward_mix_ratios_batched,
    stack_initial_states,
    step_env_batched,
    step_env_masked_batched,
)
from orbit_wars_pt.compressed_observation import (
    CompressedObservationBuffer,
    decode_observation,
    index_compressed_observation_rows,
    init_compressed_observation_buffer,
    store_precompressed_observation_rows,
)
from orbit_wars_pt.constants import (
    BOARD_SIZE,
    CENTER,
    FEATURE_DIM,
    FRACTIONS,
    MAX_PLANETS,
    SUN_RADIUS,
    obs_feature_dim_for_num_agents,
)
from orbit_wars_pt.env_wrapper import OrbitWarsEnvConfig
from orbit_wars_pt.exploiter_reset import (
    EXPLOITER_MODE_SELFPLAY_2P,
    EXPLOITER_MODE_SELFPLAY_4P,
    EXPLOITER_MODE_VS_2P,
    EXPLOITER_MODE_VS_4P,
    build_unified_exploiter_reset,
    sample_unified_exploiter_mode_layout,
    unified_exploiter_active_seat_count,
)
from orbit_wars_pt.gpu_mem import log_cuda_mem
from orbit_wars_pt.micro_jax import (
    apply_micro_step_batched_per_ego,
    must_halt_no_owned_ships_per_ego,
    selected_origin_fraction_targets_batched,
)
from orbit_wars_pt.model import OrbitWarsPolicy
from orbit_wars_pt.observation_jax import (
    _NORMALIZED_OWNER_SLOT_4P,
    build_compressed_observation_batched_jax,
    build_compressed_observation_batched_jax_per_ego,
)
from orbit_wars_pt.transition_buffer import (
    TorchTransitionBuffer,
    append_active_to_torch_buffer,
    init_torch_transition_buffer,
)

_MAX_FLEET_EXPAND_RETRIES = 48


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


def _sample_population_assignments_for_env(seed: int, num_agents: int, population_size: int) -> np.ndarray:
    if int(population_size) <= 1:
        return np.zeros((num_agents,), dtype=np.int32)
    rng = np.random.default_rng(np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15))
    return rng.integers(0, int(population_size), size=(num_agents,), dtype=np.int32)


def _population_rows_per_member(total_rows: int, population_size: int) -> int:
    pop = int(population_size)
    if pop <= 1:
        return int(total_rows)
    if int(total_rows) % pop != 0:
        raise ValueError(
            f"num_envs * num_agents must be divisible by population_size for grouped rollout batching "
            f"(got total_rows={int(total_rows)}, population_size={pop})"
        )
    return int(total_rows) // pop


def _population_assignments_from_policy_rows(
    policy_row_for_seat: np.ndarray,
    rows_per_member: int,
    population_size: int,
) -> np.ndarray:
    if int(population_size) <= 1:
        return np.zeros_like(policy_row_for_seat, dtype=np.int32)
    return (np.asarray(policy_row_for_seat, dtype=np.int32) // int(rows_per_member)).astype(np.int32)


def _init_policy_row_mapping(
    *,
    num_envs: int,
    num_agents: int,
    population_size: int,
    seed: int,
) -> np.ndarray:
    total_rows = int(num_envs) * int(num_agents)
    if int(population_size) <= 1:
        return np.arange(total_rows, dtype=np.int32).reshape(int(num_agents), int(num_envs))
    _population_rows_per_member(total_rows, int(population_size))
    rng = np.random.default_rng(np.uint64(seed) + np.uint64(0xD1B54A32D192ED03))
    rows = np.arange(total_rows, dtype=np.int32)
    rng.shuffle(rows)
    return rows.reshape(int(num_agents), int(num_envs))


def _invert_policy_row_mapping(policy_row_for_seat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mapping = np.asarray(policy_row_for_seat, dtype=np.int32)
    num_agents, num_envs = mapping.shape
    total_rows = int(num_agents) * int(num_envs)
    row_env = np.empty((total_rows,), dtype=np.int32)
    row_ego = np.empty((total_rows,), dtype=np.int32)
    for ego in range(num_agents):
        for env_i in range(num_envs):
            row = int(mapping[ego, env_i])
            row_env[row] = env_i
            row_ego[row] = ego
    return row_env, row_ego


def _reassign_policy_rows_for_reset_envs(
    *,
    policy_row_for_seat: np.ndarray,
    done_envs: np.ndarray,
    seed: int,
) -> np.ndarray:
    mapping = np.asarray(policy_row_for_seat, dtype=np.int32).copy()
    done_envs = np.asarray(done_envs, dtype=np.int32).reshape(-1)
    if done_envs.size == 0:
        return mapping
    released_rows = mapping[:, done_envs].reshape(-1).copy()
    rng = np.random.default_rng(np.uint64(seed) + np.uint64(0x94D049BB133111EB))
    rng.shuffle(released_rows)
    mapping[:, done_envs] = released_rows.reshape(mapping.shape[0], done_envs.size)
    return mapping


def _reward_coef_matrix_for_population(
    population_assignments: np.ndarray,
    global_coef: float,
    member_coefs: Optional[list[float]],
    num_agents: int,
) -> np.ndarray:
    env_count = int(population_assignments.shape[1])
    out = np.zeros((env_count, 4), dtype=np.float32)
    out[:, :num_agents] = float(global_coef)
    if not member_coefs:
        return out
    member_arr = np.asarray(member_coefs, dtype=np.float32).reshape(-1)
    for ego in range(int(num_agents)):
        pop = np.asarray(population_assignments[ego], dtype=np.int32)
        valid = (pop >= 0) & (pop < int(member_arr.shape[0]))
        if not np.any(valid):
            continue
        out[valid, ego] = member_arr[pop[valid]]
    return out


import jax_orbit_wars as jow


def _sample_controller_assignments_for_env(seed: int, num_agents: int, controller_counts: tuple[int, ...]) -> np.ndarray:
    counts = tuple(int(c) for c in controller_counts)
    if sum(counts) != int(num_agents):
        raise ValueError(f"controller_counts {counts} must sum to num_agents={int(num_agents)}")
    if len(counts) <= 1:
        return np.zeros((num_agents,), dtype=np.int32)
    seats = np.arange(int(num_agents), dtype=np.int32)
    rng = np.random.default_rng(np.uint64(seed) + np.uint64(0xA0761D6478BD642F))
    rng.shuffle(seats)
    out = np.zeros((int(num_agents),), dtype=np.int32)
    start = 0
    for controller_id, count in enumerate(counts):
        stop = start + int(count)
        if stop > start:
            out[seats[start:stop]] = int(controller_id)
        start = stop
    return out


def _sample_main_player_mask_for_env(
    controller_assignments: np.ndarray,
    termination_controller: Optional[int],
) -> np.ndarray:
    out = np.zeros_like(np.asarray(controller_assignments, dtype=np.bool_), dtype=np.bool_)
    if termination_controller is None:
        return out
    out = np.asarray(controller_assignments, dtype=np.int32) == int(termination_controller)
    return out.astype(np.bool_)


def _broadcast_controller_assignment_template(
    template: np.ndarray,
    *,
    num_envs: int,
    num_agents: int,
) -> np.ndarray:
    templ = np.asarray(template, dtype=np.int32)
    if templ.ndim == 1:
        if templ.shape[0] != int(num_agents):
            raise ValueError(
                f"controller_assignment_template length {templ.shape[0]} != num_agents={int(num_agents)}"
            )
        return np.broadcast_to(templ[:, None], (int(num_agents), int(num_envs))).astype(np.int32, copy=True)
    if templ.ndim == 2 and templ.shape == (int(num_agents), int(num_envs)):
        return templ.astype(np.int32, copy=True)
    raise ValueError(
        "controller_assignment_template must have shape "
        f"({int(num_agents)},) or ({int(num_agents)}, {int(num_envs)}), got {templ.shape}"
    )


def _broadcast_main_player_mask_template(
    template: np.ndarray,
    *,
    num_envs: int,
    num_agents: int,
) -> np.ndarray:
    templ = np.asarray(template, dtype=np.bool_)
    if templ.ndim == 1:
        if templ.shape[0] != int(num_agents):
            raise ValueError(
                f"main_player_mask_template length {templ.shape[0]} != num_agents={int(num_agents)}"
            )
        return np.broadcast_to(templ[:, None], (int(num_agents), int(num_envs))).astype(np.bool_, copy=True)
    if templ.ndim == 2 and templ.shape == (int(num_agents), int(num_envs)):
        return templ.astype(np.bool_, copy=True)
    raise ValueError(
        "main_player_mask_template must have shape "
        f"({int(num_agents)},) or ({int(num_agents)}, {int(num_envs)}), got {templ.shape}"
    )


def _relative_main_value_head_idx_for_rows(
    *,
    ego_idx: np.ndarray,
    env_idx: np.ndarray,
    controller_assignments: Optional[np.ndarray],
    main_player_mask: Optional[np.ndarray],
    normalize_obs_to_p0: bool,
) -> np.ndarray:
    """Critic-only egocentric main-opponent index: 0=p1, 1=p2, 2=p3."""

    ego_np = np.asarray(ego_idx, dtype=np.int32).reshape(-1)
    env_np = np.asarray(env_idx, dtype=np.int32).reshape(-1)
    if ego_np.shape != env_np.shape:
        raise ValueError("ego_idx and env_idx must have matching shape")
    out = np.zeros_like(ego_np, dtype=np.int32)
    if controller_assignments is None or main_player_mask is None:
        return out
    norm_table = np.asarray(jax.device_get(_NORMALIZED_OWNER_SLOT_4P), dtype=np.int32)
    for i, (ego, env_i) in enumerate(zip(ego_np.tolist(), env_np.tolist())):
        ctrl_col = np.asarray(controller_assignments[:, env_i], dtype=np.int32)
        if ego < 0 or ego >= ctrl_col.shape[0] or int(ctrl_col[ego]) != 1:
            continue
        if int(np.count_nonzero(ctrl_col >= 0)) <= 2:
            out[i] = 0
            continue
        main_slots = np.flatnonzero(np.asarray(main_player_mask[:, env_i], dtype=np.bool_))
        if main_slots.size != 1:
            continue
        main_abs = int(main_slots[0])
        if bool(normalize_obs_to_p0):
            out[i] = int(norm_table[int(np.clip(ego, 0, 3)), int(np.clip(main_abs, 0, 3))] - 2)
        else:
            out[i] = int(main_abs if main_abs < ego else main_abs - 1)
    return np.clip(out, 0, 2).astype(np.int32, copy=False)


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


@jax.jit
def _copy_state_slices_between(
    dst_b: OrbitWarsState,
    dst_idx: jnp.ndarray,
    src_b: OrbitWarsState,
    src_idx: jnp.ndarray,
) -> OrbitWarsState:
    return jax.tree.map(lambda dst, src: dst.at[dst_idx].set(src[src_idx]), dst_b, src_b)


@jax.jit
def _refresh_virt_from_state_masked_non_grouped(
    virt_b: OrbitWarsState,
    state_b: OrbitWarsState,
    ready_mask: jnp.ndarray,
) -> OrbitWarsState:
    """Refresh only stepped env rows in non-grouped ``virt_b`` from ``state_b``."""

    num_rows = int(virt_b.planets.shape[0])
    num_envs = int(state_b.planets.shape[0])
    if num_envs <= 0:
        return virt_b
    row_env = jnp.mod(jnp.arange(num_rows, dtype=jnp.int32), jnp.asarray(num_envs, dtype=jnp.int32))

    def _blend(virt_leaf: jnp.ndarray, state_leaf: jnp.ndarray) -> jnp.ndarray:
        refreshed = state_leaf[row_env]
        mask = ready_mask[row_env]
        for _ in range(virt_leaf.ndim - 1):
            mask = mask[..., None]
        return jnp.where(mask, refreshed, virt_leaf)

    return jax.tree.map(_blend, virt_b, state_b)


@jax.jit
def _refresh_virt_from_state_masked_grouped(
    virt_b: OrbitWarsState,
    state_b: OrbitWarsState,
    row_env: jnp.ndarray,
    ready_mask: jnp.ndarray,
) -> OrbitWarsState:
    """Refresh only stepped rows in grouped-population ``virt_b`` from ``state_b``."""

    def _blend(virt_leaf: jnp.ndarray, state_leaf: jnp.ndarray) -> jnp.ndarray:
        refreshed = state_leaf[row_env]
        mask = ready_mask[row_env]
        for _ in range(virt_leaf.ndim - 1):
            mask = mask[..., None]
        return jnp.where(mask, refreshed, virt_leaf)

    return jax.tree.map(_blend, virt_b, state_b)


@jax.jit
def _gather_state_rows(
    state_b: OrbitWarsState,
    row_env_idx: jnp.ndarray,
) -> OrbitWarsState:
    return jax.tree.map(lambda leaf: leaf[row_env_idx], state_b)


@dataclass
class RolloutCarry:
    """Batched env state carried across PPO iterations (continues unfinished games)."""

    state_b: OrbitWarsState
    cfg: OrbitWarsEnvConfig
    #: Env-turn counter for the current episode per env (for stats across rollout segments).
    episode_turns: List[int]
    #: Per-player local terminal flags for unfinished 4p games. The env may continue
    #: after a player is eliminated; that player's PPO stream should not.
    player_done: Optional[np.ndarray] = None
    #: Active population member per seat/env for the ongoing episode.
    population_assignments: Optional[np.ndarray] = None
    #: Persistent policy-batch row owned by each seat/env for grouped population rollout.
    policy_row_for_seat: Optional[np.ndarray] = None
    #: Controller/policy owner per seat/env for the ongoing episode.
    controller_assignments: Optional[np.ndarray] = None
    #: Marks the seat whose elimination ends the episode early in versus mode.
    main_player_mask: Optional[np.ndarray] = None
    #: Optional per-env matchup mode code used by unified exploiter-mode rollouts.
    env_mode_by_env: Optional[np.ndarray] = None
    #: Mixed-mode exploiters that died before the env resolved and still need
    #: a deferred terminal team-outcome row appended when the episode ends.
    pending_exploiter_terminal: Optional[np.ndarray] = None


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
    member_episode_count: Optional[np.ndarray] = None
    member_positive_reward_count: Optional[np.ndarray] = None
    main_vs_exploiter_games: int = 0
    main_vs_exploiter_wins: int = 0
    main_vs_exploiter_games_2p: int = 0
    main_vs_exploiter_wins_2p: int = 0
    main_vs_exploiter_games_4p: int = 0
    main_vs_exploiter_wins_4p: int = 0
    main_vs_exploiter_sum_episode_turns_2p: float = 0.0
    main_vs_exploiter_sum_episode_turns_4p: float = 0.0
    main_vs_exploiter_timeout_2p: int = 0
    main_vs_exploiter_timeout_4p: int = 0
    main_vs_exploiter_main_eliminated_2p: int = 0
    main_vs_exploiter_main_eliminated_4p: int = 0

    def record_completion(
        self,
        *,
        step_limit: bool,
        ships_p0: float,
        ships_p1: float,
        episode_turns: int,
        reward0: float,
        reward1: float,
        reward_by_player: Optional[np.ndarray] = None,
        population_assignment: Optional[np.ndarray] = None,
        main_policy_win: Optional[bool] = None,
        env_mode: Optional[int] = None,
        main_eliminated: bool = False,
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
        if main_policy_win is not None:
            self.main_vs_exploiter_games += 1
            if bool(main_policy_win):
                self.main_vs_exploiter_wins += 1
            if env_mode == EXPLOITER_MODE_VS_2P:
                self.main_vs_exploiter_games_2p += 1
                self.main_vs_exploiter_sum_episode_turns_2p += float(episode_turns)
                if bool(main_policy_win):
                    self.main_vs_exploiter_wins_2p += 1
                if step_limit:
                    self.main_vs_exploiter_timeout_2p += 1
                if bool(main_eliminated):
                    self.main_vs_exploiter_main_eliminated_2p += 1
            elif env_mode == EXPLOITER_MODE_VS_4P:
                self.main_vs_exploiter_games_4p += 1
                self.main_vs_exploiter_sum_episode_turns_4p += float(episode_turns)
                if bool(main_policy_win):
                    self.main_vs_exploiter_wins_4p += 1
                if step_limit:
                    self.main_vs_exploiter_timeout_4p += 1
                if bool(main_eliminated):
                    self.main_vs_exploiter_main_eliminated_4p += 1
        if (
            reward_by_player is not None
            and population_assignment is not None
            and self.member_episode_count is not None
            and self.member_positive_reward_count is not None
        ):
            rewards_np = np.asarray(reward_by_player, dtype=np.float32).reshape(-1)
            pop_np = np.asarray(population_assignment, dtype=np.int32).reshape(-1)
            if rewards_np.shape == pop_np.shape:
                for player_i, member_i in enumerate(pop_np):
                    if 0 <= int(member_i) < int(self.member_episode_count.shape[0]):
                        self.member_episode_count[int(member_i)] += 1
                        if float(rewards_np[player_i]) > 0.0:
                            self.member_positive_reward_count[int(member_i)] += 1


@dataclass
class RolloutSegment:
    """One rollout segment's data, mostly on device.

    Parallel lists (length = ``num_agents`` in the env) hold per-player rollout
    mirrors: transition buffers, compressed observations, and host GAE metadata.
    ``bufs[p]`` / ``obs_bufs[p]`` are player ``p``'s egocentric stream;
    ``write_idx[p][n]`` counts valid transitions for env ``n`` in this segment.
    """

    bufs: List[TorchTransitionBuffer]
    obs_bufs: List[CompressedObservationBuffer]
    write_idx: List[np.ndarray]
    valid: List[np.ndarray]
    old_logprob: List[np.ndarray]
    old_value: List[np.ndarray]
    reward: List[np.ndarray]
    done: List[np.ndarray]
    bootstrap: List[np.ndarray]
    bootstrap_valid: List[np.ndarray]
    trunc_bootstrap: List[np.ndarray]
    trunc_bootstrap_valid: List[np.ndarray]
    env_steps_per_env: np.ndarray
    #: Optional per-env matchup mode code aligned with env index ``n``.
    env_mode_by_env: Optional[np.ndarray] = None
    #: Host-only sidecar for the consistency check: ``(env_i, new_seed,
    #: write_idx_at_reset_per_seat)`` of the first env that resets during this
    #: segment, or ``None`` if no env reset. ``new_seed`` is the seed used to
    #: reset ``env_i``; ``write_idx_at_reset_per_seat[p]`` is the buffer row
    #: index where the freshly-reset game's first row lands for seat ``p``.
    first_reset_event: Optional[Tuple[int, int, np.ndarray]] = None


@dataclass
class RolloutTiming:
    """Wall and per-phase times inside `collect_parallel_micro_rollouts` (perf_counter, host-side)."""

    init_s: float = 0.0
    init_reset_bank_s: float = 0.0
    init_reset_host_ready_n: int = 0
    init_reset_host_pending_n: int = 0
    init_reset_bank_drain_s: float = 0.0
    init_reset_bank_ready_pop_s: float = 0.0
    init_reset_bank_wait_s: float = 0.0
    init_reset_bank_stack_s: float = 0.0
    init_reset_bank_append_s: float = 0.0
    init_reset_bank_submit_s: float = 0.0
    init_buffer_alloc_s: float = 0.0
    init_state_setup_s: float = 0.0
    env_step_s: float = 0.0
    env_prep_s: float = 0.0
    env_state_gather_s: float = 0.0
    env_coef_s: float = 0.0
    env_step_core_s: float = 0.0
    env_reward_s: float = 0.0
    env_post_stats_s: float = 0.0
    env_host_transfer_s: float = 0.0
    env_reset_s: float = 0.0
    env_reset_bank_slice_s: float = 0.0
    env_reset_host_resolve_s: float = 0.0
    env_reset_host_stack_s: float = 0.0
    env_reset_concat_s: float = 0.0
    env_reset_apply_s: float = 0.0
    env_reset_fallback_host_s: float = 0.0
    env_reset_count: int = 0
    env_reset_mode_2p_count: int = 0
    env_reset_mode_4p_count: int = 0
    env_bookkeeping_s: float = 0.0
    env_state_scatter_s: float = 0.0
    env_python_s: float = 0.0
    reset_prefetch_pop_s: float = 0.0
    reset_prefetch_pop_init_s: float = 0.0
    reset_prefetch_pop_episode_s: float = 0.0
    reset_prefetch_bank_hit_n: int = 0
    reset_prefetch_wait_n: int = 0
    reset_prefetch_fallback_n: int = 0
    reset_prefetch_drained_results: int = 0
    reset_prefetch_banked_other_results: int = 0
    reset_prefetch_mode_2p_n: int = 0
    reset_prefetch_mode_4p_n: int = 0
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
    micro_apply_obs_store_s: float = 0.0
    micro_apply_numpy_s: float = 0.0
    micro_prep_non_grouped_s: float = 0.0
    micro_prep_grouped_s: float = 0.0
    micro_post_apply_extract_s: float = 0.0
    micro_post_apply_host_bookkeeping_s: float = 0.0
    micro_post_apply_row_stats_s: float = 0.0
    micro_post_apply_pending_actions_s: float = 0.0
    micro_post_apply_halt_block_indices_s: float = 0.0
    micro_post_apply_device_index_s: float = 0.0
    bootstrap_obs_build_s: float = 0.0
    bootstrap_policy_batch_s: float = 0.0
    bootstrap_policy_forward_s: float = 0.0
    loop_control_s: float = 0.0
    loop_post_micro_s: float = 0.0
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
            + self.micro_prep_non_grouped_s
            + self.micro_prep_grouped_s
            + self.micro_post_apply_extract_s
            + self.micro_post_apply_host_bookkeeping_s
            + self.micro_post_apply_device_index_s
            + self.loop_control_s
            + self.loop_post_micro_s
            + self.state_unstack_s
        )


@dataclass
class _DeviceResetBank:
    """A fixed-capacity device-resident reset bank."""

    capacity: int
    num_agents: int = -1
    max_fleets: int = -1
    states: Optional[OrbitWarsState] = None
    seeds: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))
    size_used: int = 0
    next_submit_seed: Optional[int] = None

    def clear(self) -> None:
        self.states = None
        self.seeds = np.zeros((0,), dtype=np.int64)
        self.size_used = 0
        self.next_submit_seed = None

    def prepare(self, *, num_agents: int, max_fleets: int, first_seed: Optional[int] = None) -> None:
        if int(num_agents) != self.num_agents or int(max_fleets) != self.max_fleets:
            self.num_agents = int(num_agents)
            self.max_fleets = int(max_fleets)
            self.clear()
        if self.next_submit_seed is None and first_seed is not None:
            self.next_submit_seed = int(first_seed)

    def pop_any(self) -> Optional[tuple[int, OrbitWarsState]]:
        if self.states is None or self.size_used <= 0:
            return None
        idx = int(self.size_used) - 1
        seed = int(self.seeds[idx])
        state = jax.tree.map(lambda leaf: leaf[idx], self.states)
        self.size_used = idx
        return seed, state

    def size(self) -> int:
        return int(self.size_used)

    def pop_many(self, count: int) -> Optional[tuple[list[int], OrbitWarsState]]:
        take = min(max(0, int(count)), int(self.size_used))
        if self.states is None or take <= 0:
            return None
        start = int(self.size_used) - take
        stop = int(self.size_used)
        seeds = self.seeds[start:stop].astype(np.int64, copy=True).tolist()
        states_b = jax.tree.map(
            lambda leaf: jax.lax.dynamic_slice_in_dim(leaf, start, take, axis=0),
            self.states,
        )
        self.size_used = start
        return seeds, states_b

    def reserve_tail(self, count: int) -> Optional[tuple[list[int], int, int]]:
        take = min(max(0, int(count)), int(self.size_used))
        if self.states is None or take <= 0:
            return None
        start = int(self.size_used) - take
        stop = int(self.size_used)
        seeds = self.seeds[start:stop].astype(np.int64, copy=True).tolist()
        self.size_used = start
        return seeds, start, take

    def append_batch(self, seeds: list[int], states_b: OrbitWarsState) -> None:
        n = len(seeds)
        if n <= 0:
            return
        if self.states is None:
            self.states = jax.tree.map(
                lambda leaf: jnp.zeros((int(self.capacity),) + tuple(leaf.shape[1:]), dtype=leaf.dtype),
                states_b,
            )
            self.seeds = np.full((int(self.capacity),), -1, dtype=np.int64)
            self.size_used = 0
        start = int(self.size_used)
        stop = start + int(n)
        if stop > int(self.capacity):
            raise RuntimeError(
                f"device reset bank overflow: stop={stop} capacity={int(self.capacity)}"
            )
        self.states = jax.tree.map(
            lambda dst, src: jax.lax.dynamic_update_slice(
                dst,
                src,
                (start,) + (0,) * (dst.ndim - 1),
            ),
            self.states,
            states_b,
        )
        self.seeds[start:stop] = np.asarray(seeds, dtype=np.int64)
        self.size_used = stop

    def append_padded_tail(self, seeds: list[int], states_full: OrbitWarsState, tail_count: int) -> None:
        n = int(tail_count)
        if n <= 0:
            return
        if self.states is None:
            self.states = jax.tree.map(
                lambda leaf: jnp.zeros((int(self.capacity),) + tuple(leaf.shape[1:]), dtype=leaf.dtype),
                states_full,
            )
            self.seeds = np.full((int(self.capacity),), -1, dtype=np.int64)
            self.size_used = 0
        start = int(self.size_used)
        stop = start + n
        if stop > int(self.capacity):
            raise RuntimeError(
                f"device reset bank overflow: stop={stop} capacity={int(self.capacity)}"
            )
        self.states = jax.tree.map(
            lambda dst, src: _append_padded_tail_leaf(dst, src, start, n),
            self.states,
            states_full,
        )
        self.seeds[start:stop] = np.asarray(seeds, dtype=np.int64)
        self.size_used = stop


def make_device_reset_bank(capacity: int) -> _DeviceResetBank:
    return _DeviceResetBank(capacity=int(capacity))


@jax.jit
def _append_padded_tail_leaf(
    dst: jnp.ndarray,
    src_full: jnp.ndarray,
    start: int,
    tail_count: int,
) -> jnp.ndarray:
    cap = int(dst.shape[0])
    pos = jnp.arange(cap, dtype=jnp.int32)
    stop = start + tail_count
    mask = (pos >= start) & (pos < stop)
    src_idx = pos - start + (cap - tail_count)
    src_idx = jnp.clip(src_idx, 0, cap - 1)
    src_rows = jnp.take(src_full, src_idx, axis=0)
    mask_shape = (cap,) + (1,) * (dst.ndim - 1)
    return jnp.where(mask.reshape(mask_shape), src_rows, dst)


def _stack_padded_tail_numpy(xs: tuple[Any, ...], capacity: int) -> np.ndarray:
    arr = np.stack([np.asarray(x) for x in xs], axis=0)
    out = np.zeros((int(capacity),) + tuple(arr.shape[1:]), dtype=arr.dtype)
    out[int(capacity) - int(arr.shape[0]) :] = arr
    return out


def _stage_plain_reset_bank(
    device_reset_bank: Optional[_DeviceResetBank],
    reset_prefetch: Optional[RolloutResetPrefetch],
    *,
    first_seed: int,
    num_agents: int,
    max_fleets: int,
    target_size: Optional[int] = None,
    block_until_full: bool = False,
    timing: Optional[RolloutTiming] = None,
    sync_policy_timing: bool = False,
    device: Optional[torch.device] = None,
) -> int:
    """Upload ready plain reset states from the CPU prefetch bank into a bounded device pool."""

    if device_reset_bank is None or reset_prefetch is None:
        return 0
    reset_prefetch.notify_max_fleets(int(max_fleets))
    device_reset_bank.prepare(
        num_agents=int(num_agents),
        max_fleets=int(max_fleets),
        first_seed=int(first_seed),
    )
    if device_reset_bank.next_submit_seed is not None and device_reset_bank.size() <= 0:
        t_submit0 = perf_counter()
        reset_prefetch.submit_plain_range(
            int(device_reset_bank.next_submit_seed),
            int(device_reset_bank.capacity),
            int(num_agents),
            int(max_fleets),
        )
        if timing is not None:
            timing.init_reset_bank_submit_s += perf_counter() - t_submit0
        device_reset_bank.next_submit_seed += int(device_reset_bank.capacity)
    t_drain0 = perf_counter()
    reset_prefetch.drain_ready()
    if timing is not None:
        timing.init_reset_bank_drain_s += perf_counter() - t_drain0
        timing.init_reset_host_ready_n = reset_prefetch.ready_banked_count(
            int(num_agents),
            int(max_fleets),
        )
        timing.init_reset_host_pending_n = reset_prefetch.outstanding_count(
            int(num_agents),
            int(max_fleets),
        )
    target = int(device_reset_bank.capacity) if target_size is None else max(0, int(target_size))
    target = min(target, int(device_reset_bank.capacity))
    staged = 0
    fill_t0 = perf_counter()
    last_fill_log_t = fill_t0

    def _log_fill_progress(*, force: bool = False) -> None:
        nonlocal last_fill_log_t
        if target <= 0:
            return
        now = perf_counter()
        if not force and (now - last_fill_log_t) < 1.0:
            return
        done = int(device_reset_bank.size())
        remaining = max(0, target - done)
        elapsed = max(now - fill_t0, 1e-9)
        rate = float(done) / elapsed
        print(
            f"[orbit_wars_pt] initial reset prefill {done}/{target} done "
            f"remaining {remaining} rate {rate:.2f}/s",
            flush=True,
        )
        last_fill_log_t = now

    if target > 0:
        print(f"[orbit_wars_pt] initial reset prefill starting target={target}", flush=True)
    while device_reset_bank.size() < target:
        host_items: list[tuple[int, Any]] = []
        target_n = target - int(device_reset_bank.size())
        while len(host_items) < target_n:
            t_pop0 = perf_counter()
            item = reset_prefetch.pop_any_banked_state(
                int(num_agents),
                int(max_fleets),
            )
            if timing is not None:
                timing.init_reset_bank_ready_pop_s += perf_counter() - t_pop0
            if item is None:
                if not block_until_full:
                    break
                t_wait0 = perf_counter()
                item = reset_prefetch.wait_any_banked_state(
                    int(num_agents),
                    int(max_fleets),
                )
                if timing is not None:
                    timing.init_reset_bank_wait_s += perf_counter() - t_wait0
                if item is None:
                    break
            host_items.append(item)
        if not host_items:
            _log_fill_progress()
            break
        seeds = [int(seed) for seed, _ in host_items]
        if sync_policy_timing and device is not None:
            _sync_rollout_policy_timing(device)
        t_stack0 = perf_counter()
        fresh_b = jax.tree.map(
            lambda *xs: jnp.asarray(_stack_padded_tail_numpy(xs, int(device_reset_bank.capacity))),
            *[fresh_np for _, fresh_np in host_items],
        )
        if sync_policy_timing and device is not None:
            _sync_rollout_policy_timing(device, fresh_b)
        if timing is not None:
            timing.init_reset_bank_stack_s += perf_counter() - t_stack0
        if sync_policy_timing and device is not None:
            _sync_rollout_policy_timing(device)
        t_append0 = perf_counter()
        device_reset_bank.append_padded_tail(seeds, fresh_b, len(host_items))
        if sync_policy_timing and device is not None and device_reset_bank.states is not None:
            _sync_rollout_policy_timing(device, device_reset_bank.states)
        if timing is not None:
            timing.init_reset_bank_append_s += perf_counter() - t_append0
        if device_reset_bank.next_submit_seed is not None:
            t_submit0 = perf_counter()
            reset_prefetch.submit_plain_range(
                int(device_reset_bank.next_submit_seed),
                len(host_items),
                int(num_agents),
                int(max_fleets),
            )
            if timing is not None:
                timing.init_reset_bank_submit_s += perf_counter() - t_submit0
            device_reset_bank.next_submit_seed += len(host_items)
        staged += len(host_items)
        _log_fill_progress()
    if target > 0:
        _log_fill_progress(force=True)
    return staged


def _resolve_plain_reset_host_batch(
    *,
    reset_count: int,
    fallback_seeds: list[int],
    reset_prefetch: Optional[RolloutResetPrefetch],
    num_agents: int,
    max_fleets: int,
    timing: RolloutTiming,
    sync_policy_timing: bool,
    device: torch.device,
) -> tuple[list[int], OrbitWarsState, list[PrefetchPopMeta]]:
    """Return host/fallback plain reset states as one batched pytree, plus realized seeds/meta."""

    if reset_count <= 0:
        raise ValueError("reset_count must be positive")
    seeds_out: list[int] = []
    metas: list[PrefetchPopMeta] = []
    host_items: list[tuple[int, Any]] = []
    for i in range(int(reset_count)):
        fallback_seed = int(fallback_seeds[i])
        if sync_policy_timing:
            _sync_rollout_policy_timing(device)
        t0 = perf_counter()
        if reset_prefetch is not None:
            sid, fresh_np, meta = reset_prefetch.pop_any_state(
                int(num_agents),
                int(max_fleets),
                fallback_seed=fallback_seed,
                return_meta=True,
            )
        else:
            sid = int(fallback_seed)
            state_i = jow.reset_from_reference(sid, int(num_agents), max_fleets=int(max_fleets))
            fresh_np = jax.device_get(state_i)
            meta = PrefetchPopMeta()
            meta.fallback_used = True
        if sync_policy_timing:
            _sync_rollout_policy_timing(device)
        dt = perf_counter() - t0
        timing.env_reset_host_resolve_s += dt
        if meta.fallback_used:
            timing.env_reset_fallback_host_s += dt
        seeds_out.append(int(sid))
        metas.append(meta)
        host_items.append((int(sid), fresh_np))
    if sync_policy_timing:
        _sync_rollout_policy_timing(device)
    t0 = perf_counter()
    host_states = jax.tree.map(
        lambda *xs: jnp.stack([jnp.asarray(x) for x in xs], axis=0),
        *[fresh_np for _, fresh_np in host_items],
    )
    if sync_policy_timing:
        _sync_rollout_policy_timing(device, host_states)
    timing.env_reset_host_stack_s += perf_counter() - t0
    return seeds_out, host_states, metas


def _accum_prefetch_pop_meta(
    timing: RolloutTiming,
    meta: PrefetchPopMeta,
    *,
    init_phase: bool,
    active_seat_count: Optional[int],
) -> None:
    timing.reset_prefetch_pop_s += float(meta.wait_s)
    if init_phase:
        timing.reset_prefetch_pop_init_s += float(meta.wait_s)
    else:
        timing.reset_prefetch_pop_episode_s += float(meta.wait_s)
    if bool(meta.immediate_bank_hit):
        timing.reset_prefetch_bank_hit_n += 1
    else:
        timing.reset_prefetch_wait_n += 1
    if bool(meta.fallback_used):
        timing.reset_prefetch_fallback_n += 1
    timing.reset_prefetch_drained_results += int(meta.drained_results)
    timing.reset_prefetch_banked_other_results += int(meta.banked_other_results)
    if active_seat_count == 2:
        timing.reset_prefetch_mode_2p_n += 1
    elif active_seat_count == 4:
        timing.reset_prefetch_mode_4p_n += 1


def _normalize_unified_exploiter_seed_state(seed_base: Any) -> dict[str, int]:
    if isinstance(seed_base, dict):
        if "two_p" in seed_base and "four_p" in seed_base:
            return {
                "two_p": int(seed_base["two_p"]),
                "four_p": int(seed_base["four_p"]),
            }
        if "2p" in seed_base and "4p" in seed_base:
            return {
                "two_p": int(seed_base["2p"]),
                "four_p": int(seed_base["4p"]),
            }
    base = int(seed_base)
    return {"two_p": base, "four_p": base}


def _take_unified_exploiter_seed(seed_state: dict[str, int], active_seat_count: int) -> int:
    if int(active_seat_count) == 2:
        logical = int(seed_state["two_p"])
        seed_state["two_p"] = logical + 1
        return 2 * logical
    if int(active_seat_count) == 4:
        logical = int(seed_state["four_p"])
        seed_state["four_p"] = logical + 1
        return 2 * logical + 1
    raise ValueError(f"unsupported active_seat_count {int(active_seat_count)}")


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
    of ``append_to_torch_buffer``. ``t6`` end of compressed observation store. ``t7`` end
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
    timing.micro_apply_obs_store_s += t6 - t5
    timing.micro_apply_numpy_s += t7 - t6


def _sync_rollout_policy_timing(device: torch.device, *jax_values: Any) -> None:
    """Fence async accelerator work for diagnostic rollout subphase timings."""

    for value in jax_values:
        if value is not None:
            jax.block_until_ready(value)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _compressed_obs_jax_to_torch(obs_jax: dict[str, jnp.ndarray]) -> CompressedObservationBuffer:
    """Zero-copy JAX -> Torch handoff for compressed observation planes."""

    production = torch.from_dlpack(obs_jax["production"])
    return CompressedObservationBuffer(
        token_meta=torch.from_dlpack(obs_jax["token_meta"]),
        owner_idx=torch.from_dlpack(obs_jax["owner_idx"]),
        production=production,
        ships=torch.from_dlpack(obs_jax["ships"]),
        velocity=torch.from_dlpack(obs_jax["velocity"]),
        xy=torch.from_dlpack(obs_jax["xy"]),
        turn_progress=torch.from_dlpack(obs_jax["turn_progress"]),
        incoming_net=torch.from_dlpack(obs_jax["incoming_net"]),
        incoming_survivor=torch.from_dlpack(obs_jax["incoming_survivor"]),
        origin_frac_blocked=(
            torch.from_dlpack(obs_jax["origin_frac_blocked"])
            if "origin_frac_blocked" in obs_jax
            else torch.zeros(
                obs_jax["production"].shape[:-1] + (MAX_PLANETS, len(FRACTIONS)),
                dtype=torch.bool,
                device=production.device,
            )
        ),
    )


def _compressed_obs_to_device(comp: CompressedObservationBuffer, device: torch.device) -> CompressedObservationBuffer:
    return CompressedObservationBuffer(
        token_meta=comp.token_meta.to(device),
        owner_idx=comp.owner_idx.to(device),
        production=comp.production.to(device),
        ships=comp.ships.to(device),
        velocity=comp.velocity.to(device),
        xy=comp.xy.to(device),
        turn_progress=comp.turn_progress.to(device),
        incoming_net=comp.incoming_net.to(device),
        incoming_survivor=comp.incoming_survivor.to(device),
        origin_frac_blocked=comp.origin_frac_blocked.to(device),
    )


def _compute_state_values_per_ego(
    *,
    state_b: OrbitWarsState,
    grouped_population_rollout: bool,
    row_env: Optional[np.ndarray],
    row_ego: Optional[np.ndarray],
    row_env_j: Optional[jnp.ndarray],
    num_envs: int,
    n_ego: int,
    ship_speed: float,
    obs_feature_dim: int,
    normalize_obs_to_p0: bool,
    population_assignments: np.ndarray,
    controller_assignments: np.ndarray,
    main_player_mask: np.ndarray,
    policies: list[OrbitWarsPolicy],
    policy: OrbitWarsPolicy,
    device: torch.device,
    timing: Optional[RolloutTiming] = None,
) -> np.ndarray:
    values = np.zeros((n_ego, num_envs), dtype=np.float32)
    if grouped_population_rollout:
        assert row_env is not None and row_ego is not None and row_env_j is not None
        t0 = perf_counter()
        policy_state_b = _gather_state_rows(state_b, row_env_j)
        obs_comp_j = build_compressed_observation_batched_jax_per_ego(
            policy_state_b,
            jnp.asarray(row_ego, dtype=jnp.int32),
            ship_speed,
            obs_feature_dim,
            normalize_to_p0=normalize_obs_to_p0,
        )
        if timing is not None:
            timing.bootstrap_obs_build_s += perf_counter() - t0
        tb0 = perf_counter()
        obs_comp_t = _compressed_obs_jax_to_torch(obs_comp_j)
        obs_t = decode_observation(obs_comp_t, feature_dim=obs_feature_dim)
        out_rows = policy.forward_dense_rollout_grouped_population(**{k: v.to(device) for k, v in obs_t.items()})
        if timing is not None:
            timing.bootstrap_policy_batch_s += perf_counter() - tb0
        tf0 = perf_counter()
        value_np_rows = out_rows["value"].float().detach().cpu().numpy()
        del out_rows
        if timing is not None:
            timing.bootstrap_policy_forward_s += perf_counter() - tf0
        for row in range(value_np_rows.shape[0]):
            values[int(row_ego[row]), int(row_env[row])] = float(value_np_rows[row])
        return values

    for ego in range(n_ego):
        t0 = perf_counter()
        obs_comp_j = build_compressed_observation_batched_jax(
            state_b,
            ego,
            ship_speed,
            obs_feature_dim,
            normalize_to_p0=normalize_obs_to_p0,
        )
        if timing is not None:
            timing.bootstrap_obs_build_s += perf_counter() - t0
        tb0 = perf_counter()
        obs_t = decode_observation(_compressed_obs_jax_to_torch(obs_comp_j), feature_dim=obs_feature_dim)
        pop_t = torch.as_tensor(population_assignments[ego], device=device, dtype=torch.long)
        controller_t = torch.as_tensor(controller_assignments[ego], device=device, dtype=torch.long)
        value_head_idx_t = torch.as_tensor(
            _relative_main_value_head_idx_for_rows(
                ego_idx=np.full((num_envs,), ego, dtype=np.int32),
                env_idx=np.arange(num_envs, dtype=np.int32),
                controller_assignments=controller_assignments,
                main_player_mask=main_player_mask,
                normalize_obs_to_p0=normalize_obs_to_p0,
            ),
            device=device,
            dtype=torch.long,
        )
        obs_t_dev = {k: v.to(device) for k, v in obs_t.items()}
        out_e = _forward_dense_rollout_by_controller(
            policies=policies,
            active_obs=obs_t_dev,
            active_population_idx_t=pop_t,
            active_controller_idx_t=controller_t,
            active_value_head_idx_t=value_head_idx_t,
        )
        if timing is not None:
            timing.bootstrap_policy_batch_s += perf_counter() - tb0
        tf0 = perf_counter()
        values[ego] = out_e["value"].float().detach().cpu().numpy()
        del out_e
        if timing is not None:
            timing.bootstrap_policy_forward_s += perf_counter() - tf0
    return values


def _merge_controller_outputs(
    controller_outputs: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
    total_rows: int,
) -> dict[str, torch.Tensor]:
    merged: dict[str, torch.Tensor] = {}
    for row_idx, out in controller_outputs:
        for key, value in out.items():
            if key not in merged:
                shape = (total_rows,) + tuple(value.shape[1:])
                merged[key] = torch.zeros(shape, dtype=value.dtype, device=value.device)
            merged[key][row_idx] = value
    return merged


def _forward_dense_rollout_by_controller(
    *,
    policies: list[OrbitWarsPolicy],
    active_obs: dict[str, torch.Tensor],
    active_population_idx_t: torch.Tensor,
    active_controller_idx_t: torch.Tensor,
    active_value_head_idx_t: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    controller_outputs: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
    for controller_id in torch.unique(active_controller_idx_t, sorted=True).tolist():
        row_idx = torch.nonzero(active_controller_idx_t == int(controller_id), as_tuple=False).squeeze(-1)
        if int(row_idx.numel()) == 0:
            continue
        policy = policies[int(controller_id)]
        obs_slice = {key: value.index_select(0, row_idx) for key, value in active_obs.items()}
        pop_slice = active_population_idx_t.index_select(0, row_idx)
        value_head_slice = None if active_value_head_idx_t is None else active_value_head_idx_t.index_select(0, row_idx)
        out = policy.forward_dense_rollout(
            **obs_slice,
            population_idx=pop_slice,
            value_head_idx=value_head_slice,
        )
        controller_outputs.append((row_idx, out))
    return _merge_controller_outputs(controller_outputs, int(active_controller_idx_t.shape[0]))


def _forward_dense_rollout_compressed_by_controller(
    *,
    policies: list[OrbitWarsPolicy],
    active_comp: CompressedObservationBuffer,
    feature_dim: int,
    active_population_idx_t: torch.Tensor,
    active_controller_idx_t: torch.Tensor,
    active_value_head_idx_t: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    controller_outputs: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
    for controller_id in torch.unique(active_controller_idx_t, sorted=True).tolist():
        row_idx = torch.nonzero(active_controller_idx_t == int(controller_id), as_tuple=False).squeeze(-1)
        if int(row_idx.numel()) == 0:
            continue
        policy = policies[int(controller_id)]
        comp_slice = index_compressed_observation_rows(active_comp, row_idx)
        pop_slice = active_population_idx_t.index_select(0, row_idx)
        value_head_slice = None if active_value_head_idx_t is None else active_value_head_idx_t.index_select(0, row_idx)
        out = policy.forward_dense_rollout_compressed(
            comp_slice.token_meta,
            comp_slice.owner_idx,
            comp_slice.production,
            comp_slice.ships,
            comp_slice.velocity,
            comp_slice.xy,
            comp_slice.turn_progress,
            comp_slice.incoming_net,
            comp_slice.incoming_survivor,
            int(feature_dim),
            population_idx=pop_slice,
            value_head_idx=value_head_slice,
        )
        controller_outputs.append((row_idx, out))
    return _merge_controller_outputs(controller_outputs, int(active_controller_idx_t.shape[0]))


def _target_logits_by_controller(
    *,
    policies: list[OrbitWarsPolicy],
    planet_hidden: torch.Tensor,
    origin_idx: torch.Tensor,
    frac_idx: torch.Tensor,
    fleet_size: torch.Tensor,
    target_eta: torch.Tensor,
    target_ships: torch.Tensor,
    active_population_idx_t: torch.Tensor,
    active_controller_idx_t: torch.Tensor,
) -> torch.Tensor:
    pieces: list[tuple[torch.Tensor, torch.Tensor]] = []
    for controller_id in torch.unique(active_controller_idx_t, sorted=True).tolist():
        row_idx = torch.nonzero(active_controller_idx_t == int(controller_id), as_tuple=False).squeeze(-1)
        if int(row_idx.numel()) == 0:
            continue
        policy = policies[int(controller_id)]
        logits = policy.target_logits_for_origin_fraction(
            planet_hidden.index_select(0, row_idx),
            origin_idx.index_select(0, row_idx),
            frac_idx.index_select(0, row_idx),
            fleet_size=fleet_size.index_select(0, row_idx),
            target_eta=target_eta.index_select(0, row_idx),
            target_ships=target_ships.index_select(0, row_idx),
            population_idx=active_population_idx_t.index_select(0, row_idx),
        )
        pieces.append((row_idx, logits))
    if pieces:
        sample_logits = pieces[0][1]
        out = torch.zeros(
            (int(active_controller_idx_t.shape[0]), int(target_eta.shape[1])),
            dtype=sample_logits.dtype,
            device=sample_logits.device,
        )
    else:
        out = torch.zeros(
            (int(active_controller_idx_t.shape[0]), int(target_eta.shape[1])),
            dtype=planet_hidden.dtype,
            device=planet_hidden.device,
        )
    for row_idx, logits in pieces:
        out[row_idx] = logits
    return out


def _assert_write_cursors_in_sync(
    write_idx: list[np.ndarray],
    write_idx_dev: list[torch.Tensor],
    n_ego: int,
) -> None:
    for p in range(n_ego):
        dev = write_idx_dev[p].detach().cpu().numpy()
        if not np.array_equal(dev, write_idx[p]):
            mism = np.flatnonzero(dev != write_idx[p])
            e0 = int(mism[0])
            raise RuntimeError(
                f"write cursor desync ego={p}: {mism.size} env(s) differ; "
                f"first env={e0} host={int(write_idx[p][e0])} dev={int(dev[e0])}"
            )


def _append_synthetic_terminal_rows(
    *,
    state_rows: OrbitWarsState,
    ego_idx: np.ndarray,
    env_idx: np.ndarray,
    controller_idx: np.ndarray,
    population_idx: np.ndarray,
    value_head_idx: np.ndarray,
    terminal_reward: np.ndarray,
    ship_speed: float,
    obs_feature_dim: int,
    normalize_obs_to_p0: bool,
    policies: list[OrbitWarsPolicy],
    bufs: list[TorchTransitionBuffer],
    obs_bufs: list[CompressedObservationBuffer],
    write_idx: list[np.ndarray],
    write_idx_dev: list[torch.Tensor],
    valid: list[np.ndarray],
    old_logprob: list[np.ndarray],
    old_value: list[np.ndarray],
    reward: list[np.ndarray],
    done: list[np.ndarray],
    device: torch.device,
) -> None:
    if int(ego_idx.size) == 0:
        return
    ego_np = np.asarray(ego_idx, dtype=np.int32).reshape(-1)
    env_np = np.asarray(env_idx, dtype=np.int32).reshape(-1)
    ctrl_np = np.asarray(controller_idx, dtype=np.int32).reshape(-1)
    pop_np = np.asarray(population_idx, dtype=np.int32).reshape(-1)
    value_head_np = np.asarray(value_head_idx, dtype=np.int32).reshape(-1)
    rew_np = np.asarray(terminal_reward, dtype=np.float32).reshape(-1)
    if not (
        ego_np.shape == env_np.shape == ctrl_np.shape == pop_np.shape == value_head_np.shape == rew_np.shape
    ):
        raise ValueError("synthetic terminal row metadata shapes must match")

    obs_comp_j = build_compressed_observation_batched_jax_per_ego(
        state_rows,
        jnp.asarray(ego_np, dtype=jnp.int32),
        ship_speed,
        obs_feature_dim,
        normalize_to_p0=normalize_obs_to_p0,
    )
    obs_comp_t = _compressed_obs_jax_to_torch(obs_comp_j)
    obs_t_dev = {key: value.to(device) for key, value in decode_observation(obs_comp_t, feature_dim=obs_feature_dim).items()}
    ctrl_t = torch.as_tensor(ctrl_np, device=device, dtype=torch.long)
    pop_t = torch.as_tensor(pop_np, device=device, dtype=torch.long)
    value_head_t = torch.as_tensor(value_head_np, device=device, dtype=torch.long)
    out = _forward_dense_rollout_by_controller(
        policies=policies,
        active_obs=obs_t_dev,
        active_population_idx_t=pop_t,
        active_controller_idx_t=ctrl_t,
        active_value_head_idx_t=value_head_t,
    )
    value_np = out["value"].float().detach().cpu().numpy().astype(np.float32, copy=False)
    del out

    by_player: dict[int, list[int]] = {}
    for row_i, ego in enumerate(ego_np.tolist()):
        by_player.setdefault(int(ego), []).append(int(row_i))

    for ego, row_ids in by_player.items():
        env_sel = env_np[row_ids].astype(np.int32, copy=False)
        row_sel = np.asarray(row_ids, dtype=np.int32)
        row_sel_t = torch.as_tensor(row_sel, device=device, dtype=torch.long)
        env_t = torch.as_tensor(env_sel, device=device, dtype=torch.long)
        write_row_t = write_idx_dev[ego].index_select(0, env_t)
        wr_np = write_row_t.detach().cpu().numpy().astype(np.int32, copy=False)
        count = int(row_sel.shape[0])
        zero_i32 = torch.zeros((count,), device=device, dtype=torch.int32)
        minus_one_i32 = torch.full((count,), -1, device=device, dtype=torch.int32)
        true_bool = torch.ones((count,), device=device, dtype=torch.bool)
        false_bool = torch.zeros((count,), device=device, dtype=torch.bool)
        zero_f32 = torch.zeros((count,), device=device, dtype=torch.float32)
        target_valid = torch.zeros((count, MAX_PLANETS), device=device, dtype=torch.bool)
        target_hit_tick = torch.zeros((count, MAX_PLANETS), device=device, dtype=torch.float32)
        bufs[ego] = append_active_to_torch_buffer(
            bufs[ego],
            env_t,
            true_bool,
            zero_f32,
            zero_f32,
            minus_one_i32,
            true_bool.to(torch.int32),
            false_bool,
            zero_i32,
            zero_i32,
            true_bool,
            true_bool,
            true_bool,
            target_valid,
            target_hit_tick,
            torch.as_tensor(pop_np[row_sel], device=device, dtype=torch.int32),
            torch.as_tensor(ctrl_np[row_sel], device=device, dtype=torch.int32),
            torch.as_tensor(value_head_np[row_sel], device=device, dtype=torch.int32),
            write_row_t,
            zero_i32,
            1,
        )
        obs_comp_ego = index_compressed_observation_rows(obs_comp_t, row_sel_t)
        obs_bufs[ego] = store_precompressed_observation_rows(
            obs_bufs[ego], write_row_t.to(torch.long), env_t, obs_comp_ego
        )
        valid[ego][wr_np, env_sel] = True
        old_logprob[ego][wr_np, env_sel] = 0.0
        old_value[ego][wr_np, env_sel] = value_np[row_sel]
        reward[ego][wr_np, env_sel] += rew_np[row_sel]
        done[ego][wr_np, env_sel] = True
        write_idx[ego][env_sel] += 1
        write_idx_dev[ego][env_t] = write_idx_dev[ego][env_t] + 1


def _selected_origin_fraction_targets_batched_maybe_chunked(
    state_b: OrbitWarsState,
    origin_idx_b: jnp.ndarray,
    frac_idx_b: jnp.ndarray,
    *,
    horizon: int,
    ship_speed: float,
    samples_per_span: int,
    n_rays: int,
    ray_chunk_size: int,
    first_hit_method: str,
    env_chunk_size: int,
):
    """Call ``selected_origin_fraction_targets_batched`` on the full env batch or env chunks."""

    total = int(origin_idx_b.shape[0])
    if env_chunk_size <= 0 or env_chunk_size >= total:
        return selected_origin_fraction_targets_batched(
            state_b,
            origin_idx_b,
            frac_idx_b,
            horizon=horizon,
            ship_speed=ship_speed,
            samples_per_span=samples_per_span,
            n_rays=n_rays,
            ray_chunk_size=ray_chunk_size,
            first_hit_method=first_hit_method,
        )

    parts = []
    for start in range(0, total, int(env_chunk_size)):
        stop = min(total, start + int(env_chunk_size))
        state_chunk = jax.tree.map(lambda x: x[start:stop], state_b)
        parts.append(
            selected_origin_fraction_targets_batched(
                state_chunk,
                origin_idx_b[start:stop],
                frac_idx_b[start:stop],
                horizon=horizon,
                ship_speed=ship_speed,
                samples_per_span=samples_per_span,
                n_rays=n_rays,
                ray_chunk_size=ray_chunk_size,
                first_hit_method=first_hit_method,
            )
        )
    return tuple(jnp.concatenate(xs, axis=0) for xs in zip(*parts))


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


def _run_async_micro_step_multi(
    *,
    n_ego: int,
    num_envs: int,
    pending: np.ndarray,
    virt_b: OrbitWarsState,
    bufs: List[TorchTransitionBuffer],
    obs_bufs: List[CompressedObservationBuffer],
    write_idx: List[np.ndarray],
    write_idx_dev: List[torch.Tensor],
    micro_k: np.ndarray,
    micro_k_dev: torch.Tensor,
    valid: List[np.ndarray],
    old_logprob: List[np.ndarray],
    old_value: List[np.ndarray],
    reward: List[np.ndarray],
    population_assignments: np.ndarray,
    pending_actions: np.ndarray,
    pending_action_count: np.ndarray,
    reward_idx: np.ndarray,
    halted: np.ndarray,
    policies: list[OrbitWarsPolicy],
    device: torch.device,
    rng: Optional[torch.Generator],
    greedy: bool,
    ship_speed: float,
    max_micro_steps: int,
    timing: RolloutTiming,
    first_hit_n_rays: int,
    micro_step_penalty: float = 0.0,
    first_hit_ray_chunk_size: int = 0,
    first_hit_env_chunk_size: int = 0,
    first_hit_method: str = "category-rays",
    obs_feature_dim: int = FEATURE_DIM,
    normalize_obs_to_p0: bool = False,
    sync_policy_timing: bool = False,
    controller_assignments: Optional[np.ndarray] = None,
    main_player_mask: Optional[np.ndarray] = None,
    value_only: Optional[np.ndarray] = None,
) -> tuple[
    OrbitWarsState,
    List[TorchTransitionBuffer],
    List[CompressedObservationBuffer],
    Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
]:
    """Run one micro decision for every pending egocentric row in a ``n_ego * num_envs`` JAX batch."""

    t_prep0 = perf_counter()
    if value_only is None:
        value_only = np.zeros_like(pending, dtype=np.bool_)
    else:
        value_only = np.asarray(value_only, dtype=np.bool_)
        if value_only.shape != pending.shape:
            raise ValueError(f"value_only shape {value_only.shape} != pending shape {pending.shape}")
    players_active = [np.flatnonzero(pending[p]).astype(np.int32) for p in range(n_ego)]
    n_list = [int(x.size) for x in players_active]
    offsets: list[int] = []
    off = 0
    for n_p in n_list:
        offsets.append(off)
        off += n_p
    n_active = int(off)
    total_env_rows = int(virt_b.planets.shape[0])
    expected_total_env_rows = int(n_ego) * int(num_envs)
    if total_env_rows != expected_total_env_rows:
        raise ValueError(
            f"virt_b env-row count {total_env_rows} != n_ego*num_envs {expected_total_env_rows}; "
            "rollout carry/state batch width is stale"
        )
    if n_active == 0:
        return virt_b, bufs, obs_bufs, None

    active_rows = np.concatenate([players_active[p] + p * num_envs for p in range(n_ego)]).astype(np.int32)
    active_population_idx = np.concatenate(
        [population_assignments[p, players_active[p]] for p in range(n_ego)]
    ).astype(np.int64)
    if controller_assignments is None:
        active_controller_idx = np.zeros((n_active,), dtype=np.int64)
    else:
        active_controller_idx = np.concatenate(
            [controller_assignments[p, players_active[p]] for p in range(n_ego)]
        ).astype(np.int64)
    active_env_idx = np.concatenate([players_active[p] for p in range(n_ego)]).astype(np.int32)
    active_ego_idx = np.concatenate(
        [np.full((int(players_active[p].size),), p, dtype=np.int32) for p in range(n_ego)]
    ).astype(np.int32)
    active_value_only = np.concatenate(
        [value_only[p, players_active[p]] for p in range(n_ego)]
    ).astype(np.bool_, copy=False)
    active_value_head_idx = _relative_main_value_head_idx_for_rows(
        ego_idx=active_ego_idx,
        env_idx=active_env_idx,
        controller_assignments=controller_assignments,
        main_player_mask=main_player_mask,
        normalize_obs_to_p0=normalize_obs_to_p0,
    ).astype(np.int64, copy=False)
    active_idx_t = torch.as_tensor(active_rows, device=device, dtype=torch.long)
    active_population_idx_t = torch.as_tensor(active_population_idx, device=device, dtype=torch.long)
    active_controller_idx_t = torch.as_tensor(active_controller_idx, device=device, dtype=torch.long)
    active_value_head_idx_t = torch.as_tensor(active_value_head_idx, device=device, dtype=torch.long)
    timing.micro_prep_non_grouped_s += perf_counter() - t_prep0

    ego_rows = [jnp.full((num_envs,), p, dtype=jnp.int32) for p in range(n_ego)]
    ego_b_j = jnp.concatenate(ego_rows, axis=0)

    t0 = perf_counter()
    obs_comp_jax = build_compressed_observation_batched_jax_per_ego(
        virt_b,
        ego_b_j,
        ship_speed,
        obs_feature_dim,
        normalize_to_p0=normalize_obs_to_p0,
    )
    must_halt_j = must_halt_no_owned_ships_per_ego(virt_b, ego_b_j)
    timing.obs_build_s += perf_counter() - t0

    t0 = perf_counter()
    obs_comp_torch = _compressed_obs_jax_to_torch(obs_comp_jax)
    must_halt_t = torch.from_dlpack(must_halt_j)
    obs_index = active_idx_t.to(obs_comp_torch.token_meta.device)
    active_comp = index_compressed_observation_rows(obs_comp_torch, obs_index)
    active_obs = {key: v.to(device) for key, v in decode_observation(active_comp, feature_dim=obs_feature_dim).items()}
    must_halt_a = must_halt_t.index_select(0, active_idx_t.to(must_halt_t.device)).to(device)
    timing.policy_batch_s += perf_counter() - t0

    if sync_policy_timing:
        _sync_rollout_policy_timing(device)
    t0 = perf_counter()
    out = _forward_dense_rollout_by_controller(
        policies=policies,
        active_obs=active_obs,
        active_population_idx_t=active_population_idx_t,
        active_controller_idx_t=active_controller_idx_t,
        active_value_head_idx_t=active_value_head_idx_t,
    )
    if sync_policy_timing:
        _sync_rollout_policy_timing(device)
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
    if sync_policy_timing:
        _sync_rollout_policy_timing(device)
    t_origin = perf_counter()
    timing.policy_sample_origin_s += t_origin - t_model

    if sync_policy_timing:
        _sync_rollout_policy_timing(device)
    t_ray_start = perf_counter()
    o_idx_all = torch.zeros(total_env_rows, dtype=torch.int32, device=device)
    frac_idx_geom_all = torch.zeros(total_env_rows, dtype=torch.int32, device=device)
    o_idx_all.index_copy_(0, active_idx_t, o_idx.to(torch.int32))
    frac_idx_geom_all.index_copy_(0, active_idx_t, frac_idx.to(torch.int32))
    origin_idx_j = jax.dlpack.from_dlpack(o_idx_all.contiguous().detach())
    frac_idx_geom_j = jax.dlpack.from_dlpack(frac_idx_geom_all.contiguous().detach())
    (
        _target_angle_j,
        _target_width_j,
        target_valid_j,
        target_overflow_j,
        target_hit_tick_j,
        target_true_planet_j,
        target_true_hit_tick_j,
    ) = _selected_origin_fraction_targets_batched_maybe_chunked(
        virt_b,
        origin_idx_j,
        frac_idx_geom_j,
        horizon=24,
        ship_speed=ship_speed,
        samples_per_span=17,
        n_rays=first_hit_n_rays,
        ray_chunk_size=first_hit_ray_chunk_size,
        first_hit_method=first_hit_method,
        env_chunk_size=first_hit_env_chunk_size,
    )
    if sync_policy_timing:
        _sync_rollout_policy_timing(
            device,
            (
                _target_angle_j,
                target_valid_j,
                target_overflow_j,
                target_hit_tick_j,
                target_true_planet_j,
                target_true_hit_tick_j,
            ),
        )
    target_valid_t = torch.from_dlpack(target_valid_j).index_select(0, active_idx_t)
    target_overflow_t = torch.from_dlpack(target_overflow_j).index_select(0, active_idx_t)
    target_hit_tick_t = torch.from_dlpack(target_hit_tick_j).index_select(0, active_idx_t)
    target_true_planet_t = torch.from_dlpack(target_true_planet_j).index_select(0, active_idx_t)
    target_true_hit_tick_t = torch.from_dlpack(target_true_hit_tick_j).index_select(0, active_idx_t)
    if sync_policy_timing:
        _sync_rollout_policy_timing(device)
    t_raycast = perf_counter()
    timing.policy_raycast_s += t_raycast - (t_ray_start if sync_policy_timing else t_origin)

    if sync_policy_timing:
        _sync_rollout_policy_timing(device)
    t_target_start = perf_counter()
    n_a_idx = torch.arange(n_active, device=device)
    planet_ships = active_obs["features"][:, 1 : 1 + MAX_PLANETS, 1] * 1000.0
    origin_ships = active_obs["features"][n_a_idx, 1 + o_idx, 1] * 1000.0
    frac_values = origin_ships.new_tensor(FRACTIONS)
    fleet_size_for_logits = torch.floor(frac_values[frac_idx] * origin_ships)
    target_logits = _target_logits_by_controller(
        policies=policies,
        planet_hidden=out["planet_hidden"],
        origin_idx=o_idx,
        frac_idx=frac_idx,
        fleet_size=fleet_size_for_logits,
        target_eta=target_hit_tick_t,
        target_ships=planet_ships,
        active_population_idx_t=active_population_idx_t,
        active_controller_idx_t=active_controller_idx_t,
    )
    abort_logits_all = out.get("abort_logits")
    abort_logits = None
    if abort_logits_all is not None:
        abort_logits = abort_logits_all[n_a_idx, o_idx, frac_idx]
    target_mask = out["pair_mask"][n_a_idx, o_idx, :] & target_valid_t & ~target_overflow_t[:, None]
    any_valid_target = target_mask.any(dim=-1)
    target_abort = torch.zeros((n_active,), dtype=torch.bool, device=device)
    if abort_logits is not None:
        combined_target = torch.cat([target_logits.masked_fill(~target_mask, -1e4), abort_logits[:, None]], dim=-1)
        target_lp = torch.log_softmax(combined_target, dim=-1)
        if greedy:
            target_choice = combined_target.argmax(dim=-1)
        else:
            target_choice = torch.multinomial(torch.softmax(combined_target, dim=-1), 1, generator=rng).squeeze(-1)
        target_abort = target_choice == MAX_PLANETS
        d_idx = target_choice.clamp_max(MAX_PLANETS - 1)
        target_logp = target_lp.gather(1, target_choice[:, None]).squeeze(-1)
    else:
        masked_target = target_logits.masked_fill(~target_mask, -1e4)
        target_lp = torch.log_softmax(masked_target, dim=-1)
        safe_target = torch.where(any_valid_target[:, None], masked_target, torch.zeros_like(masked_target))
        if greedy:
            d_idx = safe_target.argmax(dim=-1)
        else:
            d_idx = torch.multinomial(torch.softmax(safe_target, dim=-1), 1, generator=rng).squeeze(-1)
        target_logp = target_lp.gather(1, d_idx[:, None]).squeeze(-1)
    pair_flat = o_idx * P + d_idx
    policy_fleet_eta = target_hit_tick_t[n_a_idx, d_idx]
    true_d_idx = target_true_planet_t[n_a_idx, d_idx].to(torch.long).clamp(0, P - 1)
    true_fleet_eta = target_true_hit_tick_t[n_a_idx, d_idx]

    dispatch_used = origin_frac_used & (~target_abort) & any_valid_target
    target_decision_used = origin_frac_used if abort_logits is not None else dispatch_used
    total_logp = halt_logp + origin_frac_used.float() * origin_frac_logp + target_decision_used.float() * target_logp
    values_active = out["value"].float()
    apply_halt_now = ~dispatch_used
    stop_now = (halt_action == 1) | ((halt_action == 0) & ~any_valid_origin_frac)
    total_logp = total_logp.detach()
    values_active = values_active.detach()
    halt_action = halt_action.detach()
    target_abort = target_abort.detach()
    pair_flat = pair_flat.detach()
    frac_idx = frac_idx.detach()
    o_idx = o_idx.detach()
    d_idx = d_idx.detach()
    origin_frac_used = origin_frac_used.detach()
    apply_halt_now = apply_halt_now.detach()
    stop_now = stop_now.detach()
    policy_fleet_eta = policy_fleet_eta.detach()
    if np.any(active_value_only):
        value_only_t = torch.as_tensor(active_value_only, device=device, dtype=torch.bool)
        apply_halt_now = torch.where(value_only_t, torch.ones_like(apply_halt_now), apply_halt_now)
        stop_now = torch.where(value_only_t, torch.ones_like(stop_now), stop_now)
    del out
    if sync_policy_timing:
        _sync_rollout_policy_timing(device)
    t_target = perf_counter()
    timing.policy_target_s += t_target - (t_target_start if sync_policy_timing else t_raycast)

    if sync_policy_timing:
        _sync_rollout_policy_timing(device)
    t_scatter_start = perf_counter()
    halt_now_all = torch.ones(total_env_rows, dtype=torch.bool, device=device)
    pair_flat_all = torch.zeros(total_env_rows, dtype=torch.int32, device=device)
    frac_idx_all = torch.zeros(total_env_rows, dtype=torch.int32, device=device)
    fleet_eta_all = torch.zeros(total_env_rows, dtype=torch.float32, device=device)
    target_abort_all = torch.zeros(total_env_rows, dtype=torch.bool, device=device)

    halt_now_all.index_copy_(0, active_idx_t, apply_halt_now)
    pair_flat_all.index_copy_(0, active_idx_t, pair_flat.to(torch.int32))
    frac_idx_all.index_copy_(0, active_idx_t, frac_idx.to(torch.int32))
    fleet_eta_all.index_copy_(0, active_idx_t, policy_fleet_eta.to(torch.float32))
    target_abort_all.index_copy_(0, active_idx_t, target_abort)
    if sync_policy_timing:
        _sync_rollout_policy_timing(device)
    t_scatter = perf_counter()
    timing.policy_scatter_s += t_scatter - (t_scatter_start if sync_policy_timing else t_target)
    timing.policy_forward_s += t_scatter - t0

    t0 = perf_counter()
    halt_now_j = jax.dlpack.from_dlpack(halt_now_all.contiguous().detach())
    pair_flat_j = jax.dlpack.from_dlpack(pair_flat_all.contiguous().detach())
    frac_idx_j = jax.dlpack.from_dlpack(frac_idx_all.contiguous().detach())
    fleet_eta_j = jax.dlpack.from_dlpack(fleet_eta_all.contiguous().detach())
    target_abort_j = jax.dlpack.from_dlpack(target_abort_all.contiguous().detach())
    t1 = perf_counter()

    virt_b, oid_j, send_j, dispatched_j, slot_j = apply_micro_step_batched_per_ego(
        virt_b, ego_b_j, halt_now_j, pair_flat_j, frac_idx_j, fleet_eta_j, target_abort_j
    )
    t2 = perf_counter()

    oid_t = torch.from_dlpack(oid_j)
    send_t = torch.from_dlpack(send_j)
    dispatched_t = torch.from_dlpack(dispatched_j)
    slot_t = torch.from_dlpack(slot_j)
    t3 = perf_counter()

    t3a = perf_counter()
    for p in range(n_ego):
        ap = players_active[p]
        n_p = n_list[p]
        if n_p:
            rows_np = write_idx[p][ap]
            micro_np = micro_k[p, ap]
            _validate_numpy_index_range(f"rollout write_idx p{p}", rows_np, int(bufs[p].micro_halt_now.shape[0]))
            _validate_numpy_index_range(f"rollout micro_k p{p}", micro_np, max_micro_steps)
    t3b = perf_counter()
    t4 = perf_counter()

    for p in range(n_ego):
        ap = players_active[p]
        n_p = n_list[p]
        offset = offsets[p]
        if n_p == 0:
            continue
        action_local = np.flatnonzero(~active_value_only[offset : offset + n_p]).astype(np.int32)
        if action_local.size == 0:
            continue
        pos_p_t = torch.as_tensor(offset + action_local, device=device, dtype=torch.long)
        active_p_idx_t = torch.as_tensor(ap[action_local], device=device, dtype=torch.long)
        active_p_rows_t = active_p_idx_t + p * num_envs
        micro_kp_t = micro_k_dev[p].index_select(0, active_p_idx_t)
        write_row_t = write_idx_dev[p].index_select(0, active_p_idx_t)
        bufs[p] = append_active_to_torch_buffer(
            bufs[p],
            active_p_idx_t,
            apply_halt_now.index_select(0, pos_p_t),
            send_t.index_select(0, active_p_rows_t),
            policy_fleet_eta.index_select(0, pos_p_t),
            slot_t.index_select(0, active_p_rows_t),
            halt_action.index_select(0, pos_p_t).to(torch.int32),
            target_abort.index_select(0, pos_p_t),
            pair_flat.index_select(0, pos_p_t).to(torch.int32),
            frac_idx.index_select(0, pos_p_t).to(torch.int32),
            (origin_frac_used & ~any_valid_target).index_select(0, pos_p_t),
            (~any_valid_origin_frac & (halt_action == 0)).index_select(0, pos_p_t),
            must_halt_a.index_select(0, pos_p_t).to(torch.bool),
            (target_valid_t & ~target_overflow_t[:, None]).index_select(0, pos_p_t).to(torch.bool),
            target_hit_tick_t.index_select(0, pos_p_t).to(torch.float32),
            active_population_idx_t.index_select(0, pos_p_t).to(torch.int32),
            active_controller_idx_t.index_select(0, pos_p_t).to(torch.int32),
            active_value_head_idx_t.index_select(0, pos_p_t).to(torch.int32),
            write_row_t,
            micro_kp_t,
            max_micro_steps,
        )
        comp_p_active = index_compressed_observation_rows(active_comp, pos_p_t)
        obs_bufs[p] = store_precompressed_observation_rows(
            obs_bufs[p], write_row_t.to(torch.long), active_p_idx_t, comp_p_active
        )
    t5 = perf_counter()

    t6 = perf_counter()

    oid_active_t = oid_t.index_select(0, active_idx_t)
    send_active_t = send_t.index_select(0, active_idx_t)
    dispatched_active_t = dispatched_t.index_select(0, active_idx_t)
    oid_np = oid_active_t.detach().cpu().numpy()
    send_np = send_active_t.detach().cpu().numpy()
    dispatched_np = dispatched_active_t.detach().cpu().numpy()
    t7 = perf_counter()
    _accum_micro_apply_breakdown(timing, t0, t1, t2, t3, t3a, t3b, t4, t5, t6, t7)

    t_post0 = perf_counter()
    total_logp_np = total_logp.detach().cpu().numpy()
    values_np = values_active.detach().cpu().numpy()
    stop_now_np = stop_now.detach().cpu().numpy()
    d_idx_np = d_idx.detach().cpu().numpy()
    true_d_idx_np = true_d_idx.detach().cpu().numpy()
    true_fleet_eta_np = true_fleet_eta.detach().cpu().numpy()
    policy_fleet_eta_np = policy_fleet_eta.detach().cpu().numpy()
    timing.micro_post_apply_extract_s += perf_counter() - t_post0

    t_post1 = perf_counter()
    for p in range(n_ego):
        ap = players_active[p]
        n_p = n_list[p]
        offset = offsets[p]
        if n_p == 0:
            continue
        action_local = np.flatnonzero(~active_value_only[offset : offset + n_p]).astype(np.int32)
        if action_local.size == 0:
            continue
        env_sel = ap[action_local]
        rows_np = write_idx[p][env_sel]
        valid[p][rows_np, env_sel] = True
        old_logprob[p][rows_np, env_sel] = total_logp_np[offset + action_local]
        old_value[p][rows_np, env_sel] = values_np[offset + action_local]
        if micro_step_penalty != 0.0:
            reward[p][rows_np, env_sel] -= float(micro_step_penalty) * dispatched_np[offset + action_local].astype(
                np.float32
            )
    t_post2 = perf_counter()
    timing.micro_post_apply_row_stats_s += t_post2 - t_post1

    t_post3 = perf_counter()
    for p in range(n_ego):
        ap = players_active[p]
        n_p = n_list[p]
        offset = offsets[p]
        widx = write_idx[p]
        micro_arr = micro_k[p]
        for local_j, env_i in enumerate(ap):
            if bool(active_value_only[offset + local_j]):
                continue
            j_arr = offset + local_j
            env_i = int(env_i)
            if bool(dispatched_np[j_arr]):
                ac = int(pending_action_count[p, env_i])
                if ac < pending_actions.shape[2]:
                    pending_actions[env_i, p, ac, 0] = float(oid_np[j_arr])
                    pending_actions[env_i, p, ac, 1] = float(send_np[j_arr])
                    pending_actions[env_i, p, ac, 2] = float(true_d_idx_np[j_arr])
                    pending_actions[env_i, p, ac, 3] = float(true_fleet_eta_np[j_arr])
                    pending_actions[env_i, p, ac, 4] = float(d_idx_np[j_arr])
                    pending_actions[env_i, p, ac, 5] = float(policy_fleet_eta_np[j_arr])
                pending_action_count[p, env_i] = ac + 1
    t_post4 = perf_counter()
    timing.micro_post_apply_pending_actions_s += t_post4 - t_post3

    t_post5 = perf_counter()
    for p in range(n_ego):
        ap = players_active[p]
        n_p = n_list[p]
        offset = offsets[p]
        widx = write_idx[p]
        micro_arr = micro_k[p]
        value_only_local = np.flatnonzero(active_value_only[offset : offset + n_p]).astype(np.int32)
        if value_only_local.size > 0:
            halted[p, ap[value_only_local]] = True
        if n_p == 0:
            continue
        action_local = np.flatnonzero(~active_value_only[offset : offset + n_p]).astype(np.int32)
        if action_local.size == 0:
            continue
        env_sel = ap[action_local]
        stop_mask = stop_now_np[offset + action_local].astype(np.bool_, copy=False)
        reward_idx[p, env_sel] = widx[env_sel]
        halted[p, env_sel] |= stop_mask
        widx[env_sel] += 1
        micro_arr[env_sel] += 1
        halted[p, env_sel] |= micro_arr[env_sel] >= max_micro_steps
    t_post6 = perf_counter()
    timing.micro_post_apply_halt_block_indices_s += t_post6 - t_post5
    timing.micro_post_apply_host_bookkeeping_s += t_post6 - t_post1

    t_post7 = perf_counter()
    for p in range(n_ego):
        ap = players_active[p]
        n_p = n_list[p]
        offset = offsets[p]
        if n_p == 0:
            continue
        action_local = np.flatnonzero(~active_value_only[offset : offset + n_p]).astype(np.int32)
        if action_local.size == 0:
            continue
        active_p_idx_t = torch.as_tensor(ap[action_local], device=device, dtype=torch.long)
        micro_k_dev[p, active_p_idx_t] = micro_k_dev[p, active_p_idx_t] + 1
        write_idx_dev[p][active_p_idx_t] = write_idx_dev[p][active_p_idx_t] + 1
    timing.micro_post_apply_device_index_s += perf_counter() - t_post7

    value_capture = (
        values_np,
        active_env_idx.astype(np.int32, copy=False),
        active_ego_idx.astype(np.int32, copy=False),
        active_value_only.astype(np.bool_, copy=False),
    )
    return virt_b, bufs, obs_bufs, value_capture


def _run_async_micro_step_multi_grouped_population(
    *,
    n_ego: int,
    num_envs: int,
    pending_rows: np.ndarray,
    virt_b: OrbitWarsState,
    bufs: List[TorchTransitionBuffer],
    obs_bufs: List[CompressedObservationBuffer],
    write_idx: List[np.ndarray],
    write_idx_dev: List[torch.Tensor],
    micro_k: np.ndarray,
    micro_k_dev: torch.Tensor,
    valid: List[np.ndarray],
    old_logprob: List[np.ndarray],
    old_value: List[np.ndarray],
    reward: List[np.ndarray],
    row_env: np.ndarray,
    row_ego: np.ndarray,
    rows_per_member: int,
    pending_actions: np.ndarray,
    pending_action_count: np.ndarray,
    reward_idx: np.ndarray,
    halted: np.ndarray,
    policies: list[OrbitWarsPolicy],
    device: torch.device,
    rng: Optional[torch.Generator],
    greedy: bool,
    ship_speed: float,
    max_micro_steps: int,
    timing: RolloutTiming,
    first_hit_n_rays: int,
    micro_step_penalty: float = 0.0,
    first_hit_ray_chunk_size: int = 0,
    first_hit_env_chunk_size: int = 0,
    first_hit_method: str = "category-rays",
    obs_feature_dim: int = FEATURE_DIM,
    normalize_obs_to_p0: bool = False,
    sync_policy_timing: bool = False,
) -> tuple[OrbitWarsState, List[TorchTransitionBuffer], List[CompressedObservationBuffer]]:
    """Run one fixed-shape grouped-population microstep over all policy rows."""

    t_prep0 = perf_counter()
    policy = policies[0]

    total_rows = int(virt_b.planets.shape[0])
    if total_rows == 0:
        return virt_b, bufs, obs_bufs

    pending_rows = np.asarray(pending_rows, dtype=np.bool_)
    if pending_rows.shape != (total_rows,):
        raise ValueError(f"pending_rows shape {pending_rows.shape} != {(total_rows,)}")
    if not bool(np.any(pending_rows)):
        return virt_b, bufs, obs_bufs

    row_env_np = np.asarray(row_env, dtype=np.int32)
    row_ego_np = np.asarray(row_ego, dtype=np.int32)
    row_idx_np = np.arange(total_rows, dtype=np.int32)
    row_idx_t = torch.as_tensor(row_idx_np, device=device, dtype=torch.long)
    row_ego_j = jnp.asarray(row_ego_np, dtype=jnp.int32)
    pending_t = torch.as_tensor(pending_rows, device=device, dtype=torch.bool)
    population_idx_t = torch.div(row_idx_t, int(rows_per_member), rounding_mode="floor").to(torch.int32)
    timing.micro_prep_grouped_s += perf_counter() - t_prep0

    t0 = perf_counter()
    obs_comp_jax = build_compressed_observation_batched_jax_per_ego(
        virt_b,
        row_ego_j,
        ship_speed,
        obs_feature_dim,
        normalize_to_p0=normalize_obs_to_p0,
    )
    must_halt_j = must_halt_no_owned_ships_per_ego(virt_b, row_ego_j)
    timing.obs_build_s += perf_counter() - t0

    t0 = perf_counter()
    obs_comp_torch = _compressed_obs_jax_to_torch(obs_comp_jax)
    must_halt_t = torch.from_dlpack(must_halt_j)
    full_obs = {key: v.to(device) for key, v in decode_observation(obs_comp_torch, feature_dim=obs_feature_dim).items()}
    must_halt = must_halt_t.to(device=device, dtype=torch.bool)
    timing.policy_batch_s += perf_counter() - t0

    if sync_policy_timing:
        _sync_rollout_policy_timing(device)
    t0 = perf_counter()
    out = policy.forward_dense_rollout_grouped_population(**full_obs)
    if sync_policy_timing:
        _sync_rollout_policy_timing(device)
    t_model = perf_counter()
    timing.policy_model_s += t_model - t0

    halt_logits = out["halt_logits"]
    halt_lp = torch.log_softmax(halt_logits, dim=-1)
    if greedy:
        halt_sampled = halt_logits.argmax(dim=-1)
    else:
        halt_sampled = torch.multinomial(halt_lp.exp(), 1, generator=rng).squeeze(-1)
    halt_action = torch.where(must_halt, torch.ones_like(halt_sampled), halt_sampled)
    halt_action = torch.where(pending_t, halt_action, torch.ones_like(halt_action))
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

    o_idx = origin_frac_flat // len(FRACTIONS)
    frac_idx = origin_frac_flat % len(FRACTIONS)
    origin_frac_used = pending_t & (halt_action == 0) & any_valid_origin_frac
    if sync_policy_timing:
        _sync_rollout_policy_timing(device)
    t_origin = perf_counter()
    timing.policy_sample_origin_s += t_origin - t_model

    if sync_policy_timing:
        _sync_rollout_policy_timing(device)
    t_ray_start = perf_counter()
    origin_idx_j = jax.dlpack.from_dlpack(o_idx.to(torch.int32).contiguous().detach())
    frac_idx_geom_j = jax.dlpack.from_dlpack(frac_idx.to(torch.int32).contiguous().detach())
    (
        _target_angle_j,
        _target_width_j,
        target_valid_j,
        target_overflow_j,
        target_hit_tick_j,
        target_true_planet_j,
        target_true_hit_tick_j,
    ) = _selected_origin_fraction_targets_batched_maybe_chunked(
        virt_b,
        origin_idx_j,
        frac_idx_geom_j,
        horizon=24,
        ship_speed=ship_speed,
        samples_per_span=17,
        n_rays=first_hit_n_rays,
        ray_chunk_size=first_hit_ray_chunk_size,
        first_hit_method=first_hit_method,
        env_chunk_size=first_hit_env_chunk_size,
    )
    if sync_policy_timing:
        _sync_rollout_policy_timing(
            device,
            (
                _target_angle_j,
                target_valid_j,
                target_overflow_j,
                target_hit_tick_j,
                target_true_planet_j,
                target_true_hit_tick_j,
            ),
        )
    target_valid_t = torch.from_dlpack(target_valid_j).to(device)
    target_overflow_t = torch.from_dlpack(target_overflow_j).to(device)
    target_hit_tick_t = torch.from_dlpack(target_hit_tick_j).to(device)
    target_true_planet_t = torch.from_dlpack(target_true_planet_j).to(device)
    target_true_hit_tick_t = torch.from_dlpack(target_true_hit_tick_j).to(device)
    if sync_policy_timing:
        _sync_rollout_policy_timing(device)
    t_raycast = perf_counter()
    timing.policy_raycast_s += t_raycast - (t_ray_start if sync_policy_timing else t_origin)

    if sync_policy_timing:
        _sync_rollout_policy_timing(device)
    t_target_start = perf_counter()
    n_idx = torch.arange(total_rows, device=device)
    planet_ships = full_obs["features"][:, 1 : 1 + MAX_PLANETS, 1] * 1000.0
    origin_ships = full_obs["features"][n_idx, 1 + o_idx, 1] * 1000.0
    frac_values = origin_ships.new_tensor(FRACTIONS)
    fleet_size_for_logits = torch.floor(frac_values[frac_idx] * origin_ships)
    target_logits = policy.target_logits_for_origin_fraction_grouped_population(
        out["planet_hidden"],
        o_idx,
        frac_idx,
        fleet_size=fleet_size_for_logits,
        target_eta=target_hit_tick_t,
        target_ships=planet_ships,
    )
    abort_logits_all = out.get("abort_logits")
    abort_logits = None
    if abort_logits_all is not None:
        abort_logits = abort_logits_all[n_idx, o_idx, frac_idx]
    target_mask = out["pair_mask"][n_idx, o_idx, :] & target_valid_t & ~target_overflow_t[:, None]
    any_valid_target = target_mask.any(dim=-1)
    target_abort = torch.zeros((total_rows,), dtype=torch.bool, device=device)
    if abort_logits is not None:
        combined_target = torch.cat([target_logits.masked_fill(~target_mask, -1e4), abort_logits[:, None]], dim=-1)
        target_lp = torch.log_softmax(combined_target, dim=-1)
        if greedy:
            target_choice = combined_target.argmax(dim=-1)
        else:
            target_choice = torch.multinomial(torch.softmax(combined_target, dim=-1), 1, generator=rng).squeeze(-1)
        target_abort = target_choice == MAX_PLANETS
        d_idx = target_choice.clamp_max(MAX_PLANETS - 1)
        target_logp = target_lp.gather(1, target_choice[:, None]).squeeze(-1)
    else:
        masked_target = target_logits.masked_fill(~target_mask, -1e4)
        target_lp = torch.log_softmax(masked_target, dim=-1)
        safe_target = torch.where(any_valid_target[:, None], masked_target, torch.zeros_like(masked_target))
        if greedy:
            d_idx = safe_target.argmax(dim=-1)
        else:
            d_idx = torch.multinomial(torch.softmax(safe_target, dim=-1), 1, generator=rng).squeeze(-1)
        target_logp = target_lp.gather(1, d_idx[:, None]).squeeze(-1)
    pair_flat = o_idx * MAX_PLANETS + d_idx
    policy_fleet_eta = target_hit_tick_t[n_idx, d_idx]
    true_d_idx = target_true_planet_t[n_idx, d_idx].to(torch.long).clamp(0, MAX_PLANETS - 1)
    true_fleet_eta = target_true_hit_tick_t[n_idx, d_idx]

    dispatch_used = origin_frac_used & (~target_abort) & any_valid_target
    total_logp = torch.where(
        pending_t,
        halt_logp
        + origin_frac_used.float() * origin_frac_logp
        + (origin_frac_used if abort_logits is not None else dispatch_used).float() * target_logp,
        torch.zeros_like(halt_logp),
    )
    values_all = out["value"].float()
    apply_halt_now = ~dispatch_used
    apply_halt_now = torch.where(pending_t, apply_halt_now, torch.ones_like(apply_halt_now))
    stop_now = torch.where(
        pending_t,
        (halt_action == 1) | ((halt_action == 0) & ~any_valid_origin_frac),
        torch.ones_like(halt_action, dtype=torch.bool),
    )
    total_logp = total_logp.detach()
    values_all = values_all.detach()
    halt_action = halt_action.detach()
    target_abort = target_abort.detach()
    pair_flat = pair_flat.detach()
    frac_idx = frac_idx.detach()
    o_idx = o_idx.detach()
    d_idx = d_idx.detach()
    origin_frac_used = origin_frac_used.detach()
    apply_halt_now = apply_halt_now.detach()
    stop_now = stop_now.detach()
    policy_fleet_eta = policy_fleet_eta.detach()
    pending_t = pending_t.detach()
    del out
    if sync_policy_timing:
        _sync_rollout_policy_timing(device)
    t_target = perf_counter()
    timing.policy_target_s += t_target - (t_target_start if sync_policy_timing else t_raycast)

    if sync_policy_timing:
        _sync_rollout_policy_timing(device)
    t_scatter_start = perf_counter()
    if sync_policy_timing:
        _sync_rollout_policy_timing(device)
    t_scatter = perf_counter()
    timing.policy_scatter_s += t_scatter - (t_scatter_start if sync_policy_timing else t_target)
    timing.policy_forward_s += t_scatter - t0

    t0 = perf_counter()
    halt_now_j = jax.dlpack.from_dlpack(apply_halt_now.to(torch.bool).contiguous().detach())
    pair_flat_j = jax.dlpack.from_dlpack(pair_flat.to(torch.int32).contiguous().detach())
    frac_idx_j = jax.dlpack.from_dlpack(frac_idx.to(torch.int32).contiguous().detach())
    fleet_eta_j = jax.dlpack.from_dlpack(policy_fleet_eta.to(torch.float32).contiguous().detach())
    target_abort_j = jax.dlpack.from_dlpack(target_abort.to(torch.bool).contiguous().detach())
    t1 = perf_counter()

    virt_b, oid_j, send_j, dispatched_j, slot_j = apply_micro_step_batched_per_ego(
        virt_b, row_ego_j, halt_now_j, pair_flat_j, frac_idx_j, fleet_eta_j, target_abort_j
    )
    t2 = perf_counter()

    oid_t = torch.from_dlpack(oid_j)
    send_t = torch.from_dlpack(send_j)
    dispatched_t = torch.from_dlpack(dispatched_j)
    slot_t = torch.from_dlpack(slot_j)
    t3 = perf_counter()

    t3a = perf_counter()
    for p in range(n_ego):
        envs_p = row_env_np[pending_rows & (row_ego_np == p)]
        if envs_p.size:
            rows_np = write_idx[p][envs_p]
            micro_np = micro_k[p, envs_p]
            _validate_numpy_index_range(f"rollout write_idx p{p}", rows_np, int(bufs[p].micro_halt_now.shape[0]))
            _validate_numpy_index_range(f"rollout micro_k p{p}", micro_np, max_micro_steps)
    t3b = perf_counter()
    t4 = perf_counter()

    for p in range(n_ego):
        row_sel = np.flatnonzero(pending_rows & (row_ego_np == p)).astype(np.int32)
        if row_sel.size == 0:
            continue
        env_sel = row_env_np[row_sel]
        row_sel_t = torch.as_tensor(row_sel, device=device, dtype=torch.long)
        env_sel_t = torch.as_tensor(env_sel, device=device, dtype=torch.long)
        micro_kp_t = micro_k_dev[p].index_select(0, env_sel_t)
        write_row_t = write_idx_dev[p].index_select(0, env_sel_t)
        bufs[p] = append_active_to_torch_buffer(
            bufs[p],
            env_sel_t,
            apply_halt_now.index_select(0, row_sel_t),
            send_t.index_select(0, row_sel_t),
            policy_fleet_eta.index_select(0, row_sel_t),
            slot_t.index_select(0, row_sel_t),
            halt_action.index_select(0, row_sel_t).to(torch.int32),
            target_abort.index_select(0, row_sel_t),
            pair_flat.index_select(0, row_sel_t).to(torch.int32),
            frac_idx.index_select(0, row_sel_t).to(torch.int32),
            (origin_frac_used & ~any_valid_target).index_select(0, row_sel_t),
            (~any_valid_origin_frac & (halt_action == 0) & pending_t).index_select(0, row_sel_t),
            must_halt.index_select(0, row_sel_t),
            (target_valid_t & ~target_overflow_t[:, None]).index_select(0, row_sel_t).to(torch.bool),
            target_hit_tick_t.index_select(0, row_sel_t).to(torch.float32),
            population_idx_t.index_select(0, row_sel_t),
            torch.zeros_like(population_idx_t.index_select(0, row_sel_t), dtype=torch.int32),
            torch.zeros_like(population_idx_t.index_select(0, row_sel_t), dtype=torch.int32),
            write_row_t,
            micro_kp_t,
            max_micro_steps,
        )
        comp_p_active = index_compressed_observation_rows(obs_comp_torch, row_sel_t)
        obs_bufs[p] = store_precompressed_observation_rows(
            obs_bufs[p], write_row_t.to(torch.long), env_sel_t, comp_p_active
        )
    t5 = perf_counter()

    t6 = perf_counter()

    oid_np = oid_t.detach().cpu().numpy()
    send_np = send_t.detach().cpu().numpy()
    dispatched_np = dispatched_t.detach().cpu().numpy()
    t7 = perf_counter()
    _accum_micro_apply_breakdown(timing, t0, t1, t2, t3, t3a, t3b, t4, t5, t6, t7)

    t_post0 = perf_counter()
    total_logp_np = total_logp.detach().cpu().numpy()
    values_np = values_all.detach().cpu().numpy()
    stop_now_np = stop_now.detach().cpu().numpy()
    d_idx_np = d_idx.detach().cpu().numpy()
    true_d_idx_np = true_d_idx.detach().cpu().numpy()
    true_fleet_eta_np = true_fleet_eta.detach().cpu().numpy()
    policy_fleet_eta_np = policy_fleet_eta.detach().cpu().numpy()
    timing.micro_post_apply_extract_s += perf_counter() - t_post0

    t_post1 = perf_counter()
    for p in range(n_ego):
        row_sel = np.flatnonzero(pending_rows & (row_ego_np == p)).astype(np.int32)
        if row_sel.size == 0:
            continue
        env_sel = row_env_np[row_sel]
        rows_np = write_idx[p][env_sel]
        valid[p][rows_np, env_sel] = True
        old_logprob[p][rows_np, env_sel] = total_logp_np[row_sel]
        old_value[p][rows_np, env_sel] = values_np[row_sel]
        if micro_step_penalty != 0.0:
            reward[p][rows_np, env_sel] -= float(micro_step_penalty) * dispatched_np[row_sel].astype(np.float32)
    t_post2 = perf_counter()
    timing.micro_post_apply_row_stats_s += t_post2 - t_post1

    t_post3 = perf_counter()
    for p in range(n_ego):
        row_sel = np.flatnonzero(pending_rows & (row_ego_np == p)).astype(np.int32)
        if row_sel.size == 0:
            continue
        env_sel = row_env_np[row_sel]
        widx = write_idx[p]
        micro_arr = micro_k[p]
        for row in row_sel:
            env_i = int(row_env_np[row])
            if bool(dispatched_np[row]):
                ac = int(pending_action_count[p, env_i])
                if ac < pending_actions.shape[2]:
                    pending_actions[env_i, p, ac, 0] = float(oid_np[row])
                    pending_actions[env_i, p, ac, 1] = float(send_np[row])
                    pending_actions[env_i, p, ac, 2] = float(true_d_idx_np[row])
                    pending_actions[env_i, p, ac, 3] = float(true_fleet_eta_np[row])
                    pending_actions[env_i, p, ac, 4] = float(d_idx_np[row])
                    pending_actions[env_i, p, ac, 5] = float(policy_fleet_eta_np[row])
                pending_action_count[p, env_i] = ac + 1
    t_post4 = perf_counter()
    timing.micro_post_apply_pending_actions_s += t_post4 - t_post3

    t_post5 = perf_counter()
    for p in range(n_ego):
        row_sel = np.flatnonzero(pending_rows & (row_ego_np == p)).astype(np.int32)
        if row_sel.size == 0:
            continue
        env_sel = row_env_np[row_sel]
        widx = write_idx[p]
        micro_arr = micro_k[p]
        reward_idx[p, env_sel] = widx[env_sel]
        halted[p, env_sel] |= stop_now_np[row_sel].astype(np.bool_, copy=False)
        widx[env_sel] += 1
        micro_arr[env_sel] += 1
        halted[p, env_sel] |= micro_arr[env_sel] >= max_micro_steps

    t_post6 = perf_counter()
    timing.micro_post_apply_halt_block_indices_s += t_post6 - t_post5
    timing.micro_post_apply_host_bookkeeping_s += t_post6 - t_post1

    t_post7 = perf_counter()
    for p in range(n_ego):
        row_sel = np.flatnonzero(pending_rows & (row_ego_np == p)).astype(np.int32)
        if row_sel.size == 0:
            continue
        env_sel = row_env_np[row_sel]
        env_sel_t = torch.as_tensor(env_sel, device=device, dtype=torch.long)
        micro_k_dev[p, env_sel_t] = micro_k_dev[p, env_sel_t] + 1
        write_idx_dev[p][env_sel_t] = write_idx_dev[p][env_sel_t] + 1
    timing.micro_post_apply_device_index_s += perf_counter() - t_post7

    return virt_b, bufs, obs_bufs


def _reset_prefetch_resync(
    reset_prefetch: Optional[RolloutResetPrefetch],
    seed_base: int,
    seeds_consumed: int,
    cfg: OrbitWarsEnvConfig,
    *,
    unified_exploiter_rollout: bool = False,
    unified_seed_state: Optional[dict[str, int]] = None,
) -> None:
    if reset_prefetch is None:
        return
    reset_prefetch.notify_max_fleets(int(cfg.max_fleets))
    if unified_exploiter_rollout:
        assert unified_seed_state is not None
        reset_prefetch.prefetch_unified_exploiter_ahead(
            int(unified_seed_state["two_p"]),
            int(unified_seed_state["four_p"]),
            int(cfg.max_fleets),
        )
    else:
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
    device_reset_bank: Optional[_DeviceResetBank] = None,
    first_hit_n_rays: int = 2048,
    first_hit_ray_chunk_size: int = 0,
    first_hit_env_chunk_size: int = 0,
    first_hit_method: str = "category-rays",
    micro_step_penalty: float = 1e-4,
    sync_policy_timing: bool = False,
    additional_policies: Optional[list[OrbitWarsPolicy]] = None,
    controller_counts: Optional[tuple[int, ...]] = None,
    termination_controller: Optional[int] = None,
    controller_assignment_template: Optional[np.ndarray] = None,
    main_player_mask_template: Optional[np.ndarray] = None,
    termination_requires_all_main_dead: bool = False,
    env_mode_by_env: Optional[np.ndarray] = None,
    earlygame_env_turn_limit: int = 0,
) -> Tuple[RolloutSegment, RolloutTiming, RolloutCarry, int, RolloutGameStats]:
    """Collect one rollout segment using device-resident transition buffers.

    Stops after a full env-step when any env's any player's micro-step count in
    this segment reaches ``rollout_micro_horizon`` (end-of-turn cut).
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

    requested_num_envs = int(num_envs)
    if carry_in is not None:
        carry_num_envs = int(carry_in.state_b.planets.shape[0])
        if carry_num_envs != requested_num_envs:
            if carry_in.env_mode_by_env is not None and env_mode_by_env is None:
                raise ValueError(
                    f"carry_in state has {carry_num_envs} envs but requested num_envs {requested_num_envs}; "
                    "cannot reset unified exploiter rollout without matching env_mode_by_env"
                )
            carry_in = None
    num_envs = requested_num_envs

    seeds_consumed = 0
    t_init0 = perf_counter()
    policies = [policy] + list(additional_policies or [])
    population_size = int(getattr(policy, "population_size", 1))
    grouped_population_rollout = population_size > 1
    if grouped_population_rollout and len(policies) > 1:
        raise ValueError("grouped population rollout does not support multiple disjoint policies")
    if controller_counts is None:
        controller_counts = (int(cfg_template.num_agents),)
    controller_counts = tuple(int(x) for x in controller_counts)
    if sum(controller_counts) != int(cfg_template.num_agents):
        raise ValueError(
            f"controller_counts {controller_counts} must sum to cfg_template.num_agents={int(cfg_template.num_agents)}"
        )
    versus_controller_rollout = len(policies) > 1 and int(controller_counts[0]) > 0 and termination_controller is not None
    fixed_controller_assignments = (
        None
        if controller_assignment_template is None
        else _broadcast_controller_assignment_template(
            np.asarray(controller_assignment_template, dtype=np.int32),
            num_envs=int(num_envs),
            num_agents=int(cfg_template.num_agents),
        )
    )
    fixed_main_player_mask = (
        None
        if main_player_mask_template is None
        else _broadcast_main_player_mask_template(
            np.asarray(main_player_mask_template, dtype=np.bool_),
            num_envs=int(num_envs),
            num_agents=int(cfg_template.num_agents),
        )
    )
    carry_mode_arr = None if carry_in is None or carry_in.env_mode_by_env is None else np.asarray(carry_in.env_mode_by_env)
    unified_exploiter_rollout = (env_mode_by_env is not None) or (carry_mode_arr is not None)
    plain_device_bank_mode = (reset_prefetch is not None) and (not unified_exploiter_rollout)
    unified_seed_state = _normalize_unified_exploiter_seed_state(seed_base) if unified_exploiter_rollout else None
    if reset_prefetch is not None:
        mf0 = int(carry_in.cfg.max_fleets) if carry_in is not None else int(cfg_template.max_fleets)
        na0 = int(cfg_template.num_agents)
        reset_prefetch.notify_max_fleets(mf0)
        if unified_exploiter_rollout:
            assert unified_seed_state is not None
            reset_prefetch.prefetch_unified_exploiter_ahead(
                int(unified_seed_state["two_p"]),
                int(unified_seed_state["four_p"]),
                mf0,
            )
        elif not plain_device_bank_mode:
            reset_prefetch.prefetch_ahead(int(seed_base) + seeds_consumed, na0, mf0)
    if carry_in is None:
        mode_arr = None
        if unified_exploiter_rollout:
            cfg = OrbitWarsEnvConfig(
                num_agents=cfg_template.num_agents,
                max_fleets=cfg_template.max_fleets,
                episode_seed=cfg_template.episode_seed,
                reward_mode=cfg_template.reward_mode,
                reward_ship_mass_share_coef=cfg_template.reward_ship_mass_share_coef,
                reward_ship_mass_share_member_coefs=cfg_template.reward_ship_mass_share_member_coefs,
                reward_production_share_coef=cfg_template.reward_production_share_coef,
                reward_production_share_member_coefs=cfg_template.reward_production_share_member_coefs,
                reward_terminal_win_loss_coef=cfg_template.reward_terminal_win_loss_coef,
                reward_terminal_win_loss_member_coefs=cfg_template.reward_terminal_win_loss_member_coefs,
                reward_terminal_loss=cfg_template.reward_terminal_loss,
                reward_terminal_draw=cfg_template.reward_terminal_draw,
                reward_terminal_win=cfg_template.reward_terminal_win,
                reward_time_bonus_coef=cfg_template.reward_time_bonus_coef,
                reward_time_bonus_member_coefs=cfg_template.reward_time_bonus_member_coefs,
                normalize_obs_to_p0=cfg_template.normalize_obs_to_p0,
            )
            mode_arr = np.asarray(env_mode_by_env, dtype=np.int32).reshape(-1)
            if mode_arr.shape[0] != num_envs:
                raise ValueError(f"env_mode_by_env length {mode_arr.shape[0]} != num_envs {num_envs}")
            states: list[OrbitWarsState] = []
            ctrl_cols: list[np.ndarray] = []
            main_cols: list[np.ndarray] = []
            init_env_seeds: list[int] = []
            for env_i, mode_code in enumerate(mode_arr.tolist()):
                active_seat_count = unified_exploiter_active_seat_count(int(mode_code))
                assert unified_seed_state is not None
                sid = _take_unified_exploiter_seed(unified_seed_state, active_seat_count)
                init_env_seeds.append(int(sid))
                ctrl_i, main_i = sample_unified_exploiter_mode_layout(sid, int(mode_code))
                if reset_prefetch is not None:
                    fresh_np, prefetch_meta = reset_prefetch.pop_unified_exploiter_state(
                        sid,
                        active_seat_count,
                        int(cfg.max_fleets),
                        return_meta=True,
                    )
                    _accum_prefetch_pop_meta(
                        timing,
                        prefetch_meta,
                        init_phase=True,
                        active_seat_count=active_seat_count,
                    )
                    state_i = jax.tree.map(jnp.asarray, fresh_np)
                else:
                    state_i, _, _ = build_unified_exploiter_reset(
                        sid, int(mode_code), int(cfg.max_fleets)
                    )
                states.append(state_i)
                ctrl_cols.append(ctrl_i.astype(np.int32, copy=False))
                main_cols.append(main_i.astype(np.bool_, copy=False))
            state_b = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *states)
            controller_assignments = np.stack(ctrl_cols, axis=1)
            main_player_mask = np.stack(main_cols, axis=1)
        else:
            state_b, cfg = stack_initial_states(
                cfg_template, num_envs, seed_base, reset_prefetch=reset_prefetch
            )
            controller_assignments = None
            main_player_mask = None
            init_env_seeds = [int(seed_base) + env_i for env_i in range(num_envs)]
        seeds_consumed += num_envs
        episode_turns = [0] * num_envs
        player_done = (
            np.asarray(controller_assignments < 0, dtype=np.bool_)
            if controller_assignments is not None
            else np.zeros((int(cfg.num_agents), num_envs), dtype=np.bool_)
        )
        pending_exploiter_terminal = np.zeros((int(cfg.num_agents), num_envs), dtype=np.bool_)
        total_rows = int(cfg.num_agents) * int(num_envs)
        rows_per_member = _population_rows_per_member(total_rows, population_size)
        if grouped_population_rollout:
            policy_row_for_seat = _init_policy_row_mapping(
                num_envs=num_envs,
                num_agents=int(cfg.num_agents),
                population_size=population_size,
                seed=seed_base,
            )
            population_assignments = _population_assignments_from_policy_rows(
                policy_row_for_seat, rows_per_member, population_size
            )
        else:
            policy_row_for_seat = None
            population_assignments = np.stack(
                [
                    _sample_population_assignments_for_env(
                        int(init_env_seeds[env_i]),
                        int(cfg.num_agents),
                        population_size,
                    )
                    for env_i in range(num_envs)
                ],
                axis=1,
            )
        if controller_assignments is None:
            if fixed_controller_assignments is not None:
                controller_assignments = np.asarray(fixed_controller_assignments, dtype=np.int32).copy()
            else:
                controller_assignments = np.stack(
                    [
                        _sample_controller_assignments_for_env(
                            int(init_env_seeds[env_i]),
                            int(cfg.num_agents),
                            controller_counts,
                        )
                        for env_i in range(num_envs)
                    ],
                    axis=1,
                )
        if main_player_mask is None:
            if fixed_main_player_mask is not None:
                main_player_mask = np.asarray(fixed_main_player_mask, dtype=np.bool_).copy()
            else:
                main_player_mask = np.stack(
                    [
                        _sample_main_player_mask_for_env(controller_assignments[:, env_i], termination_controller)
                        for env_i in range(num_envs)
                    ],
                    axis=1,
                )
    else:
        state_b, cfg = carry_in.state_b, carry_in.cfg
        cfg.reward_mode = cfg_template.reward_mode
        cfg.reward_ship_mass_share_coef = cfg_template.reward_ship_mass_share_coef
        cfg.reward_ship_mass_share_member_coefs = cfg_template.reward_ship_mass_share_member_coefs
        cfg.reward_production_share_coef = cfg_template.reward_production_share_coef
        cfg.reward_production_share_member_coefs = cfg_template.reward_production_share_member_coefs
        cfg.reward_terminal_win_loss_coef = cfg_template.reward_terminal_win_loss_coef
        cfg.reward_terminal_win_loss_member_coefs = cfg_template.reward_terminal_win_loss_member_coefs
        cfg.reward_terminal_loss = cfg_template.reward_terminal_loss
        cfg.reward_terminal_draw = cfg_template.reward_terminal_draw
        cfg.reward_terminal_win = cfg_template.reward_terminal_win
        cfg.reward_time_bonus_coef = cfg_template.reward_time_bonus_coef
        cfg.reward_time_bonus_member_coefs = cfg_template.reward_time_bonus_member_coefs
        cfg.normalize_obs_to_p0 = cfg_template.normalize_obs_to_p0
        episode_turns = list(carry_in.episode_turns)
        if len(episode_turns) != num_envs:
            episode_turns = [0] * num_envs
        mode_arr = None if carry_in.env_mode_by_env is None else np.asarray(carry_in.env_mode_by_env, dtype=np.int32)
        ca = carry_in.controller_assignments
        if ca is None:
            if fixed_controller_assignments is not None:
                controller_assignments = np.asarray(fixed_controller_assignments, dtype=np.int32).copy()
            else:
                controller_assignments = np.stack(
                    [
                        _sample_controller_assignments_for_env(seed_base + env_i, int(cfg.num_agents), controller_counts)
                        for env_i in range(num_envs)
                    ],
                    axis=1,
                )
        else:
            controller_assignments = np.asarray(ca, dtype=np.int32)
            if controller_assignments.shape != (int(cfg.num_agents), num_envs):
                if fixed_controller_assignments is not None:
                    controller_assignments = np.asarray(fixed_controller_assignments, dtype=np.int32).copy()
                else:
                    controller_assignments = np.stack(
                        [
                            _sample_controller_assignments_for_env(seed_base + env_i, int(cfg.num_agents), controller_counts)
                            for env_i in range(num_envs)
                        ],
                        axis=1,
                    )
        pd = carry_in.player_done
        if pd is None:
            player_done = np.asarray(controller_assignments < 0, dtype=np.bool_)
        else:
            player_done = np.asarray(pd, dtype=np.bool_)
            if player_done.shape != (int(cfg.num_agents), num_envs):
                player_done = np.asarray(controller_assignments < 0, dtype=np.bool_)
        pet = carry_in.pending_exploiter_terminal
        if pet is None:
            pending_exploiter_terminal = np.zeros((int(cfg.num_agents), num_envs), dtype=np.bool_)
        else:
            pending_exploiter_terminal = np.asarray(pet, dtype=np.bool_)
            if pending_exploiter_terminal.shape != (int(cfg.num_agents), num_envs):
                pending_exploiter_terminal = np.zeros((int(cfg.num_agents), num_envs), dtype=np.bool_)
        pop = carry_in.population_assignments
        if pop is None:
            population_assignments = np.zeros((int(cfg.num_agents), num_envs), dtype=np.int32)
        else:
            population_assignments = np.asarray(pop, dtype=np.int32)
            if population_assignments.shape != (int(cfg.num_agents), num_envs):
                population_assignments = np.zeros((int(cfg.num_agents), num_envs), dtype=np.int32)
        total_rows = int(cfg.num_agents) * int(num_envs)
        rows_per_member = _population_rows_per_member(total_rows, population_size)
        prs = carry_in.policy_row_for_seat
        mpm = carry_in.main_player_mask
        if mpm is None:
            if fixed_main_player_mask is not None:
                main_player_mask = np.asarray(fixed_main_player_mask, dtype=np.bool_).copy()
            else:
                main_player_mask = np.stack(
                    [
                        _sample_main_player_mask_for_env(controller_assignments[:, env_i], termination_controller)
                        for env_i in range(num_envs)
                    ],
                    axis=1,
                )
        else:
            main_player_mask = np.asarray(mpm, dtype=np.bool_)
            if main_player_mask.shape != (int(cfg.num_agents), num_envs):
                if fixed_main_player_mask is not None:
                    main_player_mask = np.asarray(fixed_main_player_mask, dtype=np.bool_).copy()
                else:
                    main_player_mask = np.stack(
                        [
                            _sample_main_player_mask_for_env(controller_assignments[:, env_i], termination_controller)
                            for env_i in range(num_envs)
                        ],
                        axis=1,
                    )
        if grouped_population_rollout:
            if prs is None:
                policy_row_for_seat = _init_policy_row_mapping(
                    num_envs=num_envs,
                    num_agents=int(cfg.num_agents),
                    population_size=population_size,
                    seed=seed_base,
                )
            else:
                policy_row_for_seat = np.asarray(prs, dtype=np.int32)
                if policy_row_for_seat.shape != (int(cfg.num_agents), num_envs):
                    policy_row_for_seat = _init_policy_row_mapping(
                        num_envs=num_envs,
                        num_agents=int(cfg.num_agents),
                        population_size=population_size,
                        seed=seed_base,
                    )
            population_assignments = _population_assignments_from_policy_rows(
                policy_row_for_seat, rows_per_member, population_size
            )
        else:
            policy_row_for_seat = None

    obs_feature_dim = int(policy.feat_proj.in_features)
    n_ego = int(cfg.num_agents)
    if device_reset_bank is None and reset_prefetch is not None and not unified_exploiter_rollout:
        device_reset_bank = _DeviceResetBank(capacity=int(reset_prefetch.lookahead))
    if device_reset_bank is None:
        _reset_prefetch_resync(
            reset_prefetch,
            seed_base,
            seeds_consumed,
            cfg,
            unified_exploiter_rollout=unified_exploiter_rollout,
            unified_seed_state=unified_seed_state,
        )
    if device_reset_bank is not None:
        t_bank0 = perf_counter()
        _stage_plain_reset_bank(
            device_reset_bank,
            reset_prefetch,
            first_seed=int(seed_base) + int(seeds_consumed),
            num_agents=int(cfg.num_agents),
            max_fleets=int(cfg.max_fleets),
            target_size=int(device_reset_bank.capacity),
            block_until_full=True,
            timing=timing,
            sync_policy_timing=sync_policy_timing,
            device=device,
        )
        timing.init_reset_bank_s += perf_counter() - t_bank0

    episode_lim = int(np.asarray(jax.device_get(jow.OrbitWarsConfig().episode_steps)))
    episode_timeout_step_count = episode_lim - 1
    game_stats = RolloutGameStats(
        member_episode_count=np.zeros((population_size,), dtype=np.int64),
        member_positive_reward_count=np.zeros((population_size,), dtype=np.int64),
    )

    # Allocate per-segment device buffers. Once any player's write index hits
    # the horizon, async collection drains every env to its next real env-step
    # boundary. In the worst phase mix, any player's buffer may still need
    # one full local turn of micro rows before that boundary.
    H_buf = rollout_micro_horizon + max_micro_steps_per_player + 2
    t_buf0 = perf_counter()
    bufs = [
        init_torch_transition_buffer(num_envs, H_buf, max_micro_steps_per_player, device=device)
        for _ in range(n_ego)
    ]
    obs_bufs = [init_compressed_observation_buffer(num_envs, H_buf, device=device) for _ in range(n_ego)]

    valid = [np.zeros((H_buf, num_envs), dtype=np.bool_) for _ in range(n_ego)]
    old_logprob = [np.zeros((H_buf, num_envs), dtype=np.float32) for _ in range(n_ego)]
    old_value = [np.zeros((H_buf, num_envs), dtype=np.float32) for _ in range(n_ego)]
    reward = [np.zeros((H_buf, num_envs), dtype=np.float32) for _ in range(n_ego)]
    done = [np.zeros((H_buf, num_envs), dtype=np.bool_) for _ in range(n_ego)]
    write_idx = [np.zeros((num_envs,), dtype=np.int32) for _ in range(n_ego)]
    write_idx_dev = [torch.zeros((num_envs,), dtype=torch.int32, device=device) for _ in range(n_ego)]
    trunc_bootstrap = [np.zeros((H_buf, num_envs), dtype=np.float32) for _ in range(n_ego)]
    trunc_bootstrap_valid = [np.zeros((H_buf, num_envs), dtype=np.bool_) for _ in range(n_ego)]
    timing.init_buffer_alloc_s += perf_counter() - t_buf0

    t_state_setup0 = perf_counter()
    env_steps_per_env = np.zeros((num_envs,), dtype=np.int32)
    horizon_fired = False
    timing.init_state_setup_s += perf_counter() - t_state_setup0
    timing.init_s = perf_counter() - t_init0

    logged_first_policy_fwd = False
    outer = 0
    t_loop0 = perf_counter()
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if amp_dtype is not None
        else nullcontext()
    )

    t_state0 = perf_counter()
    if grouped_population_rollout:
        assert policy_row_for_seat is not None
        row_env, row_ego = _invert_policy_row_mapping(policy_row_for_seat)
        row_env_j = jnp.asarray(row_env, dtype=jnp.int32)
        virt_b = _gather_state_rows(state_b, row_env_j)
    else:
        row_env = None
        row_ego = None
        row_env_j = None
        virt_b = jax.tree.map(
            lambda leaf: jnp.concatenate([leaf] * n_ego, axis=0),
            state_b,
        )
    timing.state_unstack_s += perf_counter() - t_state0
    halted = np.zeros((n_ego, num_envs), dtype=np.bool_)
    micro_k = np.zeros((n_ego, num_envs), dtype=np.int32)
    micro_k_dev = torch.zeros((n_ego, num_envs), dtype=torch.int32, device=device)
    pending_actions = np.zeros((num_envs, int(cfg.num_agents), DEFAULT_MAX_ACTIONS, 6), dtype=np.float32)
    pending_actions[..., 2] = -1.0
    pending_actions[..., 3] = 500.0
    pending_actions[..., 4] = -1.0
    pending_actions[..., 5] = 500.0
    pending_action_count = np.zeros((n_ego, num_envs), dtype=np.int32)
    reward_idx = np.full((n_ego, num_envs), -1, dtype=np.int32)
    earlygame_bootstrap_row = np.full((n_ego, num_envs), -1, dtype=np.int32)
    segment_done = np.zeros((num_envs,), dtype=np.bool_)
    # First env to reset during this segment, recorded for the optional
    # background consistency check. Stored as ``(env_i, new_seed,
    # write_idx_at_reset_per_seat)``; ``write_idx_at_reset`` is the per-seat
    # row count BEFORE the reset (i.e. the first post-reset row index).
    first_reset_event: Optional[Tuple[int, int, np.ndarray]] = None

    # Keep grad mode consistent for torch.compile policy forwards (inference_mode /
    # no_grad elsewhere caused grad_mode guard failures and recompiles). Detach
    # stored logprob/value immediately after compute (see micro-step helpers).
    with amp_ctx:
        while outer < max_outer_iters:
            outer += 1

            if reset_prefetch is not None:
                timing.reset_prefetch_drained_results += int(reset_prefetch.drain_ready(max_items=32))

            t_ctrl0 = perf_counter()
            if horizon_fired and np.all(segment_done):
                timing.loop_control_s += perf_counter() - t_ctrl0
                break

            ready_mask = (~segment_done) & np.all(halted | player_done, axis=0)
            pending = (~segment_done)[None, :] & (~halted) & (~player_done)
            earlygame_value_only = pending & (earlygame_bootstrap_row >= 0)
            if grouped_population_rollout:
                assert row_env is not None and row_ego is not None
                pending_rows = (~segment_done[row_env]) & (~halted[row_ego, row_env]) & (~player_done[row_ego, row_env])
            else:
                pending_rows = None
            ready_count = int(np.sum(ready_mask))
            pending_total = int(np.sum(pending))

            # Env stepping is cheap relative to policy/geometry work, but
            # tiny step buckets still pay host/JAX dispatch overhead. Step
            # once roughly a quarter-batch is ready, or force the step when
            # every remaining player is already halted.
            ready_step_threshold = max(2, num_envs // 4)
            should_step = ready_count >= ready_step_threshold or (ready_count > 0 and pending_total == 0)
            timing.loop_control_s += perf_counter() - t_ctrl0
            if should_step:
                t0 = perf_counter()
                t_prep0 = perf_counter()
                step_envs = np.flatnonzero(ready_mask).astype(np.int32)
                ready_mask_j = jnp.asarray(ready_mask, dtype=jnp.bool_)
                actions_bucket = pending_actions.copy()
                actions_bucket[~ready_mask] = 0.0
                actions_bucket[~ready_mask, :, :, 2] = -1.0
                actions_bucket[~ready_mask, :, :, 3] = 500.0
                actions_bucket[~ready_mask, :, :, 4] = -1.0
                actions_bucket[~ready_mask, :, :, 5] = 500.0
                timing.env_prep_s += perf_counter() - t_prep0

                t_coef0 = perf_counter()
                ship_reward_coef_bucket = _reward_coef_matrix_for_population(
                    population_assignments,
                    cfg.reward_ship_mass_share_coef,
                    cfg.reward_ship_mass_share_member_coefs,
                    int(cfg.num_agents),
                )
                production_reward_coef_bucket = _reward_coef_matrix_for_population(
                    population_assignments,
                    cfg.reward_production_share_coef,
                    cfg.reward_production_share_member_coefs,
                    int(cfg.num_agents),
                )
                terminal_reward_coef_bucket = _reward_coef_matrix_for_population(
                    population_assignments,
                    cfg.reward_terminal_win_loss_coef,
                    cfg.reward_terminal_win_loss_member_coefs,
                    int(cfg.num_agents),
                )
                time_reward_coef_bucket = _reward_coef_matrix_for_population(
                    population_assignments,
                    cfg.reward_time_bonus_coef,
                    cfg.reward_time_bonus_member_coefs,
                    int(cfg.num_agents),
                )
                timing.env_coef_s += perf_counter() - t_coef0
                t_reward0 = perf_counter()
                ratios_pre_jax = reward_mix_ratios_batched(
                    state_b,
                    reward_ship_mass_share_coef=ship_reward_coef_bucket,
                    reward_production_share_coef=production_reward_coef_bucket,
                )
                if sync_policy_timing:
                    _sync_rollout_policy_timing(device, ratios_pre_jax)
                timing.env_reward_s += perf_counter() - t_reward0

                t_core0 = perf_counter()
                next_bucket = step_env_masked_batched(
                    state_b,
                    actions_bucket,
                    ready_mask_j,
                    cfg.reward_terminal_loss,
                    cfg.reward_terminal_draw,
                    cfg.reward_terminal_win,
                )
                if sync_policy_timing:
                    _sync_rollout_policy_timing(device, next_bucket)
                timing.env_step_core_s += perf_counter() - t_core0

                t_reward1 = perf_counter()
                dr_jax = reward_delta_from_state_pair_batched(
                    state_b,
                    next_bucket,
                    ratios_pre_jax,
                    reward_ship_mass_share_coef=ship_reward_coef_bucket,
                    reward_production_share_coef=production_reward_coef_bucket,
                    reward_terminal_win_loss_coef=terminal_reward_coef_bucket,
                    reward_terminal_win=cfg.reward_terminal_win,
                    reward_time_bonus_coef=time_reward_coef_bucket,
                )
                if sync_policy_timing:
                    _sync_rollout_policy_timing(device, dr_jax)
                timing.env_reward_s += perf_counter() - t_reward1

                t_stats0 = perf_counter()
                alive_post_jax, s0_post, s1_post, min_garrison_jax = post_step_stats_batched(next_bucket)
                if sync_policy_timing:
                    _sync_rollout_policy_timing(device, alive_post_jax, s0_post, s1_post, min_garrison_jax)
                timing.env_post_stats_s += perf_counter() - t_stats0

                t_host0 = perf_counter()
                dr_np, alive_post_np, done_np, step_count_np, rewards_np, s0_fin_np, s1_fin_np, min_garrison_np = jax.device_get(
                    (
                        dr_jax,
                        alive_post_jax,
                        next_bucket.done,
                        next_bucket.step_count,
                        next_bucket.rewards,
                        s0_post,
                        s1_post,
                        min_garrison_jax,
                    )
                )
                timing.env_host_transfer_s += perf_counter() - t_host0
                dr_np = np.asarray(dr_np)
                alive_post_np = np.asarray(alive_post_np, dtype=np.bool_)
                done_np = np.asarray(done_np)
                step_count_np = np.asarray(step_count_np)
                rewards_np = np.asarray(rewards_np)
                s0_fin_np = np.asarray(s0_fin_np)
                s1_fin_np = np.asarray(s1_fin_np)
                min_garrison = float(np.asarray(min_garrison_np, dtype=np.float32))
                if min_garrison < -1e-6:
                    raise RuntimeError(
                        f"negative env-step garrison detected after step_env_with_scores_batched: "
                        f"min_garrison={min_garrison:+g}"
                    )
                state_b = next_bucket

                t_py0 = perf_counter()
                done_envs: list[int] = []
                done_env_seed: dict[int, int] = {}
                reset_envs_apply: list[int] = []
                reset_fresh_states: list[Any] = []
                plain_reset_fallback_seeds: list[int] = []
                trunc_reset_rows: list[tuple[int, int, int]] = []
                bootstrap_controller_assignments = np.asarray(controller_assignments, dtype=np.int32).copy()
                bootstrap_main_player_mask = np.asarray(main_player_mask, dtype=np.bool_).copy()
                reset_total_s = 0.0
                for env_i in step_envs:
                    env_i = int(env_i)
                    env_steps_per_env[env_i] += 1
                    episode_turns[env_i] += 1
                    main_alive_mask = np.asarray(alive_post_np[env_i, : int(cfg.num_agents)], dtype=np.bool_)
                    env_controller_assignments = np.asarray(controller_assignments[:, env_i], dtype=np.int32)
                    env_has_secondary_policy = bool(np.any(env_controller_assignments == 1))
                    main_slots_mask = np.asarray(main_player_mask[:, env_i], dtype=np.bool_)
                    main_dead = False
                    if env_has_secondary_policy and np.any(main_slots_mask):
                        if bool(termination_requires_all_main_dead) or int(np.count_nonzero(main_slots_mask)) > 1:
                            main_dead = bool(np.all(~main_alive_mask[main_slots_mask]))
                        else:
                            main_dead = bool(np.any(~main_alive_mask[main_slots_mask]))
                    env_done_now = bool(done_np[env_i]) or main_dead
                    grouped_earlygame_truncate_now = (
                        grouped_population_rollout
                        and int(earlygame_env_turn_limit) > 0
                        and (env_i % 2) == 0
                        and int(episode_turns[env_i]) >= int(earlygame_env_turn_limit)
                        and not env_done_now
                    )
                    earlygame_bootstrap_now = (
                        (not grouped_population_rollout)
                        and int(earlygame_env_turn_limit) > 0
                        and (env_i % 2) == 0
                        and int(episode_turns[env_i]) == int(earlygame_env_turn_limit)
                        and not env_done_now
                    )
                    earlygame_reset_now = (
                        (not grouped_population_rollout)
                        and int(earlygame_env_turn_limit) > 0
                        and (env_i % 2) == 0
                        and int(episode_turns[env_i]) > int(earlygame_env_turn_limit)
                        and not env_done_now
                    )
                    reset_env_now = env_done_now or grouped_earlygame_truncate_now or earlygame_reset_now
                    main_policy_win: Optional[bool] = None
                    time_bonus_scale = 0.0
                    if env_has_secondary_policy and env_done_now:
                        main_slots = np.flatnonzero(main_player_mask[:, env_i]).astype(np.int32)
                        if main_dead:
                            main_policy_win = False
                        else:
                            if main_slots.size > 0:
                                main_policy_win = bool(
                                    np.any(np.asarray(rewards_np[env_i, main_slots], dtype=np.float32) > 0.0)
                                )
                            else:
                                main_policy_win = False
                        timeout_step = int(step_count_np[env_i]) >= episode_timeout_step_count
                        if (not timeout_step) and (not bool(main_policy_win)):
                            timeout_turn = max(1.0, float(episode_lim - 2))
                            pre_turn = max(0.0, float(step_count_np[env_i]) - 1.0)
                            time_bonus_scale = max(0.0, 1.0 - (pre_turn / timeout_turn))
                    synthetic_dead_exploiters: list[int] = []
                    for p in range(n_ego):
                        if reward_idx[p, env_i] >= 0:
                            seat_dead = not bool(alive_post_np[env_i, p])
                            local_done = reset_env_now or earlygame_bootstrap_now or seat_dead
                            delta_r = float(dr_np[env_i, p])
                            if env_has_secondary_policy and int(controller_assignments[p, env_i]) == 1:
                                delta_r = 0.0
                                if env_done_now and main_policy_win is not None:
                                    terminal_coef = float(terminal_reward_coef_bucket[env_i, p])
                                    member_time_bonus = float(time_reward_coef_bucket[env_i, p]) * time_bonus_scale
                                    delta_r = (
                                        terminal_coef + member_time_bonus if not bool(main_policy_win) else -terminal_coef
                                    )
                                elif seat_dead:
                                    local_done = False
                                    pending_exploiter_terminal[p, env_i] = True
                            reward[p][reward_idx[p, env_i], env_i] += delta_r
                            done[p][reward_idx[p, env_i], env_i] = local_done or bool(
                                done[p][reward_idx[p, env_i], env_i]
                            )
                            if (grouped_earlygame_truncate_now or earlygame_bootstrap_now) and not seat_dead:
                                if grouped_population_rollout:
                                    trunc_reset_rows.append((p, env_i, int(reward_idx[p, env_i])))
                                else:
                                    earlygame_bootstrap_row[p, env_i] = int(reward_idx[p, env_i])
                        if not bool(alive_post_np[env_i, p]):
                            player_done[p, env_i] = True
                        if (
                            env_done_now
                            and env_has_secondary_policy
                            and int(controller_assignments[p, env_i]) == 1
                            and bool(pending_exploiter_terminal[p, env_i])
                        ):
                            synthetic_dead_exploiters.append(int(p))
                    if synthetic_dead_exploiters:
                        row_count = len(synthetic_dead_exploiters)
                        term_state = _gather_state_rows(
                            next_bucket,
                            jnp.asarray([env_i] * row_count, dtype=jnp.int32),
                        )
                        synthetic_dead_exploiters_np = np.asarray(synthetic_dead_exploiters, dtype=np.int32)
                        synthetic_terminal_coef = terminal_reward_coef_bucket[
                            env_i, synthetic_dead_exploiters_np
                        ].astype(np.float32, copy=False)
                        synthetic_time_bonus = (
                            time_reward_coef_bucket[env_i, synthetic_dead_exploiters_np].astype(np.float32, copy=False)
                            * np.float32(time_bonus_scale)
                        )
                        synthetic_reward = (
                            (
                                synthetic_terminal_coef + synthetic_time_bonus
                                if not bool(main_policy_win)
                                else -synthetic_terminal_coef
                            ).astype(np.float32, copy=False)
                            if main_policy_win is not None
                            else np.zeros((row_count,), dtype=np.float32)
                        )
                        _append_synthetic_terminal_rows(
                            state_rows=term_state,
                            ego_idx=synthetic_dead_exploiters_np,
                            env_idx=np.full((row_count,), env_i, dtype=np.int32),
                            controller_idx=np.ones((row_count,), dtype=np.int32),
                            population_idx=population_assignments[
                                synthetic_dead_exploiters_np, env_i
                            ].astype(np.int32, copy=False),
                            value_head_idx=_relative_main_value_head_idx_for_rows(
                                ego_idx=synthetic_dead_exploiters_np,
                                env_idx=np.full((row_count,), env_i, dtype=np.int32),
                                controller_assignments=controller_assignments,
                                main_player_mask=main_player_mask,
                                normalize_obs_to_p0=cfg.normalize_obs_to_p0,
                            ),
                            terminal_reward=synthetic_reward,
                            ship_speed=ship_speed,
                            obs_feature_dim=obs_feature_dim,
                            normalize_obs_to_p0=cfg.normalize_obs_to_p0,
                            policies=policies,
                            bufs=bufs,
                            obs_bufs=obs_bufs,
                            write_idx=write_idx,
                            write_idx_dev=write_idx_dev,
                            valid=valid,
                            old_logprob=old_logprob,
                            old_value=old_value,
                            reward=reward,
                            done=done,
                            device=device,
                        )
                        pending_exploiter_terminal[
                            np.asarray(synthetic_dead_exploiters, dtype=np.int32), env_i
                        ] = False
                    if env_done_now:
                        done_envs.append(env_i)
                        sc_i = int(step_count_np[env_i])
                        game_stats.record_completion(
                            step_limit=sc_i >= episode_timeout_step_count,
                            ships_p0=float(s0_fin_np[env_i]),
                            ships_p1=float(s1_fin_np[env_i]),
                            episode_turns=int(episode_turns[env_i]),
                            reward0=float(rewards_np[env_i, 0]),
                            reward1=float(rewards_np[env_i, 1]),
                            reward_by_player=np.asarray(rewards_np[env_i, : int(cfg.num_agents)]),
                            population_assignment=np.asarray(population_assignments[: int(cfg.num_agents), env_i]),
                            main_policy_win=main_policy_win,
                            env_mode=None if mode_arr is None else int(mode_arr[env_i]),
                            main_eliminated=main_dead,
                        )
                        episode_turns[env_i] = 0
                        t_reset0 = perf_counter()
                        if mode_arr is not None:
                            active_seat_count = unified_exploiter_active_seat_count(int(mode_arr[env_i]))
                            assert unified_seed_state is not None
                            sid = _take_unified_exploiter_seed(unified_seed_state, active_seat_count)
                            ctrl_i, main_i = sample_unified_exploiter_mode_layout(sid, int(mode_arr[env_i]))
                            if reset_prefetch is not None:
                                fresh_np, prefetch_meta = reset_prefetch.pop_unified_exploiter_state(
                                    sid,
                                    active_seat_count,
                                    int(cfg.max_fleets),
                                    return_meta=True,
                                )
                                _accum_prefetch_pop_meta(
                                    timing,
                                    prefetch_meta,
                                    init_phase=False,
                                    active_seat_count=active_seat_count,
                                )
                            else:
                                state_i, _, _ = build_unified_exploiter_reset(
                                    sid, int(mode_arr[env_i]), int(cfg.max_fleets)
                                )
                                fresh_np = jax.device_get(state_i)
                            reset_envs_apply.append(env_i)
                            reset_fresh_states.append(fresh_np)
                            controller_assignments[:, env_i] = ctrl_i.astype(np.int32, copy=False)
                            main_player_mask[:, env_i] = main_i.astype(np.bool_, copy=False)
                        elif reset_prefetch is not None:
                            sid = int(seed_base + seeds_consumed)
                            reset_envs_apply.append(env_i)
                            plain_reset_fallback_seeds.append(sid)
                        else:
                            sid = int(seed_base + seeds_consumed)
                            state_i = jow.reset_from_reference(sid, int(cfg.num_agents), max_fleets=int(cfg.max_fleets))
                            reset_envs_apply.append(env_i)
                            reset_fresh_states.append(jax.device_get(state_i))
                        reset_dt = perf_counter() - t_reset0
                        reset_total_s += reset_dt
                        timing.env_reset_count += 1
                        if mode_arr is not None:
                            if active_seat_count == 2:
                                timing.env_reset_mode_2p_count += 1
                            elif active_seat_count == 4:
                                timing.env_reset_mode_4p_count += 1
                        seeds_consumed += 1
                        if mode_arr is not None or reset_prefetch is None:
                            done_env_seed[env_i] = sid
                        if (mode_arr is not None or reset_prefetch is None) and first_reset_event is None:
                            first_reset_event = (
                                int(env_i),
                                int(sid),
                                np.asarray(
                                    [int(write_idx[p][env_i]) for p in range(n_ego)],
                                    dtype=np.int64,
                                ),
                            )
                    elif reset_env_now:
                        done_envs.append(env_i)
                        episode_turns[env_i] = 0
                        t_reset0 = perf_counter()
                        if mode_arr is not None:
                            active_seat_count = unified_exploiter_active_seat_count(int(mode_arr[env_i]))
                            assert unified_seed_state is not None
                            sid = _take_unified_exploiter_seed(unified_seed_state, active_seat_count)
                            ctrl_i, main_i = sample_unified_exploiter_mode_layout(sid, int(mode_arr[env_i]))
                            if reset_prefetch is not None:
                                fresh_np, prefetch_meta = reset_prefetch.pop_unified_exploiter_state(
                                    sid,
                                    active_seat_count,
                                    int(cfg.max_fleets),
                                    return_meta=True,
                                )
                                _accum_prefetch_pop_meta(
                                    timing,
                                    prefetch_meta,
                                    init_phase=False,
                                    active_seat_count=active_seat_count,
                                )
                            else:
                                state_i, _, _ = build_unified_exploiter_reset(
                                    sid, int(mode_arr[env_i]), int(cfg.max_fleets)
                                )
                                fresh_np = jax.device_get(state_i)
                            reset_envs_apply.append(env_i)
                            reset_fresh_states.append(fresh_np)
                            controller_assignments[:, env_i] = ctrl_i.astype(np.int32, copy=False)
                            main_player_mask[:, env_i] = main_i.astype(np.bool_, copy=False)
                        elif reset_prefetch is not None:
                            sid = int(seed_base + seeds_consumed)
                            reset_envs_apply.append(env_i)
                            plain_reset_fallback_seeds.append(sid)
                        else:
                            sid = int(seed_base + seeds_consumed)
                            state_i = jow.reset_from_reference(sid, int(cfg.num_agents), max_fleets=int(cfg.max_fleets))
                            reset_envs_apply.append(env_i)
                            reset_fresh_states.append(jax.device_get(state_i))
                        reset_dt = perf_counter() - t_reset0
                        reset_total_s += reset_dt
                        timing.env_reset_count += 1
                        if mode_arr is not None:
                            if active_seat_count == 2:
                                timing.env_reset_mode_2p_count += 1
                            elif active_seat_count == 4:
                                timing.env_reset_mode_4p_count += 1
                        seeds_consumed += 1
                        if mode_arr is not None or reset_prefetch is None:
                            done_env_seed[env_i] = sid
                        if (mode_arr is not None or reset_prefetch is None) and first_reset_event is None:
                            first_reset_event = (
                                int(env_i),
                                int(sid),
                                np.asarray(
                                    [int(write_idx[p][env_i]) for p in range(n_ego)],
                                    dtype=np.int64,
                                ),
                            )

                    halted[:, env_i] = player_done[:, env_i]
                    micro_k[:, env_i] = 0
                    pending_actions[env_i] = 0.0
                    pending_actions[env_i, :, :, 2] = -1.0
                    pending_actions[env_i, :, :, 3] = 500.0
                    pending_actions[env_i, :, :, 4] = -1.0
                    pending_actions[env_i, :, :, 5] = 500.0
                    pending_action_count[:, env_i] = 0
                    reward_idx[:, env_i] = -1
                    if reset_env_now:
                        earlygame_bootstrap_row[:, env_i] = -1
                    if horizon_fired:
                        segment_done[env_i] = True
                if trunc_reset_rows:
                    value_np_per_ego = _compute_state_values_per_ego(
                        state_b=state_b,
                        grouped_population_rollout=grouped_population_rollout,
                        row_env=row_env,
                        row_ego=row_ego,
                        row_env_j=row_env_j,
                        num_envs=num_envs,
                        n_ego=n_ego,
                        ship_speed=ship_speed,
                        obs_feature_dim=obs_feature_dim,
                        normalize_obs_to_p0=cfg.normalize_obs_to_p0,
                        population_assignments=population_assignments,
                        controller_assignments=bootstrap_controller_assignments,
                        main_player_mask=bootstrap_main_player_mask,
                        policies=policies,
                        policy=policy,
                        device=device,
                    )
                    for ego, env_i, row_t in trunc_reset_rows:
                        trunc_bootstrap[ego][row_t, env_i] = float(value_np_per_ego[ego, env_i])
                        trunc_bootstrap_valid[ego][row_t, env_i] = True
                if mode_arr is None and plain_reset_fallback_seeds:
                    batch_seeds: list[int] = []
                    batch_metas: list[PrefetchPopMeta] = []
                    bank_take = (
                        0
                        if device_reset_bank is None
                        else min(len(plain_reset_fallback_seeds), int(device_reset_bank.size()))
                    )
                    if bank_take > 0:
                        bank_reserved = device_reset_bank.reserve_tail(bank_take)
                        if bank_reserved is None or device_reset_bank.states is None:
                            raise RuntimeError("device reset bank reserve_tail unexpectedly returned None")
                        bank_seeds, bank_start, bank_take = bank_reserved
                        bank_env_idx = np.asarray(reset_envs_apply[:bank_take], dtype=np.int32)
                        if sync_policy_timing:
                            _sync_rollout_policy_timing(device)
                        t_reset_bank0 = perf_counter()
                        state_b = reset_envs_from_bank_tail(
                            state_b,
                            bank_env_idx,
                            device_reset_bank.states,
                            bank_start,
                        )
                        if sync_policy_timing:
                            _sync_rollout_policy_timing(device, state_b)
                        bank_dt = perf_counter() - t_reset_bank0
                        timing.env_reset_bank_slice_s += bank_dt
                        reset_total_s += bank_dt
                        batch_seeds.extend(int(x) for x in bank_seeds)
                        batch_metas.extend([PrefetchPopMeta(immediate_bank_hit=True) for _ in range(bank_take)])
                    remaining = len(plain_reset_fallback_seeds) - len(batch_seeds)
                    if remaining > 0:
                        host_seeds, host_states, host_metas = _resolve_plain_reset_host_batch(
                            reset_count=remaining,
                            fallback_seeds=plain_reset_fallback_seeds[len(batch_seeds):],
                            reset_prefetch=reset_prefetch,
                            num_agents=int(cfg.num_agents),
                            max_fleets=int(cfg.max_fleets),
                            timing=timing,
                            sync_policy_timing=sync_policy_timing,
                            device=device,
                        )
                        host_env_idx = np.asarray(reset_envs_apply[len(batch_seeds):], dtype=np.int32)
                        t_reset_apply0 = perf_counter()
                        state_b = reset_envs_at_indices_batched(
                            state_b,
                            host_env_idx,
                            host_states,
                        )
                        if sync_policy_timing:
                            _sync_rollout_policy_timing(device, state_b)
                        apply_dt = perf_counter() - t_reset_apply0
                        timing.env_reset_apply_s += apply_dt
                        reset_total_s += apply_dt
                        batch_seeds.extend(host_seeds)
                        batch_metas.extend(host_metas)
                    for env_i, sid, meta in zip(reset_envs_apply, batch_seeds, batch_metas):
                        _accum_prefetch_pop_meta(
                            timing,
                            meta,
                            init_phase=False,
                            active_seat_count=int(cfg.num_agents),
                        )
                        done_env_seed[int(env_i)] = int(sid)
                        if first_reset_event is None:
                            first_reset_event = (
                                int(env_i),
                                int(sid),
                                np.asarray(
                                    [int(write_idx[p][env_i]) for p in range(n_ego)],
                                    dtype=np.int64,
                                ),
                            )
                    reset_envs_apply = []
                    reset_fresh_states = []
                    plain_reset_fallback_seeds = []
                if reset_envs_apply:
                    t_reset_apply0 = perf_counter()
                    state_b = reset_envs_at_indices(
                        state_b,
                        np.asarray(reset_envs_apply, dtype=np.int32),
                        reset_fresh_states,
                    )
                    if sync_policy_timing:
                        _sync_rollout_policy_timing(device, state_b)
                    apply_dt = perf_counter() - t_reset_apply0
                    timing.env_reset_apply_s += apply_dt
                    reset_total_s += apply_dt
                timing.env_reset_s += reset_total_s
                t_py0 += reset_total_s
                if grouped_population_rollout and done_envs:
                    assert policy_row_for_seat is not None
                    done_envs_np = np.asarray(done_envs, dtype=np.int32)
                    policy_row_for_seat = _reassign_policy_rows_for_reset_envs(
                        policy_row_for_seat=policy_row_for_seat,
                        done_envs=done_envs_np,
                        seed=seed_base + seeds_consumed,
                    )
                    population_assignments = _population_assignments_from_policy_rows(
                        policy_row_for_seat, rows_per_member, population_size
                    )
                    for env_i in done_envs_np:
                        if fixed_controller_assignments is not None:
                            controller_assignments[:, env_i] = fixed_controller_assignments[:, env_i]
                        else:
                            controller_assignments[:, env_i] = _sample_controller_assignments_for_env(
                                done_env_seed[int(env_i)], int(cfg.num_agents), controller_counts
                            )
                        if fixed_main_player_mask is not None:
                            main_player_mask[:, env_i] = fixed_main_player_mask[:, env_i]
                        else:
                            main_player_mask[:, env_i] = _sample_main_player_mask_for_env(
                                controller_assignments[:, env_i], termination_controller
                            )
                    pending_exploiter_terminal[:, done_envs_np] = False
                    player_done[:, done_envs_np] = False
                    halted[:, done_envs_np] = False
                elif done_envs:
                    for env_i in done_envs:
                        player_done[:, env_i] = controller_assignments[:, env_i] < 0 if mode_arr is not None else False
                        pending_exploiter_terminal[:, env_i] = False
                        population_assignments[:, env_i] = _sample_population_assignments_for_env(
                            done_env_seed[env_i], int(cfg.num_agents), population_size
                        )
                        if mode_arr is None:
                            if fixed_controller_assignments is not None:
                                controller_assignments[:, env_i] = fixed_controller_assignments[:, env_i]
                            else:
                                controller_assignments[:, env_i] = _sample_controller_assignments_for_env(
                                    done_env_seed[env_i], int(cfg.num_agents), controller_counts
                                )
                            if fixed_main_player_mask is not None:
                                main_player_mask[:, env_i] = fixed_main_player_mask[:, env_i]
                            else:
                                main_player_mask[:, env_i] = _sample_main_player_mask_for_env(
                                    controller_assignments[:, env_i], termination_controller
                                )
                timing.env_python_s += perf_counter() - t_py0

                t_book0 = perf_counter()
                step_env_t = torch.as_tensor(step_envs, dtype=torch.long, device=device)
                micro_k_dev[:, step_env_t] = 0
                ready_mask_j = jnp.asarray(ready_mask, dtype=jnp.bool_)
                if device_reset_bank is None:
                    _reset_prefetch_resync(
                        reset_prefetch,
                        seed_base,
                        seeds_consumed,
                        cfg,
                        unified_exploiter_rollout=unified_exploiter_rollout,
                        unified_seed_state=unified_seed_state,
                    )
                if grouped_population_rollout:
                    assert policy_row_for_seat is not None
                    row_env, row_ego = _invert_policy_row_mapping(policy_row_for_seat)
                    row_env_j = jnp.asarray(row_env, dtype=jnp.int32)
                    virt_b = _refresh_virt_from_state_masked_grouped(
                        virt_b,
                        state_b,
                        row_env_j,
                        ready_mask_j,
                    )
                else:
                    virt_b = _refresh_virt_from_state_masked_non_grouped(
                        virt_b,
                        state_b,
                        ready_mask_j,
                    )
                if sync_policy_timing:
                    _sync_rollout_policy_timing(device, virt_b)
                timing.env_bookkeeping_s += perf_counter() - t_book0
                timing.env_step_s += perf_counter() - t0
                continue

            t_ctrl1 = perf_counter()
            if pending_total == 0:
                timing.loop_control_s += perf_counter() - t_ctrl1
                break
            timing.loop_control_s += perf_counter() - t_ctrl1

            if grouped_population_rollout:
                assert row_env is not None and row_ego is not None and pending_rows is not None
                virt_b, bufs, obs_bufs = _run_async_micro_step_multi_grouped_population(
                    n_ego=n_ego,
                    num_envs=num_envs,
                    pending_rows=pending_rows,
                    virt_b=virt_b,
                    bufs=bufs,
                    obs_bufs=obs_bufs,
                    write_idx=write_idx,
                    write_idx_dev=write_idx_dev,
                    micro_k=micro_k,
                    micro_k_dev=micro_k_dev,
                    valid=valid,
                    old_logprob=old_logprob,
                    old_value=old_value,
                    reward=reward,
                    row_env=row_env,
                    row_ego=row_ego,
                    rows_per_member=rows_per_member,
                    pending_actions=pending_actions,
                    pending_action_count=pending_action_count,
                    reward_idx=reward_idx,
                    halted=halted,
                    policies=policies,
                    device=device,
                    rng=rng,
                    greedy=greedy,
                    ship_speed=ship_speed,
                    max_micro_steps=max_micro_steps_per_player,
                    timing=timing,
                    first_hit_n_rays=first_hit_n_rays,
                    micro_step_penalty=micro_step_penalty,
                    first_hit_ray_chunk_size=first_hit_ray_chunk_size,
                    first_hit_env_chunk_size=first_hit_env_chunk_size,
                    first_hit_method=first_hit_method,
                    obs_feature_dim=obs_feature_dim,
                    normalize_obs_to_p0=cfg.normalize_obs_to_p0,
                    sync_policy_timing=sync_policy_timing,
                    )
            else:
                virt_b, bufs, obs_bufs, value_capture = _run_async_micro_step_multi(
                        n_ego=n_ego,
                        num_envs=num_envs,
                        pending=pending,
                        virt_b=virt_b,
                        bufs=bufs,
                        obs_bufs=obs_bufs,
                        write_idx=write_idx,
                        write_idx_dev=write_idx_dev,
                        micro_k=micro_k,
                        micro_k_dev=micro_k_dev,
                        valid=valid,
                        old_logprob=old_logprob,
                        old_value=old_value,
                        reward=reward,
                        population_assignments=population_assignments,
                        pending_actions=pending_actions,
                        pending_action_count=pending_action_count,
                        reward_idx=reward_idx,
                        halted=halted,
                        policies=policies,
                        device=device,
                        rng=rng,
                        greedy=greedy,
                        ship_speed=ship_speed,
                        max_micro_steps=max_micro_steps_per_player,
                        timing=timing,
                        first_hit_n_rays=first_hit_n_rays,
                        micro_step_penalty=micro_step_penalty,
                        first_hit_ray_chunk_size=first_hit_ray_chunk_size,
                        first_hit_env_chunk_size=first_hit_env_chunk_size,
                        first_hit_method=first_hit_method,
                        obs_feature_dim=obs_feature_dim,
                        normalize_obs_to_p0=cfg.normalize_obs_to_p0,
                        sync_policy_timing=sync_policy_timing,
                        controller_assignments=controller_assignments,
                        main_player_mask=main_player_mask,
                        value_only=earlygame_value_only,
                    )
                if value_capture is not None and np.any(earlygame_bootstrap_row >= 0):
                    t_eg0 = perf_counter()
                    values_np, active_env_np, active_ego_np, active_value_only_np = value_capture
                    value_only_rows = np.flatnonzero(active_value_only_np).astype(np.int32)
                    for row in value_only_rows.tolist():
                        ego = int(active_ego_np[row])
                        env_i = int(active_env_np[row])
                        row_t = int(earlygame_bootstrap_row[ego, env_i])
                        if row_t >= 0:
                            trunc_bootstrap[ego][row_t, env_i] = float(values_np[row])
                            trunc_bootstrap_valid[ego][row_t, env_i] = True
                    timing.env_python_s += perf_counter() - t_eg0
            t_post0 = perf_counter()
            if profile_rollout and device.type == "cuda" and not logged_first_policy_fwd:
                log_cuda_mem("rollout after first batched policy forward", device)
                logged_first_policy_fwd = True

            if any(np.any(write_idx[p] >= rollout_micro_horizon) for p in range(n_ego)):
                horizon_fired = True
            if sync_policy_timing:
                _sync_rollout_policy_timing(device)
            timing.loop_post_micro_s += perf_counter() - t_post0

    if (not grouped_population_rollout) and np.any(earlygame_bootstrap_row >= 0):
        t_fallback0 = perf_counter()
        with amp_ctx:
            value_np_per_ego = _compute_state_values_per_ego(
                state_b=state_b,
                grouped_population_rollout=grouped_population_rollout,
                row_env=row_env,
                row_ego=row_ego,
                row_env_j=row_env_j,
                num_envs=num_envs,
                n_ego=n_ego,
                ship_speed=ship_speed,
                obs_feature_dim=obs_feature_dim,
                normalize_obs_to_p0=cfg.normalize_obs_to_p0,
                population_assignments=population_assignments,
                controller_assignments=controller_assignments,
                main_player_mask=main_player_mask,
                policies=policies,
                policy=policy,
                device=device,
                timing=timing,
            )
        reset_envs = np.flatnonzero(np.any(earlygame_bootstrap_row >= 0, axis=0)).astype(np.int32)
        for env_i in reset_envs.tolist():
            for ego in range(n_ego):
                row_t = int(earlygame_bootstrap_row[ego, env_i])
                if row_t >= 0:
                    trunc_bootstrap[ego][row_t, env_i] = float(value_np_per_ego[ego, env_i])
                    trunc_bootstrap_valid[ego][row_t, env_i] = True

        deferred_seed: dict[int, int] = {}
        deferred_fresh_states: list[Any] = []
        deferred_plain_fallback_seeds: list[int] = []
        for env_i in reset_envs.tolist():
            episode_turns[env_i] = 0
            if mode_arr is not None:
                active_seat_count = unified_exploiter_active_seat_count(int(mode_arr[env_i]))
                assert unified_seed_state is not None
                sid = _take_unified_exploiter_seed(unified_seed_state, active_seat_count)
                ctrl_i, main_i = sample_unified_exploiter_mode_layout(sid, int(mode_arr[env_i]))
                if reset_prefetch is not None:
                    fresh_np, prefetch_meta = reset_prefetch.pop_unified_exploiter_state(
                        sid,
                        active_seat_count,
                        int(cfg.max_fleets),
                        return_meta=True,
                    )
                    _accum_prefetch_pop_meta(
                        timing,
                        prefetch_meta,
                        init_phase=False,
                        active_seat_count=active_seat_count,
                    )
                else:
                    state_i, _, _ = build_unified_exploiter_reset(
                        sid, int(mode_arr[env_i]), int(cfg.max_fleets)
                    )
                    fresh_np = jax.device_get(state_i)
                controller_assignments[:, env_i] = ctrl_i.astype(np.int32, copy=False)
                main_player_mask[:, env_i] = main_i.astype(np.bool_, copy=False)
            elif reset_prefetch is not None:
                sid = int(seed_base + seeds_consumed)
                deferred_plain_fallback_seeds.append(sid)
            else:
                sid = int(seed_base + seeds_consumed)
                state_i = jow.reset_from_reference(sid, int(cfg.num_agents), max_fleets=int(cfg.max_fleets))
                fresh_np = jax.device_get(state_i)
            if mode_arr is not None or reset_prefetch is None:
                deferred_fresh_states.append(fresh_np)
            if mode_arr is not None or reset_prefetch is None:
                deferred_seed[env_i] = sid
            timing.env_reset_count += 1
            if mode_arr is not None:
                if active_seat_count == 2:
                    timing.env_reset_mode_2p_count += 1
                elif active_seat_count == 4:
                    timing.env_reset_mode_4p_count += 1
            seeds_consumed += 1
            if (mode_arr is not None or reset_prefetch is None) and first_reset_event is None:
                first_reset_event = (
                    int(env_i),
                    int(sid),
                    np.asarray([int(write_idx[p][env_i]) for p in range(n_ego)], dtype=np.int64),
                )
        if reset_envs.size:
            if mode_arr is None and deferred_plain_fallback_seeds:
                batch_seeds: list[int] = []
                batch_metas: list[PrefetchPopMeta] = []
                bank_take = (
                    0
                    if device_reset_bank is None
                    else min(len(deferred_plain_fallback_seeds), int(device_reset_bank.size()))
                )
                if bank_take > 0:
                    bank_reserved = device_reset_bank.reserve_tail(bank_take)
                    if bank_reserved is None or device_reset_bank.states is None:
                        raise RuntimeError("device reset bank reserve_tail unexpectedly returned None")
                    bank_seeds, bank_start, bank_take = bank_reserved
                    bank_env_idx = reset_envs[:bank_take]
                    if sync_policy_timing:
                        _sync_rollout_policy_timing(device)
                    t_reset_bank0 = perf_counter()
                    state_b = reset_envs_from_bank_tail(
                        state_b,
                        bank_env_idx,
                        device_reset_bank.states,
                        bank_start,
                    )
                    if sync_policy_timing:
                        _sync_rollout_policy_timing(device, state_b)
                    timing.env_reset_bank_slice_s += perf_counter() - t_reset_bank0
                    batch_seeds.extend(int(x) for x in bank_seeds)
                    batch_metas.extend([PrefetchPopMeta(immediate_bank_hit=True) for _ in range(bank_take)])
                remaining = len(deferred_plain_fallback_seeds) - len(batch_seeds)
                if remaining > 0:
                    host_seeds, host_states, host_metas = _resolve_plain_reset_host_batch(
                        reset_count=remaining,
                        fallback_seeds=deferred_plain_fallback_seeds[len(batch_seeds):],
                        reset_prefetch=reset_prefetch,
                        num_agents=int(cfg.num_agents),
                        max_fleets=int(cfg.max_fleets),
                        timing=timing,
                        sync_policy_timing=sync_policy_timing,
                        device=device,
                    )
                    host_env_idx = reset_envs[len(batch_seeds):]
                    t_reset_apply0 = perf_counter()
                    state_b = reset_envs_at_indices_batched(state_b, host_env_idx, host_states)
                    if sync_policy_timing:
                        _sync_rollout_policy_timing(device, state_b)
                    timing.env_reset_apply_s += perf_counter() - t_reset_apply0
                    batch_seeds.extend(host_seeds)
                    batch_metas.extend(host_metas)
                for env_i, sid, meta in zip(reset_envs.tolist(), batch_seeds, batch_metas):
                    _accum_prefetch_pop_meta(
                        timing,
                        meta,
                        init_phase=False,
                        active_seat_count=int(cfg.num_agents),
                    )
                    deferred_seed[int(env_i)] = int(sid)
                    if first_reset_event is None:
                        first_reset_event = (
                            int(env_i),
                            int(sid),
                            np.asarray([int(write_idx[p][env_i]) for p in range(n_ego)], dtype=np.int64),
                        )
            else:
                t_reset_apply0 = perf_counter()
                state_b = reset_envs_at_indices(state_b, reset_envs, deferred_fresh_states)
                if sync_policy_timing:
                    _sync_rollout_policy_timing(device, state_b)
                timing.env_reset_apply_s += perf_counter() - t_reset_apply0
            for env_i in reset_envs.tolist():
                pending_exploiter_terminal[:, env_i] = False
                population_assignments[:, env_i] = _sample_population_assignments_for_env(
                    deferred_seed[env_i], int(cfg.num_agents), population_size
                )
                if mode_arr is None:
                    if fixed_controller_assignments is not None:
                        controller_assignments[:, env_i] = fixed_controller_assignments[:, env_i]
                    else:
                        controller_assignments[:, env_i] = _sample_controller_assignments_for_env(
                            deferred_seed[env_i], int(cfg.num_agents), controller_counts
                        )
                    if fixed_main_player_mask is not None:
                        main_player_mask[:, env_i] = fixed_main_player_mask[:, env_i]
                    else:
                        main_player_mask[:, env_i] = _sample_main_player_mask_for_env(
                            controller_assignments[:, env_i], termination_controller
                        )
                player_done[:, env_i] = controller_assignments[:, env_i] < 0 if mode_arr is not None else False
                earlygame_bootstrap_row[:, env_i] = -1
        fallback_dt = perf_counter() - t_fallback0
        timing.env_reset_s += fallback_dt
        timing.env_python_s += fallback_dt

    timing.loop_s = perf_counter() - t_loop0
    timing.outer_iters = outer
    timing.wall_s = perf_counter() - t_wall0

    _assert_write_cursors_in_sync(write_idx, write_idx_dev, n_ego)

    if profile_rollout:
        log_cuda_mem("rollout exit (segment finished)", device)

    bootstrap = [np.zeros((num_envs,), dtype=np.float32) for _ in range(n_ego)]
    bootstrap_valid = [np.zeros((num_envs,), dtype=np.bool_) for _ in range(n_ego)]
    if horizon_fired:
        with amp_ctx:
            value_np_per_ego = _compute_state_values_per_ego(
                state_b=state_b,
                grouped_population_rollout=grouped_population_rollout,
                row_env=row_env,
                row_ego=row_ego,
                row_env_j=row_env_j,
                num_envs=num_envs,
                n_ego=n_ego,
                ship_speed=ship_speed,
                obs_feature_dim=obs_feature_dim,
                normalize_obs_to_p0=cfg.normalize_obs_to_p0,
                population_assignments=population_assignments,
                controller_assignments=controller_assignments,
                main_player_mask=main_player_mask,
                policies=policies,
                policy=policy,
                device=device,
                timing=timing,
            )
            for i in range(num_envs):
                for ego in range(n_ego):
                    last_t = int(write_idx[ego][i]) - 1
                    if last_t >= 0 and not bool(done[ego][last_t, i]):
                        bootstrap[ego][i] = float(value_np_per_ego[ego, i])
                        bootstrap_valid[ego][i] = True

    segment = RolloutSegment(
        bufs=bufs,
        obs_bufs=obs_bufs,
        write_idx=write_idx,
        valid=valid,
        old_logprob=old_logprob,
        old_value=old_value,
        reward=reward,
        done=done,
        bootstrap=bootstrap,
        bootstrap_valid=bootstrap_valid,
        trunc_bootstrap=trunc_bootstrap,
        trunc_bootstrap_valid=trunc_bootstrap_valid,
        env_steps_per_env=env_steps_per_env,
        env_mode_by_env=mode_arr if mode_arr is None else np.asarray(mode_arr, dtype=np.int32),
        first_reset_event=first_reset_event,
    )

    next_carry = RolloutCarry(
        state_b=state_b,
        cfg=cfg,
        episode_turns=episode_turns,
        player_done=player_done,
        population_assignments=population_assignments,
        policy_row_for_seat=policy_row_for_seat,
        controller_assignments=controller_assignments,
        main_player_mask=main_player_mask,
        env_mode_by_env=mode_arr if carry_in is not None else (None if env_mode_by_env is None else np.asarray(env_mode_by_env, dtype=np.int32)),
        pending_exploiter_terminal=pending_exploiter_terminal,
    )
    next_seed_state = unified_seed_state if unified_exploiter_rollout else seeds_consumed
    return segment, timing, next_carry, next_seed_state, game_stats

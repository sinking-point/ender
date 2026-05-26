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

from jax_orbit_wars import DEFAULT_MAX_ACTIONS, FLEET_ETA, FLEET_TARGET_PLANET, OrbitWarsState

from orbit_wars_pt.reset_prefetch import PrefetchPopMeta, RolloutResetPrefetch

from orbit_wars_pt.batched_env import (
    obs_jax_to_torch,
    post_step_stats_batched,
    reset_env_at_index,
    reset_envs_at_indices,
    reward_delta_from_state_pair_batched,
    reward_mix_ratios_batched,
    stack_initial_states,
    step_env_batched,
    step_env_masked_batched,
)
from orbit_wars_pt.compressed_observation import (
    CompressedObservationBuffer,
    compress_observation,
    decode_observation,
    init_compressed_observation_buffer,
    store_compressed_observation_rows,
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
from orbit_wars_pt.observation_jax import build_observation_batched_jax, build_observation_batched_jax_per_ego
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
    env_step_s: float = 0.0
    env_prep_s: float = 0.0
    env_state_gather_s: float = 0.0
    env_coef_s: float = 0.0
    env_step_core_s: float = 0.0
    env_reward_s: float = 0.0
    env_post_stats_s: float = 0.0
    env_host_transfer_s: float = 0.0
    env_reset_s: float = 0.0
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
) -> dict[str, torch.Tensor]:
    controller_outputs: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
    for controller_id in torch.unique(active_controller_idx_t, sorted=True).tolist():
        row_idx = torch.nonzero(active_controller_idx_t == int(controller_id), as_tuple=False).squeeze(-1)
        if int(row_idx.numel()) == 0:
            continue
        policy = policies[int(controller_id)]
        obs_slice = {key: value.index_select(0, row_idx) for key, value in active_obs.items()}
        pop_slice = active_population_idx_t.index_select(0, row_idx)
        out = policy.forward_dense_rollout(**obs_slice, population_idx=pop_slice)
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


@torch.no_grad()
def _append_synthetic_terminal_rows(
    *,
    state_rows: OrbitWarsState,
    ego_idx: np.ndarray,
    env_idx: np.ndarray,
    controller_idx: np.ndarray,
    population_idx: np.ndarray,
    terminal_reward: np.ndarray,
    ship_speed: float,
    obs_feature_dim: int,
    normalize_obs_to_p0: bool,
    policies: list[OrbitWarsPolicy],
    bufs: list[TorchTransitionBuffer],
    obs_bufs: list[CompressedObservationBuffer],
    write_idx: list[np.ndarray],
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
    rew_np = np.asarray(terminal_reward, dtype=np.float32).reshape(-1)
    if not (
        ego_np.shape == env_np.shape == ctrl_np.shape == pop_np.shape == rew_np.shape
    ):
        raise ValueError("synthetic terminal row metadata shapes must match")

    obs_j = build_observation_batched_jax_per_ego(
        state_rows,
        jnp.asarray(ego_np, dtype=jnp.int32),
        ship_speed,
        obs_feature_dim,
        normalize_to_p0=normalize_obs_to_p0,
    )
    obs_t = obs_jax_to_torch(obs_j)
    obs_t_dev = {key: value.to(device) for key, value in obs_t.items()}
    ctrl_t = torch.as_tensor(ctrl_np, device=device, dtype=torch.long)
    pop_t = torch.as_tensor(pop_np, device=device, dtype=torch.long)
    out = _forward_dense_rollout_by_controller(
        policies=policies,
        active_obs=obs_t_dev,
        active_population_idx_t=pop_t,
        active_controller_idx_t=ctrl_t,
    )
    value_np = out["value"].float().detach().cpu().numpy().astype(np.float32, copy=False)

    by_player: dict[int, list[int]] = {}
    for row_i, ego in enumerate(ego_np.tolist()):
        by_player.setdefault(int(ego), []).append(int(row_i))

    for ego, row_ids in by_player.items():
        env_sel = env_np[row_ids].astype(np.int32, copy=False)
        row_sel = np.asarray(row_ids, dtype=np.int32)
        row_sel_t = torch.as_tensor(row_sel, device=device, dtype=torch.long)
        wr_np = write_idx[ego][env_sel].astype(np.int32, copy=False)
        wr_t = torch.as_tensor(wr_np, device=device, dtype=torch.long)
        env_t = torch.as_tensor(env_sel, device=device, dtype=torch.long)
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
            zero_i32,
            zero_i32,
            true_bool,
            true_bool,
            true_bool,
            target_valid,
            target_hit_tick,
            torch.as_tensor(pop_np[row_sel], device=device, dtype=torch.int32),
            torch.as_tensor(ctrl_np[row_sel], device=device, dtype=torch.int32),
            wr_t,
            zero_i32,
            1,
        )
        obs_ego = {key: value.index_select(0, row_sel_t) for key, value in obs_t.items()}
        obs_bufs[ego] = store_compressed_observation_rows(obs_bufs[ego], wr_t, env_t, obs_ego)
        valid[ego][wr_np, env_sel] = True
        old_logprob[ego][wr_np, env_sel] = 0.0
        old_value[ego][wr_np, env_sel] = value_np[row_sel]
        reward[ego][wr_np, env_sel] += rew_np[row_sel]
        done[ego][wr_np, env_sel] = True
        write_idx[ego][env_sel] += 1


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
) -> tuple[OrbitWarsState, List[TorchTransitionBuffer], List[CompressedObservationBuffer]]:
    """Run one micro decision for every pending egocentric row in a ``n_ego * num_envs`` JAX batch."""

    players_active = [np.flatnonzero(pending[p]).astype(np.int32) for p in range(n_ego)]
    n_list = [int(x.size) for x in players_active]
    offsets: list[int] = []
    off = 0
    for n_p in n_list:
        offsets.append(off)
        off += n_p
    n_active = int(off)
    total_env_rows = int(virt_b.planets.shape[0])
    if n_active == 0:
        return virt_b, bufs, obs_bufs

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
    active_idx_t = torch.as_tensor(active_rows, device=device, dtype=torch.long)
    active_population_idx_t = torch.as_tensor(active_population_idx, device=device, dtype=torch.long)
    active_controller_idx_t = torch.as_tensor(active_controller_idx, device=device, dtype=torch.long)

    ego_rows = [jnp.full((num_envs,), p, dtype=jnp.int32) for p in range(n_ego)]
    ego_b_j = jnp.concatenate(ego_rows, axis=0)

    t0 = perf_counter()
    obs_jax = build_observation_batched_jax_per_ego(
        virt_b,
        ego_b_j,
        ship_speed,
        obs_feature_dim,
        normalize_to_p0=normalize_obs_to_p0,
    )
    must_halt_j = must_halt_no_owned_ships_per_ego(virt_b, ego_b_j)
    timing.obs_build_s += perf_counter() - t0

    t0 = perf_counter()
    obs_torch = obs_jax_to_torch(obs_jax)
    must_halt_t = torch.from_dlpack(must_halt_j)
    obs_torch = decode_observation(compress_observation(obs_torch), feature_dim=obs_feature_dim)
    obs_index = active_idx_t.to(next(iter(obs_torch.values())).device)
    active_obs = {key: v.index_select(0, obs_index).to(device) for key, v in obs_torch.items()}
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
    policy_fleet_eta = target_hit_tick_t[n_a_idx, d_idx]
    true_d_idx = target_true_planet_t[n_a_idx, d_idx].to(torch.long).clamp(0, P - 1)
    true_fleet_eta = target_true_hit_tick_t[n_a_idx, d_idx]

    dispatch_used = origin_frac_used & any_valid_target
    total_logp = halt_logp + origin_frac_used.float() * origin_frac_logp + dispatch_used.float() * target_logp
    values_active = out["value"].float()
    halt_now = ~dispatch_used
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

    halt_now_all.index_copy_(0, active_idx_t, halt_now)
    pair_flat_all.index_copy_(0, active_idx_t, pair_flat.to(torch.int32))
    frac_idx_all.index_copy_(0, active_idx_t, frac_idx.to(torch.int32))
    fleet_eta_all.index_copy_(0, active_idx_t, policy_fleet_eta.to(torch.float32))
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
    t1 = perf_counter()

    virt_b, oid_j, send_j, dispatched_j, slot_j = apply_micro_step_batched_per_ego(
        virt_b, ego_b_j, halt_now_j, pair_flat_j, frac_idx_j, fleet_eta_j
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
        pos_p_t = torch.arange(offset, offset + n_p, device=device, dtype=torch.long)
        active_p_idx_t = torch.as_tensor(ap, device=device, dtype=torch.long)
        active_p_rows_t = active_p_idx_t + p * num_envs
        micro_kp_t = micro_k_dev[p].index_select(0, active_p_idx_t)
        write_row_t = write_idx_dev[p].index_select(0, active_p_idx_t)
        bufs[p] = append_active_to_torch_buffer(
            bufs[p],
            active_p_idx_t,
            halt_now.index_select(0, pos_p_t),
            send_t.index_select(0, active_p_rows_t),
            policy_fleet_eta.index_select(0, pos_p_t),
            slot_t.index_select(0, active_p_rows_t),
            halt_action.index_select(0, pos_p_t).to(torch.int32),
            pair_flat.index_select(0, pos_p_t).to(torch.int32),
            frac_idx.index_select(0, pos_p_t).to(torch.int32),
            (origin_frac_used & ~any_valid_target).index_select(0, pos_p_t),
            (~any_valid_origin_frac & (halt_action == 0)).index_select(0, pos_p_t),
            must_halt_a.index_select(0, pos_p_t).to(torch.bool),
            (target_valid_t & ~target_overflow_t[:, None]).index_select(0, pos_p_t).to(torch.bool),
            target_hit_tick_t.index_select(0, pos_p_t).to(torch.float32),
            active_population_idx_t.index_select(0, pos_p_t).to(torch.int32),
            active_controller_idx_t.index_select(0, pos_p_t).to(torch.int32),
            write_row_t,
            micro_kp_t,
            max_micro_steps,
        )
        obs_p_active = {key: v.index_select(0, pos_p_t) for key, v in active_obs.items()}
        obs_bufs[p] = store_compressed_observation_rows(
            obs_bufs[p], write_row_t.to(torch.long), active_p_idx_t, obs_p_active
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

    total_logp_np = total_logp.detach().cpu().numpy()
    values_np = values_active.detach().cpu().numpy()
    halt_now_np = halt_now.detach().cpu().numpy()
    d_idx_np = d_idx.detach().cpu().numpy()
    true_d_idx_np = true_d_idx.detach().cpu().numpy()
    true_fleet_eta_np = true_fleet_eta.detach().cpu().numpy()
    policy_fleet_eta_np = policy_fleet_eta.detach().cpu().numpy()

    for p in range(n_ego):
        ap = players_active[p]
        n_p = n_list[p]
        offset = offsets[p]
        if n_p == 0:
            continue
        rows_np = write_idx[p][ap]
        valid[p][rows_np, ap] = True
        old_logprob[p][rows_np, ap] = total_logp_np[offset : offset + n_p]
        old_value[p][rows_np, ap] = values_np[offset : offset + n_p]
        if micro_step_penalty != 0.0:
            reward[p][rows_np, ap] -= float(micro_step_penalty) * dispatched_np[offset : offset + n_p].astype(
                np.float32
            )

    for p in range(n_ego):
        ap = players_active[p]
        n_p = n_list[p]
        offset = offsets[p]
        widx = write_idx[p]
        micro_arr = micro_k[p]
        for local_j, env_i in enumerate(ap):
            j_arr = offset + local_j
            env_i = int(env_i)
            reward_idx[p, env_i] = int(widx[env_i])
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
            if bool(halt_now_np[j_arr]):
                halted[p, env_i] = True
            widx[env_i] += 1
            micro_arr[env_i] += 1
            if micro_arr[env_i] >= max_micro_steps:
                halted[p, env_i] = True

    for p in range(n_ego):
        ap = players_active[p]
        n_p = n_list[p]
        if n_p == 0:
            continue
        active_p_idx_t = torch.as_tensor(ap, device=device, dtype=torch.long)
        micro_k_dev[p, active_p_idx_t] = micro_k_dev[p, active_p_idx_t] + 1
        write_idx_dev[p][active_p_idx_t] = write_idx_dev[p][active_p_idx_t] + 1

    return virt_b, bufs, obs_bufs


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
    policy: OrbitWarsPolicy,
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

    t0 = perf_counter()
    obs_jax = build_observation_batched_jax_per_ego(
        virt_b,
        row_ego_j,
        ship_speed,
        obs_feature_dim,
        normalize_to_p0=normalize_obs_to_p0,
    )
    must_halt_j = must_halt_no_owned_ships_per_ego(virt_b, row_ego_j)
    timing.obs_build_s += perf_counter() - t0

    t0 = perf_counter()
    obs_torch = obs_jax_to_torch(obs_jax)
    must_halt_t = torch.from_dlpack(must_halt_j)
    obs_torch = decode_observation(compress_observation(obs_torch), feature_dim=obs_feature_dim)
    full_obs = {key: v.to(device) for key, v in obs_torch.items()}
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
    target_mask = out["pair_mask"][n_idx, o_idx, :] & target_valid_t & ~target_overflow_t[:, None]
    any_valid_target = target_mask.any(dim=-1)
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

    dispatch_used = origin_frac_used & any_valid_target
    total_logp = torch.where(
        pending_t,
        halt_logp + origin_frac_used.float() * origin_frac_logp + dispatch_used.float() * target_logp,
        torch.zeros_like(halt_logp),
    )
    values_all = out["value"].float()
    halt_now = ~dispatch_used
    halt_now = torch.where(pending_t, halt_now, torch.ones_like(halt_now))
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
    halt_now_j = jax.dlpack.from_dlpack(halt_now.to(torch.bool).contiguous().detach())
    pair_flat_j = jax.dlpack.from_dlpack(pair_flat.to(torch.int32).contiguous().detach())
    frac_idx_j = jax.dlpack.from_dlpack(frac_idx.to(torch.int32).contiguous().detach())
    fleet_eta_j = jax.dlpack.from_dlpack(policy_fleet_eta.to(torch.float32).contiguous().detach())
    t1 = perf_counter()

    virt_b, oid_j, send_j, dispatched_j, slot_j = apply_micro_step_batched_per_ego(
        virt_b, row_ego_j, halt_now_j, pair_flat_j, frac_idx_j, fleet_eta_j
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
            halt_now.index_select(0, row_sel_t),
            send_t.index_select(0, row_sel_t),
            policy_fleet_eta.index_select(0, row_sel_t),
            slot_t.index_select(0, row_sel_t),
            halt_action.index_select(0, row_sel_t).to(torch.int32),
            pair_flat.index_select(0, row_sel_t).to(torch.int32),
            frac_idx.index_select(0, row_sel_t).to(torch.int32),
            (origin_frac_used & ~any_valid_target).index_select(0, row_sel_t),
            (~any_valid_origin_frac & (halt_action == 0) & pending_t).index_select(0, row_sel_t),
            must_halt.index_select(0, row_sel_t),
            (target_valid_t & ~target_overflow_t[:, None]).index_select(0, row_sel_t).to(torch.bool),
            target_hit_tick_t.index_select(0, row_sel_t).to(torch.float32),
            population_idx_t.index_select(0, row_sel_t),
            torch.zeros_like(population_idx_t.index_select(0, row_sel_t), dtype=torch.int32),
            write_row_t,
            micro_kp_t,
            max_micro_steps,
        )
        obs_p_active = {key: v.index_select(0, row_sel_t) for key, v in full_obs.items()}
        obs_bufs[p] = store_compressed_observation_rows(
            obs_bufs[p], write_row_t.to(torch.long), env_sel_t, obs_p_active
        )
    t5 = perf_counter()

    t6 = perf_counter()

    oid_np = oid_t.detach().cpu().numpy()
    send_np = send_t.detach().cpu().numpy()
    dispatched_np = dispatched_t.detach().cpu().numpy()
    t7 = perf_counter()
    _accum_micro_apply_breakdown(timing, t0, t1, t2, t3, t3a, t3b, t4, t5, t6, t7)

    total_logp_np = total_logp.detach().cpu().numpy()
    values_np = values_all.detach().cpu().numpy()
    halt_now_np = halt_now.detach().cpu().numpy()
    d_idx_np = d_idx.detach().cpu().numpy()
    true_d_idx_np = true_d_idx.detach().cpu().numpy()
    true_fleet_eta_np = true_fleet_eta.detach().cpu().numpy()
    policy_fleet_eta_np = policy_fleet_eta.detach().cpu().numpy()

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

    for p in range(n_ego):
        row_sel = np.flatnonzero(pending_rows & (row_ego_np == p)).astype(np.int32)
        if row_sel.size == 0:
            continue
        env_sel = row_env_np[row_sel]
        widx = write_idx[p]
        micro_arr = micro_k[p]
        for row in row_sel:
            env_i = int(row_env_np[row])
            reward_idx[p, env_i] = int(widx[env_i])
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
            if bool(halt_now_np[row]):
                halted[p, env_i] = True
            widx[env_i] += 1
            micro_arr[env_i] += 1
            if micro_arr[env_i] >= max_micro_steps:
                halted[p, env_i] = True

        env_sel_t = torch.as_tensor(env_sel, device=device, dtype=torch.long)
        micro_k_dev[p, env_sel_t] = micro_k_dev[p, env_sel_t] + 1
        write_idx_dev[p][env_sel_t] = write_idx_dev[p][env_sel_t] + 1

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
    first_hit_n_rays: int = 2048,
    first_hit_ray_chunk_size: int = 0,
    first_hit_env_chunk_size: int = 0,
    first_hit_method: str = "category-rays",
    micro_step_penalty: float = 1e-4,
    sync_policy_timing: bool = False,
    additional_policies: Optional[list[OrbitWarsPolicy]] = None,
    controller_counts: Optional[tuple[int, ...]] = None,
    termination_controller: Optional[int] = None,
    env_mode_by_env: Optional[np.ndarray] = None,
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
    carry_mode_arr = None if carry_in is None or carry_in.env_mode_by_env is None else np.asarray(carry_in.env_mode_by_env)
    unified_exploiter_rollout = (env_mode_by_env is not None) or (carry_mode_arr is not None)
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
        else:
            reset_prefetch.prefetch_ahead(int(seed_base) + seeds_consumed, na0, mf0)
    if carry_in is None:
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
        cfg.reward_time_bonus_coef = cfg_template.reward_time_bonus_coef
        cfg.reward_time_bonus_member_coefs = cfg_template.reward_time_bonus_member_coefs
        cfg.normalize_obs_to_p0 = cfg_template.normalize_obs_to_p0
        episode_turns = list(carry_in.episode_turns)
        if len(episode_turns) != num_envs:
            episode_turns = [0] * num_envs
        mode_arr = None if carry_in.env_mode_by_env is None else np.asarray(carry_in.env_mode_by_env, dtype=np.int32)
        pd = carry_in.player_done
        if pd is None:
            if carry_in.controller_assignments is not None:
                player_done = np.asarray(np.asarray(carry_in.controller_assignments, dtype=np.int32) < 0, dtype=np.bool_)
            else:
                player_done = np.zeros((int(cfg.num_agents), num_envs), dtype=np.bool_)
        else:
            player_done = np.asarray(pd, dtype=np.bool_)
            if player_done.shape != (int(cfg.num_agents), num_envs):
                if carry_in.controller_assignments is not None:
                    player_done = np.asarray(np.asarray(carry_in.controller_assignments, dtype=np.int32) < 0, dtype=np.bool_)
                else:
                    player_done = np.zeros((int(cfg.num_agents), num_envs), dtype=np.bool_)
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
        ca = carry_in.controller_assignments
        if ca is None:
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
                controller_assignments = np.stack(
                    [
                        _sample_controller_assignments_for_env(seed_base + env_i, int(cfg.num_agents), controller_counts)
                        for env_i in range(num_envs)
                    ],
                    axis=1,
                )
        mpm = carry_in.main_player_mask
        if mpm is None:
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
    _reset_prefetch_resync(
        reset_prefetch,
        seed_base,
        seeds_consumed,
        cfg,
        unified_exploiter_rollout=unified_exploiter_rollout,
        unified_seed_state=unified_seed_state,
    )

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
    segment_done = np.zeros((num_envs,), dtype=np.bool_)
    # First env to reset during this segment, recorded for the optional
    # background consistency check. Stored as ``(env_i, new_seed,
    # write_idx_at_reset_per_seat)``; ``write_idx_at_reset`` is the per-seat
    # row count BEFORE the reset (i.e. the first post-reset row index).
    first_reset_event: Optional[Tuple[int, int, np.ndarray]] = None

    with torch.inference_mode(), amp_ctx:
        while outer < max_outer_iters:
            outer += 1

            if horizon_fired and np.all(segment_done):
                break

            ready_mask = (~segment_done) & np.all(halted | player_done, axis=0)
            pending = (~segment_done)[None, :] & (~halted) & (~player_done)
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
                next_bucket = step_env_masked_batched(state_b, actions_bucket, ready_mask_j)
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
                reset_total_s = 0.0
                for env_i in step_envs:
                    env_i = int(env_i)
                    env_steps_per_env[env_i] += 1
                    episode_turns[env_i] += 1
                    main_dead = bool(
                        versus_controller_rollout
                        and np.any(main_player_mask[:, env_i] & (~alive_post_np[env_i, : int(cfg.num_agents)]))
                    )
                    env_done_now = bool(done_np[env_i]) or main_dead
                    main_policy_win: Optional[bool] = None
                    if versus_controller_rollout and env_done_now:
                        main_slots = np.flatnonzero(main_player_mask[:, env_i]).astype(np.int32)
                        main_slot = int(main_slots[0]) if main_slots.size else 0
                        if main_dead:
                            main_policy_win = False
                        else:
                            main_policy_win = bool(float(rewards_np[env_i, main_slot]) > 0.0)
                        timeout_step = int(step_count_np[env_i]) >= episode_timeout_step_count
                        time_bonus = 0.0
                        if (not timeout_step) and (not bool(main_policy_win)):
                            timeout_turn = max(1.0, float(episode_lim - 2))
                            pre_turn = max(0.0, float(step_count_np[env_i]) - 1.0)
                            time_bonus = float(cfg.reward_time_bonus_coef) * max(0.0, 1.0 - (pre_turn / timeout_turn))
                    synthetic_dead_exploiters: list[int] = []
                    for p in range(n_ego):
                        if reward_idx[p, env_i] >= 0:
                            seat_dead = not bool(alive_post_np[env_i, p])
                            local_done = env_done_now or seat_dead
                            delta_r = float(dr_np[env_i, p])
                            if versus_controller_rollout and int(controller_assignments[p, env_i]) == 1:
                                delta_r = 0.0
                                if env_done_now and main_policy_win is not None:
                                    delta_r = (1.0 + time_bonus) if not bool(main_policy_win) else -1.0
                                elif seat_dead:
                                    local_done = False
                                    pending_exploiter_terminal[p, env_i] = True
                            reward[p][reward_idx[p, env_i], env_i] += delta_r
                            done[p][reward_idx[p, env_i], env_i] = local_done or bool(
                                done[p][reward_idx[p, env_i], env_i]
                            )
                        if not bool(alive_post_np[env_i, p]):
                            player_done[p, env_i] = True
                        if (
                            env_done_now
                            and versus_controller_rollout
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
                        synthetic_reward = (
                            np.full(
                                (row_count,),
                                (1.0 + time_bonus) if not bool(main_policy_win) else -1.0,
                                dtype=np.float32,
                            )
                            if main_policy_win is not None
                            else np.zeros((row_count,), dtype=np.float32)
                        )
                        _append_synthetic_terminal_rows(
                            state_rows=term_state,
                            ego_idx=np.asarray(synthetic_dead_exploiters, dtype=np.int32),
                            env_idx=np.full((row_count,), env_i, dtype=np.int32),
                            controller_idx=np.ones((row_count,), dtype=np.int32),
                            population_idx=population_assignments[
                                np.asarray(synthetic_dead_exploiters, dtype=np.int32), env_i
                            ].astype(np.int32, copy=False),
                            terminal_reward=synthetic_reward,
                            ship_speed=ship_speed,
                            obs_feature_dim=obs_feature_dim,
                            normalize_obs_to_p0=cfg.normalize_obs_to_p0,
                            policies=policies,
                            bufs=bufs,
                            obs_bufs=obs_bufs,
                            write_idx=write_idx,
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
                            fresh_np, prefetch_meta = reset_prefetch.pop_state(
                                sid,
                                int(cfg.num_agents),
                                int(cfg.max_fleets),
                                return_meta=True,
                            )
                            _accum_prefetch_pop_meta(
                                timing,
                                prefetch_meta,
                                init_phase=False,
                                active_seat_count=int(cfg.num_agents),
                            )
                            reset_envs_apply.append(env_i)
                            reset_fresh_states.append(fresh_np)
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
                        done_env_seed[env_i] = sid
                        if first_reset_event is None:
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
                    if horizon_fired:
                        segment_done[env_i] = True
                if reset_envs_apply:
                    t_reset_apply0 = perf_counter()
                    state_b = reset_envs_at_indices(
                        state_b,
                        np.asarray(reset_envs_apply, dtype=np.int32),
                        reset_fresh_states,
                    )
                    if sync_policy_timing:
                        _sync_rollout_policy_timing(device, state_b)
                    reset_total_s += perf_counter() - t_reset_apply0
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
                        controller_assignments[:, env_i] = _sample_controller_assignments_for_env(
                            done_env_seed[int(env_i)], int(cfg.num_agents), controller_counts
                        )
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
                            controller_assignments[:, env_i] = _sample_controller_assignments_for_env(
                                done_env_seed[env_i], int(cfg.num_agents), controller_counts
                            )
                            main_player_mask[:, env_i] = _sample_main_player_mask_for_env(
                                controller_assignments[:, env_i], termination_controller
                            )
                timing.env_python_s += perf_counter() - t_py0

                t_book0 = perf_counter()
                step_env_t = torch.as_tensor(step_envs, dtype=torch.long, device=device)
                micro_k_dev[:, step_env_t] = 0
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
                    step_rows = np.flatnonzero(np.isin(row_env, step_envs)).astype(np.int32)
                    if step_rows.size:
                        virt_b = _copy_state_slices_between(
                            virt_b,
                            jnp.asarray(step_rows, dtype=jnp.int32),
                            state_b,
                            jnp.asarray(row_env[step_rows], dtype=jnp.int32),
                        )
                else:
                    actual_idx_j = jnp.asarray(step_envs, dtype=jnp.int32)
                    virt_dst_idx_j = jnp.concatenate(
                        [actual_idx_j + jnp.asarray(k * num_envs, dtype=jnp.int32) for k in range(n_ego)],
                        axis=0,
                    )
                    virt_src_idx_j = jnp.concatenate([actual_idx_j] * n_ego, axis=0)
                    virt_b = _copy_state_slices_between(
                        virt_b,
                        virt_dst_idx_j,
                        state_b,
                        virt_src_idx_j,
                    )
                if sync_policy_timing:
                    _sync_rollout_policy_timing(device, virt_b)
                timing.env_bookkeeping_s += perf_counter() - t_book0
                timing.env_step_s += perf_counter() - t0
                continue

            if pending_total == 0:
                break

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
                virt_b, bufs, obs_bufs = _run_async_micro_step_multi(
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
                    population_assignments=population_assignments,
                )
            if profile_rollout and device.type == "cuda" and not logged_first_policy_fwd:
                log_cuda_mem("rollout after first batched policy forward", device)
                logged_first_policy_fwd = True

            if any(np.any(write_idx[p] >= rollout_micro_horizon) for p in range(n_ego)):
                horizon_fired = True

    timing.loop_s = perf_counter() - t_loop0
    timing.outer_iters = outer
    timing.wall_s = perf_counter() - t_wall0

    if profile_rollout:
        log_cuda_mem("rollout exit (segment finished)", device)

    bootstrap = [np.zeros((num_envs,), dtype=np.float32) for _ in range(n_ego)]
    bootstrap_valid = [np.zeros((num_envs,), dtype=np.bool_) for _ in range(n_ego)]
    if horizon_fired:
        if grouped_population_rollout:
            assert row_env is not None and row_ego is not None and row_env_j is not None
            t0 = perf_counter()
            policy_state_b = _gather_state_rows(state_b, row_env_j)
            obs_j = build_observation_batched_jax_per_ego(
                policy_state_b,
                jnp.asarray(row_ego, dtype=jnp.int32),
                ship_speed,
                obs_feature_dim,
                normalize_to_p0=cfg.normalize_obs_to_p0,
            )
            timing.obs_build_s += perf_counter() - t0
            tb0 = perf_counter()
            obs_t = decode_observation(compress_observation(obs_jax_to_torch(obs_j)), feature_dim=obs_feature_dim)
            with amp_ctx:
                out_rows = policy.forward_dense_rollout_grouped_population(**{k: v.to(device) for k, v in obs_t.items()})
            timing.policy_batch_s += perf_counter() - tb0
            tf0 = perf_counter()
            value_np_rows = out_rows["value"].float().detach().cpu().numpy()
            timing.policy_forward_s += perf_counter() - tf0
            for row in range(value_np_rows.shape[0]):
                env_i = int(row_env[row])
                ego = int(row_ego[row])
                last_t = int(write_idx[ego][env_i]) - 1
                if last_t >= 0 and not bool(done[ego][last_t, env_i]):
                    bootstrap[ego][env_i] = float(value_np_rows[row])
                    bootstrap_valid[ego][env_i] = True
        else:
            value_np_per_ego: list[np.ndarray] = []
            for ego in range(n_ego):
                t0 = perf_counter()
                obs_j = build_observation_batched_jax(
                    state_b,
                    ego,
                    ship_speed,
                    obs_feature_dim,
                    normalize_to_p0=cfg.normalize_obs_to_p0,
                )
                timing.obs_build_s += perf_counter() - t0
                tb0 = perf_counter()
                obs_t = decode_observation(compress_observation(obs_jax_to_torch(obs_j)), feature_dim=obs_feature_dim)
                pop_t = torch.as_tensor(population_assignments[ego], device=device, dtype=torch.long)
                controller_t = torch.as_tensor(controller_assignments[ego], device=device, dtype=torch.long)
                obs_t_dev = {k: v.to(device) for k, v in obs_t.items()}
                with amp_ctx:
                    out_e = _forward_dense_rollout_by_controller(
                        policies=policies,
                        active_obs=obs_t_dev,
                        active_population_idx_t=pop_t,
                        active_controller_idx_t=controller_t,
                    )
                timing.policy_batch_s += perf_counter() - tb0
                tf0 = perf_counter()
                value_np_per_ego.append(out_e["value"].float().detach().cpu().numpy())
                timing.policy_forward_s += perf_counter() - tf0

            for i in range(num_envs):
                for ego in range(n_ego):
                    last_t = int(write_idx[ego][i]) - 1
                    if last_t >= 0 and not bool(done[ego][last_t, i]):
                        bootstrap[ego][i] = float(value_np_per_ego[ego][i])
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

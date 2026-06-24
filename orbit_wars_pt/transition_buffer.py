"""Phase 5: device-resident rollout transition buffer.

Each buffer row stores a length-``M`` prefix of in-phase micro deltas (halt, stored
``send``, ``slot``, ``pair_flat``, ``frac_idx``), padded with no-ops, plus the
scalar policy bookkeeping fields, ``target_planet_reachable`` (snapshot of
which planets are valid first-hit targets for the sampled origin/fraction), and
``target_hit_tick`` (the matching per-target ETA feature used by the target head).
Canonical ``planets`` / ``fleets`` / ``fleet_active`` live in ``turn_state_cache`` per
turn.  PPO gather replays a prefix with :func:`apply_prefix_micro_deltas_batched`
(no scan over ``H_buf``). Launch angle is intentionally not stored because the
training rollout env schedules incoming mass by target/ETA metadata only.
"""

from __future__ import annotations

import os
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import torch

from jax_orbit_wars import OrbitWarsState

from orbit_wars_pt.constants import MAX_PLANETS
from orbit_wars_pt.micro_jax import apply_prefix_micro_deltas_batched


class TransitionBuffer(NamedTuple):
    """One player's transitions for one segment, all on device.

    ``target_planet_reachable`` and ``target_hit_tick`` store rollout-time
    first-hit geometry per planet for the sampled origin/fraction (used at PPO
    replay instead of recomputing first-hit rays).
    """

    micro_halt_now: jnp.ndarray
    send: jnp.ndarray
    fleet_eta: jnp.ndarray
    slot: jnp.ndarray
    halt_action: jnp.ndarray
    target_abort: jnp.ndarray
    pair_flat: jnp.ndarray
    frac_idx: jnp.ndarray
    no_valid_pairs: jnp.ndarray
    no_valid_fracs: jnp.ndarray
    must_halt_no_ships: jnp.ndarray
    target_planet_reachable: jnp.ndarray
    target_hit_tick: jnp.ndarray
    phase_micro_idx: jnp.ndarray
    population_idx: jnp.ndarray
    policy_id: jnp.ndarray
    value_head_idx: jnp.ndarray


class TorchTransitionBuffer(NamedTuple):
    """One player's transitions stored in PyTorch tensors.

    This mirrors ``TransitionBuffer`` field-for-field, but keeps persistent
    rollout records out of XLA's allocator. PPO replay selects minibatch rows
    from these tensors and transfers only the selected rows to the accelerator.
    """

    micro_halt_now: torch.Tensor
    send: torch.Tensor
    fleet_eta: torch.Tensor
    slot: torch.Tensor
    halt_action: torch.Tensor
    target_abort: torch.Tensor
    pair_flat: torch.Tensor
    frac_idx: torch.Tensor
    no_valid_pairs: torch.Tensor
    no_valid_fracs: torch.Tensor
    must_halt_no_ships: torch.Tensor
    target_planet_reachable: torch.Tensor
    target_hit_tick: torch.Tensor
    phase_micro_idx: torch.Tensor
    population_idx: torch.Tensor
    policy_id: torch.Tensor
    value_head_idx: torch.Tensor


def init_transition_buffer(num_envs: int, H_buf: int, max_micro_steps: int) -> TransitionBuffer:
    """Allocate prefix tensors ``(H_buf, num_envs, max_micro_steps)`` and scalars ``(H_buf, num_envs)``."""

    m = int(max_micro_steps)
    noop_halt = jnp.ones((H_buf, num_envs, m), dtype=jnp.bool_)
    return TransitionBuffer(
        micro_halt_now=noop_halt,
        send=jnp.zeros((H_buf, num_envs, m), dtype=jnp.float32),
        fleet_eta=jnp.zeros((H_buf, num_envs, m), dtype=jnp.float32),
        slot=jnp.full((H_buf, num_envs, m), -1, dtype=jnp.int32),
        halt_action=jnp.zeros((H_buf, num_envs), dtype=jnp.int32),
        target_abort=jnp.zeros((H_buf, num_envs), dtype=jnp.bool_),
        pair_flat=jnp.zeros((H_buf, num_envs, m), dtype=jnp.int32),
        frac_idx=jnp.zeros((H_buf, num_envs, m), dtype=jnp.int32),
        no_valid_pairs=jnp.zeros((H_buf, num_envs), dtype=jnp.bool_),
        no_valid_fracs=jnp.zeros((H_buf, num_envs), dtype=jnp.bool_),
        must_halt_no_ships=jnp.zeros((H_buf, num_envs), dtype=jnp.bool_),
        target_planet_reachable=jnp.zeros((H_buf, num_envs, MAX_PLANETS), dtype=jnp.bool_),
        target_hit_tick=jnp.zeros((H_buf, num_envs, MAX_PLANETS), dtype=jnp.float32),
        phase_micro_idx=jnp.zeros((H_buf, num_envs), dtype=jnp.int32),
        population_idx=jnp.zeros((H_buf, num_envs), dtype=jnp.int32),
        policy_id=jnp.zeros((H_buf, num_envs), dtype=jnp.int32),
        value_head_idx=jnp.zeros((H_buf, num_envs), dtype=jnp.int32),
    )


def init_torch_transition_buffer(
    num_envs: int,
    H_buf: int,
    max_micro_steps: int,
    *,
    device: torch.device,
) -> TorchTransitionBuffer:
    """Allocate a PyTorch-backed transition buffer with the same layout as ``TransitionBuffer``."""

    m = int(max_micro_steps)
    return TorchTransitionBuffer(
        micro_halt_now=torch.ones((H_buf, num_envs, m), dtype=torch.bool, device=device),
        send=torch.zeros((H_buf, num_envs, m), dtype=torch.float32, device=device),
        fleet_eta=torch.zeros((H_buf, num_envs, m), dtype=torch.float32, device=device),
        slot=torch.full((H_buf, num_envs, m), -1, dtype=torch.int32, device=device),
        halt_action=torch.zeros((H_buf, num_envs), dtype=torch.int32, device=device),
        target_abort=torch.zeros((H_buf, num_envs), dtype=torch.bool, device=device),
        pair_flat=torch.zeros((H_buf, num_envs, m), dtype=torch.int32, device=device),
        frac_idx=torch.zeros((H_buf, num_envs, m), dtype=torch.int32, device=device),
        no_valid_pairs=torch.zeros((H_buf, num_envs), dtype=torch.bool, device=device),
        no_valid_fracs=torch.zeros((H_buf, num_envs), dtype=torch.bool, device=device),
        must_halt_no_ships=torch.zeros((H_buf, num_envs), dtype=torch.bool, device=device),
        target_planet_reachable=torch.zeros((H_buf, num_envs, MAX_PLANETS), dtype=torch.bool, device=device),
        target_hit_tick=torch.zeros((H_buf, num_envs, MAX_PLANETS), dtype=torch.float32, device=device),
        phase_micro_idx=torch.zeros((H_buf, num_envs), dtype=torch.int32, device=device),
        population_idx=torch.zeros((H_buf, num_envs), dtype=torch.int32, device=device),
        policy_id=torch.zeros((H_buf, num_envs), dtype=torch.int32, device=device),
        value_head_idx=torch.zeros((H_buf, num_envs), dtype=torch.int32, device=device),
    )


@torch.no_grad()
def append_to_torch_buffer(
    buf: TorchTransitionBuffer,
    micro_halt_now: torch.Tensor,
    send_now: torch.Tensor,
    fleet_eta_now: torch.Tensor,
    slot_now: torch.Tensor,
    halt_action: torch.Tensor,
    target_abort: torch.Tensor,
    pair_flat: torch.Tensor,
    frac_idx: torch.Tensor,
    no_valid_pairs: torch.Tensor,
    no_valid_fracs: torch.Tensor,
    must_halt_no_ships: torch.Tensor,
    target_planet_reachable_now: torch.Tensor,
    target_hit_tick_now: torch.Tensor,
    population_idx: torch.Tensor,
    policy_id: torch.Tensor,
    value_head_idx: torch.Tensor,
    write_row: torch.Tensor,
    micro_k: torch.Tensor,
    active: torch.Tensor,
    max_micro_steps: int,
) -> TorchTransitionBuffer:
    """Write one transition per active env into a PyTorch-backed buffer."""

    device = buf.micro_halt_now.device
    n = int(write_row.shape[0])
    n_idx = torch.arange(n, device=device, dtype=torch.long)
    wr = write_row.to(device=device, dtype=torch.long)
    mk = micro_k.to(device=device, dtype=torch.long)
    active_b = active.to(device=device, dtype=torch.bool)
    safe_wr = torch.where(active_b, wr, torch.zeros_like(wr))
    safe_mk = torch.where(active_b, mk, torch.zeros_like(mk))
    if os.environ.get("ORBIT_WARS_VALIDATE_CUDA_INDEXES") == "1" and bool(torch.any(active_b).detach().cpu()):
        active_wr = wr[active_b].detach().cpu()
        active_mk = mk[active_b].detach().cpu()
        h_buf = int(buf.micro_halt_now.shape[0])
        m_buf = int(buf.micro_halt_now.shape[2])
        if bool(torch.any((active_wr < 0) | (active_wr >= h_buf))):
            raise RuntimeError(
                f"append_to_torch_buffer write_row out of range: "
                f"min={int(active_wr.min())} max={int(active_wr.max())} H_buf={h_buf}"
            )
        if bool(torch.any((active_mk < 0) | (active_mk >= m_buf) | (active_mk >= int(max_micro_steps)))):
            raise RuntimeError(
                f"append_to_torch_buffer micro_k out of range: "
                f"min={int(active_mk.min())} max={int(active_mk.max())} "
                f"M_buf={m_buf} max_micro_steps={int(max_micro_steps)}"
            )
    prev_row = torch.where((safe_mk > 0) & active_b, safe_wr - 1, torch.full_like(safe_wr, -1))
    safe_prev = torch.clamp(prev_row, min=0)
    has_prev = prev_row >= 0

    def _grow(field: torch.Tensor, new_value: torch.Tensor, fill_value: int | float | bool) -> torch.Tensor:
        base = field[safe_prev, n_idx, :].clone()
        base[~has_prev] = fill_value
        base[n_idx, safe_mk] = new_value.to(device=device, dtype=field.dtype)
        return base

    new_halt = _grow(buf.micro_halt_now, micro_halt_now, True)
    new_send = _grow(buf.send, send_now, 0.0)
    new_fleet_eta = _grow(buf.fleet_eta, fleet_eta_now, 0.0)
    new_slot = _grow(buf.slot, slot_now, -1)
    new_pf = _grow(buf.pair_flat, pair_flat, 0)
    new_fi = _grow(buf.frac_idx, frac_idx, 0)

    env = n_idx[active_b]
    row = wr[active_b]
    buf.micro_halt_now[row, env, :] = new_halt[active_b]
    buf.send[row, env, :] = new_send[active_b]
    buf.fleet_eta[row, env, :] = new_fleet_eta[active_b]
    buf.slot[row, env, :] = new_slot[active_b]
    buf.pair_flat[row, env, :] = new_pf[active_b]
    buf.frac_idx[row, env, :] = new_fi[active_b]
    buf.halt_action[row, env] = halt_action.to(device=device, dtype=torch.int32)[active_b]
    buf.target_abort[row, env] = target_abort.to(device=device, dtype=torch.bool)[active_b]
    buf.no_valid_pairs[row, env] = no_valid_pairs.to(device=device, dtype=torch.bool)[active_b]
    buf.no_valid_fracs[row, env] = no_valid_fracs.to(device=device, dtype=torch.bool)[active_b]
    buf.must_halt_no_ships[row, env] = must_halt_no_ships.to(device=device, dtype=torch.bool)[active_b]
    buf.target_planet_reachable[row, env, :] = target_planet_reachable_now.to(device=device, dtype=torch.bool)[active_b]
    buf.target_hit_tick[row, env, :] = target_hit_tick_now.to(device=device, dtype=torch.float32)[active_b]
    buf.phase_micro_idx[row, env] = micro_k.to(device=device, dtype=torch.int32)[active_b]
    buf.population_idx[row, env] = population_idx.to(device=device, dtype=torch.int32)[active_b]
    buf.policy_id[row, env] = policy_id.to(device=device, dtype=torch.int32)[active_b]
    buf.value_head_idx[row, env] = value_head_idx.to(device=device, dtype=torch.int32)[active_b]
    return buf


@torch.no_grad()
def append_active_to_torch_buffer(
    buf: TorchTransitionBuffer,
    env_idx: torch.Tensor,
    micro_halt_now: torch.Tensor,
    send_now: torch.Tensor,
    fleet_eta_now: torch.Tensor,
    slot_now: torch.Tensor,
    halt_action: torch.Tensor,
    target_abort: torch.Tensor,
    pair_flat: torch.Tensor,
    frac_idx: torch.Tensor,
    no_valid_pairs: torch.Tensor,
    no_valid_fracs: torch.Tensor,
    must_halt_no_ships: torch.Tensor,
    target_planet_reachable_now: torch.Tensor,
    target_hit_tick_now: torch.Tensor,
    population_idx: torch.Tensor,
    policy_id: torch.Tensor,
    value_head_idx: torch.Tensor,
    write_row: torch.Tensor,
    micro_k: torch.Tensor,
    max_micro_steps: int,
) -> TorchTransitionBuffer:
    """Write one transition per active env into a PyTorch-backed buffer."""

    device = buf.micro_halt_now.device
    env = env_idx.to(device=device, dtype=torch.long)
    row = write_row.to(device=device, dtype=torch.long)
    mk = micro_k.to(device=device, dtype=torch.long)
    if os.environ.get("ORBIT_WARS_VALIDATE_CUDA_INDEXES") == "1" and int(env.numel()) > 0:
        active_wr = row.detach().cpu()
        active_mk = mk.detach().cpu()
        h_buf = int(buf.micro_halt_now.shape[0])
        m_buf = int(buf.micro_halt_now.shape[2])
        if bool(torch.any((active_wr < 0) | (active_wr >= h_buf))):
            raise RuntimeError(
                f"append_active_to_torch_buffer write_row out of range: "
                f"min={int(active_wr.min())} max={int(active_wr.max())} H_buf={h_buf}"
            )
        if bool(torch.any((active_mk < 0) | (active_mk >= m_buf) | (active_mk >= int(max_micro_steps)))):
            raise RuntimeError(
                f"append_active_to_torch_buffer micro_k out of range: "
                f"min={int(active_mk.min())} max={int(active_mk.max())} "
                f"M_buf={m_buf} max_micro_steps={int(max_micro_steps)}"
            )

    n_idx = torch.arange(env.shape[0], device=device, dtype=torch.long)
    prev_row = torch.where(mk > 0, row - 1, torch.full_like(row, -1))
    safe_prev = torch.clamp(prev_row, min=0)
    has_prev = prev_row >= 0

    def _grow(field: torch.Tensor, new_value: torch.Tensor, fill_value: int | float | bool) -> torch.Tensor:
        base = field[safe_prev, env, :].clone()
        base[~has_prev] = fill_value
        base[n_idx, mk] = new_value.to(device=device, dtype=field.dtype)
        return base

    new_halt = _grow(buf.micro_halt_now, micro_halt_now, True)
    new_send = _grow(buf.send, send_now, 0.0)
    new_fleet_eta = _grow(buf.fleet_eta, fleet_eta_now, 0.0)
    new_slot = _grow(buf.slot, slot_now, -1)
    new_pf = _grow(buf.pair_flat, pair_flat, 0)
    new_fi = _grow(buf.frac_idx, frac_idx, 0)

    buf.micro_halt_now[row, env, :] = new_halt
    buf.send[row, env, :] = new_send
    buf.fleet_eta[row, env, :] = new_fleet_eta
    buf.slot[row, env, :] = new_slot
    buf.pair_flat[row, env, :] = new_pf
    buf.frac_idx[row, env, :] = new_fi
    buf.halt_action[row, env] = halt_action.to(device=device, dtype=torch.int32)
    buf.target_abort[row, env] = target_abort.to(device=device, dtype=torch.bool)
    buf.no_valid_pairs[row, env] = no_valid_pairs.to(device=device, dtype=torch.bool)
    buf.no_valid_fracs[row, env] = no_valid_fracs.to(device=device, dtype=torch.bool)
    buf.must_halt_no_ships[row, env] = must_halt_no_ships.to(device=device, dtype=torch.bool)
    buf.target_planet_reachable[row, env, :] = target_planet_reachable_now.to(device=device, dtype=torch.bool)
    buf.target_hit_tick[row, env, :] = target_hit_tick_now.to(device=device, dtype=torch.float32)
    buf.phase_micro_idx[row, env] = micro_k.to(device=device, dtype=torch.int32)
    buf.population_idx[row, env] = population_idx.to(device=device, dtype=torch.int32)
    buf.policy_id[row, env] = policy_id.to(device=device, dtype=torch.int32)
    buf.value_head_idx[row, env] = value_head_idx.to(device=device, dtype=torch.int32)
    return buf


def _noop_prefix_planes(num_envs: int, max_micro_steps: int):
    m = int(max_micro_steps)
    return (
        jnp.ones((num_envs, m), dtype=jnp.bool_),
        jnp.zeros((num_envs, m), dtype=jnp.float32),
        jnp.full((num_envs, m), -1, dtype=jnp.int32),
        jnp.zeros((num_envs, m), dtype=jnp.int32),
        jnp.zeros((num_envs, m), dtype=jnp.int32),
    )


@partial(jax.jit, static_argnames=("max_micro_steps",))
def append_to_buffer(
    buf: TransitionBuffer,
    micro_halt_now,
    send_now,
    fleet_eta_now,
    slot_now,
    halt_action,
    target_abort,
    pair_flat,
    frac_idx,
    no_valid_pairs,
    no_valid_fracs,
    must_halt_no_ships,
    target_planet_reachable_now,
    target_hit_tick_now,
    population_idx,
    policy_id,
    value_head_idx,
    write_row: jnp.ndarray,
    micro_k: jnp.ndarray,
    active: jnp.ndarray,
    max_micro_steps: int,
) -> TransitionBuffer:
    """Write one transition per env at ``write_row[n]``, extending the in-phase prefix at slot ``micro_k[n]``."""

    n_idx = jnp.arange(write_row.shape[0], dtype=jnp.int32)
    noop_h, noop_s, noop_sl, noop_pf, noop_fi = _noop_prefix_planes(write_row.shape[0], max_micro_steps)

    prev_row = jnp.where(micro_k > 0, write_row - 1, -1)
    safe_prev = jnp.maximum(prev_row, 0)

    def _grow(field_prev, new_k, noop_p):
        prev_vals = field_prev[safe_prev, n_idx, :]
        base = jnp.where((prev_row >= 0)[:, None], prev_vals, noop_p)
        return jax.vmap(lambda plane, kk, v: plane.at[kk].set(v))(base, micro_k, new_k)

    new_halt = _grow(buf.micro_halt_now, micro_halt_now.astype(jnp.bool_), noop_h)
    new_send = _grow(buf.send, send_now.astype(jnp.float32), noop_s)
    new_fleet_eta = _grow(buf.fleet_eta, fleet_eta_now.astype(jnp.float32), noop_s)
    new_slot = _grow(buf.slot, slot_now.astype(jnp.int32), noop_sl)
    new_pf = _grow(buf.pair_flat, pair_flat.astype(jnp.int32), noop_pf)
    new_fi = _grow(buf.frac_idx, frac_idx.astype(jnp.int32), noop_fi)

    def _scatter_rows(field_plane, new_plane):
        old = field_plane[write_row, n_idx, :]
        merged = jnp.where(active[:, None], new_plane, old)
        return field_plane.at[write_row, n_idx, :].set(merged)

    out_halt = _scatter_rows(buf.micro_halt_now, new_halt)
    out_send = _scatter_rows(buf.send, new_send)
    out_fleet_eta = _scatter_rows(buf.fleet_eta, new_fleet_eta)
    out_slot = _scatter_rows(buf.slot, new_slot)
    out_pf = _scatter_rows(buf.pair_flat, new_pf)
    out_fi = _scatter_rows(buf.frac_idx, new_fi)

    out_ha = buf.halt_action.at[write_row, n_idx].set(halt_action.astype(jnp.int32))
    out_ta = buf.target_abort.at[write_row, n_idx].set(target_abort.astype(jnp.bool_))
    out_nvp = buf.no_valid_pairs.at[write_row, n_idx].set(no_valid_pairs.astype(jnp.bool_))
    out_nvf = buf.no_valid_fracs.at[write_row, n_idx].set(no_valid_fracs.astype(jnp.bool_))
    out_mh = buf.must_halt_no_ships.at[write_row, n_idx].set(must_halt_no_ships.astype(jnp.bool_))
    out_tpr = buf.target_planet_reachable.at[write_row, n_idx, :].set(
        target_planet_reachable_now.astype(jnp.bool_)
    )
    out_tht = buf.target_hit_tick.at[write_row, n_idx, :].set(target_hit_tick_now.astype(jnp.float32))

    old_pm = buf.phase_micro_idx[write_row, n_idx]
    new_pm = jnp.where(active, micro_k.astype(jnp.int32), old_pm)
    out_pm = buf.phase_micro_idx.at[write_row, n_idx].set(new_pm)
    old_pop = buf.population_idx[write_row, n_idx]
    new_pop = jnp.where(active, population_idx.astype(jnp.int32), old_pop)
    out_pop = buf.population_idx.at[write_row, n_idx].set(new_pop)
    old_policy = buf.policy_id[write_row, n_idx]
    new_policy = jnp.where(active, policy_id.astype(jnp.int32), old_policy)
    out_policy = buf.policy_id.at[write_row, n_idx].set(new_policy)
    old_value_head = buf.value_head_idx[write_row, n_idx]
    new_value_head = jnp.where(active, value_head_idx.astype(jnp.int32), old_value_head)
    out_value_head = buf.value_head_idx.at[write_row, n_idx].set(new_value_head)

    return TransitionBuffer(
        micro_halt_now=out_halt,
        send=out_send,
        fleet_eta=out_fleet_eta,
        slot=out_slot,
        halt_action=out_ha,
        target_abort=out_ta,
        pair_flat=out_pf,
        frac_idx=out_fi,
        no_valid_pairs=out_nvp,
        no_valid_fracs=out_nvf,
        must_halt_no_ships=out_mh,
        target_planet_reachable=out_tpr,
        target_hit_tick=out_tht,
        phase_micro_idx=out_pm,
        population_idx=out_pop,
        policy_id=out_policy,
        value_head_idx=out_value_head,
    )


@jax.jit
def scatter_turn_tags(
    turn_tag: jnp.ndarray,
    t_per_env: jnp.ndarray,
    turn_slot_per_env: jnp.ndarray,
) -> jnp.ndarray:
    """Write ``turn_slot_per_env[n]`` into ``turn_tag[t_per_env[n], n]``."""

    n_idx = jnp.arange(t_per_env.shape[0], dtype=jnp.int32)
    return turn_tag.at[t_per_env, n_idx].set(turn_slot_per_env.astype(jnp.int32))


@partial(jax.jit, static_argnames=("max_micro_steps",))
def gather_minibatch(
    buf0: TransitionBuffer,
    buf1: TransitionBuffer,
    player_b: jnp.ndarray,
    t_b: jnp.ndarray,
    n_b: jnp.ndarray,
    turn_state_cache: Any,
    turn_tag_p0: jnp.ndarray,
    turn_tag_p1: jnp.ndarray,
    max_micro_steps: int,
):
    """Reconstruct pre-action ``OrbitWarsState`` per sample, then return action fields."""

    is_p0 = player_b == 0
    turn_idx_mb = jnp.where(is_p0, turn_tag_p0[t_b, n_b], turn_tag_p1[t_b, n_b])
    state_mb = jax.tree.map(lambda leaf: leaf[turn_idx_mb, n_b], turn_state_cache)

    m_sel = jnp.where(is_p0, buf0.phase_micro_idx[t_b, n_b], buf1.phase_micro_idx[t_b, n_b])
    k_ar = jnp.arange(max_micro_steps, dtype=jnp.int32)
    apply_mask_m = k_ar[None, :] < m_sel[:, None]

    def _gp(field0, field1):
        a = field0[t_b, n_b, :]
        b = field1[t_b, n_b, :]
        return jnp.where(is_p0[:, None], a, b)

    halt_m = _gp(buf0.micro_halt_now, buf1.micro_halt_now)
    send_m = _gp(buf0.send, buf1.send)
    fleet_eta_m = _gp(buf0.fleet_eta, buf1.fleet_eta)
    slot_m = _gp(buf0.slot, buf1.slot)
    pf_m = _gp(buf0.pair_flat, buf1.pair_flat)
    fi_m = _gp(buf0.frac_idx, buf1.frac_idx)

    state_mb = apply_prefix_micro_deltas_batched(
        state_mb,
        player_b.astype(jnp.int32),
        max_micro_steps,
        halt_m,
        send_m,
        slot_m,
        pf_m,
        fi_m,
        fleet_eta_m,
        apply_mask_m,
    )

    bb = jnp.arange(player_b.shape[0], dtype=jnp.int32)
    pair_flat_act = pf_m[bb, m_sel]
    frac_idx_act = fi_m[bb, m_sel]

    def _sel(b0, b1):
        from0 = b0[t_b, n_b]
        from1 = b1[t_b, n_b]
        broadcast_shape = (player_b.shape[0],) + (1,) * (from0.ndim - 1)
        mask = is_p0.reshape(broadcast_shape)
        return jnp.where(mask, from0, from1)

    halt_action = _sel(buf0.halt_action, buf1.halt_action)
    pair_flat = pair_flat_act
    frac_idx = frac_idx_act
    no_valid_pairs = _sel(buf0.no_valid_pairs, buf1.no_valid_pairs)
    no_valid_fracs = _sel(buf0.no_valid_fracs, buf1.no_valid_fracs)
    must_halt_no_ships = _sel(buf0.must_halt_no_ships, buf1.must_halt_no_ships)
    tpr0 = buf0.target_planet_reachable[t_b, n_b, :]
    tpr1 = buf1.target_planet_reachable[t_b, n_b, :]
    target_planet_reachable = jnp.where(is_p0[:, None], tpr0, tpr1)
    tht0 = buf0.target_hit_tick[t_b, n_b, :]
    tht1 = buf1.target_hit_tick[t_b, n_b, :]
    target_hit_tick = jnp.where(is_p0[:, None], tht0, tht1)

    return (
        state_mb,
        halt_action,
        pair_flat,
        frac_idx,
        no_valid_pairs,
        no_valid_fracs,
        must_halt_no_ships,
        target_planet_reachable,
        target_hit_tick,
    )

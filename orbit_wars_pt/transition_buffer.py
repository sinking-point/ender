"""Phase 5: device-resident rollout transition buffer.

Each buffer row stores a length-``M`` prefix of in-phase micro deltas (halt, stored
``send``, ``slot``, ``pair_flat``, ``frac_idx``), padded with no-ops, plus the
scalar policy bookkeeping fields for that transition.  Canonical
``planets`` / ``fleets`` / ``fleet_active`` live in ``turn_state_cache`` per
turn.  PPO gather replays a prefix with :func:`apply_prefix_micro_deltas_batched`
(no scan over ``H_buf``).
"""

from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from jax_orbit_wars import OrbitWarsState

from orbit_wars_pt.micro_jax import apply_prefix_micro_deltas_batched


class TransitionBuffer(NamedTuple):
    """One player's transitions for one segment, all on device."""

    micro_halt_now: jnp.ndarray
    send: jnp.ndarray
    angle: jnp.ndarray
    slot: jnp.ndarray
    halt_action: jnp.ndarray
    pair_flat: jnp.ndarray
    frac_idx: jnp.ndarray
    no_valid_pairs: jnp.ndarray
    no_valid_fracs: jnp.ndarray
    must_halt_no_ships: jnp.ndarray
    phase_micro_idx: jnp.ndarray


def init_transition_buffer(num_envs: int, H_buf: int, max_micro_steps: int) -> TransitionBuffer:
    """Allocate prefix tensors ``(H_buf, num_envs, max_micro_steps)`` and scalars ``(H_buf, num_envs)``."""

    m = int(max_micro_steps)
    noop_halt = jnp.ones((H_buf, num_envs, m), dtype=jnp.bool_)
    return TransitionBuffer(
        micro_halt_now=noop_halt,
        send=jnp.zeros((H_buf, num_envs, m), dtype=jnp.float32),
        angle=jnp.zeros((H_buf, num_envs, m), dtype=jnp.float32),
        slot=jnp.full((H_buf, num_envs, m), -1, dtype=jnp.int32),
        halt_action=jnp.zeros((H_buf, num_envs), dtype=jnp.int32),
        pair_flat=jnp.zeros((H_buf, num_envs, m), dtype=jnp.int32),
        frac_idx=jnp.zeros((H_buf, num_envs, m), dtype=jnp.int32),
        no_valid_pairs=jnp.zeros((H_buf, num_envs), dtype=jnp.bool_),
        no_valid_fracs=jnp.zeros((H_buf, num_envs), dtype=jnp.bool_),
        must_halt_no_ships=jnp.zeros((H_buf, num_envs), dtype=jnp.bool_),
        phase_micro_idx=jnp.zeros((H_buf, num_envs), dtype=jnp.int32),
    )


def _noop_prefix_planes(num_envs: int, max_micro_steps: int):
    m = int(max_micro_steps)
    return (
        jnp.ones((num_envs, m), dtype=jnp.bool_),
        jnp.zeros((num_envs, m), dtype=jnp.float32),
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
    angle_now,
    slot_now,
    halt_action,
    pair_flat,
    frac_idx,
    no_valid_pairs,
    no_valid_fracs,
    must_halt_no_ships,
    write_row: jnp.ndarray,
    micro_k: jnp.ndarray,
    active: jnp.ndarray,
    max_micro_steps: int,
) -> TransitionBuffer:
    """Write one transition per env at ``write_row[n]``, extending the in-phase prefix at slot ``micro_k[n]``."""

    n_idx = jnp.arange(write_row.shape[0], dtype=jnp.int32)
    noop_h, noop_s, noop_a, noop_sl, noop_pf, noop_fi = _noop_prefix_planes(write_row.shape[0], max_micro_steps)

    prev_row = jnp.where(micro_k > 0, write_row - 1, -1)
    safe_prev = jnp.maximum(prev_row, 0)

    def _grow(field_prev, new_k, noop_p):
        prev_vals = field_prev[safe_prev, n_idx, :]
        base = jnp.where((prev_row >= 0)[:, None], prev_vals, noop_p)
        return jax.vmap(lambda plane, kk, v: plane.at[kk].set(v))(base, micro_k, new_k)

    new_halt = _grow(buf.micro_halt_now, micro_halt_now.astype(jnp.bool_), noop_h)
    new_send = _grow(buf.send, send_now.astype(jnp.float32), noop_s)
    new_angle = _grow(buf.angle, angle_now.astype(jnp.float32), noop_a)
    new_slot = _grow(buf.slot, slot_now.astype(jnp.int32), noop_sl)
    new_pf = _grow(buf.pair_flat, pair_flat.astype(jnp.int32), noop_pf)
    new_fi = _grow(buf.frac_idx, frac_idx.astype(jnp.int32), noop_fi)

    def _scatter_rows(field_plane, new_plane):
        old = field_plane[write_row, n_idx, :]
        merged = jnp.where(active[:, None], new_plane, old)
        return field_plane.at[write_row, n_idx, :].set(merged)

    out_halt = _scatter_rows(buf.micro_halt_now, new_halt)
    out_send = _scatter_rows(buf.send, new_send)
    out_angle = _scatter_rows(buf.angle, new_angle)
    out_slot = _scatter_rows(buf.slot, new_slot)
    out_pf = _scatter_rows(buf.pair_flat, new_pf)
    out_fi = _scatter_rows(buf.frac_idx, new_fi)

    out_ha = buf.halt_action.at[write_row, n_idx].set(halt_action.astype(jnp.int32))
    out_nvp = buf.no_valid_pairs.at[write_row, n_idx].set(no_valid_pairs.astype(jnp.bool_))
    out_nvf = buf.no_valid_fracs.at[write_row, n_idx].set(no_valid_fracs.astype(jnp.bool_))
    out_mh = buf.must_halt_no_ships.at[write_row, n_idx].set(must_halt_no_ships.astype(jnp.bool_))

    old_pm = buf.phase_micro_idx[write_row, n_idx]
    new_pm = jnp.where(active, micro_k.astype(jnp.int32), old_pm)
    out_pm = buf.phase_micro_idx.at[write_row, n_idx].set(new_pm)

    return TransitionBuffer(
        micro_halt_now=out_halt,
        send=out_send,
        angle=out_angle,
        slot=out_slot,
        halt_action=out_ha,
        pair_flat=out_pf,
        frac_idx=out_fi,
        no_valid_pairs=out_nvp,
        no_valid_fracs=out_nvf,
        must_halt_no_ships=out_mh,
        phase_micro_idx=out_pm,
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
    angle_m = _gp(buf0.angle, buf1.angle)
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
        angle_m,
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

    return (
        state_mb,
        halt_action,
        pair_flat,
        frac_idx,
        no_valid_pairs,
        no_valid_fracs,
        must_halt_no_ships,
    )

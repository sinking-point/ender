"""Torch-only PPO replay helpers.

Rollout/env stepping can stay in JAX, but PPO minibatch replay should not need
to hand state back to XLA.  This module reconstructs pre-action states from the
Torch rollout buffers and builds policy observations directly in Torch.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

from jax_orbit_wars import (
    OrbitWarsState,
    PLANET_ID,
    PLANET_OWNER,
    PLANET_PRODUCTION,
    PLANET_RADIUS,
    PLANET_SHIPS,
    PLANET_X,
    PLANET_Y,
)

from orbit_wars_pt.constants import (
    BOARD_SIZE,
    CENTER,
    ENTITY_CLS,
    ENTITY_COMET,
    FEATURE_DIM,
    INCOMING_TA_BINS,
    ENTITY_PLANET,
    MAX_PLANETS,
    NUM_OWNER_SLOTS,
    ROTATION_RADIUS_LIMIT,
)
from orbit_wars_pt.parallel_rollout import RolloutSegment


def _remap_owner_2p(owner: torch.Tensor, ego: torch.Tensor) -> torch.Tensor:
    o = owner.to(torch.int64)
    e = ego.to(torch.int64)
    while e.ndim < o.ndim:
        e = e.unsqueeze(-1)
    out = torch.where(o < 0, torch.zeros_like(o), torch.where(o == e, torch.ones_like(o), torch.full_like(o, 2)))
    return out.clamp_max(NUM_OWNER_SLOTS - 1).to(torch.long)


def apply_prefix_micro_deltas_torch(
    state: OrbitWarsState,
    ego: torch.Tensor,
    micro_halt: torch.Tensor,
    send_m: torch.Tensor,
    slot_m: torch.Tensor,
    pair_flat_m: torch.Tensor,
    angle_m: torch.Tensor,
    fleet_eta_m: torch.Tensor,
    phase_micro_idx: torch.Tensor,
) -> OrbitWarsState:
    """Torch equivalent of ``apply_prefix_micro_deltas_batched``."""

    del slot_m, angle_m
    b, m = micro_halt.shape
    device = state.planets.device
    k = torch.arange(m, device=device)
    apply_mask = k.unsqueeze(0) < phase_micro_idx.to(torch.long).unsqueeze(1)
    dispatch_bm = apply_mask & (~micro_halt.to(torch.bool))
    send_eff = torch.where(dispatch_bm, send_m, torch.zeros_like(send_m))
    o_idx_m = (pair_flat_m // MAX_PLANETS).to(torch.long).clamp(0, MAX_PLANETS - 1)
    deduct = torch.zeros((b, MAX_PLANETS), dtype=state.planets.dtype, device=device)
    deduct.scatter_add_(1, o_idx_m, send_eff.to(state.planets.dtype))
    planets = state.planets.clone()
    planets[..., PLANET_SHIPS] = planets[..., PLANET_SHIPS] - deduct

    ta_m = (fleet_eta_m - 1.0).clamp_min(0.0).floor().to(torch.long).clamp(0, INCOMING_TA_BINS - 1)
    d_idx_m = (pair_flat_m % MAX_PLANETS).to(torch.long).clamp(0, MAX_PLANETS - 1)
    num_players = int(state.incoming_fleets.shape[1])
    ego_m = ego.to(torch.long).clamp(0, num_players - 1)[:, None].expand(b, m)
    flat_idx = (ego_m * (MAX_PLANETS * INCOMING_TA_BINS)) + (d_idx_m * INCOMING_TA_BINS) + ta_m
    add = torch.where(
        dispatch_bm,
        send_m.clamp_min(0.0).clamp_max(65535.0).to(torch.int32),
        torch.zeros_like(send_m, dtype=torch.int32),
    )
    incoming_i32 = state.incoming_fleets.to(torch.int32).reshape(b, -1).clone()
    incoming_i32.scatter_add_(1, flat_idx, add)
    incoming = incoming_i32.clamp_(0, 65535).reshape_as(state.incoming_fleets).to(state.incoming_fleets.dtype)
    return state._replace(planets=planets, incoming_fleets=incoming)


def select_and_replay_minibatch_torch(
    segment: RolloutSegment,
    mb_player: np.ndarray,
    mb_t: np.ndarray,
    mb_n: np.ndarray,
    turn_tag_cache: tuple[torch.Tensor, torch.Tensor],
    replay_device: Optional[torch.device] = None,
    timing: Optional[object] = None,
) -> tuple[OrbitWarsState, dict[str, torch.Tensor]]:
    """Select rows from Torch rollout buffers and reconstruct pre-action state."""

    buf0, buf1 = segment.buf0, segment.buf1
    storage_device = buf0.micro_halt_now.device
    out_device = replay_device if replay_device is not None else storage_device
    mb_t_t = torch.as_tensor(mb_t, dtype=torch.long, device=storage_device)
    mb_n_t = torch.as_tensor(mb_n, dtype=torch.long, device=storage_device)
    player_t = torch.as_tensor(mb_player, dtype=torch.long, device=storage_device)
    is_p0 = player_t == 0

    state_device = segment.turn_state_cache.planets.device
    mb_t_s = mb_t_t.to(state_device)
    mb_n_s = mb_n_t.to(state_device)
    player_s = player_t.to(state_device)
    is_p0_s = is_p0.to(state_device)
    tag0, tag1 = turn_tag_cache
    turn_idx = torch.where(is_p0_s, tag0[mb_t_s, mb_n_s], tag1[mb_t_s, mb_n_s]).to(torch.long)
    if os.environ.get("ORBIT_WARS_VALIDATE_CUDA_INDEXES") == "1":
        turn_idx_cpu = turn_idx.detach().cpu()
        cache_rows = int(segment.turn_state_cache.planets.shape[0])
        if bool(torch.any((turn_idx_cpu < 0) | (turn_idx_cpu >= cache_rows))):
            raise RuntimeError(
                f"select_and_replay_minibatch_torch turn tag out of range: "
                f"min={int(turn_idx_cpu.min())} max={int(turn_idx_cpu.max())} cache_rows={cache_rows}"
            )
    import time

    t_select0 = time.perf_counter()
    state = OrbitWarsState(
        **{
            field: getattr(segment.turn_state_cache, field)[turn_idx, mb_n_s].to(out_device)
            for field in OrbitWarsState._fields
        }
    )

    def plane(f0: torch.Tensor, f1: torch.Tensor) -> torch.Tensor:
        return torch.where(is_p0[:, None], f0[mb_t_t, mb_n_t, :], f1[mb_t_t, mb_n_t, :])

    def scalar(f0: torch.Tensor, f1: torch.Tensor) -> torch.Tensor:
        return torch.where(is_p0, f0[mb_t_t, mb_n_t], f1[mb_t_t, mb_n_t])

    phase = scalar(buf0.phase_micro_idx, buf1.phase_micro_idx).to(torch.long)
    if os.environ.get("ORBIT_WARS_VALIDATE_CUDA_INDEXES") == "1":
        phase_cpu = phase.detach().cpu()
        m_buf = int(buf0.micro_halt_now.shape[2])
        if bool(torch.any((phase_cpu < 0) | (phase_cpu >= m_buf))):
            raise RuntimeError(
                f"select_and_replay_minibatch_torch phase index out of range: "
                f"min={int(phase_cpu.min())} max={int(phase_cpu.max())} M={m_buf}"
            )
    if timing is not None:
        timing.gather_select_s += time.perf_counter() - t_select0
    t_prefix0 = time.perf_counter()
    state = apply_prefix_micro_deltas_torch(
        state,
        player_s.to(out_device),
        plane(buf0.micro_halt_now, buf1.micro_halt_now).to(out_device),
        plane(buf0.send, buf1.send).to(out_device),
        plane(buf0.slot, buf1.slot).to(out_device),
        plane(buf0.pair_flat, buf1.pair_flat).to(out_device),
        plane(buf0.angle, buf1.angle).to(out_device),
        plane(buf0.fleet_eta, buf1.fleet_eta).to(out_device),
        phase.to(out_device),
    )
    if timing is not None:
        timing.prefix_replay_s += time.perf_counter() - t_prefix0

    t_select0 = time.perf_counter()
    row = torch.arange(player_t.shape[0], device=storage_device)
    pf_m = plane(buf0.pair_flat, buf1.pair_flat)
    fi_m = plane(buf0.frac_idx, buf1.frac_idx)
    actions = {
        "halt_action": scalar(buf0.halt_action, buf1.halt_action).to(out_device, dtype=torch.long),
        "pair_flat": pf_m[row, phase.to(storage_device)].to(out_device, dtype=torch.long),
        "frac_idx": fi_m[row, phase.to(storage_device)].to(out_device, dtype=torch.long),
        "no_valid_pairs": scalar(buf0.no_valid_pairs, buf1.no_valid_pairs).to(out_device, dtype=torch.bool),
        "no_valid_fracs": scalar(buf0.no_valid_fracs, buf1.no_valid_fracs).to(out_device, dtype=torch.bool),
        "must_halt_no_ships": scalar(buf0.must_halt_no_ships, buf1.must_halt_no_ships).to(out_device, dtype=torch.bool),
        "target_planet_reachable": torch.where(
            is_p0[:, None],
            buf0.target_planet_reachable[mb_t_t, mb_n_t, :],
            buf1.target_planet_reachable[mb_t_t, mb_n_t, :],
        ).to(out_device, dtype=torch.bool),
        "ego": player_t.to(out_device),
    }
    if timing is not None:
        timing.gather_select_s += time.perf_counter() - t_select0
    return state, actions


def _planet_velocities_torch(state: OrbitWarsState, is_comet: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    planets = state.planets
    init = state.initial_planets
    active = state.planet_active
    init_active = state.initial_active
    cur_xy = planets[..., PLANET_X : PLANET_Y + 1]
    init_xy = init[..., PLANET_X : PLANET_Y + 1]
    radius = planets[..., PLANET_RADIUS]
    dx0 = init_xy[..., 0] - CENTER
    dy0 = init_xy[..., 1] - CENTER
    orbital_r = torch.hypot(dx0, dy0)
    rotates = (orbital_r >= 1e-6) & ((orbital_r + radius) < ROTATION_RADIUS_LIMIT) & init_active & active
    th0 = torch.atan2(dy0, dx0)
    th_next = th0 + state.angular_velocity[:, None] * (state.step_count.to(torch.float32)[:, None] + 1.0)
    nx = CENTER + orbital_r * torch.cos(th_next)
    ny = CENTER + orbital_r * torch.sin(th_next)
    rot_vx = torch.where(rotates, nx - cur_xy[..., 0], torch.zeros_like(nx))
    rot_vy = torch.where(rotates, ny - cur_xy[..., 1], torch.zeros_like(ny))

    paths = state.comet_paths
    b, g, kk, tmax, _ = paths.shape
    safe_idx = state.comet_path_index[:, :, None].clamp(0, tmax - 2).expand(b, g, kk)
    idx0 = safe_idx[..., None, None].expand(b, g, kk, 1, 2).long()
    idx1 = (safe_idx + 1)[..., None, None].expand(b, g, kk, 1, 2).long()
    p0 = torch.gather(paths, 3, idx0).squeeze(3)
    p1 = torch.gather(paths, 3, idx1).squeeze(3)
    diff = p1 - p0
    valid_gk = (
        (state.comet_path_lengths > 1)
        & (state.comet_path_index[:, :, None] >= 0)
        & (state.comet_path_index[:, :, None] < state.comet_path_lengths - 1)
    )
    pid = planets[..., PLANET_ID].to(torch.int32)
    match = state.comet_planet_ids[:, None, :, :] == pid[:, :, None, None]
    weight = (match & valid_gk[:, None, :, :]).to(paths.dtype)
    comet_vx = (diff[:, None, :, :, 0] * weight).sum(dim=(2, 3))
    comet_vy = (diff[:, None, :, :, 1] * weight).sum(dim=(2, 3))
    vx = torch.where(is_comet & active, comet_vx, rot_vx)
    vy = torch.where(is_comet & active, comet_vy, rot_vy)
    return torch.where(active, vx, torch.zeros_like(vx)), torch.where(active, vy, torch.zeros_like(vy))


def build_observation_torch(state: OrbitWarsState, ego: torch.Tensor, ship_speed: float = 6.0) -> dict[str, torch.Tensor]:
    del ship_speed
    planets = state.planets
    active = state.planet_active
    b = planets.shape[0]
    device = planets.device
    dtype = planets.dtype
    pid = planets[..., PLANET_ID].to(torch.int32)
    is_comet = (state.comet_planet_ids[:, None, :, :] == pid[:, :, None, None]).any(dim=(2, 3))
    vx, vy = _planet_velocities_torch(state, is_comet)

    planet_etype = torch.where(is_comet, torch.full_like(pid, ENTITY_COMET), torch.full_like(pid, ENTITY_PLANET)).long()
    planet_owner = _remap_owner_2p(planets[..., PLANET_OWNER], ego)
    incoming = state.incoming_fleets.to(dtype)
    batch_idx = torch.arange(b, device=device)
    ego_i = ego.to(torch.long).clamp(0, int(state.incoming_fleets.shape[1]) - 1)
    self_incoming = incoming[batch_idx, ego_i]
    enemy_incoming = incoming.sum(dim=1) - self_incoming
    incoming_net = (self_incoming - enemy_incoming) / 1000.0

    planet_features = torch.zeros((b, MAX_PLANETS, FEATURE_DIM), dtype=dtype, device=device)
    planet_features[..., 0] = torch.log1p(planets[..., PLANET_PRODUCTION].clamp_min(0.0))
    planet_features[..., 1] = planets[..., PLANET_SHIPS] / 1000.0
    planet_features[..., 2] = vx / 5.0
    planet_features[..., 3] = vy / 5.0
    planet_features[..., 4] = active.to(dtype)
    planet_features[..., 5] = planets[..., PLANET_RADIUS] / 10.0
    planet_features[..., 8:] = incoming_net
    planet_rope = torch.zeros((b, MAX_PLANETS, 3), dtype=dtype, device=device)
    xy = torch.where(active[..., None], planets[..., PLANET_X : PLANET_Y + 1], torch.zeros_like(planets[..., PLANET_X : PLANET_Y + 1]))
    planet_rope[..., 0] = xy[..., 0] / BOARD_SIZE
    planet_rope[..., 1] = xy[..., 1] / BOARD_SIZE

    cls_type = torch.full((b, 1), ENTITY_CLS, dtype=torch.long, device=device)
    cls_owner = torch.ones((b, 1), dtype=torch.long, device=device)
    cls_features = torch.zeros((b, 1, FEATURE_DIM), dtype=dtype, device=device)
    cls_features[:, 0, 6] = (state.step_count.to(dtype) / 498.0).clamp(0.0, 1.0)
    cls_rope = torch.tensor([CENTER / BOARD_SIZE, CENTER / BOARD_SIZE, 0.0], dtype=dtype, device=device).view(1, 1, 3).expand(b, 1, 3)
    return {
        "entity_type": torch.cat([cls_type, planet_etype], dim=1),
        "owner_idx": torch.cat([cls_owner, planet_owner], dim=1),
        "features": torch.cat([cls_features, planet_features], dim=1),
        "rope_pos": torch.cat([cls_rope, planet_rope], dim=1),
        "entity_mask": torch.cat([torch.ones((b, 1), dtype=torch.bool, device=device), active], dim=1),
        "planet_mask": torch.cat(
            [
                torch.zeros((b, 1), dtype=torch.bool, device=device),
                torch.ones((b, MAX_PLANETS), dtype=torch.bool, device=device),
            ],
            dim=1,
        ),
    }

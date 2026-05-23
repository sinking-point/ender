"""Torch-only PPO replay helpers.

Rollout/env stepping can stay in JAX, but PPO minibatch replay should not need
to hand state back to XLA.  This module reconstructs pre-action states from the
Torch rollout buffers and builds policy observations directly in Torch.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from orbit_wars_pt.compressed_observation import (
    CompressedObservationBuffer,
    decode_observation,
)
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
    FEATURE_DIM_MULTI,
    INCOMING_SURVIVOR_FLAT,
    INCOMING_TA_BINS,
    ENTITY_PLANET,
    MAX_PLANETS,
    NUM_OWNER_SLOTS,
    ROTATION_RADIUS_LIMIT,
    obs_feature_dim_for_num_agents,
)
from orbit_wars_pt.parallel_rollout import RolloutSegment
from orbit_wars_pt.transition_buffer import TorchTransitionBuffer


def _remap_owner_2p(owner: torch.Tensor, ego: torch.Tensor) -> torch.Tensor:
    o = owner.to(torch.int64)
    e = ego.to(torch.int64)
    while e.ndim < o.ndim:
        e = e.unsqueeze(-1)
    out = torch.where(o < 0, torch.zeros_like(o), torch.where(o == e, torch.ones_like(o), torch.full_like(o, 2)))
    return out.clamp_max(NUM_OWNER_SLOTS - 1).to(torch.long)


def _incoming_interfleet_torch(
    incoming: torch.Tensor,
    ego: torch.Tensor,
    num_agents: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``incoming`` ``[B, A, P, T]`` — same interfleet reduction as ``observation_jax``."""

    b, a, p, t_bins = incoming.shape
    pad = 4 - a
    padded = torch.nn.functional.pad(incoming.float(), (0, 0, 0, 0, 0, pad))
    ships = padded.permute(0, 2, 3, 1)
    order = ships.argsort(dim=-1, descending=True)
    top_s = ships.gather(-1, order[..., :1]).squeeze(-1)
    second_s = ships.gather(-1, order[..., 1:2]).squeeze(-1)
    top_p = order[..., 0]
    survivor = torch.where(top_s == second_s, torch.zeros_like(top_s), top_s - second_s)
    ego_e = ego.to(torch.long).view(b, 1, 1).expand(b, p, t_bins)
    signed = torch.where(survivor <= 0.0, torch.zeros_like(survivor), torch.where(top_p == ego_e, survivor, -survivor))
    is_self = top_p == ego_e
    slot_le2 = torch.where(survivor <= 0.0, torch.zeros_like(top_p), torch.where(is_self, torch.ones_like(top_p), torch.full_like(top_p, 2)))
    slot_gt2 = torch.where(
        survivor <= 0.0,
        torch.zeros_like(top_p),
        torch.where(is_self, torch.ones_like(top_p), torch.where(top_p < ego_e, 2 + top_p, 2 + (top_p - 1))),
    )
    survivor_slot = slot_gt2 if num_agents > 2 else slot_le2
    survivor_slot = survivor_slot.clamp(0, NUM_OWNER_SLOTS - 1).to(torch.long)
    return signed / 1000.0, survivor_slot


def apply_prefix_micro_deltas_torch(
    state: OrbitWarsState,
    ego: torch.Tensor,
    micro_halt: torch.Tensor,
    send_m: torch.Tensor,
    slot_m: torch.Tensor,
    pair_flat_m: torch.Tensor,
    fleet_eta_m: torch.Tensor,
    phase_micro_idx: torch.Tensor,
) -> OrbitWarsState:
    """Torch equivalent of ``apply_prefix_micro_deltas_batched``."""

    del slot_m
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

    # Mirror ``micro_jax._per_env_apply_one`` / ``_launch_fleets``: bin = floor(eta).
    ta_m = fleet_eta_m.floor().to(torch.long).clamp(0, INCOMING_TA_BINS - 1)
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


def _select_multi_compressed_observation(
    obs_bufs: list[CompressedObservationBuffer],
    mb_player: torch.Tensor,
    mb_t: torch.Tensor,
    mb_n: torch.Tensor,
    *,
    device: torch.device,
) -> CompressedObservationBuffer:
    """Gather compressed obs rows from the per-ego buffer selected by ``mb_player``."""

    storage_device = obs_bufs[0].token_meta.device
    t = mb_t.to(device=storage_device, dtype=torch.long)
    n = mb_n.to(device=storage_device, dtype=torch.long)
    player = mb_player.to(device=storage_device, dtype=torch.long)
    bsz = int(t.shape[0])

    first = obs_bufs[0]
    token_meta = torch.zeros((bsz,) + first.token_meta.shape[2:], device=device, dtype=first.token_meta.dtype)
    owner_idx = torch.zeros((bsz,) + first.owner_idx.shape[2:], device=device, dtype=first.owner_idx.dtype)
    production = torch.zeros((bsz,) + first.production.shape[2:], device=device, dtype=first.production.dtype)
    ships = torch.zeros((bsz,) + first.ships.shape[2:], device=device, dtype=first.ships.dtype)
    velocity = torch.zeros((bsz,) + first.velocity.shape[2:], device=device, dtype=first.velocity.dtype)
    xy = torch.zeros((bsz,) + first.xy.shape[2:], device=device, dtype=first.xy.dtype)
    turn_progress = torch.zeros((bsz,) + first.turn_progress.shape[2:], device=device, dtype=first.turn_progress.dtype)
    incoming_net = torch.zeros((bsz,) + first.incoming_net.shape[2:], device=device, dtype=first.incoming_net.dtype)
    incoming_survivor = torch.zeros(
        (bsz,) + first.incoming_survivor.shape[2:], device=device, dtype=first.incoming_survivor.dtype
    )

    for p, obs in enumerate(obs_bufs):
        m = player == p
        if not bool(m.any().item()):
            continue
        tp, np_ = t[m], n[m]
        token_meta[m] = obs.token_meta[tp, np_].to(device)
        owner_idx[m] = obs.owner_idx[tp, np_].to(device)
        production[m] = obs.production[tp, np_].to(device)
        ships[m] = obs.ships[tp, np_].to(device)
        velocity[m] = obs.velocity[tp, np_].to(device)
        xy[m] = obs.xy[tp, np_].to(device)
        turn_progress[m] = obs.turn_progress[tp, np_].to(device)
        incoming_net[m] = obs.incoming_net[tp, np_].to(device)
        incoming_survivor[m] = obs.incoming_survivor[tp, np_].to(device)

    return CompressedObservationBuffer(
        token_meta=token_meta,
        owner_idx=owner_idx,
        production=production,
        ships=ships,
        velocity=velocity,
        xy=xy,
        turn_progress=turn_progress,
        incoming_net=incoming_net,
        incoming_survivor=incoming_survivor,
    )


def _gather_transition_fields_for_players(
    bufs: list[TorchTransitionBuffer],
    mb_player: torch.Tensor,
    mb_t: torch.Tensor,
    mb_n_v: torch.Tensor,
    *,
    storage_device: torch.device,
    out_device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build per-minibatch action records by gathering from each player's rollout buffer."""

    t = mb_t.to(device=storage_device, dtype=torch.long)
    n = mb_n_v.to(device=storage_device, dtype=torch.long)
    player = mb_player.to(device=storage_device, dtype=torch.long)
    bsz = int(t.shape[0])
    row = torch.arange(bsz, device=storage_device)

    phase = torch.zeros((bsz,), device=storage_device, dtype=torch.long)
    pair_flat_acc = torch.zeros((bsz, bufs[0].pair_flat.shape[-1]), device=storage_device, dtype=torch.int32)
    frac_acc = torch.zeros((bsz, bufs[0].frac_idx.shape[-1]), device=storage_device, dtype=torch.int32)
    halt_action = torch.zeros((bsz,), device=storage_device, dtype=torch.long)
    no_valid_pairs = torch.zeros((bsz,), device=storage_device, dtype=torch.bool)
    no_valid_fracs = torch.zeros((bsz,), device=storage_device, dtype=torch.bool)
    must_halt_no_ships = torch.zeros((bsz,), device=storage_device, dtype=torch.bool)
    population_idx = torch.zeros((bsz,), device=storage_device, dtype=torch.long)
    tpr = torch.zeros((bsz, MAX_PLANETS), device=storage_device, dtype=torch.bool)
    tht = torch.zeros((bsz, MAX_PLANETS), device=storage_device, dtype=torch.float32)

    for p, buf in enumerate(bufs):
        m = player == p
        if not bool(m.any().item()):
            continue
        tp, np_ = t[m], n[m]
        phase[m] = buf.phase_micro_idx[tp, np_].to(torch.long)
        pair_flat_acc[m] = buf.pair_flat[tp, np_, :]
        frac_acc[m] = buf.frac_idx[tp, np_, :]
        halt_action[m] = buf.halt_action[tp, np_].to(torch.long)
        no_valid_pairs[m] = buf.no_valid_pairs[tp, np_]
        no_valid_fracs[m] = buf.no_valid_fracs[tp, np_]
        must_halt_no_ships[m] = buf.must_halt_no_ships[tp, np_]
        population_idx[m] = buf.population_idx[tp, np_].to(torch.long)
        tpr[m] = buf.target_planet_reachable[tp, np_, :]
        tht[m] = buf.target_hit_tick[tp, np_, :]

    actions = {
        "halt_action": halt_action.to(out_device, dtype=torch.long),
        "pair_flat": pair_flat_acc[row, phase.to(storage_device)].to(out_device, dtype=torch.long),
        "frac_idx": frac_acc[row, phase.to(storage_device)].to(out_device, dtype=torch.long),
        "no_valid_pairs": no_valid_pairs.to(out_device, dtype=torch.bool),
        "no_valid_fracs": no_valid_fracs.to(out_device, dtype=torch.bool),
        "must_halt_no_ships": must_halt_no_ships.to(out_device, dtype=torch.bool),
        "population_idx": population_idx.to(out_device, dtype=torch.long),
        "target_planet_reachable": tpr.to(out_device, dtype=torch.bool),
        "target_hit_tick": tht.to(out_device, dtype=torch.float32),
        "ego": player.to(out_device),
    }
    return actions


def select_stored_observation_minibatch_torch(
    segment: RolloutSegment,
    mb_player: np.ndarray,
    mb_t: np.ndarray,
    mb_n: np.ndarray,
    replay_device: Optional[torch.device] = None,
    timing: Optional[object] = None,
    *,
    obs_feature_dim: int = FEATURE_DIM,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Select stored compressed observations and action records for PPO."""

    import time

    bufs = segment.bufs
    storage_device = bufs[0].micro_halt_now.device
    out_device = replay_device if replay_device is not None else storage_device
    mb_t_t = torch.as_tensor(mb_t, dtype=torch.long, device=storage_device)
    mb_n_t = torch.as_tensor(mb_n, dtype=torch.long, device=storage_device)
    player_t = torch.as_tensor(mb_player, dtype=torch.long, device=storage_device)

    t_select0 = time.perf_counter()
    comp = _select_multi_compressed_observation(
        segment.obs_bufs,
        player_t,
        mb_t_t,
        mb_n_t,
        device=out_device,
    )
    obs = decode_observation(comp, feature_dim=obs_feature_dim)

    actions = _gather_transition_fields_for_players(
        bufs,
        player_t,
        mb_t_t,
        mb_n_t,
        storage_device=storage_device,
        out_device=out_device,
    )
    if timing is not None:
        timing.gather_select_s += time.perf_counter() - t_select0
    return obs, actions


def select_stored_compressed_minibatch_torch(
    segment: RolloutSegment,
    mb_player: np.ndarray,
    mb_t: np.ndarray,
    mb_n: np.ndarray,
    replay_device: Optional[torch.device] = None,
    timing: Optional[object] = None,
) -> tuple[CompressedObservationBuffer, dict[str, torch.Tensor]]:
    """Select stored compressed observations and action records for PPO."""

    import time

    bufs = segment.bufs
    storage_device = bufs[0].micro_halt_now.device
    out_device = replay_device if replay_device is not None else storage_device
    mb_t_t = torch.as_tensor(mb_t, dtype=torch.long, device=storage_device)
    mb_n_t = torch.as_tensor(mb_n, dtype=torch.long, device=storage_device)
    player_t = torch.as_tensor(mb_player, dtype=torch.long, device=storage_device)

    t_select0 = time.perf_counter()
    comp = _select_multi_compressed_observation(
        segment.obs_bufs,
        player_t,
        mb_t_t,
        mb_n_t,
        device=out_device,
    )
    actions = _gather_transition_fields_for_players(
        bufs,
        player_t,
        mb_t_t,
        mb_n_t,
        storage_device=storage_device,
        out_device=out_device,
    )
    if timing is not None:
        timing.gather_select_s += time.perf_counter() - t_select0
    return comp, actions


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
    na = int(incoming.shape[1])
    fdim = obs_feature_dim_for_num_agents(na)
    incoming_net, survivor_slot = _incoming_interfleet_torch(incoming, ego, na)

    planet_features = torch.zeros((b, MAX_PLANETS, fdim), dtype=dtype, device=device)
    planet_features[..., 0] = torch.log1p(planets[..., PLANET_PRODUCTION].clamp_min(0.0))
    planet_features[..., 1] = planets[..., PLANET_SHIPS] / 1000.0
    planet_features[..., 2] = vx / 5.0
    planet_features[..., 3] = vy / 5.0
    planet_features[..., 4] = active.to(dtype)
    planet_features[..., 5] = planets[..., PLANET_RADIUS] / 10.0
    planet_features[..., 8 : 8 + INCOMING_TA_BINS] = incoming_net
    if fdim == FEATURE_DIM_MULTI and na > 2:
        idx = survivor_slot.clamp(0, NUM_OWNER_SLOTS - 1)
        oh = torch.nn.functional.one_hot(idx, NUM_OWNER_SLOTS).to(dtype=dtype)
        oh_flat = oh.reshape(b, MAX_PLANETS, INCOMING_SURVIVOR_FLAT)
        planet_features[
            ..., 8 + INCOMING_TA_BINS : 8 + INCOMING_TA_BINS + INCOMING_SURVIVOR_FLAT
        ] = oh_flat
    planet_rope = torch.zeros((b, MAX_PLANETS, 3), dtype=dtype, device=device)
    xy = torch.where(active[..., None], planets[..., PLANET_X : PLANET_Y + 1], torch.zeros_like(planets[..., PLANET_X : PLANET_Y + 1]))
    planet_rope[..., 0] = xy[..., 0] / BOARD_SIZE
    planet_rope[..., 1] = xy[..., 1] / BOARD_SIZE

    cls_type = torch.full((b, 1), ENTITY_CLS, dtype=torch.long, device=device)
    cls_owner = torch.ones((b, 1), dtype=torch.long, device=device)
    cls_features = torch.zeros((b, 1, fdim), dtype=dtype, device=device)
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

"""Adapter for running a trained policy in the official Kaggle Orbit Wars env.

The Kaggle agent API calls ``agent(obs, config)`` and expects a Python list of
``[from_planet_id, angle, num_ships]`` launches.  Training uses a fixed-table
``OrbitWarsState`` plus a PyTorch policy, so this module bridges between the two.
"""

from __future__ import annotations

import hashlib
import math
import os
import traceback
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Optional

import numpy as np
import torch

from orbit_wars_pt.constants import (
    BOARD_SIZE,
    CENTER,
    ENTITY_CLS,
    ENTITY_COMET,
    ENTITY_PLANET,
    FEATURE_DIM,
    FRACTIONS,
    INCOMING_TA_BINS,
    MAX_PLANETS,
    NUM_OWNER_SLOTS,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
)
from orbit_wars_pt.geometry import estimate_time_to_hit, planet_pred_velocity
from orbit_wars_pt.model import OrbitWarsPolicy


DEFAULT_CHECKPOINT = "checkpoint.pt"
DEFAULT_RAYCAST_RAYS = 256
DEFAULT_MAX_ACTIONS = 64
MAX_COMET_GROUPS = 5
MAX_COMET_PATH = 40

PLANET_X = 2
PLANET_Y = 3
PLANET_RADIUS = 4
FLEET_ID = 0
FLEET_OWNER = 1
FLEET_X = 2
FLEET_Y = 3
FLEET_ANGLE = 4
FLEET_FROM_PLANET = 5
FLEET_SHIPS = 6
FLEET_ROW_WIDTH = 9


class OrbitWarsState(NamedTuple):
    planets: np.ndarray
    planet_active: np.ndarray
    initial_planets: np.ndarray
    initial_active: np.ndarray
    fleets: np.ndarray
    fleet_active: np.ndarray
    incoming_fleets: np.ndarray
    comet_paths: np.ndarray
    comet_path_lengths: np.ndarray
    comet_ships: np.ndarray
    comet_group_active: np.ndarray
    comet_path_index: np.ndarray
    comet_planet_ids: np.ndarray
    comet_slots: np.ndarray
    next_fleet_id: np.ndarray
    angular_velocity: np.ndarray
    step_count: np.ndarray
    num_agents: np.ndarray
    rewards: np.ndarray
    done: np.ndarray
    overflow: np.ndarray


def _cfg_get(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _as_array(rows: Any, width: int) -> np.ndarray:
    if rows is None:
        rows = []
    arr = np.asarray(rows, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, width), dtype=np.float32)
    arr = arr.reshape((-1, width))
    return arr.astype(np.float32, copy=False)


def _place_rows_by_id(rows: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    table = np.zeros((MAX_PLANETS, width), dtype=np.float32)
    active = np.zeros((MAX_PLANETS,), dtype=np.bool_)
    id_to_slot: dict[int, int] = {}
    next_free = 0

    for row in rows[:MAX_PLANETS]:
        pid = int(row[0])
        if 0 <= pid < MAX_PLANETS and not active[pid]:
            slot = pid
        else:
            while next_free < MAX_PLANETS and active[next_free]:
                next_free += 1
            if next_free >= MAX_PLANETS:
                break
            slot = next_free
        table[slot, : min(width, row.shape[0])] = row[:width]
        active[slot] = True
        id_to_slot[pid] = slot

    return table, active, id_to_slot


def _fleet_speed(ships: float, max_speed: float = 6.0) -> float:
    if ships <= 1.0:
        return 1.0
    return float(min(max_speed, 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5))


def _remap_owner(owner: float, ego: int, num_agents: int) -> int:
    o = int(owner)
    if o < 0:
        return 0
    if o == ego:
        return 1
    if num_agents <= 2:
        return 2
    return 2 + o if o < ego else 2 + (o - 1)


def _point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    delta = end - start
    l2 = float(np.dot(delta, delta))
    if l2 <= 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, delta) / l2, 0.0, 1.0))
    projection = start + t * delta
    return float(np.linalg.norm(point - projection))


def _swept_pair_hit(a0: np.ndarray, a1: np.ndarray, p0: np.ndarray, p1: np.ndarray, radius: float) -> bool:
    d0 = a0 - p0
    dv = (a1 - a0) - (p1 - p0)
    qa = float(np.dot(dv, dv))
    qb = float(2.0 * np.dot(d0, dv))
    qc = float(np.dot(d0, d0) - radius * radius)
    if qa < 1e-12:
        return qc <= 0.0
    disc = qb * qb - 4.0 * qa * qc
    if disc < 0.0:
        return False
    sqrt_disc = math.sqrt(max(disc, 0.0))
    t1 = (-qb - sqrt_disc) / (2.0 * qa)
    t2 = (-qb + sqrt_disc) / (2.0 * qa)
    return t2 >= 0.0 and t1 <= 1.0


def _next_planet_positions(
    planets: np.ndarray,
    planet_active: np.ndarray,
    initial_planets: np.ndarray,
    initial_active: np.ndarray,
    comet_paths: np.ndarray,
    comet_path_lengths: np.ndarray,
    comet_group_active: np.ndarray,
    comet_path_index: np.ndarray,
    comet_slots: np.ndarray,
    angular_velocity: float,
    step_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Host mirror of the path-only part used by the JAX raycast forecast."""

    old_pos = planets[:, PLANET_X : PLANET_Y + 1].copy()
    new_pos = old_pos.copy()

    init_pos = initial_planets[:, PLANET_X : PLANET_Y + 1]
    delta = init_pos - np.asarray([CENTER, CENTER], dtype=np.float32)
    orbital_r = np.linalg.norm(delta, axis=1)
    initial_angle = np.arctan2(delta[:, 1], delta[:, 0])
    rotating = planet_active & initial_active & (orbital_r + planets[:, PLANET_RADIUS] < ROTATION_RADIUS_LIMIT)
    current_angle = initial_angle + float(angular_velocity) * float(step_count)
    new_pos[rotating, 0] = CENTER + orbital_r[rotating] * np.cos(current_angle[rotating])
    new_pos[rotating, 1] = CENTER + orbital_r[rotating] * np.sin(current_angle[rotating])

    collision_enabled = planet_active.copy()
    next_path_index = comet_path_index + comet_group_active.astype(np.int32)
    expired_after_move = np.zeros_like(planet_active)

    for g in range(MAX_COMET_GROUPS):
        if not comet_group_active[g]:
            continue
        idx = int(next_path_index[g])
        for k in range(4):
            slot = int(comet_slots[g, k])
            if slot < 0 or slot >= MAX_PLANETS:
                continue
            length = int(comet_path_lengths[g, k])
            expired = idx >= length
            in_path = idx < length
            if in_path:
                new_pos[slot] = comet_paths[g, k, max(idx, 0)]
            first_placement = planets[slot, PLANET_X] < 0.0
            collision_enabled[slot] = (not first_placement) or expired
            expired_after_move[slot] = expired_after_move[slot] or expired

    return old_pos, new_pos, collision_enabled, next_path_index, planet_active & ~expired_after_move, initial_active & ~expired_after_move


def _forecast_incoming_fleets(
    planets: np.ndarray,
    planet_active: np.ndarray,
    initial_planets: np.ndarray,
    initial_active: np.ndarray,
    fleets_in: np.ndarray,
    comet_paths: np.ndarray,
    comet_path_lengths: np.ndarray,
    comet_group_active: np.ndarray,
    comet_path_index: np.ndarray,
    comet_slots: np.ndarray,
    num_agents: int,
    step_count: int,
    angular_velocity: float,
    *,
    ship_speed: float = 6.0,
    horizon: int = INCOMING_TA_BINS,
) -> np.ndarray:
    """Fill policy incoming bins with only public fleets that hit a planet in the forecast."""

    incoming = np.zeros((num_agents, MAX_PLANETS, INCOMING_TA_BINS), dtype=np.uint16)
    if fleets_in.size == 0:
        return incoming

    positions = fleets_in[:, 2:4].astype(np.float32, copy=True)
    alive = np.ones((len(fleets_in),), dtype=np.bool_)
    angles = fleets_in[:, 4].astype(np.float32, copy=False)
    owners = fleets_in[:, 1].astype(np.int32, copy=False)
    ships = np.floor(fleets_in[:, 6].astype(np.float32, copy=False))
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    speeds = np.asarray([_fleet_speed(float(s), ship_speed) for s in ships], dtype=np.float32)

    p = planets.copy()
    pa = planet_active.copy()
    ia = initial_active.copy()
    cpi = comet_path_index.copy()

    for t in range(min(horizon, INCOMING_TA_BINS)):
        old_pos, new_pos, collision_enabled, cpi_next, pa_next, ia_next = _next_planet_positions(
            p,
            pa,
            initial_planets,
            ia,
            comet_paths,
            comet_path_lengths,
            comet_group_active,
            cpi,
            comet_slots,
            angular_velocity,
            step_count + t,
        )

        for f in range(len(fleets_in)):
            if not alive[f]:
                continue
            owner = int(owners[f])
            if owner < 0 or owner >= num_agents or ships[f] <= 0:
                alive[f] = False
                continue
            a0 = positions[f]
            a1 = a0 + speeds[f] * dirs[f]

            hit_slot = -1
            for i in range(MAX_PLANETS):
                if not collision_enabled[i]:
                    continue
                if _swept_pair_hit(a0, a1, old_pos[i], new_pos[i], float(p[i, PLANET_RADIUS])):
                    hit_slot = i
                    break
            if hit_slot >= 0:
                add = int(min(max(int(ships[f]), 0), 65535))
                cur = int(incoming[owner, hit_slot, t])
                incoming[owner, hit_slot, t] = min(cur + add, 65535)
                alive[f] = False
            else:
                if _point_to_segment_distance(np.asarray([CENTER, CENTER], dtype=np.float32), a0, a1) < SUN_RADIUS:
                    alive[f] = False
                    continue
                if a1[0] < 0.0 or a1[0] > BOARD_SIZE or a1[1] < 0.0 or a1[1] > BOARD_SIZE:
                    alive[f] = False
                    continue
                positions[f] = a1

        p[:, PLANET_X : PLANET_Y + 1] = new_pos
        pa = pa_next
        ia = ia_next
        cpi = cpi_next

    return incoming


def _forecast_planet_paths_np(state: OrbitWarsState, horizon: int = INCOMING_TA_BINS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    planets = np.asarray(state.planets).copy()
    planet_active = np.asarray(state.planet_active).astype(bool).copy()
    initial_planets = np.asarray(state.initial_planets)
    initial_active = np.asarray(state.initial_active).astype(bool).copy()
    comet_paths = np.asarray(state.comet_paths)
    comet_path_lengths = np.asarray(state.comet_path_lengths)
    comet_group_active = np.asarray(state.comet_group_active).astype(bool)
    comet_path_index = np.asarray(state.comet_path_index).astype(np.int32).copy()
    comet_slots = np.asarray(state.comet_slots).astype(np.int32)
    angular_velocity = float(np.asarray(state.angular_velocity))
    step_count = int(np.asarray(state.step_count))

    p0_rows: list[np.ndarray] = []
    p1_rows: list[np.ndarray] = []
    active_rows: list[np.ndarray] = []
    for t in range(horizon):
        old_pos, new_pos, collision_enabled, cpi_next, pa_next, ia_next = _next_planet_positions(
            planets,
            planet_active,
            initial_planets,
            initial_active,
            comet_paths,
            comet_path_lengths,
            comet_group_active,
            comet_path_index,
            comet_slots,
            angular_velocity,
            step_count + t,
        )
        p0_rows.append(old_pos)
        p1_rows.append(new_pos)
        active_rows.append(collision_enabled)
        planets[:, PLANET_X : PLANET_Y + 1] = new_pos
        planet_active = pa_next
        initial_active = ia_next
        comet_path_index = cpi_next
    return (
        np.stack(p0_rows, axis=0).astype(np.float32),
        np.stack(p1_rows, axis=0).astype(np.float32),
        np.stack(active_rows, axis=0).astype(np.bool_),
    )


def _raycast_targets_np(
    state: OrbitWarsState,
    origin_idx: int,
    frac_idx: int,
    *,
    ship_speed: float = 6.0,
    horizon: int = INCOMING_TA_BINS,
    n_rays: int = DEFAULT_RAYCAST_RAYS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """NumPy version of the rollout discrete first-hit ray target sampler."""

    planets = np.asarray(state.planets)
    current_active = np.asarray(state.planet_active).astype(bool)
    p0, p1, active_by_tick = _forecast_planet_paths_np(state, horizon=horizon)
    radii = planets[:, PLANET_RADIUS].astype(np.float32)
    origin_xy = planets[origin_idx, PLANET_X : PLANET_Y + 1].astype(np.float32)
    origin_radius = float(planets[origin_idx, PLANET_RADIUS])
    ships_avail = float(planets[origin_idx, 5])
    send = math.floor(float(FRACTIONS[frac_idx]) * ships_avail)
    speed = _fleet_speed(float(send), ship_speed)

    angles = np.arange(n_rays, dtype=np.float32) * (2.0 * math.pi / float(n_rays))
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    pos = origin_xy[None, :] + (origin_radius + 0.1) * dirs
    done_policy = np.zeros((n_rays,), dtype=np.bool_)
    done_true = np.zeros((n_rays,), dtype=np.bool_)
    policy_code = np.full((n_rays,), -1, dtype=np.int32)
    policy_tick = np.full((n_rays,), 10_000, dtype=np.int32)
    true_code = np.full((n_rays,), -1, dtype=np.int32)
    true_tick = np.full((n_rays,), 10_000, dtype=np.int32)

    sun = np.asarray([CENTER, CENTER], dtype=np.float32)
    for t in range(horizon):
        if bool(np.all(done_policy & done_true)):
            break
        a0 = pos
        a1 = pos + speed * dirs

        d0 = a0[:, None, :] - p0[t][None, :, :]
        dv = (a1[:, None, :] - a0[:, None, :]) - (p1[t][None, :, :] - p0[t][None, :, :])
        qa = np.sum(dv * dv, axis=-1)
        qb = 2.0 * np.sum(d0 * dv, axis=-1)
        qc = np.sum(d0 * d0, axis=-1) - radii[None, :] ** 2
        disc = qb * qb - 4.0 * qa * qc
        static_hit = qc <= 0.0
        qa_safe = np.where(qa < 1e-12, 1.0, qa)
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
        t1 = (-qb - sqrt_disc) / (2.0 * qa_safe)
        t2 = (-qb + sqrt_disc) / (2.0 * qa_safe)
        moving_hit = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
        hit_raw = np.where(qa < 1e-12, static_hit, moving_hit)
        hit_true = hit_raw & active_by_tick[t][None, :]
        hit_policy = hit_true & current_active[None, :]

        any_true = np.any(hit_true, axis=1)
        idx_true = np.argmax(hit_true, axis=1).astype(np.int32)
        any_policy = np.any(hit_policy, axis=1)
        idx_policy = np.argmax(hit_policy, axis=1).astype(np.int32)

        delta = a1 - a0
        l2 = np.sum(delta * delta, axis=1)
        proj = np.zeros((n_rays,), dtype=np.float32)
        nonzero = l2 > 1e-12
        proj[nonzero] = np.sum((sun[None, :] - a0[nonzero]) * delta[nonzero], axis=1) / l2[nonzero]
        proj = np.clip(proj, 0.0, 1.0)
        closest = a0 + proj[:, None] * delta
        sun_hit = np.linalg.norm(closest - sun[None, :], axis=1) < SUN_RADIUS
        in_bounds = (a1[:, 0] >= 0.0) & (a1[:, 0] <= BOARD_SIZE) & (a1[:, 1] >= 0.0) & (a1[:, 1] <= BOARD_SIZE)
        oob = ~in_bounds

        had_policy = any_policy | sun_hit | oob
        new_policy = (~done_policy) & had_policy
        policy_code[new_policy] = np.where(any_policy[new_policy], idx_policy[new_policy], -1)
        policy_tick[new_policy] = t
        done_policy |= had_policy

        had_true = any_true | sun_hit | oob
        new_true = (~done_true) & had_true
        true_code[new_true] = np.where(any_true[new_true], idx_true[new_true], -1)
        true_tick[new_true] = t
        done_true |= had_true

        pos = a1

    out_angle = np.zeros((MAX_PLANETS,), dtype=np.float32)
    valid = np.zeros((MAX_PLANETS,), dtype=np.bool_)
    hit_tick = np.zeros((MAX_PLANETS,), dtype=np.float32)
    true_planet = np.full((MAX_PLANETS,), -1, dtype=np.int32)
    true_hit_tick = np.full((MAX_PLANETS,), 500.0, dtype=np.float32)
    for target in range(MAX_PLANETS):
        ray_idx = np.flatnonzero((policy_code == target) & done_policy)
        if ray_idx.size == 0:
            continue
        best_pos = int(np.lexsort((ray_idx, policy_tick[ray_idx]))[0])
        ray = int(ray_idx[best_pos])
        out_angle[target] = float(angles[ray] % (2.0 * math.pi))
        valid[target] = True
        hit_tick[target] = float(policy_tick[ray])
        if 0 <= int(true_code[ray]) < MAX_PLANETS:
            true_planet[target] = int(true_code[ray])
            true_hit_tick[target] = float(true_tick[ray])
    valid &= current_active
    valid[origin_idx] = False
    true_planet = np.where(valid, true_planet, -1).astype(np.int32)
    true_hit_tick = np.where(valid, true_hit_tick, 500.0).astype(np.float32)
    return out_angle, valid, hit_tick, true_planet, true_hit_tick


def _obs_tensors_for_state(state: OrbitWarsState, ego_player: int, device: torch.device) -> dict[str, torch.Tensor]:
    planets = np.asarray(state.planets)
    planet_active = np.asarray(state.planet_active)
    initial_planets = np.asarray(state.initial_planets)
    initial_active = np.asarray(state.initial_active)
    incoming_fleets = np.asarray(state.incoming_fleets)
    angular_velocity = float(np.asarray(state.angular_velocity))
    step_count = int(np.asarray(state.step_count))
    num_agents = int(np.asarray(state.num_agents))
    comet_ids = np.asarray(state.comet_planet_ids)
    comet_set = set(float(x) for x in comet_ids.flatten() if int(x) >= 0)

    entity_type = np.zeros((1 + MAX_PLANETS,), dtype=np.int64)
    owner_idx = np.zeros((1 + MAX_PLANETS,), dtype=np.int64)
    features = np.zeros((1 + MAX_PLANETS, FEATURE_DIM), dtype=np.float32)
    rope_pos = np.zeros((1 + MAX_PLANETS, 3), dtype=np.float32)
    entity_mask = np.zeros((1 + MAX_PLANETS,), dtype=np.bool_)
    planet_mask = np.zeros((1 + MAX_PLANETS,), dtype=np.bool_)

    entity_type[0] = ENTITY_CLS
    owner_idx[0] = 1
    features[0, 6] = np.float32(np.clip(float(step_count) / 498.0, 0.0, 1.0))
    rope_pos[0] = np.asarray([CENTER / BOARD_SIZE, CENTER / BOARD_SIZE, 0.0], dtype=np.float32)
    entity_mask[0] = True

    incoming = incoming_fleets.astype(np.float32)
    self_incoming = incoming[int(ego_player)]
    enemy_incoming = incoming[np.arange(incoming.shape[0]) != int(ego_player)].sum(axis=0)
    incoming_net = (self_incoming - enemy_incoming) / 1000.0

    for i in range(MAX_PLANETS):
        j = 1 + i
        active = bool(planet_active[i])
        pid = float(planets[i, 0])
        is_comet = pid in comet_set
        entity_type[j] = ENTITY_COMET if is_comet else ENTITY_PLANET
        owner_idx[j] = min(_remap_owner(float(planets[i, 1]), ego_player, num_agents), NUM_OWNER_SLOTS - 1)
        vx, vy = planet_pred_velocity(
            initial_planets[i, 2:4].astype(np.float64),
            planets[i, 2:4].astype(np.float64),
            float(planets[i, 4]),
            angular_velocity,
            step_count,
            bool(initial_active[i]),
            active,
        )
        if is_comet and active:
            group_row = np.where(comet_ids == int(pid))
            if group_row[0].size > 0:
                g, k = int(group_row[0][0]), int(group_row[1][0])
                paths = np.asarray(state.comet_paths[g, k])
                lens = int(np.asarray(state.comet_path_lengths[g, k]))
                idx = int(np.asarray(state.comet_path_index[g]))
                if lens > 1 and 0 <= idx < lens - 1:
                    vx = float(paths[idx + 1, 0] - paths[idx, 0])
                    vy = float(paths[idx + 1, 1] - paths[idx, 1])

        features[j, 0] = np.log1p(max(float(planets[i, 6]), 0.0))
        features[j, 1] = float(planets[i, 5]) / 1000.0
        features[j, 2] = float(vx) / 5.0
        features[j, 3] = float(vy) / 5.0
        features[j, 4] = float(active)
        features[j, 5] = float(planets[i, 4]) / 10.0
        features[j, 8:] = incoming_net[i]
        if active:
            rope_pos[j, 0] = float(planets[i, 2]) / BOARD_SIZE
            rope_pos[j, 1] = float(planets[i, 3]) / BOARD_SIZE
        entity_mask[j] = active
        planet_mask[j] = True

    def tensor(x: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(x)).to(device=device, dtype=dtype).unsqueeze(0)

    return {
        "entity_type": tensor(entity_type, torch.long),
        "owner_idx": tensor(owner_idx, torch.long),
        "features": tensor(features, torch.float32),
        "rope_pos": tensor(rope_pos, torch.float32),
        "entity_mask": tensor(entity_mask, torch.bool),
        "planet_mask": tensor(planet_mask, torch.bool),
    }


def _build_turn_actions_torch_only(
    policy: OrbitWarsPolicy,
    state: OrbitWarsState,
    ego_player: int,
    device: torch.device,
    *,
    ship_speed: float = 6.0,
    max_micro_steps: int = DEFAULT_MAX_ACTIONS,
    greedy: bool = False,
    rng: Optional[torch.Generator] = None,
    n_rays: int = DEFAULT_RAYCAST_RAYS,
) -> list[list[float]]:
    planets = np.array(np.asarray(state.planets), copy=True)
    incoming_fleets = np.array(np.asarray(state.incoming_fleets), copy=True)
    planet_active = np.asarray(state.planet_active).astype(bool)
    actions: list[list[float]] = []

    for _ in range(max_micro_steps):
        virt = state._replace(planets=planets, incoming_fleets=incoming_fleets)
        batch = _obs_tensors_for_state(virt, ego_player, device)
        out = policy(**batch)
        halt_logits = out["halt_logits"][0]
        if greedy:
            halt_action = int(torch.argmax(halt_logits, dim=-1).item())
        else:
            halt_probs = torch.softmax(halt_logits, dim=-1)
            halt_action = int(torch.multinomial(halt_probs, 1, generator=rng).item())
        if halt_action == 1:
            break

        flat_mask = out["origin_frac_mask"].flatten(start_dim=1)[0]
        if not bool(flat_mask.any().item()):
            break
        flat_logits = out["origin_frac_logits"].flatten(start_dim=1)[0]
        masked_origin_frac = flat_logits.masked_fill(~flat_mask, -1e4)
        if greedy:
            origin_frac_flat = int(torch.argmax(masked_origin_frac).item())
        else:
            origin_frac_probs = torch.softmax(masked_origin_frac, dim=-1)
            origin_frac_flat = int(torch.multinomial(origin_frac_probs, 1, generator=rng).item())
        o_idx = origin_frac_flat // len(FRACTIONS)
        frac_idx = origin_frac_flat % len(FRACTIONS)

        ray_angle, ray_valid, ray_hit_tick, true_planet, true_hit_tick = _raycast_targets_np(
            virt,
            int(o_idx),
            int(frac_idx),
            ship_speed=ship_speed,
            horizon=INCOMING_TA_BINS,
            n_rays=n_rays,
        )

        target_logits = policy.target_logits_for_origin_fraction(
            out["planet_hidden"],
            torch.tensor([o_idx], device=device, dtype=torch.long),
            torch.tensor([frac_idx], device=device, dtype=torch.long),
            torch.tensor([math.floor(float(FRACTIONS[frac_idx]) * float(planets[o_idx, 5]))], device=device, dtype=torch.float32),
            torch.from_numpy(ray_hit_tick[None, :]).to(device=device, dtype=torch.float32),
            torch.from_numpy(planets[None, :, 5]).to(device=device, dtype=torch.float32),
        )[0]
        target_mask = out["pair_mask"][0, o_idx].clone()
        ray_valid_t = torch.from_numpy(ray_valid).to(device=device, dtype=torch.bool)
        target_mask &= ray_valid_t
        if not bool(target_mask.any().item()):
            break
        masked_target = target_logits.masked_fill(~target_mask, -1e4)
        if greedy:
            d_idx = int(torch.argmax(masked_target).item())
        else:
            target_probs = torch.softmax(masked_target, dim=-1)
            d_idx = int(torch.multinomial(target_probs, 1, generator=rng).item())

        ships_avail = float(planets[o_idx, 5])
        send = max(1, math.floor(float(FRACTIONS[frac_idx]) * ships_avail))
        send = min(send, int(ships_avail))
        if send <= 0:
            break
        angle = float(ray_angle[d_idx])
        actions.append([float(planets[o_idx, 0]), float(angle), int(send)])
        planets[o_idx, 5] -= float(send)
        env_target = int(true_planet[d_idx])
        if 0 <= env_target < MAX_PLANETS:
            ta = int(math.floor(max(float(true_hit_tick[d_idx]) - 1.0, 0.0)))
            ta = max(0, min(ta, incoming_fleets.shape[2] - 1))
            owner = max(0, min(int(ego_player), incoming_fleets.shape[0] - 1))
            cur = int(incoming_fleets[owner, env_target, ta])
            incoming_fleets[owner, env_target, ta] = min(cur + int(send), 65535)
        if not planet_active[o_idx] or planets[o_idx, 5] < 1.0:
            continue

    return actions


def observation_to_state(
    obs: Mapping[str, Any],
    config: Any = None,
    *,
    max_fleets: int = 512,
    step_count_override: Optional[int] = None,
) -> OrbitWarsState:
    """Convert an official Kaggle observation dict into a padded ``OrbitWarsState``.

    The policy only needs the current public state.  Fields absent from the
    official observation, such as rollout-only fleet ETA metadata, are filled
    with neutral defaults.
    """

    planets_in = _as_array(obs.get("planets", []), 7)
    planets, planet_active, id_to_slot = _place_rows_by_id(planets_in, 7)

    initial_in = _as_array(obs.get("initial_planets", []), 7)
    initial_planets = np.zeros((MAX_PLANETS, 7), dtype=np.float32)
    initial_active = np.zeros((MAX_PLANETS,), dtype=np.bool_)
    for row in initial_in[:MAX_PLANETS]:
        pid = int(row[0])
        slot = id_to_slot.get(pid, pid if 0 <= pid < MAX_PLANETS else -1)
        if 0 <= slot < MAX_PLANETS:
            initial_planets[slot, :7] = row[:7]
            initial_active[slot] = True

    # Comets can be present in planets without being present in initial_planets.
    missing_initial = planet_active & ~initial_active
    initial_planets[missing_initial] = planets[missing_initial]
    initial_active[missing_initial] = True

    fleets_in = _as_array(obs.get("fleets", []), 7)
    max_fleets = max(int(max_fleets), len(fleets_in) + DEFAULT_MAX_ACTIONS)
    fleets = np.zeros((max_fleets, FLEET_ROW_WIDTH), dtype=np.float32)
    fleet_active = np.zeros((max_fleets,), dtype=np.bool_)
    for i, row in enumerate(fleets_in[:max_fleets]):
        fleets[i, FLEET_ID] = row[0]
        fleets[i, FLEET_OWNER] = row[1]
        fleets[i, FLEET_X] = row[2]
        fleets[i, FLEET_Y] = row[3]
        fleets[i, FLEET_ANGLE] = row[4]
        fleets[i, FLEET_FROM_PLANET] = row[5]
        fleets[i, FLEET_SHIPS] = row[6]
        fleet_active[i] = True

    num_agents = int(_cfg_get(config, "agentCount", 2))
    num_agents = max(num_agents, int(obs.get("player", 0)) + 1, 2)

    comet_paths = np.zeros((MAX_COMET_GROUPS, 4, MAX_COMET_PATH, 2), dtype=np.float32)
    comet_path_lengths = np.zeros((MAX_COMET_GROUPS, 4), dtype=np.int32)
    comet_ships = np.zeros((MAX_COMET_GROUPS,), dtype=np.float32)
    comet_group_active = np.zeros((MAX_COMET_GROUPS,), dtype=np.bool_)
    comet_path_index = np.full((MAX_COMET_GROUPS,), -1, dtype=np.int32)
    comet_planet_ids = np.full((MAX_COMET_GROUPS, 4), -1, dtype=np.int32)
    comet_slots = np.full((MAX_COMET_GROUPS, 4), -1, dtype=np.int32)

    for g, comet in enumerate((obs.get("comets") or [])[:MAX_COMET_GROUPS]):
        ids = list(comet.get("planet_ids", []))[:4]
        paths = list(comet.get("paths", []))[:4]
        comet_group_active[g] = True
        comet_path_index[g] = int(comet.get("path_index", -1))
        for k, pid_raw in enumerate(ids):
            pid = int(pid_raw)
            comet_planet_ids[g, k] = pid
            comet_slots[g, k] = id_to_slot.get(pid, -1)
        for k, path in enumerate(paths):
            p = np.asarray(path, dtype=np.float32).reshape((-1, 2))
            n = min(len(p), MAX_COMET_PATH)
            comet_paths[g, k, :n] = p[:n]
            comet_path_lengths[g, k] = n
        slot0 = comet_slots[g, 0]
        if 0 <= slot0 < MAX_PLANETS:
            comet_ships[g] = planets[slot0, 5]

    angular_velocity = float(obs.get("angular_velocity", 0.0))
    step_count = int(step_count_override if step_count_override is not None else obs.get("step", obs.get("step_count", 0)))
    incoming_fleets = _forecast_incoming_fleets(
        planets,
        planet_active,
        initial_planets,
        initial_active,
        fleets_in,
        comet_paths,
        comet_path_lengths,
        comet_group_active,
        comet_path_index,
        comet_slots,
        num_agents,
        step_count,
        angular_velocity,
        ship_speed=float(_cfg_get(config, "shipSpeed", 6.0)),
        horizon=INCOMING_TA_BINS,
    )
    rewards = np.zeros((max(num_agents, 4),), dtype=np.float32)

    return OrbitWarsState(
        planets=planets,
        planet_active=planet_active,
        initial_planets=initial_planets,
        initial_active=initial_active,
        fleets=fleets,
        fleet_active=fleet_active,
        incoming_fleets=incoming_fleets,
        comet_paths=comet_paths,
        comet_path_lengths=comet_path_lengths,
        comet_ships=comet_ships,
        comet_group_active=comet_group_active,
        comet_path_index=comet_path_index,
        comet_planet_ids=comet_planet_ids,
        comet_slots=comet_slots,
        next_fleet_id=np.asarray(len(fleets_in), dtype=np.int32),
        angular_velocity=np.asarray(angular_velocity, dtype=np.float32),
        step_count=np.asarray(step_count, dtype=np.int32),
        num_agents=np.asarray(num_agents, dtype=np.int32),
        rewards=rewards,
        done=np.asarray(False),
        overflow=np.asarray(False),
    )


def _infer_policy_kwargs(payload: Any) -> dict[str, Any]:
    training_args = payload.get("training_args", {}) if isinstance(payload, Mapping) else {}
    policy_state = payload.get("policy", payload) if isinstance(payload, Mapping) else payload
    kwargs = {
        "d_model": int(training_args.get("d_model", 384)),
        "n_heads": int(training_args.get("n_heads", 8)),
        "n_layers": int(training_args.get("n_layers", 4)),
        "activation_checkpointing": False,
    }
    if isinstance(policy_state, Mapping):
        w = policy_state.get("feat_proj.weight")
        if hasattr(w, "shape"):
            kwargs["d_model"] = int(w.shape[0])
        layer_ids = []
        for key in policy_state:
            if key.startswith("blocks."):
                try:
                    layer_ids.append(int(key.split(".")[1]))
                except (IndexError, ValueError):
                    pass
        if layer_ids:
            kwargs["n_layers"] = max(layer_ids) + 1
    return kwargs


def _checkpoint_training_args(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        training_args = payload.get("training_args", {})
        if isinstance(training_args, Mapping):
            return training_args
    return {}


def load_policy(
    checkpoint_path: str | os.PathLike[str] = DEFAULT_CHECKPOINT,
    *,
    device: Optional[str | torch.device] = None,
) -> tuple[OrbitWarsPolicy, torch.device, Mapping[str, Any]]:
    """Load a training checkpoint or raw policy state dict for inference."""

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    policy_state = payload.get("policy", payload) if isinstance(payload, Mapping) else payload
    policy = OrbitWarsPolicy(**_infer_policy_kwargs(payload)).to(torch_device)
    policy.load_state_dict(policy_state)
    policy.eval()
    return policy, torch_device, _checkpoint_training_args(payload)


class KaggleOrbitWarsAgent:
    """Callable adapter object suitable for Kaggle's ``agent(obs, config)`` API."""

    def __init__(
        self,
        checkpoint_path: str | os.PathLike[str] = DEFAULT_CHECKPOINT,
        *,
        device: Optional[str | torch.device] = None,
        greedy: bool = False,
        max_micro_steps: Optional[int] = None,
        max_fleets: int = 512,
        seed: Optional[int] = None,
        raycast_rays: Optional[int] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.policy, self.device, training_args = load_policy(self.checkpoint_path, device=device)
        self.greedy = bool(greedy)
        self.max_micro_steps = int(
            max_micro_steps
            if max_micro_steps is not None
            else training_args.get("max_micro_steps", DEFAULT_MAX_ACTIONS)
        )
        self.max_fleets = int(max_fleets)
        self.raycast_rays = int(
            raycast_rays
            if raycast_rays is not None
            else training_args.get("first_hit_n_rays", DEFAULT_RAYCAST_RAYS)
        )
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(int(seed if seed is not None else os.environ.get("ORBIT_WARS_AGENT_SEED", "0")))
        self._game_key: Optional[str] = None
        self._next_step_count = 0

    def _obs_game_key(self, obs: Mapping[str, Any]) -> str:
        initial = np.asarray(obs.get("initial_planets", obs.get("planets", [])), dtype=np.float32)
        h = hashlib.blake2b(digest_size=16)
        h.update(initial.tobytes())
        h.update(str(obs.get("angular_velocity", 0.0)).encode("ascii", errors="ignore"))
        h.update(str(obs.get("player", 0)).encode("ascii", errors="ignore"))
        return h.hexdigest()

    def _step_count_for_obs(self, obs: Mapping[str, Any]) -> int:
        step_raw = obs.get("step", obs.get("step_count", None))
        if step_raw is not None:
            self._next_step_count = int(step_raw) + 1
            self._game_key = self._obs_game_key(obs)
            return int(step_raw)

        key = self._obs_game_key(obs)
        if key != self._game_key:
            self._game_key = key
            self._next_step_count = 0
        step_count = self._next_step_count
        self._next_step_count += 1
        return step_count

    @torch.inference_mode()
    def __call__(self, obs: Mapping[str, Any], config: Any = None) -> list[list[float]]:
        state = observation_to_state(
            obs,
            config,
            max_fleets=self.max_fleets,
            step_count_override=self._step_count_for_obs(obs),
        )
        ship_speed = float(_cfg_get(config, "shipSpeed", 6.0))
        return _build_turn_actions_torch_only(
            self.policy,
            state,
            int(obs.get("player", 0)),
            self.device,
            ship_speed=ship_speed,
            max_micro_steps=self.max_micro_steps,
            greedy=self.greedy,
            rng=self.rng,
            n_rays=self.raycast_rays,
        )


_AGENT: Optional[KaggleOrbitWarsAgent] = None
_ERROR_REPORTED = False


def _report_once(exc: BaseException) -> None:
    global _ERROR_REPORTED
    if not _ERROR_REPORTED:
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        _ERROR_REPORTED = True


def agent(obs: Mapping[str, Any], config: Any = None) -> list[list[float]]:
    """Kaggle entry point.

    Set ``ORBIT_WARS_CHECKPOINT`` to choose a checkpoint path.  By default the
    adapter looks for ``checkpoint.pt`` next to the submission's working dir.
    """

    global _AGENT
    if _AGENT is None:
        ckpt = os.environ.get("ORBIT_WARS_CHECKPOINT", DEFAULT_CHECKPOINT)
        device = os.environ.get("ORBIT_WARS_DEVICE")
        greedy = os.environ.get("ORBIT_WARS_GREEDY", "0").lower() in {"1", "true", "yes", "on"}
        seed_raw = os.environ.get("ORBIT_WARS_AGENT_SEED")
        seed = int(seed_raw) if seed_raw is not None else None
        rays_raw = os.environ.get("ORBIT_WARS_RAYCAST_RAYS")
        rays = int(rays_raw) if rays_raw is not None else None
        max_micro_raw = os.environ.get("ORBIT_WARS_MAX_MICRO_STEPS")
        max_micro_steps = int(max_micro_raw) if max_micro_raw is not None else None
        try:
            _AGENT = KaggleOrbitWarsAgent(
                ckpt,
                device=device,
                greedy=greedy,
                max_micro_steps=max_micro_steps,
                seed=seed,
                raycast_rays=rays,
            )
        except Exception as exc:
            _report_once(exc)
            return []
    try:
        return _AGENT(obs, config)
    except Exception as exc:
        _report_once(exc)
        return []

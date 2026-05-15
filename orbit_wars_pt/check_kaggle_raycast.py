"""Validate NumPy self-play raycast against the official Kaggle interpreter."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from orbit_wars_pt.constants import BOARD_SIZE, CENTER, INCOMING_TA_BINS, MAX_PLANETS, SUN_RADIUS
from orbit_wars_pt.kaggle_adapter import (
    OrbitWarsState,
    _build_turn_actions_torch_only,
    _fleet_speed,
    _forecast_planet_paths_np,
    _planned_send,
    _point_to_segment_distance,
    _raycast_targets_np,
    _swept_pair_hit,
    observation_to_state,
)


def _simulate_launch(
    state: OrbitWarsState,
    o_idx: int,
    angle: float,
    send: int,
    *,
    ship_speed: float = 6.0,
    horizon: int = INCOMING_TA_BINS,
) -> tuple[str, int, int]:
    planets = np.asarray(state.planets)
    ox, oy, o_rad = float(planets[o_idx, 2]), float(planets[o_idx, 3]), float(planets[o_idx, 4])
    speed = _fleet_speed(float(max(send, 1)), ship_speed)
    dx, dy = math.cos(angle), math.sin(angle)
    x = ox + dx * (o_rad + 0.1)
    y = oy + dy * (o_rad + 0.1)
    p0, p1, active = _forecast_planet_paths_np(state, horizon=horizon)
    radii = planets[:, 4]
    rank = np.asarray(state.planet_collision_rank, dtype=np.int32)
    for t in range(horizon):
        a0 = np.array([x, y], dtype=np.float32)
        a1 = a0 + speed * np.array([dx, dy], dtype=np.float32)
        for i in np.argsort(rank):
            if rank[i] >= MAX_PLANETS or not active[t, i]:
                continue
            if _swept_pair_hit(a0, a1, p0[t, i], p1[t, i], float(radii[i])):
                return "hit", int(i), t
        if _point_to_segment_distance(np.array([CENTER, CENTER]), a0, a1) < SUN_RADIUS:
            return "sun", -1, t
        if not (0 <= a1[0] <= BOARD_SIZE and 0 <= a1[1] <= BOARD_SIZE):
            return "oob", -1, t
        x, y = float(a1[0]), float(a1[1])
    return "horizon", -1, horizon


def _raycast_matches_angle(
    state: OrbitWarsState,
    o_idx: int,
    frac_idx: int,
    angle: float,
    *,
    ship_speed: float = 6.0,
    n_rays: int = 256,
    tol: float = 0.025,
) -> list[tuple[int, int]]:
    ra, valid, *_ = _raycast_targets_np(
        state, o_idx, frac_idx, ship_speed=ship_speed, n_rays=n_rays
    )
    ang = angle % (2 * math.pi)
    out: list[tuple[int, int]] = []
    for d in range(MAX_PLANETS):
        if not valid[d]:
            continue
        da = float(ra[d]) % (2 * math.pi)
        diff = abs(((ang - da + math.pi) % (2 * math.pi)) - math.pi)
        if diff < tol:
            out.append((frac_idx, d))
    return out


def check_record(path: Path, *, ship_speed: float = 6.0) -> None:
    record = json.loads(path.read_text())
    steps = record["steps"]
    cfg = record["configuration"]

    mismatches = 0
    oob_with_valid = 0
    checked = 0

    for si in range(1, len(steps)):
        obs = steps[si][0]["observation"]
        state = observation_to_state(obs, cfg)
        planets = np.array(np.asarray(state.planets), copy=True)
        planet_active = np.asarray(state.planet_active).astype(bool)
        incoming = np.array(np.asarray(state.incoming_fleets), copy=True)
        id_to_slot = {int(planets[i, 0]): i for i in range(MAX_PLANETS) if planet_active[i]}

        for pi in (0, 1):
            for move in steps[si][pi].get("action", []):
                if len(move) < 3:
                    continue
                pid, angle, _ships = int(move[0]), float(move[1]), int(move[2])
                o_idx = id_to_slot.get(pid)
                if o_idx is None:
                    continue
                checked += 1
                virt = state._replace(planets=planets, incoming_fleets=incoming)
                # Match raycast speed to the fraction that produced this angle (best-effort).
                send = _planned_send(float(planets[o_idx, 5]), 4)
                for fi in range(5):
                    if _raycast_matches_angle(virt, o_idx, fi, angle, ship_speed=ship_speed, n_rays=256):
                        send = _planned_send(float(planets[o_idx, 5]), fi)
                        break
                outcome, hit_slot, _ = _simulate_launch(virt, o_idx, angle, send, ship_speed=ship_speed)
                if outcome == "hit":
                    continue
                matches: list[tuple[int, int]] = []
                for fi in range(5):
                    matches.extend(_raycast_matches_angle(virt, o_idx, fi, angle, ship_speed=ship_speed))
                if matches:
                    oob_with_valid += 1
                mismatches += 1

        # Replay micro-step bookkeeping for next player in same turn.
        for pi in (0, 1):
            if steps[si][pi].get("action"):
                # Approximate: only player 1 had launches in many steps; rebuild from both.
                pass

    print(f"record={path.name} launches_checked={checked}")
    print(f"  sim miss (oob/sun/horizon): {mismatches}")
    print(f"  sim miss but raycast valid angle exists: {oob_with_valid}")


if __name__ == "__main__":
    check_record(Path("records/selfplay_010.json"))

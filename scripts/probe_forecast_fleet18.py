#!/usr/bin/env python3
"""Probe forecast vs geometry vs game record for fleet 18 (selfplay_tangent step 9)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from orbit_wars_pt.kaggle_adapter import (
    MAX_PLANETS,
    PLANET_RADIUS,
    PLANET_X,
    PLANET_Y,
    _first_hit_targets_np,
    _fleet_speed,
    _forecast_incoming_fleets,
    _next_planet_positions,
    _simulate_discrete_ray_policy_hits_np,
    _swept_pair_hit,
    observation_to_state,
)

RECORD = Path("records/selfplay_tangent.json")
FLEET_ID = 18
FROM_PID = 15
LAUNCH_STEP = 8
OBSERVE_STEP = 9
ANGLE = 1.8679654598236084
SHIPS = 25


def obs_at(record: dict, step_idx: int) -> dict:
    return record["steps"][step_idx][0]["observation"]


def pid_to_slot(obs: dict) -> dict[int, int]:
    return {int(r[0]): i for i, r in enumerate(obs.get("planets", []))}


def planet_xy(obs: dict, pid: int) -> tuple[float, float] | None:
    for r in obs.get("planets", []):
        if int(r[0]) == pid:
            return float(r[2]), float(r[3])
    return None


def simulate_fleet_hits(
    *,
    start_pos: np.ndarray,
    start_step: int,
    angle: float,
    ships: float,
    ship_speed: float,
    st,
    rank: np.ndarray,
    horizon: int = 20,
) -> list[dict]:
    ux, uy = math.cos(angle), math.sin(angle)
    spd = _fleet_speed(ships, ship_speed)
    pos = start_pos.astype(np.float32).copy()
    p = np.asarray(st.planets).copy()
    pa = np.asarray(st.planet_active).copy()
    ia = np.asarray(st.initial_active).copy()
    cpi = np.asarray(st.comet_path_index).copy()
    rows: list[dict] = []
    for t in range(horizon):
        old_pos, new_pos, coll_en, cpi_next, pa_next, ia_next = _next_planet_positions(
            p,
            pa,
            np.asarray(st.initial_planets),
            ia,
            np.asarray(st.comet_paths),
            np.asarray(st.comet_path_lengths),
            np.asarray(st.comet_group_active),
            cpi,
            np.asarray(st.comet_slots),
            float(st.angular_velocity),
            start_step + t,
        )
        a1 = pos + spd * np.array([ux, uy], dtype=np.float32)
        hits: list[tuple[int, float]] = []
        for i in np.argsort(rank):
            if int(rank[i]) >= MAX_PLANETS or not coll_en[i]:
                continue
            if _swept_pair_hit(pos, a1, old_pos[i], new_pos[i], float(p[i, PLANET_RADIUS])):
                hits.append((int(i), float(p[i, 0])))
        rows.append(
            {
                "t_forecast": t,
                "game_step": start_step + t,
                "fleet_a0": pos.copy(),
                "fleet_a1": a1.copy(),
                "first_hit": hits[0] if hits else None,
                "all_hits": hits,
                "p13_old": old_pos.copy(),
                "p13_new": new_pos.copy(),
                "p17_old": old_pos.copy(),
                "p17_new": new_pos.copy(),
            }
        )
        if hits:
            break
        pos = a1
        p[:, PLANET_X : PLANET_Y + 1] = new_pos
        pa = pa_next
        ia = ia_next
        cpi = cpi_next
    return rows


def main() -> None:
    record = json.loads(RECORD.read_text())
    cfg = record["configuration"]
    ship_speed = float(cfg["shipSpeed"])

    obs8 = obs_at(record, LAUNCH_STEP)
    obs9 = obs_at(record, OBSERVE_STEP)
    st8 = observation_to_state(obs8, cfg, step_count_override=LAUNCH_STEP)
    st9 = observation_to_state(obs9, cfg, step_count_override=OBSERVE_STEP)
    rank = np.asarray(st9.planet_collision_rank, dtype=np.int32)
    slots9 = pid_to_slot(obs9)
    s15 = slots9[FROM_PID]
    s13 = slots9.get(13, -1)
    s17 = slots9.get(17, -1)

    f18 = next(f for f in obs9["fleets"] if int(f[0]) == FLEET_ID)
    print("=== Fleet 18 @ observe step 9 ===")
    print(f"  record pos=({f18[2]:.6f}, {f18[3]:.6f}) angle={f18[4]:.6f} ships={f18[6]:.0f}")

    ox8 = st8.planets[s15, PLANET_X : PLANET_Y + 1]
    r15 = float(st8.planets[s15, PLANET_RADIUS])
    ux, uy = math.cos(ANGLE), math.sin(ANGLE)
    rim8 = ox8 + (r15 + 0.1) * np.array([ux, uy], dtype=np.float32)
    spd = _fleet_speed(SHIPS, ship_speed)
    print(f"\n=== Launch geometry @ step {LAUNCH_STEP} ===")
    print(f"  planet 15 center@8: ({ox8[0]:.4f}, {ox8[1]:.4f}) r={r15:.3f}")
    print(f"  launch rim: ({rim8[0]:.4f}, {rim8[1]:.4f})")
    print(f"  fleet@9 minus rim@8: dist={np.linalg.norm(np.array([f18[2], f18[3]]) - rim8):.4f} (speed={spd:.4f})")

    ang, valid, ht, tp, tht = _first_hit_targets_np(
        st8, s15, 0, ship_speed=ship_speed, n_rays=256, target_method="interval"
    )
    print("  interval targets from launch:")
    for s in sorted(np.flatnonzero(valid)):
        print(
            f"    slot {s} pid={st8.planets[s, 0]:.0f} "
            f"angle={ang[s]:.4f} hit_tick={ht[s]:.0f} true={tp[s]}"
        )

    arrivals = np.full((1, 2), -1, dtype=np.int32)
    _forecast_incoming_fleets(
        np.asarray(st9.planets),
        np.asarray(st9.planet_active),
        np.asarray(st9.initial_planets),
        np.asarray(st9.initial_active),
        np.array([f18], dtype=np.float32),
        np.asarray(st9.comet_paths),
        np.asarray(st9.comet_path_lengths),
        np.asarray(st9.comet_group_active),
        np.asarray(st9.comet_path_index),
        np.asarray(st9.comet_slots),
        int(st9.num_agents),
        OBSERVE_STEP,
        float(st9.angular_velocity),
        rank,
        ship_speed=ship_speed,
        horizon=24,
        per_fleet_arrival=arrivals,
    )
    fc_slot, fc_tick = int(arrivals[0, 0]), int(arrivals[0, 1])
    print(f"\n=== _forecast_incoming_fleets from step-9 fleet pos ===")
    print(f"  first hit: slot={fc_slot} pid={st9.planets[fc_slot, 0]:.0f} tick={fc_tick}")

    print("\n=== Planet position: forecast path vs game record (pid 13 & 17) ===")
    p = np.asarray(st9.planets).copy()
    pa = np.asarray(st9.planet_active).copy()
    ia = np.asarray(st9.initial_active).copy()
    cpi = np.asarray(st9.comet_path_index).copy()
    for t in range(20):
        gs = OBSERVE_STEP + t
        old_pos, new_pos, coll_en, cpi, pa, ia = _next_planet_positions(
            p,
            pa,
            np.asarray(st9.initial_planets),
            ia,
            np.asarray(st9.comet_paths),
            np.asarray(st9.comet_path_lengths),
            np.asarray(st9.comet_group_active),
            cpi,
            np.asarray(st9.comet_slots),
            float(st9.angular_velocity),
            gs,
        )
        if gs >= len(record["steps"]):
            break
        obs_g = obs_at(record, gs)
        for pid, slot in [(13, s13), (17, s17)]:
            if slot < 0:
                continue
            rec = planet_xy(obs_g, pid)
            fc_old = (float(old_pos[slot, 0]), float(old_pos[slot, 1]))
            fc_new = (float(new_pos[slot, 0]), float(new_pos[slot, 1]))
            if rec is None:
                continue
            d_old = math.hypot(fc_old[0] - rec[0], fc_old[1] - rec[1])
            d_new = math.hypot(fc_new[0] - rec[0], fc_new[1] - rec[1])
            if d_old > 0.05 or d_new > 0.05 or t < 3:
                print(
                    f"  game {gs} pid {pid}: "
                    f"fc_old drift={d_old:.4f} fc_new drift={d_new:.4f} "
                    f"coll_en={bool(coll_en[slot])}"
                )
        p[:, PLANET_X : PLANET_Y + 1] = new_pos
        pa, ia, cpi = pa, ia, cpi

    print("\n=== Fleet hits: from launch rim@8 ===")
    for row in simulate_fleet_hits(
        start_pos=rim8,
        start_step=LAUNCH_STEP,
        angle=ANGLE,
        ships=SHIPS,
        ship_speed=ship_speed,
        st=st8,
        rank=rank,
    ):
        hit = row["first_hit"]
        htxt = f"slot={hit[0]} pid={hit[1]:.0f}" if hit else "none"
        print(f"  forecast t={row['t_forecast']} game {row['game_step']}: {htxt}")

    print("\n=== Fleet hits: from record fleet pos@9 ===")
    for row in simulate_fleet_hits(
        start_pos=np.array([f18[2], f18[3]], dtype=np.float32),
        start_step=OBSERVE_STEP,
        angle=ANGLE,
        ships=SHIPS,
        ship_speed=ship_speed,
        st=st9,
        rank=rank,
    ):
        hit = row["first_hit"]
        htxt = f"slot={hit[0]} pid={hit[1]:.0f}" if hit else "none"
        print(f"  forecast t={row['t_forecast']} game {row['game_step']}: {htxt}")

    print("\n=== Fleet hits: game record positions stepped (ground truth replay) ===")
    pos = rim8.copy()
    for t in range(20):
        gs = LAUNCH_STEP + t
        if gs > OBSERVE_STEP + 5:
            break
        obs_g = obs_at(record, gs) if gs < len(record["steps"]) else None
        if gs == LAUNCH_STEP:
            a0 = rim8
        elif obs_g:
            # find fleet in record at this step
            ff = next((f for f in obs_g.get("fleets", []) if int(f[0]) == FLEET_ID), None)
            if ff is None:
                print(f"  game {gs}: fleet {FLEET_ID} gone (after hit?)")
                break
            a0 = np.array([ff[2], ff[3]], dtype=np.float32)
            pos = a0
        a1 = pos + spd * np.array([ux, uy], dtype=np.float32)
        if obs_g and gs + 1 < len(record["steps"]):
            ff1 = next(
                (f for f in obs_at(record, gs + 1).get("fleets", []) if int(f[0]) == FLEET_ID),
                None,
            )
            if ff1:
                rec_a1 = np.array([ff1[2], ff1[3]], dtype=np.float32)
                move_err = np.linalg.norm(rec_a1 - a1)
                if move_err > 0.1 or gs <= OBSERVE_STEP:
                    print(
                        f"  game {gs}: fleet move forecast=({a1[0]:.2f},{a1[1]:.2f}) "
                        f"record@+1=({rec_a1[0]:.2f},{rec_a1[1]:.2f}) err={move_err:.4f}"
                    )
        if gs >= OBSERVE_STEP:
            pos = a1

    print("\n=== Fleet presence in record steps 8-15 ===")
    for si in range(8, 16):
        obs = obs_at(record, si)
        fleets = [f for f in obs.get("fleets", []) if int(f[0]) == FLEET_ID]
        if fleets:
            print(f"  step {si}: pos=({fleets[0][2]:.4f}, {fleets[0][3]:.4f})")
        else:
            print(f"  step {si}: absent")


if __name__ == "__main__":
    main()

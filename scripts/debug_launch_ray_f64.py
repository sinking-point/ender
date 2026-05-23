"""Float64 replay of a single launch ray vs Kaggle ground truth."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# noqa: E402 — repo root on path when run as script
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from orbit_wars_pt.constants import FRACTIONS, INCOMING_TA_BINS
from orbit_wars_pt.kaggle_adapter import (
    PLANET_RADIUS,
    PLANET_X,
    PLANET_Y,
    _discrete_first_hit_at_angle_np,
    _fleet_speed,
    _forecast_planet_paths_with_geometry_np,
    _planned_send,
    _raycast_targets_np,
    observation_to_state,
)
from scripts.replay_consistency_dump import _load_trajectory_from_dump


def _fmt(x: float) -> str:
    return f"{float(x):.17g}"


def _replay_to_turn(seed: int, num_agents: int, target_turn: int, actions_by_turn: list) -> object:
    from kaggle_environments import make

    env = make("orbit_wars", configuration={"maxTurns": 500}, debug=False)
    env.reset(num_agents)
    for t, acts in enumerate(actions_by_turn):
        if t >= target_turn:
            break
        env.step(acts)
    return env


def main() -> int:
    dump = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "experiments/4p-007/consistency_mismatches/iter_001380_env_0171.npz"
    )
    traj = _load_trajectory_from_dump(dump)

    # Rebuild actions through turn 66 from dump JSON is heavy; replay worker turns from npz-only
    # by running the same consistency walk but stopping before turn 67 step.
    import json

    json_path = dump.with_suffix(".json")
    turns_data = json.loads(json_path.read_text())["turns"]
    actions_by_turn: list = []
    for td in turns_data:
        if int(td["turn_index"]) >= 67:
            break
        actions_by_turn.append(td.get("final_actions") or [[] for _ in range(traj.num_agents)])

    env = _replay_to_turn(traj.seed, traj.num_agents, 67, actions_by_turn)
    turn_index = 67
    origin_slot = 5
    target_slots = (18, 19)
    angle_resolved = 6.135923151542564
    angle_frac3 = 6.160466844148735
    ships = 108
    frac_idx = 3

    print(f"seed={traj.seed} turn={turn_index} origin_slot={origin_slot} ships={ships} frac_idx={frac_idx}")
    print(f"angles: resolved={_fmt(angle_resolved)} frac3={_fmt(angle_frac3)}")

    kobs = env.state[1].observation
    state = observation_to_state(
        kobs,
        config={"agentCount": int(traj.num_agents)},
        step_count_override=turn_index,
    )
    planets = np.asarray(state.planets, dtype=np.float64)
    origin_xy = planets[origin_slot, PLANET_X : PLANET_Y + 1].copy()
    origin_r = float(planets[origin_slot, PLANET_RADIUS])
    ships_avail = float(planets[origin_slot, 5])
    send_frac = _planned_send(ships_avail, frac_idx)
    speed_actual = _fleet_speed(float(ships), traj.ship_speed)
    speed_frac = _fleet_speed(float(max(send_frac, 1)), traj.ship_speed)

    print(f"origin_xy=({_fmt(origin_xy[0])}, {_fmt(origin_xy[1])}) origin_r={_fmt(origin_r)}")
    print(f"ships_avail={ships_avail} planned_send(frac3)={send_frac}")
    print(f"speed(actual {ships})={_fmt(speed_actual)} speed(frac send)={_fmt(speed_frac)}")

    p0, p1, active = _forecast_planet_paths_with_geometry_np(state, None, horizon=INCOMING_TA_BINS)
    radii = planets[:, PLANET_RADIUS].astype(np.float64)
    collision_rank = np.asarray(state.planet_collision_rank, dtype=np.int32)

    for label, angle, speed in (
        ("resolved_angle+actual_ships_speed", angle_resolved, speed_actual),
        ("frac3_angle+frac_planned_speed", angle_frac3, speed_frac),
        ("frac3_angle+actual_ships_speed", angle_frac3, speed_actual),
        ("resolved_angle+frac_planned_speed", angle_resolved, speed_frac),
    ):
        kind, code, tick = _discrete_first_hit_at_angle_np(
            angle,
            origin_xy,
            origin_r,
            speed,
            p0,
            p1,
            radii,
            active,
            collision_rank,
            horizon=INCOMING_TA_BINS,
        )
        print(f"  _discrete_first_hit [{label}]: kind={kind} code={code} tick={tick}")

    out_angle, valid, hit_tick, true_planet, true_hit_tick = _raycast_targets_np(
        state, origin_slot, frac_idx, ship_speed=traj.ship_speed, n_rays=traj.n_rays
    )
    for t in target_slots:
        print(
            f"  _raycast_targets_np frac={frac_idx} target={t}: "
            f"valid={valid[t]} angle={_fmt(out_angle[t])} hit_tick={hit_tick[t]} "
            f"true_planet={true_planet[t]} true_hit_tick={true_hit_tick[t]}"
        )

    theta = angle_resolved % (2.0 * math.pi)
    direction = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
    launch_pos = origin_xy + (origin_r + 0.1) * direction
    print(f"\nlaunch_pos (resolved angle) = ({_fmt(launch_pos[0])}, {_fmt(launch_pos[1])})")

    print("\nPer-tick planet 18/19 (p0 -> p1) and fleet segment (resolved, actual speed):")
    pos = launch_pos.copy()
    for tick in range(min(20, INCOMING_TA_BINS)):
        a0 = pos.copy()
        a1 = pos + speed_actual * direction
        parts = [f"tick={tick}"]
        for slot in target_slots:
            if not active[tick, slot]:
                parts.append(f"p{slot}=inactive")
                continue
            p0s = p0[tick, slot]
            p1s = p1[tick, slot]
            parts.append(
                f"p{slot}_p0=({_fmt(p0s[0])},{_fmt(p0s[1])}) "
                f"p1=({_fmt(p1s[0])},{_fmt(p1s[1])}) r={_fmt(radii[slot])}"
            )
        parts.append(f"fleet_a0=({_fmt(a0[0])},{_fmt(a0[1])}) a1=({_fmt(a1[0])},{_fmt(a1[1])})")
        print("  " + " | ".join(parts))
        pos = a1

    # Ground truth: step turn 67 with actions from dump
    acts67 = turns_data[67]["final_actions"]
    print(f"\nKaggle step turn 67 actions seat1: {acts67[1]}")
    env.step(acts67)
    fleets = env.state[0].observation.get("fleets", [])
    f53 = next((f for f in fleets if int(f.get("id", -1)) == 53), None)
    if f53:
        print("Fleet 53 after launch (Kaggle):")
        for k in ("x", "y", "angle", "ships", "from_planet", "owner"):
            print(f"  {k}={f53.get(k)}")
    else:
        print("Fleet 53 not found; fleet ids:", [int(f.get("id", -1)) for f in fleets])

    # Advance to hit (~turn 72)
    for t in range(68, 76):
        if env.done:
            break
        td = turns_data[t] if t < len(turns_data) else None
        acts = td["final_actions"] if td else [[] for _ in range(traj.num_agents)]
        env.step(acts)
        fleets = env.state[0].observation.get("fleets", [])
        f53 = next((f for f in fleets if int(f.get("id", -1)) == 53), None)
        if f53:
            print(
                f"turn {t} fleet53 pos=({_fmt(float(f53['x']))},{_fmt(float(f53['y']))}) "
                f"ships={f53.get('ships')}"
            )
        else:
            print(f"turn {t}: fleet 53 gone (hit or destroyed)")
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Ground-truth first hit: official Kaggle fleet step vs interval vs adapter raycast.

The installed interpreter (``kaggle_environments.envs.orbit_wars.orbit_wars``) moves
each fleet one tick at a time and scans ``obs0.planets`` in list order; the first
``swept_pair_hit`` wins (then sun, then board).  This script replays that loop on
forecast paths and compares to interval occlusion and ``_simulate_discrete`` raycast.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np

from orbit_wars_pt.constants import BOARD_SIZE, CENTER, SUN_RADIUS
from orbit_wars_pt.geometry import TAU, fleet_speed
from orbit_wars_pt.interval_geometry_np import (
    first_hit_at_angle_interval,
    precompute_tick_planet_hits,
)
from orbit_wars_pt.check_interval_vs_raycast import (
    first_hit_at_angle_raycast,
    _make_case,
    _swept_pair_hit,
    _point_to_segment_distance,
)
from orbit_wars_pt.first_hit_metrics import HitCompareCounts, hit_incoming_ta


def _load_kaggle_swept():
    spec = importlib.util.find_spec("kaggle_environments.envs.orbit_wars.orbit_wars")
    if spec is None or spec.origin is None:
        raise RuntimeError("kaggle_environments orbit_wars not installed")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.swept_pair_hit, mod.point_to_segment_distance


def first_hit_official_kaggle_loop(
    angle: float,
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    p0_by_tick: np.ndarray,
    p1_by_tick: np.ndarray,
    radii: np.ndarray,
    active_by_tick: np.ndarray,
    planet_slot_order: list[int],
    *,
    swept_pair_hit,
    point_to_segment_distance,
    include_sun: bool = True,
    include_board: bool = True,
) -> tuple[str, int, int]:
    """Mirror ``orbit_wars.py`` fleet movement (lines ~574–609)."""

    theta = float(angle) % TAU
    old_pos = (
        float(origin_xy[0]) + math.cos(theta) * (float(origin_radius) + 0.1),
        float(origin_xy[1]) + math.sin(theta) * (float(origin_radius) + 0.1),
    )
    ticks = int(p0_by_tick.shape[0])

    for tick in range(ticks):
        new_pos = (
            old_pos[0] + math.cos(theta) * float(speed),
            old_pos[1] + math.sin(theta) * float(speed),
        )
        a0 = np.asarray(old_pos, dtype=np.float64)
        a1 = np.asarray(new_pos, dtype=np.float64)

        hit_planet = False
        for slot in planet_slot_order:
            if slot < 0 or slot >= int(radii.shape[0]):
                continue
            if not active_by_tick[tick, slot]:
                continue
            p0 = p0_by_tick[tick, slot]
            p1 = p1_by_tick[tick, slot]
            if swept_pair_hit(a0, a1, p0, p1, float(radii[slot])):
                return ("planet", int(slot), tick)

        if include_board:
            if not (0.0 <= new_pos[0] <= BOARD_SIZE and 0.0 <= new_pos[1] <= BOARD_SIZE):
                return ("board", -1, tick)

        if include_sun:
            if point_to_segment_distance((CENTER, CENTER), tuple(a0), tuple(a1)) < float(SUN_RADIUS):
                return ("sun", -1, tick)

        old_pos = new_pos

    return ("none", -1, -1)


def first_hit_official_numpy(
    angle: float,
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    p0_by_tick: np.ndarray,
    p1_by_tick: np.ndarray,
    radii: np.ndarray,
    active_by_tick: np.ndarray,
    planet_slot_order: list[int],
) -> tuple[str, int, int]:
    return first_hit_official_kaggle_loop(
        angle,
        origin_xy,
        origin_radius,
        speed,
        p0_by_tick,
        p1_by_tick,
        radii,
        active_by_tick,
        planet_slot_order,
        swept_pair_hit=_swept_pair_hit,
        point_to_segment_distance=lambda p, a, b: _point_to_segment_distance(
            np.asarray(p), np.asarray(a), np.asarray(b)
        ),
    )


def _fleet_segment_at_tick(
    angle: float,
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    tick: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fleet segment for official tick ``tick`` (cumulative motion from launch)."""

    theta = float(angle) % TAU
    direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    launch = np.asarray(origin_xy, dtype=np.float64) + (float(origin_radius) + 0.1) * direction
    a0 = launch + float(tick) * float(speed) * direction
    a1 = a0 + float(speed) * direction
    return a0, a1


def _hits_at_tick(
    angle: float,
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    p0_by_tick: np.ndarray,
    p1_by_tick: np.ndarray,
    radii: np.ndarray,
    active_by_tick: np.ndarray,
    tick: int,
    planet_slot_order: list[int],
) -> list[tuple[int, float]]:
    a0, a1 = _fleet_segment_at_tick(angle, origin_xy, origin_radius, speed, tick)
    out: list[tuple[int, float]] = []
    for slot in planet_slot_order:
        if not active_by_tick[tick, slot]:
            continue
        d0 = a0 - p0_by_tick[tick, slot]
        dv = (a1 - a0) - (p1_by_tick[tick, slot] - p0_by_tick[tick, slot])
        qa = float(np.dot(dv, dv))
        qb = float(2.0 * np.dot(d0, dv))
        qc = float(np.dot(d0, d0) - float(radii[slot]) ** 2)
        if qa < 1e-12:
            if qc <= 0.0:
                out.append((slot, 0.0))
            continue
        disc = qb * qb - 4.0 * qa * qc
        if disc < 0.0:
            continue
        sd = math.sqrt(max(disc, 0.0))
        t1 = (-qb - sd) / (2.0 * qa)
        t2 = (-qb + sd) / (2.0 * qa)
        if t2 >= 0.0 and t1 <= 1.0:
            out.append((slot, float(np.clip(t1, 0.0, 1.0))))
    return out


def run_mismatch_audit(
    *,
    case_seed: int,
    theta: float,
    ticks: int,
    planets: int,
    samples_per_span: int,
) -> None:
    origin_xy, origin_radius, speed, p0, p1, radii, active, collision_rank = _make_case(
        case_seed, ticks=ticks, planets=planets
    )
    order = [int(i) for i in np.argsort(collision_rank)]
    pre = precompute_tick_planet_hits(
        origin_xy,
        origin_radius,
        speed,
        p0,
        p1,
        radii,
        active,
        samples_per_span=samples_per_span,
    )

    iv = first_hit_at_angle_interval(
        theta,
        pre,
        object_order=order,
        origin_xy=origin_xy,
        origin_radius=origin_radius,
        speed=speed,
        samples_per_span=samples_per_span,
    )
    rv = first_hit_at_angle_raycast(
        theta,
        origin_xy,
        origin_radius,
        speed,
        p0,
        p1,
        radii,
        active,
        collision_rank,
    )
    kv = first_hit_official_numpy(
        theta, origin_xy, origin_radius, speed, p0, p1, radii, active, order
    )
    try:
        k_swept, k_ptd = _load_kaggle_swept()
        kk = first_hit_official_kaggle_loop(
            theta,
            origin_xy,
            origin_radius,
            speed,
            p0,
            p1,
            radii,
            active,
            order,
            swept_pair_hit=k_swept,
            point_to_segment_distance=k_ptd,
        )
    except Exception as exc:
        kk = ("error", -1, -1)
        print(f"  (kaggle module loop failed: {exc})")

    print(f"case_seed={case_seed} theta={theta:.6f} speed={speed:.4f}")
    print(f"  interval (occlusion):     {iv}")
    print(f"  adapter raycast (obs order): {rv}")
    print(f"  official loop (numpy):    {kv}")
    print(f"  official loop (kaggle):   {kk}")
    if kv[0] == "planet":
        print(
            f"  incoming TA bins (floor(hit_tick-1)): "
            f"official={hit_incoming_ta(kv)} interval={hit_incoming_ta(iv)} raycast={hit_incoming_ta(rv)}"
        )

    if iv[0] == "planet" and iv[2] >= 0:
        tick = int(iv[2])
        hits = _hits_at_tick(
            theta, origin_xy, origin_radius, speed, p0, p1, radii, active, tick, order
        )
        print(f"  swept hits at tick {tick} (slot, t_enter) in obs order:")
        for slot, te in hits[:16]:
            print(f"    slot {slot:3d}  t_enter={te:.4f}")
        if len(hits) > 16:
            print(f"    ... +{len(hits) - 16} more")


def run_batch(
    *,
    cases: int,
    n_angles: int,
    ticks: int,
    planets: int,
    samples_per_span: int,
    seed: int,
) -> None:
    iv_vs_kv = HitCompareCounts()
    rv_vs_kv = HitCompareCounts()
    iv_vs_rv = HitCompareCounts()
    for case_i in range(cases):
        case_seed = seed + case_i * 9973
        origin_xy, origin_radius, speed, p0, p1, radii, active, collision_rank = _make_case(
            case_seed, ticks=ticks, planets=planets
        )
        order = [int(i) for i in np.argsort(collision_rank)]
        pre = precompute_tick_planet_hits(
            origin_xy,
            origin_radius,
            speed,
            p0,
            p1,
            radii,
            active,
            samples_per_span=samples_per_span,
        )
        angles = np.linspace(0.0, TAU, n_angles, endpoint=False)
        for theta in angles:
            iv = first_hit_at_angle_interval(
                float(theta),
                pre,
                object_order=order,
                origin_xy=origin_xy,
                origin_radius=origin_radius,
                speed=speed,
                samples_per_span=samples_per_span,
            )
            rv = first_hit_at_angle_raycast(
                float(theta),
                origin_xy,
                origin_radius,
                speed,
                p0,
                p1,
                radii,
                active,
                collision_rank,
            )
            kv = first_hit_official_numpy(
                float(theta),
                origin_xy,
                origin_radius,
                speed,
                p0,
                p1,
                radii,
                active,
                order,
            )
            iv_vs_kv.record(kv, iv)
            rv_vs_kv.record(kv, rv)
            iv_vs_rv.record(rv, iv)

    print(
        f"batch: cases={cases} angles={n_angles} ticks={ticks} planets={planets} "
        f"samples={samples_per_span}"
    )
    print("  (incoming TA = floor(max(hit_tick - 1, 0)) for planet hits)")
    for line in iv_vs_kv.format_lines("interval vs official"):
        print(line)
    for line in rv_vs_kv.format_lines("raycast vs official"):
        print(line)
    for line in iv_vs_rv.format_lines("interval vs raycast"):
        print(line)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-case", type=int, default=None, help="print one case in detail")
    p.add_argument("--theta", type=float, default=0.0)
    p.add_argument("--cases", type=int, default=12)
    p.add_argument("--angles", type=int, default=720)
    p.add_argument("--ticks", type=int, default=24)
    p.add_argument("--planets", type=int, default=40)
    p.add_argument("--samples-per-span", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.audit_case is not None:
        run_mismatch_audit(
            case_seed=args.seed + int(args.audit_case) * 9973,
            theta=float(args.theta),
            ticks=args.ticks,
            planets=args.planets,
            samples_per_span=args.samples_per_span,
        )
    else:
        run_batch(
            cases=args.cases,
            n_angles=args.angles,
            ticks=args.ticks,
            planets=args.planets,
            samples_per_span=args.samples_per_span,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

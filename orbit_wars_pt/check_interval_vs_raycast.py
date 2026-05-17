"""Per-ray first-hit agreement: interval occlusion vs discrete raycast (Kaggle sim)."""

from __future__ import annotations

import argparse

import numpy as np

from orbit_wars_pt.constants import BOARD_SIZE, CENTER, SUN_RADIUS
from orbit_wars_pt.geometry import TAU, fleet_speed
from orbit_wars_pt.first_hit_metrics import HitCompareCounts, hit_incoming_ta
from orbit_wars_pt.interval_geometry_np import (
    first_hit_at_angle_interval,
    precompute_tick_planet_hits,
)


def _swept_pair_hit(
    a0: np.ndarray,
    a1: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    radius: float,
) -> bool:
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
    sd = float(np.sqrt(max(disc, 0.0)))
    t1 = (-qb - sd) / (2.0 * qa)
    t2 = (-qb + sd) / (2.0 * qa)
    return t2 >= 0.0 and t1 <= 1.0


def _point_to_segment_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> float:
    delta = end - start
    l2 = float(np.dot(delta, delta))
    if l2 <= 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, delta) / l2, 0.0, 1.0))
    projection = start + t * delta
    return float(np.linalg.norm(point - projection))


def _first_planet_slot(hit_mask: np.ndarray, collision_rank: np.ndarray) -> int:
    """First hit in Kaggle planet list order (matches ``_first_hit_planet_index``)."""

    for slot in np.argsort(np.asarray(collision_rank, dtype=np.int32)):
        if hit_mask[slot]:
            return int(slot)
    raise RuntimeError("no planet hit in mask")


def first_hit_at_angle_raycast(
    angle: float,
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    p0_by_tick: np.ndarray,
    p1_by_tick: np.ndarray,
    radii: np.ndarray,
    active_by_tick: np.ndarray,
    collision_rank: np.ndarray,
    *,
    include_sun: bool = True,
    include_board: bool = True,
    board_size: float = BOARD_SIZE,
    sun_radius: float = SUN_RADIUS,
) -> tuple[str, int, int]:
    """Discrete per-tick raycast (``kaggle_adapter._simulate_discrete_ray_policy_hits_np`` rules)."""

    origin = np.asarray(origin_xy, dtype=np.float64)
    theta = float(angle) % TAU
    direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    launch_off = float(origin_radius) + 0.1
    pos = origin + launch_off * direction
    sun = np.asarray([CENTER, CENTER], dtype=np.float64)
    ticks = int(active_by_tick.shape[0])
    planets = int(radii.shape[0])
    rank = np.asarray(collision_rank, dtype=np.int32)

    for tick in range(ticks):
        a0 = pos.astype(np.float64)
        a1 = a0 + float(speed) * direction

        hit_raw = np.zeros((planets,), dtype=bool)
        for j in range(planets):
            if not active_by_tick[tick, j]:
                continue
            d0 = a0 - p0_by_tick[tick, j]
            dv = (a1 - a0) - (p1_by_tick[tick, j] - p0_by_tick[tick, j])
            qa = float(np.dot(dv, dv))
            qb = float(2.0 * np.dot(d0, dv))
            qc = float(np.dot(d0, d0) - float(radii[j]) ** 2)
            if qa < 1e-12:
                if qc <= 0.0:
                    hit_raw[j] = True
                continue
            disc = qb * qb - 4.0 * qa * qc
            if disc < 0.0:
                continue
            sd = float(np.sqrt(max(disc, 0.0)))
            t1 = (-qb - sd) / (2.0 * qa)
            t2 = (-qb + sd) / (2.0 * qa)
            if t2 >= 0.0 and t1 <= 1.0:
                hit_raw[j] = True

        if np.any(hit_raw):
            slot = _first_planet_slot(hit_raw, rank)
            return ("planet", slot, tick)

        if include_sun:
            if _point_to_segment_distance(sun, a0, a1) < float(sun_radius):
                return ("sun", -1, tick)

        if include_board:
            if not (
                0.0 <= a1[0] <= board_size
                and 0.0 <= a1[1] <= board_size
            ):
                return ("board", -1, tick)

        pos = a1

    return ("none", -1, -1)


def _make_case(
    seed: int,
    *,
    ticks: int,
    planets: int,
) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    origin_xy = rng.uniform(18.0, 82.0, size=(2,)).astype(np.float64)
    origin_radius = float(rng.uniform(1.5, 4.0))
    speed = float(fleet_speed(float(rng.uniform(10.0, 500.0))))
    p0 = rng.uniform(12.0, 88.0, size=(ticks, planets, 2)).astype(np.float64)
    p1 = p0 + rng.uniform(-2.5, 2.5, size=(ticks, planets, 2))
    active = rng.random((ticks, planets)) > 0.15
    radii = rng.uniform(1.2, 4.5, size=(planets,)).astype(np.float64)
    collision_rank = np.arange(planets, dtype=np.int32)
    return origin_xy, origin_radius, speed, p0, p1, radii, active, collision_rank


def run_check(
    *,
    cases: int,
    n_angles: int,
    ticks: int,
    planets: int,
    samples_per_span: int,
    seed: int,
) -> int:
    cmp = HitCompareCounts()
    examples: list[str] = []

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
        angles = np.linspace(0.0, TAU, int(n_angles), endpoint=False, dtype=np.float64)

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
            cmp.record(rv, iv)
            if iv != rv:
                if len(examples) < 12:
                    ta_i, ta_r = hit_incoming_ta(iv), hit_incoming_ta(rv)
                    examples.append(
                        f"case={case_i} theta={theta:.5f} interval={iv} raycast={rv} "
                        f"ta={ta_i}/{ta_r}"
                    )

    print(
        f"interval vs raycast first-hit: cases={cases} angles/case={n_angles} "
        f"ticks={ticks} planets={planets} samples_per_span={samples_per_span}"
    )
    print("  (incoming TA = floor(max(hit_tick - 1, 0)) for planet hits)")
    for line in cmp.format_lines("interval vs raycast"):
        print(line)
    for line in examples:
        print(f"  {line}")
    return cmp.mismatches()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=int, default=24)
    p.add_argument("--angles", type=int, default=720)
    p.add_argument("--ticks", type=int, default=24)
    p.add_argument("--planets", type=int, default=40)
    p.add_argument("--samples-per-span", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--expect-zero",
        action="store_true",
        help="exit 1 unless every probe angle agrees (strict; sampled hull may fail).",
    )
    args = p.parse_args()
    mismatches = run_check(
        cases=max(1, args.cases),
        n_angles=max(8, args.angles),
        ticks=max(1, args.ticks),
        planets=max(2, args.planets),
        samples_per_span=max(2, args.samples_per_span),
        seed=args.seed,
    )
    if args.expect_zero and mismatches:
        raise SystemExit(1)
    if mismatches == 0:
        print("ok")


if __name__ == "__main__":
    main()

"""Per-angle first-hit agreement: orthogonal interval vs discrete raycast."""

from __future__ import annotations

import argparse
import math

import numpy as np

from orbit_wars_pt.constants import CENTER, SUN_RADIUS
from orbit_wars_pt.first_hit_metrics import HitCompareCounts, hit_incoming_ta
from orbit_wars_pt.geometry import TAU, fleet_speed
from orbit_wars_pt.interval_geometry_np import (
    _interval_use_tangent_geometry,
    collect_hit_events,
    first_hit_at_angle_orthogonal,
)
from orbit_wars_pt.check_interval_vs_raycast import (
    first_hit_at_angle_raycast,
    _make_case,
)
from orbit_wars_pt.verify_first_hit_env import first_hit_official_numpy


def _make_orbiting_case(
    seed: int,
    *,
    ticks: int,
    planets: int,
    angular_velocity: float = 0.012,
) -> tuple[
    np.ndarray,
    float,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    int,
]:
    """Synthetic case with circular orbits (orthogonal + raycast share the same chords)."""

    rng = np.random.default_rng(seed)
    origin_xy = rng.uniform(22.0, 78.0, size=(2,)).astype(np.float64)
    origin_radius = float(rng.uniform(1.5, 4.0))
    speed = float(fleet_speed(float(rng.uniform(10.0, 500.0))))

    planet_tab = np.zeros((planets, 7), dtype=np.float64)
    initial_tab = np.zeros((planets, 7), dtype=np.float64)
    p0 = np.zeros((ticks, planets, 2), dtype=np.float64)
    p1 = np.zeros((ticks, planets, 2), dtype=np.float64)
    active = np.ones((ticks, planets), dtype=bool)
    radii = rng.uniform(1.2, 4.5, size=(planets,)).astype(np.float64)
    collision_rank = np.arange(planets, dtype=np.int32)

    for j in range(planets):
        orbit_r = float(rng.uniform(8.0, 42.0))
        th0 = float(rng.uniform(0.0, TAU))
        planet_tab[j, 0] = j
        planet_tab[j, 4] = radii[j]
        initial_tab[j, 0] = j
        initial_tab[j, 2] = CENTER + orbit_r * math.cos(th0)
        initial_tab[j, 3] = CENTER + orbit_r * math.sin(th0)
        initial_tab[j, 4] = radii[j]
        for t in range(ticks):
            th_a = th0 + angular_velocity * float(t)
            th_b = th0 + angular_velocity * float(t + 1)
            p0[t, j, 0] = CENTER + orbit_r * math.cos(th_a)
            p0[t, j, 1] = CENTER + orbit_r * math.sin(th_a)
            p1[t, j, 0] = CENTER + orbit_r * math.cos(th_b)
            p1[t, j, 1] = CENTER + orbit_r * math.sin(th_b)
        planet_tab[j, 2:4] = p0[0, j]

    step_count = 0
    return (
        origin_xy,
        origin_radius,
        speed,
        p0,
        p1,
        radii,
        active,
        collision_rank,
        planet_tab,
        initial_tab,
        active[0].copy(),
        angular_velocity,
        step_count,
    )


def run_check(
    *,
    cases: int,
    n_angles: int,
    ticks: int,
    planets: int,
    seed: int,
    motion: str,
) -> int:
    cmp = HitCompareCounts()
    cmp_planet = HitCompareCounts()
    examples: list[str] = []
    planet_examples: list[str] = []

    empty_paths = np.zeros((0, 4, 1, 2), dtype=np.float64)
    empty_lens = np.zeros((0, 4), dtype=np.int32)
    empty_bool = np.zeros((0,), dtype=bool)
    empty_idx = np.zeros((0,), dtype=np.int32)
    empty_slots = np.zeros((0, 4), dtype=np.int32)
    empty_pids = np.zeros((0, 4), dtype=np.int32)
    geom_label = "tangent" if _interval_use_tangent_geometry() else "orthogonal"

    for case_i in range(cases):
        case_seed = seed + case_i * 9973
        if motion == "orbit":
            (
                origin_xy,
                origin_radius,
                speed,
                p0,
                p1,
                radii,
                active,
                collision_rank,
                planet_tab,
                initial_tab,
                planet_active,
                angular_velocity,
                step_count,
            ) = _make_orbiting_case(case_seed, ticks=ticks, planets=planets)
        else:
            origin_xy, origin_radius, speed, p0, p1, radii, active, collision_rank = _make_case(
                case_seed, ticks=ticks, planets=planets
            )
            planet_tab = np.zeros((planets, 7), dtype=np.float64)
            for j in range(planets):
                planet_tab[j, 0] = j
                planet_tab[j, 2:4] = p0[0, j]
                planet_tab[j, 4] = radii[j]
            initial_tab = planet_tab.copy()
            planet_active = active[0].copy()
            angular_velocity = 0.0
            step_count = 0

        order = [int(i) for i in np.argsort(collision_rank)]
        events = collect_hit_events(
            origin_xy,
            origin_radius,
            speed,
            planet_tab,
            planet_active,
            initial_tab,
            planet_active,
            empty_paths,
            empty_lens,
            empty_bool,
            empty_idx,
            empty_slots,
            empty_pids,
            angular_velocity,
            step_count,
            horizon=float(ticks),
            include_sun=True,
            sun_radius=SUN_RADIUS,
        )
        angles = np.linspace(0.0, TAU, int(n_angles), endpoint=False, dtype=np.float64)

        for theta in angles:
            ov = first_hit_at_angle_orthogonal(
                float(theta),
                events,
                object_order=order,
                origin_xy=origin_xy,
                origin_radius=origin_radius,
                speed=speed,
                horizon=ticks,
                p0_by_tick=p0,
                p1_by_tick=p1,
                radii=radii,
                active_by_tick=active,
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
            cmp.record(rv, ov)
            if ov[0] == "planet" and rv[0] == "planet":
                cmp_planet.record(rv, ov)
            if ov != rv and len(examples) < 16:
                ta_o, ta_r = hit_incoming_ta(ov), hit_incoming_ta(rv)
                examples.append(
                    f"case={case_i} theta={theta:.5f} orth={ov} raycast={rv} ta={ta_o}/{ta_r}"
                )
            if (
                ov[0] == "planet"
                and rv[0] == "planet"
                and ov != rv
                and len(planet_examples) < 12
            ):
                planet_examples.append(
                    f"case={case_i} theta={theta:.5f} orth={ov} raycast={rv}"
                )

    print(
        f"{geom_label} vs raycast first-hit: cases={cases} angles/case={n_angles} "
        f"ticks={ticks} planets={planets} motion={motion}"
    )
    print("  (incoming TA = floor(max(hit_tick - 1, 0)) for planet hits)")
    for line in cmp.format_lines(f"{geom_label} vs raycast (all events)"):
        print(line)
    for line in cmp_planet.format_lines("both planet hits only"):
        print(line)
    for line in examples:
        print(f"  {line}")
    if planet_examples:
        print("  planet-only examples:")
        for line in planet_examples:
            print(f"    {line}")
    return cmp.mismatches()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=int, default=24)
    p.add_argument("--angles", type=int, default=720)
    p.add_argument("--ticks", type=int, default=24)
    p.add_argument("--planets", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--motion",
        choices=("orbit", "random"),
        default="orbit",
        help="orbit: circular planet paths (fair for orthogonal); random: legacy random chords.",
    )
    p.add_argument(
        "--expect-zero",
        action="store_true",
        help="exit 1 unless every probe angle agrees.",
    )
    p.add_argument(
        "--vs-official",
        action="store_true",
        help="also compare orthogonal and raycast against official swept loop (orbit motion).",
    )
    args = p.parse_args()
    if args.vs_official:
        run_official_comparison(
            cases=max(1, args.cases),
            n_angles=max(8, args.angles),
            ticks=max(1, args.ticks),
            planets=max(2, args.planets),
            seed=args.seed,
        )
    mismatches = run_check(
        cases=max(1, args.cases),
        n_angles=max(8, args.angles),
        ticks=max(1, args.ticks),
        planets=max(2, args.planets),
        seed=args.seed,
        motion=args.motion,
    )
    if args.expect_zero and mismatches:
        raise SystemExit(1)
    if mismatches == 0:
        print("ok")


def run_official_comparison(
    *,
    cases: int,
    n_angles: int,
    ticks: int,
    planets: int,
    seed: int,
) -> None:
    cmp_or = HitCompareCounts()
    cmp_rr = HitCompareCounts()

    empty_paths = np.zeros((0, 4, 1, 2), dtype=np.float64)
    empty_lens = np.zeros((0, 4), dtype=np.int32)
    empty_bool = np.zeros((0,), dtype=bool)
    empty_idx = np.zeros((0,), dtype=np.int32)
    empty_slots = np.zeros((0, 4), dtype=np.int32)
    empty_pids = np.zeros((0, 4), dtype=np.int32)

    for case_i in range(cases):
        case_seed = seed + case_i * 9973
        (
            origin_xy,
            origin_radius,
            speed,
            p0,
            p1,
            radii,
            active,
            collision_rank,
            planet_tab,
            initial_tab,
            planet_active,
            angular_velocity,
            step_count,
        ) = _make_orbiting_case(case_seed, ticks=ticks, planets=planets)
        order = [int(i) for i in np.argsort(collision_rank)]
        events = collect_hit_events(
            origin_xy,
            origin_radius,
            speed,
            planet_tab,
            planet_active,
            initial_tab,
            planet_active,
            empty_paths,
            empty_lens,
            empty_bool,
            empty_idx,
            empty_slots,
            empty_pids,
            angular_velocity,
            step_count,
            horizon=float(ticks),
        )
        angles = np.linspace(0.0, TAU, int(n_angles), endpoint=False, dtype=np.float64)
        for theta in angles:
            ov = first_hit_at_angle_orthogonal(
                float(theta),
                events,
                object_order=order,
                origin_xy=origin_xy,
                origin_radius=origin_radius,
                speed=speed,
                horizon=ticks,
                p0_by_tick=p0,
                p1_by_tick=p1,
                radii=radii,
                active_by_tick=active,
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
            cmp_or.record(kv, ov)
            cmp_rr.record(kv, rv)

    print(
        f"vs official (orbit motion): cases={cases} angles={n_angles} ticks={ticks} planets={planets}"
    )
    for line in cmp_or.format_lines("orthogonal vs official"):
        print(line)
    for line in cmp_rr.format_lines("raycast vs official"):
        print(line)


if __name__ == "__main__":
    main()

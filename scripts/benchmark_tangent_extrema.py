#!/usr/bin/env python3
"""Benchmark tangent overlap extrema: scan vs sextic exact, and full collect."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from orbit_wars_pt.check_orthogonal_vs_raycast import _make_orbiting_case
from orbit_wars_pt.interval_geometry_np import (
    LAUNCH_RIM_OFFSET,
    _rotating_planet_chord_polyline,
    collect_tangent_hit_events,
)
from orbit_wars_pt.tangent_geometry_np import (
    angular_intersection_extrema,
    angular_intersection_extrema_polyline_exact,
    intersection_windows,
    make_polyline_motion,
)


@dataclass
class BenchRow:
    name: str
    seconds: float
    calls: int

    @property
    def per_call_us(self) -> float:
        return 1e6 * self.seconds / max(self.calls, 1)


def _timeit(fn, *, repeats: int = 5, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats


def _bench_windows(case_id: int = 0, repeats: int = 8) -> list[BenchRow]:
    (
        origin_xy,
        origin_radius,
        speed,
        _p0,
        _p1,
        radii,
        _active,
        _rank,
        planet_tab,
        initial_tab,
        _pa,
        av,
        _sc,
    ) = _make_orbiting_case(case_id, ticks=24, planets=40)
    g = np.asarray(origin_xy, dtype=np.float64)
    lo = origin_radius + LAUNCH_RIM_OFFSET
    v = speed

    scan_jobs: list[tuple] = []
    sextic_jobs: list[tuple] = []

    for slot in range(40):
        if planet_tab[slot, 0] < 0.5:
            continue
        pos = planet_tab[slot, 2:4]
        r = float(radii[slot])
        orbital_r = np.linalg.norm(initial_tab[slot, 2:4] - [50.0, 50.0])
        if orbital_r < 1e-6:
            continue
        pts = _rotating_planet_chord_polyline(pos, orbital_r, av, horizon=24)
        center_at, velocity_at, knot_times = make_polyline_motion(pts)
        for t0, t1 in intersection_windows(
            center_at, g, r, lo, v, 24.0, polyline_points=pts
        ):
            scan_jobs.append(
                (center_at, velocity_at, r, g, v, lo, t0, t1, knot_times)
            )
            sextic_jobs.append((pts, r, g, v, lo, t0, t1))

    n = len(scan_jobs)

    def run_scan() -> None:
        for ca, va, r, gg, vv, l, t0, t1, kt in scan_jobs:
            angular_intersection_extrema(
                ca, va, r, gg, vv, l, t0, t1, split_times=kt, scan_per_piece=16
            )

    def run_sextic() -> None:
        for pts, r, gg, vv, l, t0, t1 in sextic_jobs:
            angular_intersection_extrema_polyline_exact(pts, r, gg, vv, l, t0, t1)

    # Warm sympy coeff lambdas (first call only).
    run_sextic()

    return [
        BenchRow("extrema scan (16/grid)", _timeit(run_scan, repeats=repeats), n),
        BenchRow("extrema sextic exact", _timeit(run_sextic, repeats=repeats), n),
    ]


def _bench_collect(case_id: int = 0, repeats: int = 5) -> BenchRow:
    (
        origin_xy,
        origin_radius,
        speed,
        _p0,
        _p1,
        _radii,
        _active,
        _rank,
        planet_tab,
        initial_tab,
        planet_active,
        av,
        step_count,
    ) = _make_orbiting_case(case_id, ticks=24, planets=40)
    empty_paths = np.zeros((0, 4, 1, 2), dtype=np.float64)
    empty_lens = np.zeros((0, 4), dtype=np.int32)
    empty_bool = np.zeros((0,), dtype=bool)
    empty_idx = np.zeros((0,), dtype=np.int32)

    def run() -> None:
        collect_tangent_hit_events(
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
            empty_idx,
            empty_idx,
            av,
            step_count,
            horizon=24.0,
        )

    run()  # warmup
    return BenchRow(
        f"collect_tangent_hit_events case={case_id}",
        _timeit(run, repeats=repeats),
        1,
    )


def main() -> None:
    print("Tangent geometry benchmarks (lower is better)\n")

    rows = _bench_windows(case_id=0, repeats=10)
    for r in rows:
        print(
            f"  {r.name:28}  {r.seconds*1000:8.2f} ms/batch  "
            f"({r.per_call_us:7.1f} µs/window, n={r.calls})"
        )
    if len(rows) == 2 and rows[1].seconds > 0:
        speedup = rows[0].seconds / rows[1].seconds
        print(f"  → sextic vs scan batch: {speedup:.2f}×\n")

    collect_rows = [_bench_collect(c, repeats=5) for c in (0, 5, 12)]
    print("Full pipeline:")
    for r in collect_rows:
        print(f"  {r.name:40}  {r.seconds*1000:8.2f} ms/call")

    # All 24 cases once
    t0 = time.perf_counter()
    for c in range(24):
        _bench_collect(c, repeats=1)
    total = time.perf_counter() - t0
    print(f"\n  24× collect_tangent_hit_events (1 repeat each): {total*1000:.1f} ms total")


if __name__ == "__main__":
    main()

"""Benchmark Kaggle Orbit Wars Python generators used for episode resets.

``jax_orbit_wars.reset_from_reference`` delegates map + comet schedule to
``kaggle_environments.envs.orbit_wars.orbit_wars`` (``generate_planets``,
``generate_comet_paths``). Training calls this whenever a batched env slice
needs a fresh game (e.g. ``heal_terminal_env_slices`` / ``reset_env_at_index``).

Run from repo root::

    python -m orbit_wars_pt.benchmark_kaggle_reset
    python -m orbit_wars_pt.benchmark_kaggle_reset --trials 100 --profile
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import random
import statistics
import sys
import time
from typing import List, Tuple

import jax.numpy as jnp

import jax_orbit_wars as jow


def _time_generate_planets_only(
    seeds: List[int],
) -> Tuple[List[float], List[float]]:
    """Per-seed wall time for ``generate_planets`` + list copies (no comets)."""

    from kaggle_environments.envs.orbit_wars.orbit_wars import generate_planets

    times: List[float] = []
    nplanets: List[float] = []
    for seed in seeds:
        init_rng = random.Random(seed)
        t0 = time.perf_counter()
        planets_list = generate_planets(init_rng)
        _ = [p.copy() for p in planets_list]
        _ = [p.copy() for p in planets_list]
        times.append(time.perf_counter() - t0)
        nplanets.append(float(len(planets_list)))
    return times, nplanets


def _time_comet_loop_only(seeds: List[int], num_agents: int = 2) -> List[float]:
    """Mirror ``reset_from_reference`` comet section (Kaggle calls only)."""

    from kaggle_environments.envs.orbit_wars.orbit_wars import (
        generate_comet_paths,
        generate_planets,
    )

    times: List[float] = []
    for seed in seeds:
        init_rng = random.Random(seed)
        angular_velocity = init_rng.uniform(0.025, 0.05)
        planets_list = generate_planets(init_rng)
        initial_planets_list = [p.copy() for p in planets_list]

        num_groups = len(planets_list) // 4
        if num_groups > 0:
            home_group = init_rng.randint(0, num_groups - 1)
            base = home_group * 4
            if num_agents == 2:
                planets_list[base][jow.PLANET_OWNER] = 0
                planets_list[base][jow.PLANET_SHIPS] = 10
                planets_list[base + 3][jow.PLANET_OWNER] = 1
                planets_list[base + 3][jow.PLANET_SHIPS] = 10
            elif num_agents == 4:
                for player in range(4):
                    planets_list[base + player][jow.PLANET_OWNER] = player
                    planets_list[base + player][jow.PLANET_SHIPS] = 10

        scratch_initial = [p.copy() for p in initial_planets_list]
        comet_planet_ids_for_generation: list[int] = []

        t0 = time.perf_counter()
        for spawn_step in jow.COMET_SPAWN_STEPS:
            comet_rng = random.Random(f"orbit_wars-comet-{seed}-{spawn_step}")
            paths = generate_comet_paths(
                scratch_initial,
                angular_velocity,
                spawn_step,
                comet_planet_ids_for_generation,
                4.0,
                rng=comet_rng,
            )
            if paths:
                for comet_idx, path in enumerate(paths):
                    length = min(len(path), jow.MAX_COMET_PATH)
                    _ = jnp.asarray(path[:length], dtype=jnp.float32)
        times.append(time.perf_counter() - t0)
    return times


def _summarize(name: str, samples: List[float]) -> None:
    s = sorted(samples)
    n = len(s)
    p50 = s[n // 2]
    p95 = s[int(0.95 * (n - 1))]
    mean = statistics.fmean(s)
    stdev = statistics.pstdev(s) if n > 1 else 0.0
    print(f"{name}:")
    print(f"  n={n}  mean={mean * 1e3:.2f} ms  stdev={stdev * 1e3:.2f} ms")
    print(f"  min={s[0] * 1e3:.2f} ms  p50={p50 * 1e3:.2f} ms  p95={p95 * 1e3:.2f} ms  max={s[-1] * 1e3:.2f} ms")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=40, help="Timed resets after warmup")
    parser.add_argument("--warmup", type=int, default=3, help="Untimed warmup resets")
    parser.add_argument("--seed-base", type=int, default=10_000)
    parser.add_argument("--num-agents", type=int, default=2, choices=(2, 4))
    parser.add_argument("--max-fleets", type=int, default=jow.DEFAULT_MAX_FLEETS)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run cProfile on a few full resets and print top callees",
    )
    args = parser.parse_args()

    # Cold import: kaggle subpackage (what first ``reset_from_reference`` pays).
    t0 = time.perf_counter()
    from kaggle_environments.envs.orbit_wars import orbit_wars as _ow  # noqa: F401

    kaggle_import_s = time.perf_counter() - t0
    print(f"Import kaggle_environments.envs.orbit_wars.orbit_wars: {kaggle_import_s * 1e3:.2f} ms\n")

    seeds = [args.seed_base + i for i in range(args.warmup + args.trials)]

    for _ in seeds[: args.warmup]:
        jow.reset_from_reference(_, args.num_agents, max_fleets=args.max_fleets)

    full_times: List[float] = []
    for seed in seeds[args.warmup :]:
        t0 = time.perf_counter()
        jow.reset_from_reference(seed, args.num_agents, max_fleets=args.max_fleets)
        full_times.append(time.perf_counter() - t0)

    _summarize(f"reset_from_reference (warmup={args.warmup}, num_agents={args.num_agents})", full_times)

    planet_times, planet_counts = _time_generate_planets_only(seeds[args.warmup :])
    _summarize("generate_planets (+ two list copies like reset)", planet_times)
    print(f"  planets per map: mean={statistics.fmean(planet_counts):.0f}\n")

    comet_times = _time_comet_loop_only(seeds[args.warmup :], num_agents=args.num_agents)
    _summarize("comet section only (5× generate_comet_paths + small JAX)", comet_times)

    # Sanity: comet + planets subset should approximate full reset minus JAX struct assembly.
    approx = [a + b for a, b in zip(planet_times, comet_times)]
    _summarize("planets_timing + comet_timing (overlap: planets gen twice in comet path)", approx)

    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
        for seed in seeds[: min(5, len(seeds))]:
            jow.reset_from_reference(seed, args.num_agents, max_fleets=args.max_fleets)
        pr.disable()
        buf = io.StringIO()
        ps = pstats.Stats(pr, stream=buf).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats(25)
        print("\ncProfile (top 25 cumulative, ~5 resets):\n")
        print(buf.getvalue())

    return 0


if __name__ == "__main__":
    sys.exit(main())

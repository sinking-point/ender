"""Benchmark JAX first-hit ray chunk sizes in fresh processes.

Run from the repo root, for example:

    python3 -m orbit_wars_pt.bench_ray_chunks --num-envs 256 --n-rays 256 --chunks 0,64,128,256

Each chunk size is measured in a child process so XLA allocator high-water
marks do not leak between variants.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any

import numpy as np

import orbit_wars_pt.xla_env  # noqa: F401


def _memory_stats() -> dict[str, Any]:
    import jax

    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:
        return {}
    if stats is None:
        return {}
    out: dict[str, Any] = {}
    for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit"):
        if key in stats:
            out[key] = int(stats[key])
    return out


def _mib(x: Any) -> str:
    if x is None:
        return "n/a"
    return f"{float(x) / (1024.0 * 1024.0):.1f}"


def _run_one(args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from orbit_wars_pt.batched_env import stack_initial_states
    from orbit_wars_pt.env_wrapper import OrbitWarsEnvConfig
    from orbit_wars_pt.jax_setup import configure_jax_for_training
    from orbit_wars_pt.micro_jax import selected_origin_fraction_targets_batched

    configure_jax_for_training(prefer_gpu=True, verbose=False)

    cfg = OrbitWarsEnvConfig(num_agents=2, max_fleets=int(args.max_fleets), episode_seed=int(args.seed))
    state_b, _cfg = stack_initial_states(cfg, int(args.num_envs), int(args.seed))
    origin_idx = jnp.zeros((int(args.num_envs),), dtype=jnp.int32)
    frac_idx = jnp.full((int(args.num_envs),), int(args.frac_idx), dtype=jnp.int32)

    def call():
        return selected_origin_fraction_targets_batched(
            state_b,
            origin_idx,
            frac_idx,
            horizon=int(args.horizon),
            ship_speed=float(args.ship_speed),
            samples_per_span=17,
            n_rays=int(args.n_rays),
            ray_chunk_size=int(args.chunk_size),
        )

    t0 = time.perf_counter()
    out = call()
    jax.block_until_ready(out)
    compile_run_s = time.perf_counter() - t0

    times = []
    for _ in range(int(args.repeat)):
        t0 = time.perf_counter()
        out = call()
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)

    angle, _width, valid, overflow, _hit_tick = out
    angle_np, valid_np, overflow_np = jax.device_get((angle, valid, overflow))
    return {
        "chunk_size": int(args.chunk_size),
        "num_envs": int(args.num_envs),
        "n_rays": int(args.n_rays),
        "compile_run_s": compile_run_s,
        "mean_run_s": float(np.mean(times)) if times else 0.0,
        "min_run_s": float(np.min(times)) if times else 0.0,
        "max_run_s": float(np.max(times)) if times else 0.0,
        "valid_frac": float(np.mean(valid_np)),
        "overflow_any": bool(np.any(overflow_np)),
        "angle_checksum": float(np.sum(angle_np)),
        "memory": _memory_stats(),
    }


def _parse_chunks(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise argparse.ArgumentTypeError("chunk list must be non-empty")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--max-fleets", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--ship-speed", type=float, default=6.0)
    p.add_argument("--frac-idx", type=int, default=2)
    p.add_argument("--n-rays", type=int, default=256)
    p.add_argument("--chunks", type=_parse_chunks, default=_parse_chunks("0,32,64,128,256"))
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--one-chunk", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--chunk-size", type=int, default=0, help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.one_chunk:
        print(json.dumps(_run_one(args), sort_keys=True), flush=True)
        return

    rows = []
    for chunk in args.chunks:
        cmd = [
            sys.executable,
            "-m",
            "orbit_wars_pt.bench_ray_chunks",
            "--one-chunk",
            "--chunk-size",
            str(chunk),
            "--num-envs",
            str(args.num_envs),
            "--max-fleets",
            str(args.max_fleets),
            "--seed",
            str(args.seed),
            "--horizon",
            str(args.horizon),
            "--ship-speed",
            str(args.ship_speed),
            "--frac-idx",
            str(args.frac_idx),
            "--n-rays",
            str(args.n_rays),
            "--repeat",
            str(args.repeat),
        ]
        env = os.environ.copy()
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True, env=env)
        line = proc.stdout.strip().splitlines()[-1]
        row = json.loads(line)
        rows.append(row)

    print("chunk compile+run_s mean_run_s min_run_s peak_mib in_use_mib valid_frac angle_sum overflow")
    for row in rows:
        mem = row.get("memory", {})
        print(
            f"{row['chunk_size']:>5d} "
            f"{row['compile_run_s']:>13.4f} "
            f"{row['mean_run_s']:>10.4f} "
            f"{row['min_run_s']:>9.4f} "
            f"{_mib(mem.get('peak_bytes_in_use')):>8} "
            f"{_mib(mem.get('bytes_in_use')):>10} "
            f"{row['valid_frac']:>10.4f} "
            f"{row['angle_checksum']:>9.3f} "
            f"{str(row['overflow_any']):>8}"
        )


if __name__ == "__main__":
    main()

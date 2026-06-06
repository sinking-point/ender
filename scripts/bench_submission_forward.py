#!/usr/bin/env python3
"""Benchmark submission-policy forward pass for one saved Kaggle observation."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from orbit_wars_pt.kaggle_adapter import (
    _infer_num_agents_from_planet_owners,
    _obs_tensors_for_state,
    _obs_tensors_for_states,
    _policy_forward_inference,
    _maybe_compile_policy_batched_forward_for_inference,
    load_policy,
    observation_to_state,
)


def _load_record(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_mode(obs: dict[str, Any], config: dict[str, Any]) -> str:
    cfg_agents = config.get("agentCount")
    if cfg_agents is not None:
        return "4p" if int(cfg_agents) > 2 else "2p"
    return "4p" if _infer_num_agents_from_planet_owners(obs, fallback=2) > 2 else "2p"


def _bundle_checkpoint(bundle_dir: Path, mode: str) -> Path:
    return bundle_dir / ("checkpoint_4p.pt" if mode == "4p" else "checkpoint_2p.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True, help="Extracted submission bundle directory.")
    parser.add_argument("--record", type=Path, required=True, help="Saved Kaggle episode JSON.")
    parser.add_argument("--seat", type=int, default=0, help="Seat index to benchmark.")
    parser.add_argument("--step", type=int, default=1, help="Step index to benchmark.")
    parser.add_argument("--batch-size", type=int, default=15, help="Batch size for forward_dense_rollout path.")
    parser.add_argument(
        "--mode",
        choices=("auto", "single", "batched"),
        default="auto",
        help="Inference path to benchmark. 'single' matches Kaggle adapter batch_size=1 policy(**batch); 'batched' uses forward_dense_rollout.",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations.")
    parser.add_argument("--repeats", type=int, default=20, help="Measured iterations.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device.")
    args = parser.parse_args()

    record = _load_record(args.record.expanduser())
    config = dict(record.get("configuration", {}))
    seat_state = record["steps"][int(args.step)][int(args.seat)]
    obs = dict(seat_state.get("observation", {}))
    ego = int(obs.get("player", args.seat))
    mode = _infer_mode(obs, config)
    bundle_dir = args.bundle_dir.expanduser().resolve()
    checkpoint = _bundle_checkpoint(bundle_dir, mode)
    if not checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {checkpoint}")

    policy, device, training_args = load_policy(checkpoint, device=args.device)
    policy = _maybe_compile_policy_batched_forward_for_inference(policy)
    policy_player_count = 4 if int(training_args.get("num_agents", 2)) > 2 else 2
    normalize_obs_to_p0 = bool(training_args.get("normalize_obs_to_p0", False))
    num_agents = _infer_num_agents_from_planet_owners(obs, fallback=2)
    state = observation_to_state(
        obs,
        config,
        max_fleets=max(512, len(obs.get("fleets", []) or []) + 64),
        num_agents_override=num_agents,
    )

    bench_mode = str(args.mode)
    if bench_mode == "auto":
        bench_mode = "single" if int(args.batch_size) <= 1 else "batched"

    with torch.inference_mode():
        if bench_mode == "single":
            batch = _obs_tensors_for_state(
                state,
                ego,
                device,
                policy_player_count=policy_player_count,
                target_abort_enabled=bool(getattr(policy, "target_abort_enabled", False)),
                normalize_obs_to_p0=normalize_obs_to_p0,
            )
            population_idx = None
        else:
            batch = _obs_tensors_for_states(
                [state] * int(args.batch_size),
                [ego] * int(args.batch_size),
                device,
                policy_player_count=policy_player_count,
                target_abort_enabled=bool(getattr(policy, "target_abort_enabled", False)),
                normalize_obs_to_p0=normalize_obs_to_p0,
            )
            population_idx = None

        for _ in range(int(args.warmup)):
            _policy_forward_inference(policy, batch, population_idx=population_idx)

        durs_ms: list[float] = []
        for _ in range(int(args.repeats)):
            t0 = time.perf_counter()
            out = _policy_forward_inference(policy, batch, population_idx=population_idx)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            durs_ms.append((time.perf_counter() - t0) * 1000.0)

    summary = {
        "mode": mode,
        "bench_mode": bench_mode,
        "seat": int(args.seat),
        "step": int(args.step),
        "batch_size": int(args.batch_size),
        "device": str(device),
        "checkpoint": str(checkpoint),
        "feature_shape": list(batch["features"].shape),
        "mean_ms": round(statistics.fmean(durs_ms), 3),
        "median_ms": round(statistics.median(durs_ms), 3),
        "min_ms": round(min(durs_ms), 3),
        "max_ms": round(max(durs_ms), 3),
        "repeats": int(args.repeats),
        "output_keys": sorted(out.keys()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

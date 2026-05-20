#!/usr/bin/env python3
"""Isolate-profile ``append_to_torch_buffer`` (rollout-sized tensors).

Examples::

    python -m orbit_wars_pt.profile_append_torch_buffer --bench-iters 500
    python -m orbit_wars_pt.profile_append_torch_buffer --profile-iters 32 --trace /tmp/append_trace

Uses one in-place buffer and advances ``micro_k`` / ``write_row`` like many
sequential micro-steps so ``_grow`` hits the ``clone`` + scatter path.
"""

from __future__ import annotations

import argparse
import os
import sys
from time import perf_counter

import torch
from torch.profiler import ProfilerActivity, profile

from orbit_wars_pt.constants import MAX_PLANETS
from orbit_wars_pt.transition_buffer import append_to_torch_buffer, init_torch_transition_buffer


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_steps(
    *,
    buf,
    num_envs: int,
    max_micro_steps: int,
    h_buf: int,
    device: torch.device,
    tensors: dict[str, torch.Tensor],
    n_steps: int,
    wr0: int,
) -> None:
    """Mutate ``buf`` with ``n_steps`` appends; rolls ``micro_k`` and ``write_row``."""

    wr = torch.full((num_envs,), wr0, dtype=torch.int32, device=device)
    mk = torch.randint(1, max(2, max_micro_steps - 1), (num_envs,), device=device)
    active = torch.ones((num_envs,), dtype=torch.bool, device=device)
    for _ in range(n_steps):
        append_to_torch_buffer(
            buf,
            tensors["micro_halt_now"],
            tensors["send_now"],
            tensors["fleet_eta_now"],
            tensors["slot_now"],
            tensors["halt_action"],
            tensors["pair_flat"],
            tensors["frac_idx"],
            tensors["no_valid_pairs"],
            tensors["no_valid_fracs"],
            tensors["must_halt_no_ships"],
            tensors["target_pr"],
            tensors["target_hit_tick"],
            wr,
            mk,
            active,
            max_micro_steps,
        )
        mk = mk + 1
        wrap = mk >= max_micro_steps
        mk = torch.where(wrap, torch.ones_like(mk), mk)
        wr = wr + wrap.to(torch.int32)
        wr = torch.clamp(wr, 1, h_buf - 2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--H-buf", type=int, default=322, help="default 256+64+2 like rollout")
    p.add_argument("--max-micro", type=int, default=64)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--bench-iters", type=int, default=400)
    p.add_argument("--profile-iters", type=int, default=0, help="if >0, run torch profiler for this many steps")
    p.add_argument("--trace", type=str, default="", help="Chrome trace dir for tensorboard_trace_handler")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[profile_append_torch_buffer] CUDA unavailable; use --device cpu", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device)
    N = int(args.num_envs)
    H = int(args.H_buf)
    M = int(args.max_micro)
    P = int(MAX_PLANETS)

    buf = init_torch_transition_buffer(N, H, M, device=device)

    tensors = {
        "micro_halt_now": torch.zeros((N,), dtype=torch.bool, device=device),
        "send_now": torch.randn((N,), dtype=torch.float32, device=device).abs() + 1.0,
        "fleet_eta_now": torch.full((N,), 12.0, dtype=torch.float32, device=device),
        "slot_now": torch.zeros((N,), dtype=torch.int32, device=device),
        "halt_action": torch.zeros((N,), dtype=torch.int32, device=device),
        "pair_flat": torch.randint(0, P * P, (N,), dtype=torch.int32, device=device),
        "frac_idx": torch.randint(0, 8, (N,), dtype=torch.int32, device=device),
        "no_valid_pairs": torch.zeros((N,), dtype=torch.bool, device=device),
        "no_valid_fracs": torch.zeros((N,), dtype=torch.bool, device=device),
        "must_halt_no_ships": torch.zeros((N,), dtype=torch.bool, device=device),
        "target_pr": torch.zeros((N, P), dtype=torch.bool, device=device),
        "target_hit_tick": torch.zeros((N, P), dtype=torch.float32, device=device),
    }

    wr0 = max(1, H // 4)

    _sync(device)
    _run_steps(
        buf=buf,
        num_envs=N,
        max_micro_steps=M,
        h_buf=H,
        device=device,
        tensors=tensors,
        n_steps=int(args.warmup),
        wr0=wr0,
    )
    _sync(device)

    if int(args.bench_iters) > 0:
        buf2 = init_torch_transition_buffer(N, H, M, device=device)
        _sync(device)
        t0 = perf_counter()
        _run_steps(
            buf=buf2,
            num_envs=N,
            max_micro_steps=M,
            h_buf=H,
            device=device,
            tensors=tensors,
            n_steps=int(args.bench_iters),
            wr0=wr0,
        )
        _sync(device)
        dt = perf_counter() - t0
        per_ms = 1000.0 * dt / float(args.bench_iters)
        print(
            f"[append_only] N={N} H={H} M={M} device={device.type} "
            f"iters={args.bench_iters} total_s={dt:.4f} per_append_ms={per_ms:.4f}"
        )

    pi = int(args.profile_iters)
    if pi > 0:
        buf3 = init_torch_transition_buffer(N, H, M, device=device)
        trace_dir = args.trace.strip()
        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            _run_steps(
                buf=buf3,
                num_envs=N,
                max_micro_steps=M,
                h_buf=H,
                device=device,
                tensors=tensors,
                n_steps=pi,
                wr0=wr0,
            )

        print(
            prof.key_averages().table(
                sort_by="self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total",
                row_limit=30,
            )
        )
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
            trace_file = os.path.join(trace_dir, "append_chrome_trace.json")
            prof.export_chrome_trace(trace_file)
            print(f"[append_only] Chrome trace: {trace_file} (chrome://tracing)")


if __name__ == "__main__":
    main()

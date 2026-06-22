#!/usr/bin/env python3
"""Replay one seat of a saved Kaggle episode against a packaged submission bundle.

This runs one persistent worker process for a single seat, mirroring Kaggle's
model of reusing the same Python process across agent calls within an episode.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

_TIMING_DURATION_RE = re.compile(r"\[timing\].*?duration=([0-9.]+)s")


def _load_record(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _import_bundle_agent(bundle_dir: Path):
    sys.path.insert(0, str(bundle_dir))
    main_mod = importlib.import_module("main")
    return main_mod.agent


def _preload_kaggle_environments() -> None:
    with open("/dev/null", "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            import kaggle_environments  # noqa: F401


def _parse_internal_duration(stderr: str) -> float | None:
    match = _TIMING_DURATION_RE.search(stderr)
    if match is None:
        return None
    return float(match.group(1))


def _iter_steps(record: dict[str, Any], seat: int, step_limit: int | None):
    steps = record.get("steps", [])
    if step_limit is not None:
        steps = steps[:step_limit]
    for step_idx, seats in enumerate(steps):
        if seat >= len(seats):
            break
        yield step_idx, seats[seat]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True, help="Extracted submission bundle directory.")
    parser.add_argument("--record", type=Path, required=True, help="Saved Kaggle episode JSON.")
    parser.add_argument("--seat", type=int, required=True, help="Seat index to replay.")
    parser.add_argument("--step-limit", type=int, default=None, help="Optional max steps to replay.")
    parser.add_argument(
        "--preload-kaggle-environments",
        action="store_true",
        help="Import kaggle_environments before timing to avoid paying lazy import cost on the first search step.",
    )
    parser.add_argument(
        "--emulate-scale",
        type=float,
        default=1.0,
        help="Multiplier for an emulated slower runtime profile. Applied to observed duration.",
    )
    parser.add_argument(
        "--emulate-fixed-s",
        type=float,
        default=0.0,
        help="Fixed seconds to add to emulated slower runtime after scaling.",
    )
    parser.add_argument(
        "--sleep-to-emulated",
        action="store_true",
        help="Sleep so wall time matches the emulated duration. Output still preserves observed timing fields.",
    )
    args = parser.parse_args()

    record = _load_record(args.record)
    config = record.get("configuration", {})
    if args.preload_kaggle_environments:
        _preload_kaggle_environments()
    agent = _import_bundle_agent(args.bundle_dir.expanduser().resolve())

    for step_idx, seat_state in _iter_steps(record, int(args.seat), args.step_limit):
        obs = seat_state.get("observation", {})
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        t_wall0 = time.perf_counter()
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            t0 = time.perf_counter()
            try:
                action = agent(obs, config)
                error = None
            except Exception as exc:  # pragma: no cover - diagnostic path
                action = []
                error = repr(exc)
            observed_duration = time.perf_counter() - t0
        stderr_text = err_buf.getvalue()
        internal_duration = _parse_internal_duration(stderr_text)
        emulation_base_duration = (
            float(internal_duration) if internal_duration is not None else float(observed_duration)
        )
        emulated_duration = max(
            float(observed_duration),
            float(args.emulate_fixed_s) + (float(args.emulate_scale) * emulation_base_duration),
        )
        emulated_sleep = max(0.0, float(emulated_duration) - float(observed_duration))
        if args.sleep_to_emulated and emulated_sleep > 0.0:
            time.sleep(emulated_sleep)
        wall_duration = time.perf_counter() - t_wall0
        row = {
            "step": int(step_idx),
            "seat": int(args.seat),
            "player": int(obs.get("player", args.seat)),
            "duration": round(float(observed_duration), 6),
            "duration_observed": round(float(observed_duration), 6),
            "duration_emulated": round(float(emulated_duration), 6),
            "duration_wall": round(float(wall_duration), 6),
            "emulated_sleep": round(float(emulated_sleep), 6),
            "emulation_base_duration": round(float(emulation_base_duration), 6),
            "internal_duration": (
                round(float(internal_duration), 6) if internal_duration is not None else None
            ),
            "remainingOverageTime": obs.get("remainingOverageTime"),
            "status": seat_state.get("status"),
            "reward": seat_state.get("reward"),
            "num_actions": len(action) if isinstance(action, list) else None,
            "action": action,
            "stdout": out_buf.getvalue(),
            "stderr": stderr_text,
        }
        if error is not None:
            row["error"] = error
        print(json.dumps(row, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()

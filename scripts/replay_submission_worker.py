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
import sys
import time
from pathlib import Path
from typing import Any


def _load_record(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _import_bundle_agent(bundle_dir: Path):
    sys.path.insert(0, str(bundle_dir))
    main_mod = importlib.import_module("main")
    return main_mod.agent


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
    args = parser.parse_args()

    record = _load_record(args.record)
    config = record.get("configuration", {})
    agent = _import_bundle_agent(args.bundle_dir.expanduser().resolve())

    for step_idx, seat_state in _iter_steps(record, int(args.seat), args.step_limit):
        obs = seat_state.get("observation", {})
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            t0 = time.perf_counter()
            try:
                action = agent(obs, config)
                error = None
            except Exception as exc:  # pragma: no cover - diagnostic path
                action = []
                error = repr(exc)
            duration = time.perf_counter() - t0
        row = {
            "step": int(step_idx),
            "seat": int(args.seat),
            "player": int(obs.get("player", args.seat)),
            "duration": round(float(duration), 6),
            "remainingOverageTime": obs.get("remainingOverageTime"),
            "status": seat_state.get("status"),
            "reward": seat_state.get("reward"),
            "num_actions": len(action) if isinstance(action, list) else None,
            "action": action,
            "stdout": out_buf.getvalue(),
            "stderr": err_buf.getvalue(),
        }
        if error is not None:
            row["error"] = error
        print(json.dumps(row, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()

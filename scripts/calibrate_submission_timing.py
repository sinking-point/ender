#!/usr/bin/env python3
"""Replay a packaged submission against a saved episode and fit a local slowdown profile."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any


def _extract_bundle(bundle: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if bundle.is_dir():
        return bundle.resolve(), None
    tempdir = tempfile.TemporaryDirectory(prefix="orbit-wars-bundle-")
    bundle_dir = Path(tempdir.name) / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as tf:
        tf.extractall(bundle_dir)
    return bundle_dir.resolve(), tempdir


def _run_worker(
    *,
    bundle_dir: Path,
    record: Path,
    seat: int,
    preload_kaggle_environments: bool,
) -> list[dict[str, Any]]:
    with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as tmp:
        out_path = Path(tmp.name)
    try:
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("replay_submission_worker.py")),
            "--bundle-dir",
            str(bundle_dir),
            "--record",
            str(record),
            "--seat",
            str(int(seat)),
        ]
        if preload_kaggle_environments:
            cmd.append("--preload-kaggle-environments")
        with out_path.open("w", encoding="utf-8") as out_f:
            subprocess.run(cmd, check=True, stdout=out_f)
        return [
            json.loads(line)
            for line in out_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    finally:
        out_path.unlink(missing_ok=True)


def _load_reference_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for item in payload:
        if not item:
            continue
        rows.append(item[0])
    return rows


def _pair_internal_durations(
    local_rows: list[dict[str, Any]],
    ref_rows: list[dict[str, Any]],
) -> tuple[list[float], list[float]]:
    local: list[float] = []
    ref: list[float] = []
    for lrow, rrow in zip(local_rows, ref_rows):
        ldur = lrow.get("internal_duration")
        rdur = _reference_internal_duration(rrow)
        if ldur is None or rdur is None:
            continue
        local.append(float(ldur))
        ref.append(float(rdur))
    return local, ref


def _reference_internal_duration(row: dict[str, Any]) -> float | None:
    stderr = str(row.get("stderr", ""))
    marker = "duration="
    idx = stderr.find(marker)
    if idx < 0:
        duration = row.get("duration")
        return float(duration) if duration is not None else None
    start = idx + len(marker)
    end = stderr.find("s", start)
    if end < 0:
        return None
    return float(stderr[start:end])


def _fit_scalar(xs: list[float], ys: list[float]) -> float:
    denom = sum(x * x for x in xs)
    if denom <= 0.0:
        return 1.0
    return sum(x * y for x, y in zip(xs, ys)) / denom


def _fit_affine(xs: list[float], ys: list[float]) -> tuple[float, float]:
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    var = sum((x - mx) * (x - mx) for x in xs)
    if var <= 0.0:
        return 1.0, 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = cov / var
    intercept = my - (slope * mx)
    return slope, intercept


def _mae(xs: list[float], ys: list[float], *, scale: float, intercept: float = 0.0) -> float:
    return statistics.fmean(abs((scale * x) + intercept - y) for x, y in zip(xs, ys))


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.fmean(values), 6),
        "median": round(statistics.median(values), 6),
        "max": round(max(values), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Submission bundle dir or .tar.gz.")
    parser.add_argument("--record", type=Path, required=True, help="Saved Kaggle episode JSON.")
    parser.add_argument("--reference-seat-log", type=Path, required=True, help="Reference seat log JSON.")
    parser.add_argument("--seat", type=int, default=3, help="Seat index to replay and compare.")
    parser.add_argument(
        "--preload-kaggle-environments",
        action="store_true",
        help="Preload kaggle_environments before the replay so lazy import cost does not skew one search step.",
    )
    args = parser.parse_args()

    bundle_dir, tempdir = _extract_bundle(args.bundle.expanduser().resolve())
    try:
        local_rows = _run_worker(
            bundle_dir=bundle_dir,
            record=args.record.expanduser().resolve(),
            seat=int(args.seat),
            preload_kaggle_environments=bool(args.preload_kaggle_environments),
        )
    finally:
        if tempdir is not None:
            tempdir.cleanup()

    ref_rows = _load_reference_rows(args.reference_seat_log.expanduser().resolve())
    local_durs, ref_durs = _pair_internal_durations(local_rows, ref_rows)
    if not local_durs or not ref_durs:
        raise SystemExit("no comparable internal durations found")

    scalar = _fit_scalar(local_durs, ref_durs)
    affine_scale, affine_intercept = _fit_affine(local_durs, ref_durs)
    ratio_median = statistics.median(r / l for l, r in zip(local_durs, ref_durs) if l > 0.0)

    result = {
        "seat": int(args.seat),
        "preload_kaggle_environments": bool(args.preload_kaggle_environments),
        "samples": len(local_durs),
        "local_internal_s": _summary(local_durs),
        "reference_internal_s": _summary(ref_durs),
        "median_ref_over_local": round(ratio_median, 6),
        "scalar_fit": {
            "scale": round(scalar, 6),
            "mae_s": round(_mae(local_durs, ref_durs, scale=scalar), 6),
        },
        "affine_fit": {
            "scale": round(affine_scale, 6),
            "fixed_s": round(affine_intercept, 6),
            "mae_s": round(
                _mae(local_durs, ref_durs, scale=affine_scale, intercept=affine_intercept), 6
            ),
        },
        "recommended_worker_flags": [
            "--preload-kaggle-environments" if args.preload_kaggle_environments else None,
            "--emulate-scale",
            f"{affine_scale:.6f}",
            "--emulate-fixed-s",
            f"{max(0.0, affine_intercept):.6f}",
            "--sleep-to-emulated",
        ],
    }
    result["recommended_worker_flags"] = [
        part for part in result["recommended_worker_flags"] if part is not None
    ]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

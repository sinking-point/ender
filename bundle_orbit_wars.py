#!/usr/bin/env python3
"""Bundle the Orbit Wars training project into a portable zip archive.

Default behavior packages the training code and supporting files needed to set
up and launch runs on a remote machine, while excluding local-only artifacts
such as ``.git``, virtualenvs, caches, and existing experiment outputs.

Use ``--include-experiment`` when you also want to ship a checkpoint tree for
resume/forking on the remote instance.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile


DEFAULT_SOURCE = Path("/home/billy/orbit-wars")
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/c/Users/Billy/Documents/Codex/2026-06-12/work-in-home-billy-orbit-wars/outputs"
)

DEFAULT_INCLUDE = (
    "agent.py",
    "chatgpt-geometry.md",
    "instructions.txt",
    "jax_orbit_wars.py",
    "requirements-train.txt",
    "orbit_wars_pt/**",
    "scripts/**",
    "docs/**",
)

DEFAULT_EXCLUDE = (
    ".git/**",
    ".agents/**",
    ".codex/**",
    ".pytest_cache/**",
    ".venv/**",
    "__pycache__/**",
    "**/__pycache__/**",
    "*.pyc",
    "*.pyo",
    "dist/**",
    "experiments/**",
    "records/**",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Project root to bundle (default: {DEFAULT_SOURCE})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to the output zip. Defaults to outputs/orbit-wars-bundle.zip in this workspace.",
    )
    p.add_argument(
        "--include-experiment",
        action="append",
        default=[],
        metavar="NAME",
        help="Include experiments/NAME in the bundle. Repeatable.",
    )
    p.add_argument(
        "--include-records",
        action="store_true",
        help="Include records/ for replay or validation debugging.",
    )
    p.add_argument(
        "--include-dist",
        action="store_true",
        help="Include dist/ artifacts.",
    )
    p.add_argument(
        "--root-name",
        type=str,
        default="orbit-wars",
        help="Top-level directory name inside the zip.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Print files that would be bundled without writing a zip.",
    )
    return p.parse_args()


def should_include(rel_path: str, include_patterns: tuple[str, ...], exclude_patterns: tuple[str, ...]) -> bool:
    rel_path = rel_path.replace(os.sep, "/")
    included = any(fnmatch.fnmatch(rel_path, pattern) for pattern in include_patterns)
    excluded = any(fnmatch.fnmatch(rel_path, pattern) for pattern in exclude_patterns)
    return included and not excluded


def build_include_patterns(args: argparse.Namespace) -> tuple[str, ...]:
    patterns = list(DEFAULT_INCLUDE)
    patterns.extend(f"experiments/{name}/**" for name in args.include_experiment)
    if args.include_records:
        patterns.append("records/**")
    if args.include_dist:
        patterns.append("dist/**")
    return tuple(patterns)


def build_exclude_patterns(args: argparse.Namespace) -> tuple[str, ...]:
    patterns = list(DEFAULT_EXCLUDE)
    if args.include_records:
        patterns.remove("records/**")
    if args.include_dist:
        patterns.remove("dist/**")
    if args.include_experiment:
        patterns.remove("experiments/**")
    return tuple(patterns)


def iter_files(source: Path, include_patterns: tuple[str, ...], exclude_patterns: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        rel_path = path.relative_to(source).as_posix()
        if should_include(rel_path, include_patterns, exclude_patterns):
            files.append(path)
    files.sort()
    return files


def default_output_path() -> Path:
    return DEFAULT_OUTPUT_DIR / "orbit-wars-bundle.zip"


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = (args.output.expanduser().resolve() if args.output is not None else default_output_path())

    if not source.is_dir():
        print(f"error: source directory not found: {source}", file=sys.stderr)
        return 1

    include_patterns = build_include_patterns(args)
    exclude_patterns = build_exclude_patterns(args)
    files = iter_files(source, include_patterns, exclude_patterns)

    if not files:
        print("error: no files matched the bundle rules", file=sys.stderr)
        return 1

    if args.list:
        for path in files:
            print(path.relative_to(source).as_posix())
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as zf:
        for path in files:
            rel_path = path.relative_to(source).as_posix()
            arcname = f"{args.root_name}/{rel_path}"
            zf.write(path, arcname=arcname)

    total_bytes = sum(path.stat().st_size for path in files)
    print(f"wrote {output}")
    print(f"files: {len(files)}")
    print(f"bytes: {total_bytes}")
    if args.include_experiment:
        print("included experiments:")
        for name in args.include_experiment:
            print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

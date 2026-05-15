"""Build a Kaggle Orbit Wars submission bundle from a training checkpoint.

The bundle is a directory (or ``.tar.gz``) with ``main.py`` at the root, a
minimal ``orbit_wars_pt`` inference package, and ``checkpoint.pt``.

Example::

    python -m orbit_wars_pt.package_kaggle_submission \\
        --checkpoint experiments/25/checkpoints/iter_00000520.pt \\
        --out dist/orbit-wars-submission.tar.gz

    kaggle competitions submit orbit-wars -f dist/orbit-wars-submission.tar.gz -m "exp25 iter520"
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import textwrap
from pathlib import Path

# Modules required by ``orbit_wars_pt.kaggle_adapter`` at inference time (no JAX).
_SUBMISSION_PACKAGE_FILES = (
    "__init__.py",
    "constants.py",
    "geometry.py",
    "model.py",
    "kaggle_adapter.py",
)

_MAIN_PY_TEMPLATE = """\
\"\"\"Kaggle Orbit Wars submission entry point.\"\"\"

from __future__ import annotations

import os

# Defaults for the competition runtime (override via env if needed).
os.environ.setdefault("ORBIT_WARS_CHECKPOINT", "checkpoint.pt")
os.environ.setdefault("ORBIT_WARS_DEVICE", {device!r})
os.environ.setdefault("ORBIT_WARS_GREEDY", {greedy!r})
os.environ.setdefault("ORBIT_WARS_COLLAPSE_OPPONENTS", {collapse_opponents!r})
{extra_env}

from orbit_wars_pt.kaggle_adapter import agent

__all__ = ["agent"]
"""


def _write_main_py(
    path: Path,
    *,
    device: str,
    greedy: bool,
    collapse_opponents: bool,
    extra_env: dict[str, str],
) -> None:
    extra_lines = []
    for key, value in extra_env.items():
        extra_lines.append(f'os.environ.setdefault({key!r}, {value!r})')
    extra_block = "\n".join(extra_lines)
    if extra_block:
        extra_block += "\n"
    path.write_text(
        _MAIN_PY_TEMPLATE.format(
            device=device,
            greedy="1" if greedy else "0",
            collapse_opponents="1" if collapse_opponents else "0",
            extra_env=extra_block,
        ),
        encoding="utf-8",
    )


def _copy_inference_package(dest_pkg: Path, *, source_pkg: Path) -> None:
    dest_pkg.mkdir(parents=True, exist_ok=True)
    for name in _SUBMISSION_PACKAGE_FILES:
        src = source_pkg / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing submission source file: {src}")
        shutil.copy2(src, dest_pkg / name)


def _archive_submission_dir(bundle_dir: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in sorted(bundle_dir.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(bundle_dir).as_posix())


def _submission_paths(out: Path) -> tuple[Path, Path | None]:
    """Return ``(bundle_dir, archive_path)``; archive_path is set when writing a tarball."""

    name = out.name.lower()
    if name.endswith(".tar.gz"):
        return out.with_name(out.name[: -len(".tar.gz")]), out
    if name.endswith(".tgz"):
        return out.with_suffix(""), out
    return out, None


def package_submission(
    checkpoint: Path,
    out: Path,
    *,
    greedy: bool = False,
    device: str = "cpu",
    collapse_opponents: bool = True,
    extra_env: dict[str, str] | None = None,
    source_pkg: Path | None = None,
) -> Path:
    """Write a submission bundle directory; return its path."""

    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    source_pkg = (source_pkg or Path(__file__).resolve().parent).resolve()
    out = out.expanduser().resolve()
    bundle_dir, archive_path = _submission_paths(out)

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    shutil.copy2(checkpoint, bundle_dir / "checkpoint.pt")
    _copy_inference_package(bundle_dir / "orbit_wars_pt", source_pkg=source_pkg)
    _write_main_py(
        bundle_dir / "main.py",
        device=device,
        greedy=greedy,
        collapse_opponents=collapse_opponents,
        extra_env=extra_env or {},
    )

    readme = textwrap.dedent(
        f"""\
        Orbit Wars Kaggle submission bundle
        ===================================

        Checkpoint: {checkpoint.name} (copied to checkpoint.pt)
        Greedy: {greedy}
        Device: {device}
        Collapse opponents (4p -> single enemy owner slot): {collapse_opponents}

        Test locally (from this directory):

            pip install "kaggle-environments>=1.28.0" torch numpy
            python -c "
        from kaggle_environments import make
        from main import agent
        env = make('orbit_wars', configuration={{'seed': 42}}, debug=True)
        env.run([agent, agent])
        print([(i, s.reward) for i, s in enumerate(env.steps[-1])])
        "

        Submit:

            kaggle competitions submit orbit-wars -f {archive_path or (bundle_dir.parent / (bundle_dir.name + '.tar.gz'))} -m "your message"
        """
    ).strip()
    (bundle_dir / "README-submission.txt").write_text(readme + "\n", encoding="utf-8")

    if archive_path is not None:
        _archive_submission_dir(bundle_dir, archive_path)
        return archive_path
    return bundle_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Training checkpoint (.pt) to ship as checkpoint.pt in the bundle.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dist/orbit-wars-submission.tar.gz"),
        help="Output directory, or .tar.gz path (default: dist/orbit-wars-submission.tar.gz).",
    )
    parser.add_argument(
        "--greedy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use argmax actions in main.py (default: stochastic sampling from the policy).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device written into main.py (default: cpu for Kaggle).",
    )
    parser.add_argument(
        "--collapse-opponents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Map every non-ego player to owner slot 2 in observations (2p-trained policy "
            "in 4p FFA). Incoming threat features already sum all opponents."
        ),
    )
    parser.add_argument(
        "--keep-dir",
        action="store_true",
        help="When --out ends with .tar.gz, keep the extracted bundle directory beside the archive.",
    )
    args = parser.parse_args()

    result = package_submission(
        args.checkpoint,
        args.out,
        greedy=bool(args.greedy),
        device=str(args.device),
        collapse_opponents=bool(args.collapse_opponents),
    )

    bundle_dir, archive_path = _submission_paths(args.out.expanduser().resolve())
    if archive_path is not None and not args.keep_dir and bundle_dir.exists():
        shutil.rmtree(bundle_dir)

    size_mb = result.stat().st_size / (1024 * 1024)
    print(f"Wrote {result} ({size_mb:.1f} MiB)")


if __name__ == "__main__":
    main()

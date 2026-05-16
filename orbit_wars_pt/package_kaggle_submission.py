"""Build a Kaggle Orbit Wars submission bundle from training checkpoints.

The bundle is a directory (or ``.tar.gz``) with ``main.py`` at the root, a
minimal ``orbit_wars_pt`` inference package, and two policy checkpoints (4-player
FFA and 2-player endgame).  Both policies are loaded at startup.

Example::

    python -m orbit_wars_pt.package_kaggle_submission \\
        --checkpoint-4p experiments/4p/checkpoints/iter_00000520.pt \\
        --checkpoint-2p experiments/2p/checkpoints/iter_00000520.pt \\
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
os.environ.setdefault("ORBIT_WARS_CHECKPOINT_4P", "checkpoint_4p.pt")
os.environ.setdefault("ORBIT_WARS_CHECKPOINT_2P", "checkpoint_2p.pt")
os.environ.setdefault("ORBIT_WARS_DEVICE", {device!r})
os.environ.setdefault("ORBIT_WARS_GREEDY", {greedy!r})
{extra_env}

from orbit_wars_pt.kaggle_adapter import agent

__all__ = ["agent"]
"""


def _write_main_py(
    path: Path,
    *,
    device: str,
    greedy: bool,
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
    checkpoint_4p: Path,
    checkpoint_2p: Path,
    out: Path,
    *,
    greedy: bool = False,
    device: str = "cpu",
    extra_env: dict[str, str] | None = None,
    source_pkg: Path | None = None,
) -> Path:
    """Write a submission bundle directory; return its path."""

    checkpoint_4p = checkpoint_4p.expanduser().resolve()
    checkpoint_2p = checkpoint_2p.expanduser().resolve()
    if not checkpoint_4p.is_file():
        raise FileNotFoundError(checkpoint_4p)
    if not checkpoint_2p.is_file():
        raise FileNotFoundError(checkpoint_2p)

    source_pkg = (source_pkg or Path(__file__).resolve().parent).resolve()
    out = out.expanduser().resolve()
    bundle_dir, archive_path = _submission_paths(out)

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    shutil.copy2(checkpoint_4p, bundle_dir / "checkpoint_4p.pt")
    shutil.copy2(checkpoint_2p, bundle_dir / "checkpoint_2p.pt")
    _copy_inference_package(bundle_dir / "orbit_wars_pt", source_pkg=source_pkg)
    _write_main_py(
        bundle_dir / "main.py",
        device=device,
        greedy=greedy,
        extra_env=extra_env or {},
    )

    readme = textwrap.dedent(
        f"""\
        Orbit Wars Kaggle submission bundle
        ===================================

        4-player checkpoint: {checkpoint_4p.name} (copied to checkpoint_4p.pt)
        2-player checkpoint: {checkpoint_2p.name} (copied to checkpoint_2p.pt)
        Greedy: {greedy}
        Device: {device}

        Policy selection: 4-player policy while two or more opponents are alive;
        switches to the 2-player policy as soon as only one opponent remains.
        Both policies are loaded at process start.

        Test locally (from this directory):

            pip install "kaggle-environments>=1.28.0" torch numpy
            python -c "
        from kaggle_environments import make
        from main import agent
        env = make('orbit_wars', configuration={{'seed': 42, 'agentCount': 4}}, debug=True)
        env.run([agent] * 4)
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
        "--checkpoint-4p",
        type=Path,
        required=True,
        help="4-player FFA training checkpoint (.pt) shipped as checkpoint_4p.pt.",
    )
    parser.add_argument(
        "--checkpoint-2p",
        type=Path,
        required=True,
        help="2-player training checkpoint (.pt) shipped as checkpoint_2p.pt.",
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
        "--keep-dir",
        action="store_true",
        help="When --out ends with .tar.gz, keep the extracted bundle directory beside the archive.",
    )
    args = parser.parse_args()

    result = package_submission(
        args.checkpoint_4p,
        args.checkpoint_2p,
        args.out,
        greedy=bool(args.greedy),
        device=str(args.device),
    )

    bundle_dir, archive_path = _submission_paths(args.out.expanduser().resolve())
    if archive_path is not None and not args.keep_dir and bundle_dir.exists():
        shutil.rmtree(bundle_dir)

    size_mb = result.stat().st_size / (1024 * 1024)
    print(f"Wrote {result} ({size_mb:.1f} MiB)")


if __name__ == "__main__":
    main()

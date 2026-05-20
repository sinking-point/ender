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
from typing import Any

import torch

# Modules required by ``orbit_wars_pt.kaggle_adapter`` at inference time (no JAX).
_SUBMISSION_PACKAGE_FILES = (
    "__init__.py",
    "constants.py",
    "geometry.py",
    "interval_geometry_np.py",
    "orthogonal_geometry_np.py",
    "tangent_geometry_np.py",
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


def _slim_checkpoint_payload(payload: Any) -> dict[str, Any]:
    """Drop optimizer / rollout state; keep only what ``load_policy`` needs."""

    if not isinstance(payload, dict) or "policy" not in payload:
        raise ValueError("Expected a training checkpoint dict with a 'policy' key")
    slim: dict[str, Any] = {
        "policy": payload["policy"],
        "training_args": payload.get("training_args", {}),
    }
    if "version" in payload:
        slim["version"] = payload["version"]
    return slim


def _collapse_population_members(
    payload: dict[str, Any],
    *,
    keep_member: int | None,
) -> dict[str, Any]:
    """Alias all population tail tensors to one selected member for smaller serialization."""

    if keep_member is None:
        return payload
    policy_state = payload.get("policy")
    if not isinstance(policy_state, dict):
        return payload
    training_args = payload.get("training_args", {})
    population_size = int(training_args.get("population_size", 1)) if isinstance(training_args, dict) else 1
    if population_size <= 1:
        return payload
    selected = int(keep_member)
    if selected < 0 or selected >= population_size:
        raise ValueError(
            f"population member {selected} out of range for population_size={population_size}"
        )

    remapped: dict[str, Any] = dict(policy_state)
    prefix = f"population_tails.{selected}."
    selected_keys = {
        key[len(prefix) :]: value
        for key, value in policy_state.items()
        if key.startswith(prefix)
    }
    if not selected_keys:
        raise ValueError(f"Selected population member {selected} not present in checkpoint policy state")

    for member in range(population_size):
        member_prefix = f"population_tails.{member}."
        if member == selected:
            continue
        for suffix, value in selected_keys.items():
            key = member_prefix + suffix
            if key in policy_state:
                remapped[key] = value

    out = dict(payload)
    out["policy"] = remapped
    return out


def _write_checkpoint(src: Path, dest: Path, *, slim: bool, keep_member: int | None = None) -> None:
    try:
        payload = torch.load(src, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(src, map_location="cpu")
    payload_out = _slim_checkpoint_payload(payload) if slim else payload
    if not isinstance(payload_out, dict):
        raise ValueError("Expected checkpoint payload to be a dict after loading")
    payload_out = _collapse_population_members(payload_out, keep_member=keep_member)
    torch.save(payload_out, dest)


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
    slim: bool = True,
    target_method: str | None = None,
    interval_geometry: str | None = None,
    population_member: int | None = None,
    population_member_4p: int | None = None,
    population_member_2p: int | None = None,
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

    keep_member_4p = population_member if population_member_4p is None else population_member_4p
    keep_member_2p = population_member if population_member_2p is None else population_member_2p
    if slim or keep_member_4p is not None or keep_member_2p is not None:
        _write_checkpoint(
            checkpoint_4p,
            bundle_dir / "checkpoint_4p.pt",
            slim=slim,
            keep_member=keep_member_4p,
        )
        _write_checkpoint(
            checkpoint_2p,
            bundle_dir / "checkpoint_2p.pt",
            slim=slim,
            keep_member=keep_member_2p,
        )
    else:
        shutil.copy2(checkpoint_4p, bundle_dir / "checkpoint_4p.pt")
        shutil.copy2(checkpoint_2p, bundle_dir / "checkpoint_2p.pt")
    _copy_inference_package(bundle_dir / "orbit_wars_pt", source_pkg=source_pkg)
    submission_env = dict(extra_env or {})
    if population_member is not None:
        submission_env["ORBIT_WARS_MEMBER"] = str(int(population_member))
    if population_member_4p is not None:
        submission_env["ORBIT_WARS_MEMBER_4P"] = str(int(population_member_4p))
    if population_member_2p is not None:
        submission_env["ORBIT_WARS_MEMBER_2P"] = str(int(population_member_2p))
    if target_method is not None:
        submission_env["ORBIT_WARS_TARGET_METHOD"] = str(target_method)
    if interval_geometry is not None:
        submission_env["ORBIT_WARS_INTERVAL_GEOMETRY"] = str(interval_geometry)

    _write_main_py(
        bundle_dir / "main.py",
        device=device,
        greedy=greedy,
        extra_env=submission_env,
    )

    readme = textwrap.dedent(
        f"""\
        Orbit Wars Kaggle submission bundle
        ===================================

        4-player checkpoint: {checkpoint_4p.name} (copied to checkpoint_4p.pt)
        2-player checkpoint: {checkpoint_2p.name} (copied to checkpoint_2p.pt)
        Greedy: {greedy}
        Device: {device}
        Population member (fallback): {population_member if population_member is not None else 'member 0 / checkpoint default'}
        Population member 4p: {population_member_4p if population_member_4p is not None else (population_member if population_member is not None else 'member 0 / checkpoint default')}
        Population member 2p: {population_member_2p if population_member_2p is not None else (population_member if population_member is not None else 'member 0 / checkpoint default')}
        Target method: {target_method or 'checkpoint/default'}
        Interval geometry: {interval_geometry or 'checkpoint/default'}

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
        "--member",
        type=int,
        default=None,
        help="Bake ORBIT_WARS_MEMBER into main.py as the fallback population member for both checkpoints.",
    )
    parser.add_argument(
        "--member-4p",
        type=int,
        default=None,
        help="Bake ORBIT_WARS_MEMBER_4P into main.py for the 4-player checkpoint. Defaults to --member if set.",
    )
    parser.add_argument(
        "--member-2p",
        type=int,
        default=None,
        help="Bake ORBIT_WARS_MEMBER_2P into main.py for the 2-player checkpoint. Defaults to --member if set.",
    )
    parser.add_argument(
        "--target-method",
        choices=("rays", "interval"),
        default=None,
        help=(
            "Bake ORBIT_WARS_TARGET_METHOD into main.py. "
            "Use 'interval' for interval first-hit targeting; default leaves checkpoint/env behavior unchanged."
        ),
    )
    parser.add_argument(
        "--interval-geometry",
        choices=("sampled", "orthogonal", "tangent"),
        default=None,
        help=(
            "Bake ORBIT_WARS_INTERVAL_GEOMETRY into main.py. "
            "Useful with --target-method interval; default leaves checkpoint/env behavior unchanged."
        ),
    )
    parser.add_argument(
        "--keep-dir",
        action="store_true",
        help="When --out ends with .tar.gz, keep the extracted bundle directory beside the archive.",
    )
    parser.add_argument(
        "--no-slim",
        action="store_true",
        help="Copy full training checkpoints (optimizer, rollout carry, etc.). "
        "Default strips to policy weights only so the bundle fits Kaggle's 100 MiB limit.",
    )
    args = parser.parse_args()

    result = package_submission(
        args.checkpoint_4p,
        args.checkpoint_2p,
        args.out,
        greedy=bool(args.greedy),
        device=str(args.device),
        slim=not args.no_slim,
        target_method=args.target_method,
        interval_geometry=args.interval_geometry,
        population_member=args.member,
        population_member_4p=args.member if args.member_4p is None else args.member_4p,
        population_member_2p=args.member if args.member_2p is None else args.member_2p,
    )

    bundle_dir, archive_path = _submission_paths(args.out.expanduser().resolve())
    if archive_path is not None and not args.keep_dir and bundle_dir.exists():
        shutil.rmtree(bundle_dir)

    size_mb = result.stat().st_size / (1024 * 1024)
    print(f"Wrote {result} ({size_mb:.1f} MiB)")


if __name__ == "__main__":
    main()

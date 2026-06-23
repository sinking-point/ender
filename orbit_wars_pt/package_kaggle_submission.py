"""Build a Kaggle Orbit Wars submission bundle from training checkpoints.

The bundle is a directory (or ``.tar.gz``) with ``main.py`` at the root, a
minimal ``orbit_wars_pt`` inference package, and two policy checkpoints (4-player
FFA and 2-player endgame). Search can optionally use each checkpoint's embedded
student model.

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
from typing import Any, Mapping

import torch

SAMPLING_MODE_CHOICES = ("stochastic", "greedy", "mixed")

# Modules required by ``orbit_wars_pt.kaggle_adapter`` at inference time (no JAX).
_SUBMISSION_PACKAGE_FILES = (
    "__init__.py",
    "constants.py",
    "compressed_observation.py",
    "geometry.py",
    "interval_geometry_np.py",
    "orthogonal_geometry_np.py",
    "tangent_geometry_np.py",
    "model.py",
    "reward_config.py",
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
os.environ.setdefault("ORBIT_WARS_CPU_THREADS", "1")
os.environ.setdefault("ORBIT_WARS_GREEDY", {greedy!r})
os.environ.setdefault("ORBIT_WARS_LOG_TIMING", "1")
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


def _slim_checkpoint_payload(
    payload: Any,
    *,
    policy_key: str = "policy",
    include_student_policy: bool = False,
) -> dict[str, Any]:
    """Drop optimizer / rollout state; keep only what ``load_policy`` needs."""

    if not isinstance(payload, dict) or policy_key not in payload:
        raise ValueError(f"Expected a training checkpoint dict with a {policy_key!r} key")
    training_args_in = payload.get("training_args", {})
    training_args = dict(training_args_in) if isinstance(training_args_in, Mapping) else {}
    policy_state = payload[policy_key]
    slim: dict[str, Any] = {
        "policy": policy_state,
        "training_args": training_args,
    }
    if include_student_policy:
        student_state = payload.get("student_policy")
        if student_state is None:
            raise ValueError(
                "Checkpoint is missing 'student_policy' but student search was requested"
            )
        slim["student_policy"] = student_state
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


def _write_checkpoint(
    src: Path,
    dest: Path,
    *,
    slim: bool,
    keep_member: int | None = None,
    policy_key: str = "policy",
    include_student_policy: bool = False,
) -> None:
    try:
        payload = torch.load(src, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(src, map_location="cpu")
    payload_out = (
        _slim_checkpoint_payload(
            payload,
            policy_key=policy_key,
            include_student_policy=include_student_policy,
        )
        if slim
        else payload
    )
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
    checkpoint_4p: Path | None,
    checkpoint_2p: Path | None,
    out: Path,
    *,
    use_student_for_search_4p: bool = False,
    use_student_for_search_2p: bool = False,
    search_main_policy_for_ego_steps_4p: int = 1,
    search_main_policy_for_ego_steps_2p: int = 1,
    greedy: bool = False,
    greedy_4p: bool | None = None,
    greedy_2p: bool | None = None,
    sampling_mode: str | None = None,
    sampling_mode_4p: str | None = None,
    sampling_mode_2p: str | None = None,
    device: str = "cpu",
    extra_env: dict[str, str] | None = None,
    source_pkg: Path | None = None,
    slim: bool = True,
    target_method: str | None = None,
    interval_geometry: str | None = None,
    model_search_steps: int | None = None,
    model_search_mode: str | None = None,
    model_search_adaptive_horizon: bool | None = None,
    model_search_adaptive_horizon_offset: int | None = None,
    model_search_min_overage_s: float | None = None,
    model_search_gamma: float | None = None,
    model_search_launch_prob_threshold: float | None = None,
    model_search_greedy_launch_threshold: float | None = None,
    model_search_branch_prob_threshold: float | None = None,
    model_search_max_branching: int | None = None,
    model_search_branch_after_first_env_step: bool | None = None,
    model_search_stop_at_turn_end: bool | None = None,
    model_search_turn_end_opponent_samples: int | None = None,
    model_search_turn_sampling_max_samples: int | None = None,
    population_member: int | None = None,
    population_member_4p: int | None = None,
    population_member_2p: int | None = None,
    policy_key_4p: str = "policy",
    policy_key_2p: str = "policy",
) -> Path:
    """Write a submission bundle directory; return its path."""

    if checkpoint_4p is None or checkpoint_2p is None:
        raise ValueError("Both 4p and 2p checkpoints must be resolved before packaging")
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
    if (
        slim
        or keep_member_4p is not None
        or keep_member_2p is not None
        or policy_key_4p != "policy"
        or policy_key_2p != "policy"
    ):
        _write_checkpoint(
            checkpoint_4p,
            bundle_dir / "checkpoint_4p.pt",
            slim=slim,
            keep_member=keep_member_4p,
            policy_key=policy_key_4p,
            include_student_policy=bool(use_student_for_search_4p),
        )
        _write_checkpoint(
            checkpoint_2p,
            bundle_dir / "checkpoint_2p.pt",
            slim=slim,
            keep_member=keep_member_2p,
            policy_key=policy_key_2p,
            include_student_policy=bool(use_student_for_search_2p),
        )
    else:
        shutil.copy2(checkpoint_4p, bundle_dir / "checkpoint_4p.pt")
        shutil.copy2(checkpoint_2p, bundle_dir / "checkpoint_2p.pt")
    _copy_inference_package(bundle_dir / "orbit_wars_pt", source_pkg=source_pkg)
    submission_env = dict(extra_env or {})
    if bool(use_student_for_search_4p):
        submission_env["ORBIT_WARS_USE_STUDENT_FOR_SEARCH_4P"] = "1"
    if bool(use_student_for_search_2p):
        submission_env["ORBIT_WARS_USE_STUDENT_FOR_SEARCH_2P"] = "1"
    submission_env["ORBIT_WARS_SEARCH_MAIN_POLICY_FOR_EGO_STEPS_4P"] = str(
        max(0, int(search_main_policy_for_ego_steps_4p))
    )
    submission_env["ORBIT_WARS_SEARCH_MAIN_POLICY_FOR_EGO_STEPS_2P"] = str(
        max(0, int(search_main_policy_for_ego_steps_2p))
    )
    if sampling_mode is not None:
        submission_env["ORBIT_WARS_SAMPLING_MODE"] = str(sampling_mode)
    if sampling_mode_4p is not None:
        submission_env["ORBIT_WARS_SAMPLING_MODE_4P"] = str(sampling_mode_4p)
    if sampling_mode_2p is not None:
        submission_env["ORBIT_WARS_SAMPLING_MODE_2P"] = str(sampling_mode_2p)
    if greedy_4p is not None:
        submission_env["ORBIT_WARS_GREEDY_4P"] = "1" if bool(greedy_4p) else "0"
    if greedy_2p is not None:
        submission_env["ORBIT_WARS_GREEDY_2P"] = "1" if bool(greedy_2p) else "0"
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
    if model_search_steps is not None:
        submission_env["ORBIT_WARS_MODEL_SEARCH_STEPS"] = str(max(0, int(model_search_steps)))
    if model_search_mode is not None:
        submission_env["ORBIT_WARS_MODEL_SEARCH_MODE"] = str(model_search_mode)
    if model_search_adaptive_horizon is not None:
        submission_env["ORBIT_WARS_MODEL_SEARCH_ADAPTIVE_HORIZON"] = (
            "1" if bool(model_search_adaptive_horizon) else "0"
        )
    if model_search_adaptive_horizon_offset is not None:
        submission_env["ORBIT_WARS_MODEL_SEARCH_ADAPTIVE_HORIZON_OFFSET"] = str(
            max(0, int(model_search_adaptive_horizon_offset))
        )
    if model_search_min_overage_s is not None:
        submission_env["ORBIT_WARS_MODEL_SEARCH_MIN_OVERAGE_S"] = str(
            max(0.0, float(model_search_min_overage_s))
        )
    if model_search_gamma is not None:
        submission_env["ORBIT_WARS_MODEL_SEARCH_GAMMA"] = str(float(model_search_gamma))
    if model_search_launch_prob_threshold is not None:
        submission_env["ORBIT_WARS_MODEL_SEARCH_LAUNCH_PROB_THRESHOLD"] = str(
            float(model_search_launch_prob_threshold)
        )
    if model_search_greedy_launch_threshold is not None:
        submission_env["ORBIT_WARS_MODEL_SEARCH_GREEDY_LAUNCH_THRESHOLD"] = str(
            float(model_search_greedy_launch_threshold)
        )
    if model_search_branch_prob_threshold is not None:
        submission_env["ORBIT_WARS_MODEL_SEARCH_BRANCH_PROB_THRESHOLD"] = str(
            float(model_search_branch_prob_threshold)
        )
    if model_search_max_branching is not None:
        submission_env["ORBIT_WARS_MODEL_SEARCH_MAX_BRANCHING"] = str(max(1, int(model_search_max_branching)))
    if model_search_branch_after_first_env_step is not None:
        submission_env["ORBIT_WARS_MODEL_SEARCH_BRANCH_AFTER_FIRST_ENV_STEP"] = (
            "1" if bool(model_search_branch_after_first_env_step) else "0"
        )
    if model_search_stop_at_turn_end is not None:
        submission_env["ORBIT_WARS_MODEL_SEARCH_STOP_AT_TURN_END"] = (
            "1" if bool(model_search_stop_at_turn_end) else "0"
        )
    if model_search_turn_end_opponent_samples is not None:
        submission_env["ORBIT_WARS_MODEL_SEARCH_TURN_END_OPPONENT_SAMPLES"] = str(
            max(0, int(model_search_turn_end_opponent_samples))
        )
    if model_search_turn_sampling_max_samples is not None:
        submission_env["ORBIT_WARS_MODEL_SEARCH_TURN_SAMPLING_MAX_SAMPLES"] = str(
            max(0, int(model_search_turn_sampling_max_samples))
        )

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
        4-player search uses embedded student: {bool(use_student_for_search_4p)}
        2-player search uses embedded student: {bool(use_student_for_search_2p)}
        4-player search ego main-model steps: {max(0, int(search_main_policy_for_ego_steps_4p))}
        2-player search ego main-model steps: {max(0, int(search_main_policy_for_ego_steps_2p))}
        Greedy (default): {greedy}
        Greedy 4p override: {greedy_4p if greedy_4p is not None else 'default'}
        Greedy 2p override: {greedy_2p if greedy_2p is not None else 'default'}
        Sampling mode (default): {sampling_mode if sampling_mode is not None else 'greedy fallback / adapter default'}
        Sampling mode 4p override: {sampling_mode_4p if sampling_mode_4p is not None else 'default'}
        Sampling mode 2p override: {sampling_mode_2p if sampling_mode_2p is not None else 'default'}
        Device: {device}
        CPU threads: 1
        Population member (fallback): {population_member if population_member is not None else 'member 0 / checkpoint default'}
        Population member 4p: {population_member_4p if population_member_4p is not None else (population_member if population_member is not None else 'member 0 / checkpoint default')}
        Population member 2p: {population_member_2p if population_member_2p is not None else (population_member if population_member is not None else 'member 0 / checkpoint default')}
        Target method: {target_method or 'checkpoint/default'}
        Interval geometry: {interval_geometry or 'checkpoint/default'}
        Model search steps: {model_search_steps if model_search_steps is not None else 'disabled / env default'}
        Model search adaptive horizon: {model_search_adaptive_horizon if model_search_adaptive_horizon is not None else 'env default'}
        Model search adaptive horizon offset: {model_search_adaptive_horizon_offset if model_search_adaptive_horizon_offset is not None else 'env default'}
        Model search min overage seconds: {model_search_min_overage_s if model_search_min_overage_s is not None else 'env default'}
        Model search gamma: {model_search_gamma if model_search_gamma is not None else 'checkpoint/default'}
        Model search launch probability threshold: {model_search_launch_prob_threshold if model_search_launch_prob_threshold is not None else 'env default'}
        Model search greedy launch threshold: {model_search_greedy_launch_threshold if model_search_greedy_launch_threshold is not None else 'env default'}
        Model search branch after first env step: {model_search_branch_after_first_env_step if model_search_branch_after_first_env_step is not None else 'env default'}
        Model search stop at turn end: {model_search_stop_at_turn_end if model_search_stop_at_turn_end is not None else 'env default'}
        Model search turn-end opponent samples: {model_search_turn_end_opponent_samples if model_search_turn_end_opponent_samples is not None else 'env default'}
        Model search turn-sampling max samples: {model_search_turn_sampling_max_samples if model_search_turn_sampling_max_samples is not None else 'env default'}

        Policy selection: 4-player matches use checkpoint_4p.pt; 2-player matches
        use checkpoint_2p.pt. The first observation selects the mode for the full
        episode; there is no mid-game policy switching. When enabled, search uses
        the embedded student model from the corresponding checkpoint, optionally
        keeping the main model on the ego seat for the first configured number of
        simulated search env steps.

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
        default=None,
        help="4-player FFA training checkpoint (.pt) shipped as checkpoint_4p.pt.",
    )
    parser.add_argument(
        "--checkpoint-2p",
        type=Path,
        default=None,
        help="2-player training checkpoint (.pt) shipped as checkpoint_2p.pt.",
    )
    parser.add_argument(
        "--search-use-student",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Default search-student setting for both packaged checkpoints. "
            "Per-mode flags override this."
        ),
    )
    parser.add_argument(
        "--use-student-for-search-4p",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use the packaged 4-player checkpoint's embedded student model for search rollouts.",
    )
    parser.add_argument(
        "--use-student-for-search-2p",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use the packaged 2-player checkpoint's embedded student model for search rollouts.",
    )
    parser.add_argument(
        "--search-main-model-ego-steps",
        type=int,
        default=1,
        help=(
            "Default number of simulated search env steps that keep the main model on the ego seat "
            "for both packaged checkpoints. Per-mode flags override this."
        ),
    )
    parser.add_argument(
        "--search-main-model-ego-steps-4p",
        type=int,
        default=None,
        help="How many simulated search env steps keep the packaged 4-player checkpoint's main model on the ego seat.",
    )
    parser.add_argument(
        "--search-main-model-ego-steps-2p",
        type=int,
        default=None,
        help="How many simulated search env steps keep the packaged 2-player checkpoint's main model on the ego seat.",
    )
    parser.add_argument(
        "--checkpoint-main",
        type=Path,
        default=None,
        help=(
            "One exploiter-mode training checkpoint whose main or exploiter policy should be used "
            "for --main-as-4p, --main-as-2p, --exploiter-as-4p, and/or --exploiter-as-2p."
        ),
    )
    parser.add_argument(
        "--main-as-4p",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the main policy from --checkpoint-main as the packaged 4p policy.",
    )
    parser.add_argument(
        "--main-as-2p",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the main policy from --checkpoint-main as the packaged 2p policy.",
    )
    parser.add_argument(
        "--exploiter-as-4p",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the exploiter policy from --checkpoint-main as the packaged 4p policy.",
    )
    parser.add_argument(
        "--exploiter-as-2p",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the exploiter policy from --checkpoint-main as the packaged 2p policy.",
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
        help="Use argmax actions in main.py by default for both policies.",
    )
    parser.add_argument(
        "--greedy-4p",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the packaged 4-player policy mode. Defaults to --greedy when omitted.",
    )
    parser.add_argument(
        "--greedy-2p",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the packaged 2-player policy mode. Defaults to --greedy when omitted.",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=SAMPLING_MODE_CHOICES,
        default=None,
        help=(
            "Bake ORBIT_WARS_SAMPLING_MODE into main.py for both policies. "
            "Choices: stochastic, greedy, mixed. Default leaves sampling mode unset so --greedy remains the fallback."
        ),
    )
    parser.add_argument(
        "--sampling-mode-4p",
        choices=SAMPLING_MODE_CHOICES,
        default=None,
        help="Override ORBIT_WARS_SAMPLING_MODE_4P in main.py. Defaults to --sampling-mode when omitted.",
    )
    parser.add_argument(
        "--sampling-mode-2p",
        choices=SAMPLING_MODE_CHOICES,
        default=None,
        help="Override ORBIT_WARS_SAMPLING_MODE_2P in main.py. Defaults to --sampling-mode when omitted.",
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
        "--model-search-steps",
        type=int,
        default=None,
        help=(
            "Bake ORBIT_WARS_MODEL_SEARCH_STEPS into main.py. "
            "Set >0 to enable fixed-horizon halt-vs-launch search."
        ),
    )
    parser.add_argument(
        "--model-search-mode",
        choices=("binary", "ego_bfs", "turn_sampling"),
        default=None,
        help=(
            "Bake ORBIT_WARS_MODEL_SEARCH_MODE into main.py. "
            "`binary` keeps the existing halt-vs-launch search; `ego_bfs` enables the breadth-first ego-only tree search; "
            "`turn_sampling` stochastically samples distinct current-turn action sequences with prefix caching."
        ),
    )
    parser.add_argument(
        "--model-search-adaptive-horizon",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Bake ORBIT_WARS_MODEL_SEARCH_ADAPTIVE_HORIZON into main.py. "
            "When enabled, rollout depth follows launch hit time plus the configured offset."
        ),
    )
    parser.add_argument(
        "--model-search-adaptive-horizon-offset",
        type=int,
        default=None,
        help=(
            "Bake ORBIT_WARS_MODEL_SEARCH_ADAPTIVE_HORIZON_OFFSET into main.py. "
            "Only used when adaptive horizon is enabled."
        ),
    )
    parser.add_argument(
        "--model-search-min-overage-s",
        type=float,
        default=None,
        help=(
            "Bake ORBIT_WARS_MODEL_SEARCH_MIN_OVERAGE_S into main.py. "
            "Search only runs when remaining Kaggle overage is at least this many seconds."
        ),
    )
    parser.add_argument(
        "--model-search-gamma",
        type=float,
        default=None,
        help=(
            "Bake ORBIT_WARS_MODEL_SEARCH_GAMMA into main.py. "
            "Overrides the reward discount used by the rollout search."
        ),
    )
    parser.add_argument(
        "--model-search-launch-prob-threshold",
        type=float,
        default=None,
        help=(
            "Bake ORBIT_WARS_MODEL_SEARCH_LAUNCH_PROB_THRESHOLD into main.py. "
            "Skip root halt-vs-launch search when the policy's launch probability is below this threshold."
        ),
    )
    parser.add_argument(
        "--model-search-greedy-launch-threshold",
        type=float,
        default=None,
        help=(
            "Bake ORBIT_WARS_MODEL_SEARCH_GREEDY_LAUNCH_THRESHOLD into main.py. "
            "Applies only inside greedy search continuations; for example 0.8 requires at least 80%% launch probability."
        ),
    )
    parser.add_argument(
        "--model-search-branch-prob-threshold",
        type=float,
        default=None,
        help=(
            "Bake ORBIT_WARS_MODEL_SEARCH_BRANCH_PROB_THRESHOLD into main.py. "
            "In ego_bfs mode, branch on ego choices whose probability is at least this threshold."
        ),
    )
    parser.add_argument(
        "--model-search-max-branching",
        type=int,
        default=None,
        help=(
            "Bake ORBIT_WARS_MODEL_SEARCH_MAX_BRANCHING into main.py. "
            "In ego_bfs mode, cap origin/fraction and target branching at this many choices per node."
        ),
    )
    parser.add_argument(
        "--model-search-branch-after-first-env-step",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Bake ORBIT_WARS_MODEL_SEARCH_BRANCH_AFTER_FIRST_ENV_STEP into main.py. "
            "Disable to branch only during the current turn and go greedy after the first simulated env step."
        ),
    )
    parser.add_argument(
        "--model-search-stop-at-turn-end",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Bake ORBIT_WARS_MODEL_SEARCH_STOP_AT_TURN_END into main.py. "
            "In ego_bfs mode, stop at end of current turn and score virtual turn-end states without env-step rollout."
        ),
    )
    parser.add_argument(
        "--model-search-turn-end-opponent-samples",
        type=int,
        default=None,
        help=(
            "Bake ORBIT_WARS_MODEL_SEARCH_TURN_END_OPPONENT_SAMPLES into main.py. "
            "In ego_bfs + stop-at-turn-end mode, average each current-turn leaf over this many shared sampled "
            "opponent joint-action sets and one env step; 0 keeps pure turn-end value scoring."
        ),
    )
    parser.add_argument(
        "--model-search-turn-sampling-max-samples",
        type=int,
        default=None,
        help=(
            "Bake ORBIT_WARS_MODEL_SEARCH_TURN_SAMPLING_MAX_SAMPLES into main.py. "
            "In turn_sampling mode, stop after this many completed ego turn plans have been sampled pre-dedup."
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

    checkpoint_4p = args.checkpoint_4p
    checkpoint_2p = args.checkpoint_2p
    policy_key_4p = "policy"
    policy_key_2p = "policy"
    if args.main_as_4p and args.exploiter_as_4p:
        raise SystemExit("use at most one of --main-as-4p and --exploiter-as-4p")
    if args.main_as_2p and args.exploiter_as_2p:
        raise SystemExit("use at most one of --main-as-2p and --exploiter-as-2p")
    if args.main_as_4p:
        if args.checkpoint_main is None:
            raise SystemExit("--main-as-4p requires --checkpoint-main")
        checkpoint_4p = args.checkpoint_main
    elif args.exploiter_as_4p:
        if args.checkpoint_main is None:
            raise SystemExit("--exploiter-as-4p requires --checkpoint-main")
        checkpoint_4p = args.checkpoint_main
        policy_key_4p = "exploiter_policy"
    if args.main_as_2p:
        if args.checkpoint_main is None:
            raise SystemExit("--main-as-2p requires --checkpoint-main")
        checkpoint_2p = args.checkpoint_main
    elif args.exploiter_as_2p:
        if args.checkpoint_main is None:
            raise SystemExit("--exploiter-as-2p requires --checkpoint-main")
        checkpoint_2p = args.checkpoint_main
        policy_key_2p = "exploiter_policy"
    if checkpoint_4p is None:
        raise SystemExit(
            "provide --checkpoint-4p, or --checkpoint-main with --main-as-4p or --exploiter-as-4p"
        )
    if checkpoint_2p is None:
        raise SystemExit(
            "provide --checkpoint-2p, or --checkpoint-main with --main-as-2p or --exploiter-as-2p"
        )
    if args.model_search_greedy_launch_threshold is not None and not (
        0.0 <= float(args.model_search_greedy_launch_threshold) <= 1.0
    ):
        raise SystemExit("--model-search-greedy-launch-threshold must be between 0 and 1")
    if args.model_search_launch_prob_threshold is not None and not (
        0.0 <= float(args.model_search_launch_prob_threshold) <= 1.0
    ):
        raise SystemExit("--model-search-launch-prob-threshold must be between 0 and 1")
    if args.model_search_branch_prob_threshold is not None and not (
        0.0 <= float(args.model_search_branch_prob_threshold) <= 1.0
    ):
        raise SystemExit("--model-search-branch-prob-threshold must be between 0 and 1")
    if args.model_search_turn_end_opponent_samples is not None and int(args.model_search_turn_end_opponent_samples) < 0:
        raise SystemExit("--model-search-turn-end-opponent-samples must be non-negative")
    if args.model_search_turn_sampling_max_samples is not None and int(args.model_search_turn_sampling_max_samples) < 0:
        raise SystemExit("--model-search-turn-sampling-max-samples must be non-negative")

    result = package_submission(
        checkpoint_4p,
        checkpoint_2p,
        args.out,
        use_student_for_search_4p=(
            bool(args.search_use_student)
            if args.use_student_for_search_4p is None
            else bool(args.use_student_for_search_4p)
        ),
        use_student_for_search_2p=(
            bool(args.search_use_student)
            if args.use_student_for_search_2p is None
            else bool(args.use_student_for_search_2p)
        ),
        search_main_policy_for_ego_steps_4p=max(
            0,
            int(
                args.search_main_model_ego_steps
                if args.search_main_model_ego_steps_4p is None
                else args.search_main_model_ego_steps_4p
            ),
        ),
        search_main_policy_for_ego_steps_2p=max(
            0,
            int(
                args.search_main_model_ego_steps
                if args.search_main_model_ego_steps_2p is None
                else args.search_main_model_ego_steps_2p
            ),
        ),
        greedy=bool(args.greedy),
        greedy_4p=(None if args.greedy_4p is None else bool(args.greedy_4p)),
        greedy_2p=(None if args.greedy_2p is None else bool(args.greedy_2p)),
        sampling_mode=args.sampling_mode,
        sampling_mode_4p=(args.sampling_mode if args.sampling_mode_4p is None else args.sampling_mode_4p),
        sampling_mode_2p=(args.sampling_mode if args.sampling_mode_2p is None else args.sampling_mode_2p),
        device=str(args.device),
        slim=not args.no_slim,
        target_method=args.target_method,
        interval_geometry=args.interval_geometry,
        model_search_steps=args.model_search_steps,
        model_search_mode=args.model_search_mode,
        model_search_adaptive_horizon=(
            None
            if args.model_search_adaptive_horizon is None
            else bool(args.model_search_adaptive_horizon)
        ),
        model_search_adaptive_horizon_offset=args.model_search_adaptive_horizon_offset,
        model_search_min_overage_s=args.model_search_min_overage_s,
        model_search_gamma=args.model_search_gamma,
        model_search_launch_prob_threshold=args.model_search_launch_prob_threshold,
        model_search_greedy_launch_threshold=args.model_search_greedy_launch_threshold,
        model_search_branch_prob_threshold=args.model_search_branch_prob_threshold,
        model_search_max_branching=args.model_search_max_branching,
        model_search_branch_after_first_env_step=args.model_search_branch_after_first_env_step,
        model_search_stop_at_turn_end=args.model_search_stop_at_turn_end,
        model_search_turn_end_opponent_samples=args.model_search_turn_end_opponent_samples,
        model_search_turn_sampling_max_samples=args.model_search_turn_sampling_max_samples,
        population_member=args.member,
        population_member_4p=args.member if args.member_4p is None else args.member_4p,
        population_member_2p=args.member if args.member_2p is None else args.member_2p,
        policy_key_4p=policy_key_4p,
        policy_key_2p=policy_key_2p,
    )

    bundle_dir, archive_path = _submission_paths(args.out.expanduser().resolve())
    if archive_path is not None and not args.keep_dir and bundle_dir.exists():
        shutil.rmtree(bundle_dir)

    size_mb = result.stat().st_size / (1024 * 1024)
    print(f"Wrote {result} ({size_mb:.1f} MiB)")


if __name__ == "__main__":
    main()

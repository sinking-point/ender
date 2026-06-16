"""Policy distillation via teacher rollouts and supervised student training."""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Must run before any orbit_wars_pt import that transitively imports jax.
import orbit_wars_pt.xla_env  # noqa: F401

from orbit_wars_pt.batched_env import heal_terminal_env_slices
from orbit_wars_pt.compressed_observation import CompressedObservationBuffer
from orbit_wars_pt.constants import FRACTIONS, MAX_PLANETS, obs_feature_dim_for_num_agents
from orbit_wars_pt.env_wrapper import OrbitWarsEnvConfig
from orbit_wars_pt.jax_setup import configure_jax_for_training
from orbit_wars_pt.kaggle_adapter import _strip_legacy_pair_head_keys
from orbit_wars_pt.model import OrbitWarsPolicy, adapt_checkpoint_state_for_model, infer_value_head_count_from_state_dict
from orbit_wars_pt.parallel_rollout import RolloutCarry, RolloutSegment, collect_parallel_micro_rollouts, make_device_reset_bank
from orbit_wars_pt.reset_prefetch import RolloutResetPrefetch
from orbit_wars_pt.torch_replay import select_stored_compressed_minibatch_torch
from orbit_wars_pt.train_ppo import (
    CHECKPOINT_VERSION as PPO_CHECKPOINT_VERSION,
    _deserialize_rollout_carry,
    _parse_fraction_init_weights,
    _serialize_rollout_carry,
    experiment_dirs,
    find_latest_checkpoint,
    resolve_reward_mix,
    resolve_member_reward_mix,
)


DISTILL_CHECKPOINT_VERSION = 1


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _teacher_training_args(ckpt: dict[str, Any]) -> dict[str, Any]:
    args = ckpt.get("training_args")
    if not isinstance(args, dict):
        raise RuntimeError("teacher checkpoint missing training_args")
    return args


def _teacher_policy_from_checkpoint(path: Path, device: torch.device) -> tuple[OrbitWarsPolicy, dict[str, Any]]:
    ckpt = _load_checkpoint(path)
    version = int(ckpt.get("version", 0))
    if version not in (6, PPO_CHECKPOINT_VERSION):
        raise RuntimeError(
            f"unsupported teacher checkpoint version {ckpt.get('version')!r}; expected 6 or {PPO_CHECKPOINT_VERSION}"
        )
    train_args = _teacher_training_args(ckpt)
    target_abort_enabled = bool(train_args.get("target_abort_enabled", False))
    num_agents = int(train_args.get("num_agents", 2))
    exploiter_mode = bool(train_args.get("exploiter_mode", False))
    feature_agents = 4 if exploiter_mode else num_agents
    rope_dims = int(train_args.get("rope_dims", 2))
    value_head_count = int(
        train_args.get(
            "value_head_count",
            infer_value_head_count_from_state_dict(ckpt["policy"]),
        )
    )
    policy = OrbitWarsPolicy(
        d_model=int(train_args["d_model"]),
        n_heads=int(train_args["n_heads"]),
        n_layers=int(train_args["n_layers"]),
        activation_checkpointing=False,
        feature_dim=obs_feature_dim_for_num_agents(feature_agents, target_abort_enabled=target_abort_enabled),
        population_size=int(train_args.get("population_size", 1)),
        rope_dims=rope_dims,
        value_head_count=value_head_count,
        disjoint_actor_critic=bool(train_args.get("disjoint_actor_critic", False)),
        target_abort_enabled=target_abort_enabled,
        halt_init_prob=train_args.get("halt_init_prob"),
        fraction_init_weights=_parse_fraction_init_weights(train_args.get("fraction_init_ratio")),
    ).to(device)
    state, _ = adapt_checkpoint_state_for_model(ckpt["policy"], policy)
    state = _strip_legacy_pair_head_keys(state)
    policy.load_state_dict(state)
    policy.eval()
    for param in policy.parameters():
        param.requires_grad_(False)
    return policy, train_args


def _student_policy_from_args(args: argparse.Namespace, teacher_args: dict[str, Any], device: torch.device) -> OrbitWarsPolicy:
    target_abort_enabled = bool(teacher_args.get("target_abort_enabled", False))
    num_agents = int(teacher_args.get("num_agents", 2))
    exploiter_mode = bool(teacher_args.get("exploiter_mode", False))
    feature_agents = 4 if exploiter_mode else num_agents
    population_size = int(teacher_args.get("population_size", 1))
    value_head_count = int(teacher_args.get("value_head_count", 1))
    fraction_init_raw = args.fraction_init_ratio
    if fraction_init_raw is None:
        fraction_init_raw = teacher_args.get("fraction_init_ratio")
    return OrbitWarsPolicy(
        d_model=int(args.d_model if args.d_model is not None else teacher_args["d_model"]),
        n_heads=int(args.n_heads if args.n_heads is not None else teacher_args["n_heads"]),
        n_layers=int(args.n_layers if args.n_layers is not None else teacher_args["n_layers"]),
        activation_checkpointing=bool(args.activation_checkpointing),
        feature_dim=obs_feature_dim_for_num_agents(feature_agents, target_abort_enabled=target_abort_enabled),
        population_size=population_size,
        rope_dims=int(args.rope_dims if args.rope_dims is not None else teacher_args.get("rope_dims", 2)),
        value_head_count=value_head_count,
        disjoint_actor_critic=bool(teacher_args.get("disjoint_actor_critic", False)),
        target_abort_enabled=target_abort_enabled,
        halt_init_prob=(args.halt_init_prob if args.halt_init_prob is not None else teacher_args.get("halt_init_prob")),
        fraction_init_weights=_parse_fraction_init_weights(fraction_init_raw),
    ).to(device)


def _teacher_env_config(teacher_args: dict[str, Any]) -> OrbitWarsEnvConfig:
    teacher_ns = argparse.Namespace(**teacher_args)
    (
        reward_ship_mass_share_coef,
        reward_production_share_coef,
        reward_terminal_win_loss_coef,
        reward_time_bonus_coef,
    ) = resolve_reward_mix(teacher_ns)
    (
        reward_ship_mass_share_member_coefs,
        reward_production_share_member_coefs,
        reward_terminal_win_loss_member_coefs,
        reward_time_bonus_member_coefs,
    ) = resolve_member_reward_mix(teacher_ns, int(teacher_args.get("population_size", 1)))
    return OrbitWarsEnvConfig(
        num_agents=int(teacher_args["num_agents"]),
        max_fleets=int(teacher_args["max_fleets"]),
        episode_seed=int(teacher_args["seed"]),
        reward_mode=str(teacher_args["reward_mode"]),
        reward_ship_mass_share_coef=reward_ship_mass_share_coef,
        reward_ship_mass_share_member_coefs=reward_ship_mass_share_member_coefs,
        reward_production_share_coef=reward_production_share_coef,
        reward_production_share_member_coefs=reward_production_share_member_coefs,
        reward_terminal_win_loss_coef=reward_terminal_win_loss_coef,
        reward_terminal_win_loss_member_coefs=reward_terminal_win_loss_member_coefs,
        reward_terminal_loss=float(teacher_args["reward_terminal_loss"]),
        reward_terminal_draw=float(teacher_args["reward_terminal_draw"]),
        reward_terminal_win=float(teacher_args["reward_terminal_win"]),
        reward_time_bonus_coef=reward_time_bonus_coef,
        reward_time_bonus_member_coefs=reward_time_bonus_member_coefs,
        normalize_obs_to_p0=bool(teacher_args.get("normalize_obs_to_p0", False)),
    )


def _compile_policy_modules(policy: OrbitWarsPolicy, helper_compile_mode: str) -> None:
    policy.forward_dense_rollout_compressed = torch.compile(  # type: ignore[assignment]
        policy.forward_dense_rollout_compressed, mode=helper_compile_mode, dynamic=True
    )
    policy.target_logits_for_origin_fraction = torch.compile(  # type: ignore[assignment]
        policy.target_logits_for_origin_fraction, mode=helper_compile_mode, dynamic=True
    )


def _resolved_checkpoint_training_args(args: argparse.Namespace, teacher_args: dict[str, Any]) -> dict[str, Any]:
    """Persist concrete student/runtime settings rather than raw CLI ``None`` values."""

    resolved = dict(teacher_args)
    resolved.update(vars(args).copy())
    resolved["d_model"] = int(args.d_model if args.d_model is not None else teacher_args["d_model"])
    resolved["n_heads"] = int(args.n_heads if args.n_heads is not None else teacher_args["n_heads"])
    resolved["n_layers"] = int(args.n_layers if args.n_layers is not None else teacher_args["n_layers"])
    resolved["rope_dims"] = int(args.rope_dims if args.rope_dims is not None else teacher_args.get("rope_dims", 2))
    resolved["halt_init_prob"] = (
        args.halt_init_prob if args.halt_init_prob is not None else teacher_args.get("halt_init_prob")
    )
    resolved["fraction_init_ratio"] = (
        args.fraction_init_ratio if args.fraction_init_ratio is not None else teacher_args.get("fraction_init_ratio")
    )
    for key in (
        "num_envs",
        "max_micro_steps",
        "rollout_micro_horizon",
        "ship_speed",
        "first_hit_n_rays",
        "first_hit_ray_chunk_size",
        "first_hit_method",
        "micro_step_penalty",
        "earlygame_env_turn_limit",
    ):
        if resolved.get(key) is None and key in teacher_args:
            resolved[key] = teacher_args[key]
    return resolved


def _save_checkpoint(
    path: Path,
    *,
    iteration: int,
    student: OrbitWarsPolicy,
    optimizer: torch.optim.Optimizer,
    rng: torch.Generator,
    rnd: np.random.Generator,
    rollout_env_seed: int,
    rollout_carry: Optional[RolloutCarry],
    args: argparse.Namespace,
    teacher_args: dict[str, Any],
    teacher_checkpoint: str,
) -> None:
    student_state = student.state_dict()
    payload = {
        "version": DISTILL_CHECKPOINT_VERSION,
        "iteration": int(iteration),
        "student": student_state,
        "policy": student_state,
        "optimizer": optimizer.state_dict(),
        "torch_rng": rng.get_state(),
        "numpy_rng_state": rnd.bit_generator.state,
        "rollout_env_seed": int(rollout_env_seed),
        "rollout_carry": _serialize_rollout_carry(rollout_carry) if rollout_carry is not None else None,
        "teacher_checkpoint": str(teacher_checkpoint),
        "training_args": _resolved_checkpoint_training_args(args, teacher_args),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _flatten_segment_indices(segment: RolloutSegment) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    player_all: list[np.ndarray] = []
    t_all: list[np.ndarray] = []
    n_all: list[np.ndarray] = []
    for player, valid in enumerate(segment.valid):
        t_idx, n_idx = np.nonzero(valid)
        if t_idx.size == 0:
            continue
        player_all.append(np.full((t_idx.size,), player, dtype=np.int32))
        t_all.append(t_idx.astype(np.int32, copy=False))
        n_all.append(n_idx.astype(np.int32, copy=False))
    if not player_all:
        empty = np.zeros((0,), dtype=np.int32)
        return empty, empty, empty
    return (
        np.concatenate(player_all, axis=0),
        np.concatenate(t_all, axis=0),
        np.concatenate(n_all, axis=0),
    )


def _masked_kl_from_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    *,
    temperature: float,
    row_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, int]:
    if row_mask is None:
        row_valid = torch.ones((student_logits.shape[0],), device=student_logits.device, dtype=torch.bool)
    else:
        row_valid = row_mask.to(device=student_logits.device, dtype=torch.bool)
    if valid_mask is not None:
        mask = valid_mask.to(device=student_logits.device, dtype=torch.bool)
        row_valid = row_valid & mask.any(dim=-1)
        if not bool(row_valid.any().item()):
            return student_logits.sum() * 0.0, 0
        mask = mask[row_valid]
        student_logits = student_logits[row_valid].masked_fill(~mask, -1e4)
        teacher_logits = teacher_logits[row_valid].masked_fill(~mask, -1e4)
    else:
        if not bool(row_valid.any().item()):
            return student_logits.sum() * 0.0, 0
        student_logits = student_logits[row_valid]
        teacher_logits = teacher_logits[row_valid]
    t = float(temperature)
    student_lp = torch.log_softmax(student_logits / t, dim=-1)
    teacher_p = torch.softmax(teacher_logits / t, dim=-1)
    loss = F.kl_div(student_lp, teacher_p, reduction="batchmean") * (t * t)
    return loss, int(student_logits.shape[0])


def _masked_top1_agreement(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    *,
    row_mask: Optional[torch.Tensor] = None,
) -> tuple[int, int]:
    if row_mask is None:
        row_valid = torch.ones((student_logits.shape[0],), device=student_logits.device, dtype=torch.bool)
    else:
        row_valid = row_mask.to(device=student_logits.device, dtype=torch.bool)
    if valid_mask is not None:
        mask = valid_mask.to(device=student_logits.device, dtype=torch.bool)
        row_valid = row_valid & mask.any(dim=-1)
        if not bool(row_valid.any().item()):
            return 0, 0
        mask = mask[row_valid]
        student_logits = student_logits[row_valid].masked_fill(~mask, -1e4)
        teacher_logits = teacher_logits[row_valid].masked_fill(~mask, -1e4)
    else:
        if not bool(row_valid.any().item()):
            return 0, 0
        student_logits = student_logits[row_valid]
        teacher_logits = teacher_logits[row_valid]
    student_top = student_logits.argmax(dim=-1)
    teacher_top = teacher_logits.argmax(dim=-1)
    matches = int((student_top == teacher_top).sum().item())
    rows = int(student_logits.shape[0])
    return matches, rows


def _distill_minibatch(
    *,
    teacher: OrbitWarsPolicy,
    student: OrbitWarsPolicy,
    comp: CompressedObservationBuffer,
    actions: dict[str, torch.Tensor],
    temperature: float,
    halt_coef: float,
    origin_frac_coef: float,
    target_coef: float,
    value_coef: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = next(student.parameters()).device
    feature_dim = int(student.feat_proj.in_features)
    pop_idx = actions["population_idx"].to(device=device, dtype=torch.long)
    value_head_idx = actions["value_head_idx"].to(device=device, dtype=torch.long)

    comp_dev = CompressedObservationBuffer(
        token_meta=comp.token_meta.to(device),
        owner_idx=comp.owner_idx.to(device),
        production=comp.production.to(device),
        ships=comp.ships.to(device),
        velocity=comp.velocity.to(device),
        xy=comp.xy.to(device),
        turn_progress=comp.turn_progress.to(device),
        incoming_net=comp.incoming_net.to(device),
        incoming_survivor=comp.incoming_survivor.to(device),
        origin_frac_blocked=comp.origin_frac_blocked.to(device),
    )
    with torch.no_grad():
        teacher_out = teacher.forward_dense_rollout_compressed(
            comp_dev.token_meta,
            comp_dev.owner_idx,
            comp_dev.production,
            comp_dev.ships,
            comp_dev.velocity,
            comp_dev.xy,
            comp_dev.turn_progress,
            comp_dev.incoming_net,
            comp_dev.incoming_survivor,
            feature_dim,
            origin_frac_blocked=comp_dev.origin_frac_blocked,
            population_idx=pop_idx,
            value_head_idx=value_head_idx,
        )
    student_out = student.forward_dense_rollout_compressed(
        comp_dev.token_meta,
        comp_dev.owner_idx,
        comp_dev.production,
        comp_dev.ships,
        comp_dev.velocity,
        comp_dev.xy,
        comp_dev.turn_progress,
        comp_dev.incoming_net,
        comp_dev.incoming_survivor,
        feature_dim,
        origin_frac_blocked=comp_dev.origin_frac_blocked,
        population_idx=pop_idx,
        value_head_idx=value_head_idx,
    )

    metrics: dict[str, float] = {}
    total_loss = student_out["value"].sum() * 0.0

    halt_row_mask = ~actions["must_halt_no_ships"].to(device=device, dtype=torch.bool)
    halt_loss, halt_rows = _masked_kl_from_logits(
        student_out["halt_logits"],
        teacher_out["halt_logits"],
        None,
        temperature=temperature,
        row_mask=halt_row_mask,
    )
    total_loss = total_loss + float(halt_coef) * halt_loss
    metrics["loss_halt"] = float(halt_loss.detach().item()) if halt_rows > 0 else 0.0
    metrics["rows_halt"] = float(halt_rows)
    halt_matches, _halt_acc_rows = _masked_top1_agreement(
        student_out["halt_logits"],
        teacher_out["halt_logits"],
        None,
        row_mask=halt_row_mask,
    )
    metrics["top1_halt_matches"] = float(halt_matches)
    metrics["top1_halt_rows"] = float(halt_rows)

    bsz = int(actions["pair_flat"].shape[0])
    flat_mask_teacher = teacher_out["origin_frac_mask"].flatten(start_dim=1)
    flat_mask_student = student_out["origin_frac_mask"].flatten(start_dim=1)
    flat_mask = flat_mask_teacher & flat_mask_student
    origin_row_mask = (~actions["must_halt_no_ships"].to(device=device, dtype=torch.bool)) & flat_mask_teacher.any(dim=-1)
    origin_loss, origin_rows = _masked_kl_from_logits(
        student_out["origin_frac_logits"].flatten(start_dim=1),
        teacher_out["origin_frac_logits"].flatten(start_dim=1),
        flat_mask,
        temperature=temperature,
        row_mask=origin_row_mask,
    )
    total_loss = total_loss + float(origin_frac_coef) * origin_loss
    metrics["loss_origin_frac"] = float(origin_loss.detach().item()) if origin_rows > 0 else 0.0
    metrics["rows_origin_frac"] = float(origin_rows)
    origin_matches, _origin_acc_rows = _masked_top1_agreement(
        student_out["origin_frac_logits"].flatten(start_dim=1),
        teacher_out["origin_frac_logits"].flatten(start_dim=1),
        flat_mask,
        row_mask=origin_row_mask,
    )
    metrics["top1_origin_frac_matches"] = float(origin_matches)
    metrics["top1_origin_frac_rows"] = float(origin_rows)

    pair_flat = actions["pair_flat"].to(device=device, dtype=torch.long)
    frac_idx = actions["frac_idx"].to(device=device, dtype=torch.long)
    origin_idx = torch.div(pair_flat, MAX_PLANETS, rounding_mode="floor")
    batch_idx = torch.arange(bsz, device=device)
    ships = comp_dev.ships.to(torch.float32)
    origin_ships = ships[batch_idx, origin_idx].clamp_min(0.0)
    frac_values = origin_ships.new_tensor(FRACTIONS)
    fleet_size = torch.floor(origin_ships * frac_values[frac_idx])
    target_hit_tick = actions["target_hit_tick"].to(device=device, dtype=torch.float32)
    target_reachable = actions["target_planet_reachable"].to(device=device, dtype=torch.bool)
    planet_ships = ships.to(torch.float32)

    with torch.no_grad():
        teacher_target_logits = teacher.target_logits_for_origin_fraction(
            teacher_out["planet_hidden"],
            origin_idx,
            frac_idx,
            fleet_size=fleet_size,
            target_eta=target_hit_tick,
            target_ships=planet_ships,
            population_idx=pop_idx,
        )
    student_target_logits = student.target_logits_for_origin_fraction(
        student_out["planet_hidden"],
        origin_idx,
        frac_idx,
        fleet_size=fleet_size,
        target_eta=target_hit_tick,
        target_ships=planet_ships,
        population_idx=pop_idx,
    )
    pair_mask_teacher = teacher_out["pair_mask"][batch_idx, origin_idx, :]
    pair_mask_student = student_out["pair_mask"][batch_idx, origin_idx, :]
    target_mask = pair_mask_teacher & pair_mask_student & target_reachable
    target_row_mask = flat_mask_teacher.any(dim=-1) & target_mask.any(dim=-1)

    if teacher.target_abort_enabled:
        teacher_abort = teacher_out["abort_logits"][batch_idx, origin_idx, frac_idx]
        student_abort = student_out["abort_logits"][batch_idx, origin_idx, frac_idx]
        teacher_combined = torch.cat([teacher_target_logits, teacher_abort[:, None]], dim=-1)
        student_combined = torch.cat([student_target_logits, student_abort[:, None]], dim=-1)
        combined_mask = torch.cat(
            [target_mask, torch.ones((bsz, 1), dtype=torch.bool, device=device)],
            dim=-1,
        )
        target_loss, target_rows = _masked_kl_from_logits(
            student_combined,
            teacher_combined,
            combined_mask,
            temperature=temperature,
            row_mask=target_row_mask,
        )
        target_matches, _target_acc_rows = _masked_top1_agreement(
            student_combined,
            teacher_combined,
            combined_mask,
            row_mask=target_row_mask,
        )
    else:
        target_loss, target_rows = _masked_kl_from_logits(
            student_target_logits,
            teacher_target_logits,
            target_mask,
            temperature=temperature,
            row_mask=target_row_mask,
        )
        target_matches, _target_acc_rows = _masked_top1_agreement(
            student_target_logits,
            teacher_target_logits,
            target_mask,
            row_mask=target_row_mask,
        )
    total_loss = total_loss + float(target_coef) * target_loss
    metrics["loss_target"] = float(target_loss.detach().item()) if target_rows > 0 else 0.0
    metrics["rows_target"] = float(target_rows)
    metrics["top1_target_matches"] = float(target_matches)
    metrics["top1_target_rows"] = float(target_rows)

    if float(value_coef) != 0.0:
        value_loss = F.mse_loss(student_out["value"], teacher_out["value"])
        total_loss = total_loss + float(value_coef) * value_loss
        metrics["loss_value"] = float(value_loss.detach().item())
    else:
        metrics["loss_value"] = 0.0

    metrics["loss_total"] = float(total_loss.detach().item())
    return total_loss, metrics


def train(args: argparse.Namespace) -> None:
    configure_jax_for_training(prefer_gpu=True, verbose=True)

    teacher_path = Path(args.teacher_checkpoint)
    if not teacher_path.is_absolute():
        teacher_path = (Path.cwd() / teacher_path).resolve()
    if not teacher_path.is_file():
        raise SystemExit(f"teacher checkpoint not found: {teacher_path}")

    exp_dir, tb_dir, ckpt_dir = experiment_dirs(args)
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[orbit_wars_pt] experiment dir {exp_dir}", flush=True)
    print(f"[orbit_wars_pt] tensorboard run dir {tb_dir}", flush=True)

    device = torch.device(args.device)
    configure_jax_for_training(prefer_gpu=(device.type == "cuda"), verbose=False)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    import torch._dynamo as torch_dynamo

    torch_dynamo.config.capture_scalar_outputs = True

    amp_dtype: Optional[torch.dtype] = torch.bfloat16 if (args.amp and device.type == "cuda") else None

    teacher, teacher_args = _teacher_policy_from_checkpoint(teacher_path, device)
    if bool(teacher_args.get("exploiter_mode", False)):
        raise SystemExit("teacher checkpoints from --exploiter-mode are not supported yet")

    student = _student_policy_from_args(args, teacher_args, device)
    optimizer = optim.Adam(student.parameters(), lr=args.lr)

    if args.compile:
        helper_compile_mode = "default" if args.compile_mode == "reduce-overhead" else args.compile_mode
        _compile_policy_modules(student, helper_compile_mode)
        _compile_policy_modules(teacher, helper_compile_mode)

    env_cfg = _teacher_env_config(teacher_args)
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)
    rnd = np.random.default_rng(args.seed)
    rollout_env_seed = int(args.seed)
    rollout_carry: Optional[RolloutCarry] = None
    start_iter = 0

    resume_path: Optional[Path] = None
    if args.resume:
        resume_path = find_latest_checkpoint(ckpt_dir)
    if resume_path is not None:
        ckpt = _load_checkpoint(resume_path)
        if int(ckpt.get("version", 0)) != DISTILL_CHECKPOINT_VERSION:
            raise RuntimeError(f"unsupported distill checkpoint version {ckpt.get('version')!r}")
        if str(ckpt.get("teacher_checkpoint")) != str(teacher_path):
            raise RuntimeError(
                "resume teacher checkpoint mismatch; use a fresh experiment or the same --teacher-checkpoint"
            )
        student.load_state_dict(ckpt["student"])
        optimizer.load_state_dict(ckpt["optimizer"])
        rng.set_state(ckpt["torch_rng"])
        rnd.bit_generator.state = ckpt["numpy_rng_state"]
        rollout_env_seed = int(ckpt["rollout_env_seed"])
        rc = ckpt.get("rollout_carry")
        rollout_carry = _deserialize_rollout_carry(rc) if rc is not None else None
        if rollout_carry is not None:
            heal_sb, heal_seeds, heal_et = heal_terminal_env_slices(
                rollout_carry.state_b,
                rollout_carry.cfg,
                rollout_carry.episode_turns,
                rollout_env_seed,
            )
            rollout_carry = RolloutCarry(
                state_b=heal_sb,
                cfg=rollout_carry.cfg,
                episode_turns=heal_et,
                player_done=rollout_carry.player_done,
                population_assignments=rollout_carry.population_assignments,
                policy_row_for_seat=rollout_carry.policy_row_for_seat,
                controller_assignments=rollout_carry.controller_assignments,
                main_player_mask=rollout_carry.main_player_mask,
                env_mode_by_env=rollout_carry.env_mode_by_env,
                pending_exploiter_terminal=rollout_carry.pending_exploiter_terminal,
            )
            rollout_env_seed += int(heal_seeds)
        start_iter = int(ckpt["iteration"])
        print(f"[orbit_wars_pt] resumed at iteration {start_iter}", flush=True)

    reset_prefetch: Optional[RolloutResetPrefetch] = None
    if int(args.reset_prefetch_depth) > 0:
        reset_prefetch = RolloutResetPrefetch(int(args.reset_prefetch_workers), int(args.reset_prefetch_depth))
        reset_prefetch.start()
    rollout_device_reset_bank = (
        None if reset_prefetch is None else make_device_reset_bank(int(args.reset_prefetch_depth))
    )

    writer = SummaryWriter(log_dir=str(tb_dir))
    amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype is not None else nullcontext()
    teacher.eval()

    try:
        for iteration in range(start_iter, start_iter + int(args.iterations)):
            student.train()
            segments: list[RolloutSegment] = []
            total_samples = 0
            for _chunk in range(int(args.rollout_host_chunks)):
                segment, _timing, rollout_carry, seeds_used, _game_stats = collect_parallel_micro_rollouts(
                    teacher,
                    env_cfg,
                    int(args.num_envs if args.num_envs is not None else teacher_args["num_envs"]),
                    device,
                    seed_base=rollout_env_seed,
                    rng=rng,
                    greedy=bool(args.greedy_teacher),
                    ship_speed=float(args.ship_speed if args.ship_speed is not None else teacher_args["ship_speed"]),
                    max_micro_steps_per_player=int(
                        args.max_micro_steps if args.max_micro_steps is not None else teacher_args["max_micro_steps"]
                    ),
                    rollout_micro_horizon=int(
                        args.rollout_micro_horizon
                        if args.rollout_micro_horizon is not None
                        else teacher_args["rollout_micro_horizon"]
                    ),
                    carry_in=rollout_carry,
                    amp_dtype=amp_dtype,
                    reset_prefetch=reset_prefetch,
                    device_reset_bank=rollout_device_reset_bank,
                    first_hit_n_rays=int(
                        args.first_hit_n_rays if args.first_hit_n_rays is not None else teacher_args["first_hit_n_rays"]
                    ),
                    first_hit_ray_chunk_size=int(
                        args.first_hit_ray_chunk_size
                        if args.first_hit_ray_chunk_size is not None
                        else teacher_args.get("first_hit_ray_chunk_size", 0)
                    ),
                    first_hit_env_chunk_size=int(args.first_hit_env_chunk_size),
                    first_hit_method=str(
                        args.first_hit_method if args.first_hit_method is not None else teacher_args["first_hit_method"]
                    ),
                    micro_step_penalty=float(
                        args.micro_step_penalty
                        if args.micro_step_penalty is not None
                        else teacher_args["micro_step_penalty"]
                    ),
                    sync_policy_timing=bool(args.sync_rollout_timing),
                    earlygame_env_turn_limit=int(
                        args.earlygame_env_turn_limit
                        if args.earlygame_env_turn_limit is not None
                        else teacher_args.get("earlygame_env_turn_limit", 0)
                    ),
                )
                rollout_env_seed += int(seeds_used)
                player_idx, t_idx, n_idx = _flatten_segment_indices(segment)
                total_samples += int(player_idx.size)
                segments.append(segment)

            if total_samples == 0:
                print("[orbit_wars_pt] rollout produced zero valid samples; skipping iteration", flush=True)
                continue

            stat_sums = {
                "loss_total": 0.0,
                "loss_halt": 0.0,
                "loss_origin_frac": 0.0,
                "loss_target": 0.0,
                "loss_value": 0.0,
                "rows_halt": 0.0,
                "rows_origin_frac": 0.0,
                "rows_target": 0.0,
                "top1_halt_matches": 0.0,
                "top1_halt_rows": 0.0,
                "top1_origin_frac_matches": 0.0,
                "top1_origin_frac_rows": 0.0,
                "top1_target_matches": 0.0,
                "top1_target_rows": 0.0,
            }
            update_count = 0

            for _epoch in range(int(args.epochs)):
                for segment in segments:
                    player_idx, t_idx, n_idx = _flatten_segment_indices(segment)
                    if player_idx.size == 0:
                        continue
                    perm = rnd.permutation(player_idx.size)
                    player_idx = player_idx[perm]
                    t_idx = t_idx[perm]
                    n_idx = n_idx[perm]
                    for start in range(0, int(player_idx.size), int(args.minibatch_size)):
                        stop = min(start + int(args.minibatch_size), int(player_idx.size))
                        mb_player = player_idx[start:stop]
                        mb_t = t_idx[start:stop]
                        mb_n = n_idx[start:stop]
                        comp, actions = select_stored_compressed_minibatch_torch(
                            segment,
                            mb_player,
                            mb_t,
                            mb_n,
                            replay_device=device,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        with amp_ctx:
                            loss, metrics = _distill_minibatch(
                                teacher=teacher,
                                student=student,
                                comp=comp,
                                actions=actions,
                                temperature=float(args.temperature),
                                halt_coef=float(args.halt_coef),
                                origin_frac_coef=float(args.origin_frac_coef),
                                target_coef=float(args.target_coef),
                                value_coef=float(args.value_coef),
                            )
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(student.parameters(), float(args.max_grad_norm))
                        optimizer.step()
                        update_count += 1
                        for key in stat_sums:
                            stat_sums[key] += float(metrics.get(key, 0.0))

            denom = max(1, update_count)
            halt_top1 = stat_sums["top1_halt_matches"] / max(1.0, stat_sums["top1_halt_rows"])
            origin_top1 = stat_sums["top1_origin_frac_matches"] / max(1.0, stat_sums["top1_origin_frac_rows"])
            target_top1 = stat_sums["top1_target_matches"] / max(1.0, stat_sums["top1_target_rows"])
            print(
                f"[orbit_wars_pt] iter={iteration + 1} samples={total_samples} updates={update_count} "
                f"loss={stat_sums['loss_total'] / denom:.5f} "
                f"halt={stat_sums['loss_halt'] / denom:.5f} "
                f"origin_frac={stat_sums['loss_origin_frac'] / denom:.5f} "
                f"target={stat_sums['loss_target'] / denom:.5f} "
                f"value={stat_sums['loss_value'] / denom:.5f} "
                f"acc_halt={halt_top1:.3f} "
                f"acc_origin_frac={origin_top1:.3f} "
                f"acc_target={target_top1:.3f}",
                flush=True,
            )
            writer.add_scalar("distill/loss_total", stat_sums["loss_total"] / denom, iteration + 1)
            writer.add_scalar("distill/loss_halt", stat_sums["loss_halt"] / denom, iteration + 1)
            writer.add_scalar("distill/loss_origin_frac", stat_sums["loss_origin_frac"] / denom, iteration + 1)
            writer.add_scalar("distill/loss_target", stat_sums["loss_target"] / denom, iteration + 1)
            writer.add_scalar("distill/loss_value", stat_sums["loss_value"] / denom, iteration + 1)
            writer.add_scalar("distill/top1_halt_accuracy", halt_top1, iteration + 1)
            writer.add_scalar("distill/top1_origin_frac_accuracy", origin_top1, iteration + 1)
            writer.add_scalar("distill/top1_target_accuracy", target_top1, iteration + 1)
            writer.add_scalar("distill/samples", total_samples, iteration + 1)

            if (iteration + 1) % int(args.checkpoint_every) == 0:
                ckpt_path = ckpt_dir / f"iter_{iteration + 1:08d}.pt"
                _save_checkpoint(
                    ckpt_path,
                    iteration=iteration + 1,
                    student=student,
                    optimizer=optimizer,
                    rng=rng,
                    rnd=rnd,
                    rollout_env_seed=rollout_env_seed,
                    rollout_carry=rollout_carry,
                    args=args,
                    teacher_args=teacher_args,
                    teacher_checkpoint=str(teacher_path),
                )
                print(f"[orbit_wars_pt] saved checkpoint {ckpt_path}", flush=True)
    finally:
        writer.close()
        if reset_prefetch is not None:
            reset_prefetch.stop()


def parse_args() -> argparse.Namespace:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=str, required=True)
    p.add_argument("--experiment-root", type=str, default="experiments")
    p.add_argument("--teacher-checkpoint", type=str, required=True)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--minibatch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--halt-coef", type=float, default=1.0)
    p.add_argument("--origin-frac-coef", type=float, default=1.0)
    p.add_argument("--target-coef", type=float, default=1.0)
    p.add_argument("--value-coef", type=float, default=0.25)
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--max-micro-steps", type=int, default=None)
    p.add_argument("--rollout-micro-horizon", type=int, default=None)
    p.add_argument("--rollout-host-chunks", type=int, default=1)
    p.add_argument("--ship-speed", type=float, default=None)
    p.add_argument("--first-hit-n-rays", type=int, default=None)
    p.add_argument("--first-hit-ray-chunk-size", type=int, default=None)
    p.add_argument("--first-hit-env-chunk-size", type=int, default=0)
    p.add_argument("--first-hit-method", type=str, default=None)
    p.add_argument("--micro-step-penalty", type=float, default=None)
    p.add_argument("--earlygame-env-turn-limit", type=int, default=None)
    p.add_argument("--greedy-teacher", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--n-heads", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--rope-dims", type=int, default=None, choices=(2, 3))
    p.add_argument("--halt-init-prob", type=float, default=None)
    p.add_argument("--fraction-init-ratio", type=str, default=None)
    p.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", type=str, default=default_device)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--compile-mode",
        type=str,
        default="default",
        choices=("default", "reduce-overhead", "max-autotune"),
    )
    p.add_argument(
        "--matmul-precision",
        type=str,
        default="high",
        choices=("highest", "high", "medium"),
    )
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sync-rollout-timing", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--reset-prefetch-depth", type=int, default=256)
    p.add_argument("--reset-prefetch-workers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())

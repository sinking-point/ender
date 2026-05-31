#!/usr/bin/env python3
"""Replay logp from a logp_mismatch dump NPZ and compare forward paths.

Compares rollout ``old_logprob`` against fresh replays using:
  - dense (forward_dense_rollout) vs packed (policy.forward)
  - fp32 vs BF16 autocast
  - optional torch.compile on policy helpers (matches training defaults)
  - PPO loss path via ``compute_ppo_loss_compressed_torch`` with ``check_rollout_logp``

On mismatch, training also writes ``*_mb0_full.npz`` with the entire first
minibatch exactly as passed into ``_torch_ppo_loss_from_replay`` (all rows,
compressed obs planes, actions, advantages, member_counts, loss hyperparams,
and the captured forward logp from the crash iteration). Point the script at
that file to replay the loss with the same batch size and wiring as training.

Example:
  PYTHONPATH=. python scripts/replay_logp_dump.py \\
    --checkpoint experiments/exp-013/checkpoints/iter_00000420.pt \\
    --dump experiments/exp-013/debug/logp_mismatch_iter00000420_exploiter_mb0_full.npz
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from typing import Any, Callable, Optional

from orbit_wars_pt.compressed_observation import CompressedObservationBuffer
from orbit_wars_pt.constants import FRACTIONS, MAX_PLANETS, obs_feature_dim_for_num_agents
from orbit_wars_pt.model import OrbitWarsPolicy
from orbit_wars_pt.ppo_replay import compute_ppo_loss_compressed_torch
from orbit_wars_pt.train_ppo import _population_member_counts_torch, _torch_ppo_loss_from_replay


def meta_scalar(d: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    name = f"meta_{key}"
    if name not in d.files:
        return default
    arr = d[name]
    if arr.shape == ():
        val = arr.item()
        if isinstance(val, bytes):
            return val.decode("utf-8")
        return val
    return arr


def is_full_mb_dump(d: np.lib.npyio.NpzFile, dump_path: Path) -> bool:
    kind = str(meta_scalar(d, "dump_kind", ""))
    if kind == "ppo_loss_mb_full":
        return True
    return dump_path.name.endswith("_full.npz")


def parse_amp_dtype(name: str) -> Optional[torch.dtype]:
    if not name:
        return None
    lowered = name.lower()
    if lowered in ("bfloat16", "bf16"):
        return torch.bfloat16
    if lowered in ("float16", "fp16", "half"):
        return torch.float16
    raise ValueError(f"unsupported amp dtype in dump: {name!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiments/exp-013/checkpoints/iter_00000420.pt"),
    )
    p.add_argument(
        "--dump",
        type=Path,
        default=Path("experiments/exp-013/debug/logp_mismatch_iter00000420_exploiter_mb0.npz"),
    )
    p.add_argument("--policy-key", choices=("exploiter_policy", "policy"), default="exploiter_policy")
    p.add_argument("--num-agents", type=int, default=4)
    p.add_argument("--target-abort-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--compile-loss",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="torch.compile compute_ppo_loss_compressed_torch (default: same as --compile)",
    )
    p.add_argument(
        "--ppo-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also replay through the training PPO compressed loss path",
    )
    p.add_argument(
        "--manual-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run hand-rolled dense/packed logp replay (disable to only run PPO loss)",
    )
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--compile-mode", default="default")
    p.add_argument("--matmul-precision", default="high", choices=("highest", "high", "medium"))
    p.add_argument(
        "--grad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run PPO loss replay with grad enabled (matches training forward; full dumps only)",
    )
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def report(label: str, arr: np.ndarray, old_np: np.ndarray, dump_new_np: np.ndarray) -> None:
    n = int(arr.shape[0])
    d_old = np.abs(arr - old_np)
    d_new = np.abs(arr - dump_new_np)
    print(label)
    print(
        f"  vs old:      max {d_old.max():.6g} mean {d_old.mean():.6g} "
        f">1e-4 {(d_old > 1e-4).sum()}/{n} >1e-3 {(d_old > 1e-3).sum()}/{n}"
    )
    print(
        f"  vs dump_new: max {d_new.max():.6g} mean {d_new.mean():.6g} "
        f">1e-4 {(d_new > 1e-4).sum()}/{n}"
    )


def load_policy_and_loss_fn(args: argparse.Namespace, device: torch.device) -> tuple[OrbitWarsPolicy, Callable[..., Any], bool, int]:
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ta = ckpt.get("training_args", {})
    fd = obs_feature_dim_for_num_agents(int(args.num_agents), target_abort_enabled=bool(args.target_abort_enabled))
    value_head_count = int(
        ta.get(
            "exploiter_value_head_count" if args.policy_key == "exploiter_policy" else "value_head_count",
            3 if args.policy_key == "exploiter_policy" else 1,
        )
    )

    policy = OrbitWarsPolicy(
        d_model=int(ta.get("d_model", 192)),
        n_heads=int(ta.get("n_heads", 8)),
        n_layers=int(ta.get("n_layers", 4)),
        activation_checkpointing=False,
        feature_dim=fd,
        population_size=int(ta.get("population_size", 1)),
        rope_dims=int(ta.get("rope_dims", 2)),
        target_abort_enabled=bool(args.target_abort_enabled),
        value_head_count=value_head_count,
    ).to(device)
    state = ckpt.get(args.policy_key)
    if state is None:
        raise SystemExit(f"checkpoint missing state dict key {args.policy_key!r}")
    policy.load_state_dict(state)

    compile_loss = args.compile if args.compile_loss is None else bool(args.compile_loss)
    if args.compile:
        import torch._dynamo as dynamo

        dynamo.config.capture_scalar_outputs = True
        helper_compile_mode = "default" if args.compile_mode == "reduce-overhead" else args.compile_mode
        policy.forward = torch.compile(policy.forward, mode=helper_compile_mode, dynamic=True)  # type: ignore[assignment]
        policy.forward_dense_rollout = torch.compile(  # type: ignore[assignment]
            policy.forward_dense_rollout, mode=helper_compile_mode, dynamic=True
        )
        policy.target_logits_for_origin_fraction = torch.compile(  # type: ignore[assignment]
            policy.target_logits_for_origin_fraction, mode=helper_compile_mode, dynamic=True
        )
        print(
            "torch.compile enabled on policy helpers "
            f"(mode={helper_compile_mode!r})"
        )

    compressed_loss_fn: Callable[..., Any] = compute_ppo_loss_compressed_torch
    if compile_loss:
        compressed_loss_fn = torch.compile(
            compute_ppo_loss_compressed_torch,
            mode=args.compile_mode,
            dynamic=True,
        )
        print(f"torch.compile enabled on compute_ppo_loss_compressed_torch (mode={args.compile_mode!r})")
    return policy, compressed_loss_fn, compile_loss, fd


def replay_full_mb_dump(
    args: argparse.Namespace,
    device: torch.device,
    policy: OrbitWarsPolicy,
    compressed_loss_fn: Callable[..., Any],
    compile_loss: bool,
    default_fd: int,
) -> None:
    d = np.load(args.dump)
    if not is_full_mb_dump(d, args.dump):
        raise SystemExit(f"expected a full-minibatch dump, got {args.dump}")

    old_np = d["old_logprob"].astype(np.float64)
    captured_new_np = d["new_logprob"].astype(np.float64)
    n = int(old_np.shape[0])
    fd = int(meta_scalar(d, "obs_feature_dim", default_fd))
    clip_eps = float(meta_scalar(d, "clip_eps", args.clip_eps))
    vf_coef = float(meta_scalar(d, "vf_coef", args.vf_coef))
    entropy_coef = float(meta_scalar(d, "entropy_coef", args.entropy_coef))
    population_size = int(meta_scalar(d, "population_size", 1))
    amp_dtype = parse_amp_dtype(str(meta_scalar(d, "amp_dtype", "bfloat16" if device.type == "cuda" else "")))

    print(f"full mb dump rows {n} from {args.dump}")
    print(
        f"meta iter={meta_scalar(d, 'train_iter')} policy={meta_scalar(d, 'policy_label')} "
        f"amp={amp_dtype} clip_eps={clip_eps} vf={vf_coef} ent={entropy_coef} fd={fd}"
    )

    comp_obs = CompressedObservationBuffer(
        token_meta=torch.as_tensor(d["obs_comp_token_meta"], device=device, dtype=torch.int16),
        owner_idx=torch.as_tensor(d["obs_comp_owner_idx"], device=device, dtype=torch.int16),
        production=torch.as_tensor(d["obs_comp_production"], device=device, dtype=torch.int16),
        ships=torch.as_tensor(d["obs_comp_ships"], device=device, dtype=torch.int16),
        velocity=torch.as_tensor(d["obs_comp_velocity"], device=device, dtype=torch.float16),
        xy=torch.as_tensor(d["obs_comp_xy"], device=device, dtype=torch.float16),
        turn_progress=torch.as_tensor(d["obs_comp_turn_progress"], device=device, dtype=torch.float16),
        incoming_net=torch.as_tensor(d["obs_comp_incoming_net"], device=device, dtype=torch.int16),
        incoming_survivor=torch.as_tensor(d["obs_comp_incoming_survivor"], device=device, dtype=torch.int16),
        origin_frac_blocked=torch.as_tensor(d["obs_comp_origin_frac_blocked"], device=device, dtype=torch.bool),
    )
    actions: dict[str, torch.Tensor] = {}
    for fname in d.files:
        if not fname.startswith("action_"):
            continue
        key = fname[len("action_") :]
        if key in ("target_abort", "no_valid_pairs", "no_valid_fracs", "must_halt_no_ships", "target_planet_reachable"):
            dtype = torch.bool
        elif key in ("target_hit_tick",):
            dtype = torch.float32
        else:
            dtype = torch.long
        actions[key] = torch.as_tensor(d[fname], device=device, dtype=dtype)

    adv = torch.as_tensor(d["advantages"], device=device, dtype=torch.float32)
    ret_t = torch.as_tensor(d["returns"], device=device, dtype=torch.float32)
    old_logp_t = torch.as_tensor(d["old_logprob"], device=device, dtype=torch.float32)
    old_v_t = torch.as_tensor(d["old_value"], device=device, dtype=torch.float32)
    policy_loss_mask = (
        torch.as_tensor(d["policy_loss_mask"], device=device, dtype=torch.float32)
        if "policy_loss_mask" in d.files
        else None
    )
    if "member_counts" in d.files:
        member_counts = torch.as_tensor(d["member_counts"], device=device, dtype=torch.long)
    else:
        member_counts = _population_member_counts_torch(actions["population_idx"], population_size)

    grad_ctx = nullcontext() if args.grad else torch.no_grad()

    def run_exact(label: str, loss_fn: Callable[..., Any]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        with grad_ctx:
            _, mb_stats = _torch_ppo_loss_from_replay(
                obs=comp_obs,
                actions=actions,
                adv=adv,
                returns=ret_t,
                old_logp=old_logp_t,
                old_v=old_v_t,
                policy=policy,
                ship_speed=0.0,
                clip_eps=clip_eps,
                vf_coef=vf_coef,
                entropy_coef=entropy_coef,
                loss_fn=None,
                compressed_loss_fn=loss_fn,
                amp_dtype=amp_dtype,
                member_counts=member_counts,
                obs_feature_dim=fd,
                policy_loss_mask=policy_loss_mask,
                check_rollout_logp=True,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        lp = mb_stats["rollout_logp_new"].numpy().astype(np.float64)
        parts = {
            "halt": mb_stats["rollout_logp_halt"].numpy(),
            "origin": mb_stats["rollout_logp_origin_frac"].numpy(),
            "target": mb_stats["rollout_logp_target"].numpy(),
            "origin_used": mb_stats["rollout_logp_origin_frac_used"].numpy(),
        }
        print(f"\n{label} (grad={'on' if args.grad else 'off'})")
        report(label, lp, old_np, captured_new_np)
        return lp, parts

    eager_fn: Callable[..., Any] = compute_ppo_loss_compressed_torch
    results: dict[str, np.ndarray] = {}
    parts_by_key: dict[str, dict[str, np.ndarray]] = {}
    for label, loss_fn in (
        ("exact_ppo_eager", eager_fn),
        ("exact_ppo_compiled", compressed_loss_fn),
    ):
        if label == "exact_ppo_compiled" and not compile_loss:
            continue
        lp, parts = run_exact(label, loss_fn)
        results[label] = lp
        parts_by_key[label] = parts

    if compile_loss and "exact_ppo_eager" in results and "exact_ppo_compiled" in results:
        diff = np.abs(results["exact_ppo_eager"] - results["exact_ppo_compiled"])
        print(f"exact_ppo_eager vs exact_ppo_compiled: max {diff.max():.6g} mean {diff.mean():.6g}")

    spotlight_key = "exact_ppo_compiled" if "exact_ppo_compiled" in results else next(iter(results))
    lp = results[spotlight_key]
    parts = parts_by_key[spotlight_key]
    worst_old = int(np.argmax(np.abs(lp - old_np)))
    worst_captured = int(np.argmax(np.abs(lp - captured_new_np)))
    print()
    print(f"spotlight key: {spotlight_key}")
    for tag, i in (("worst_vs_old", worst_old), ("worst_vs_captured", worst_captured)):
        print(f"{tag} mb_row={i}")
        print(
            f"  old={old_np[i]:.6f} captured={captured_new_np[i]:.6f} "
            f"replay={lp[i]:.6f}"
        )
        print(
            f"  replay halt/orig/tgt={parts['halt'][i]:.6f}/{parts['origin'][i]:.6f}/{parts['target'][i]:.6f} "
            f"dump halt/orig/tgt={d['halt_logprob'][i]:.6f}/{d['origin_frac_logprob'][i]:.6f}/{d['target_logprob'][i]:.6f}"
        )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"CUDA required but unavailable (device={args.device})")

    torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == "cuda":
        print("device", device, torch.cuda.get_device_name(0))
    else:
        print("device", device)

    if is_full_mb_dump(np.load(args.dump), args.dump):
        policy, compressed_loss_fn, compile_loss, fd = load_policy_and_loss_fn(args, device)
        replay_full_mb_dump(args, device, policy, compressed_loss_fn, compile_loss, fd)
        return

    policy, compressed_loss_fn, compile_loss, fd = load_policy_and_loss_fn(args, device)

    d = np.load(args.dump)
    old_np = d["old_logprob"].astype(np.float64)
    dump_new_np = d["new_logprob"].astype(np.float64)
    row_idx_np = d["row_idx"]
    n = int(old_np.shape[0])
    print(f"dump rows {n} from {args.dump}")

    entity_type = torch.as_tensor(d["obs_entity_type"], device=device, dtype=torch.long)
    owner_idx = torch.as_tensor(d["obs_owner_idx"], device=device, dtype=torch.long)
    features = torch.as_tensor(d["obs_features"], device=device, dtype=torch.float32)
    rope_pos = torch.as_tensor(d["obs_rope_pos"], device=device, dtype=torch.float32)
    entity_mask = torch.as_tensor(d["obs_entity_mask"], device=device, dtype=torch.bool)
    planet_mask = torch.as_tensor(d["obs_planet_mask"], device=device, dtype=torch.bool)
    origin_frac_blocked = torch.as_tensor(d["obs_comp_origin_frac_blocked"], device=device, dtype=torch.bool)

    halt_action = torch.as_tensor(d["action_halt_action"], device=device, dtype=torch.long)
    target_abort = torch.as_tensor(d["action_target_abort"], device=device, dtype=torch.bool)
    pair_flat = torch.as_tensor(d["action_pair_flat"], device=device, dtype=torch.long)
    frac_idx = torch.as_tensor(d["action_frac_idx"], device=device, dtype=torch.long)
    no_valid_fracs = torch.as_tensor(d["action_no_valid_fracs"], device=device, dtype=torch.bool)
    target_valid = torch.as_tensor(d["action_target_planet_reachable"], device=device, dtype=torch.bool)
    target_hit_tick = torch.as_tensor(d["action_target_hit_tick"], device=device, dtype=torch.float32)
    population_idx = torch.as_tensor(d["action_population_idx"], device=device, dtype=torch.long)
    value_head_idx = torch.as_tensor(d["action_value_head_idx"], device=device, dtype=torch.long)
    no_valid_pairs = torch.as_tensor(d["action_no_valid_pairs"], device=device, dtype=torch.bool)
    if "action_must_halt_no_ships" in d.files:
        must_halt_no_ships = torch.as_tensor(d["action_must_halt_no_ships"], device=device, dtype=torch.bool)
    else:
        must_halt_no_ships = torch.zeros((n,), device=device, dtype=torch.bool)
    if "policy_loss_mask" in d.files:
        policy_loss_mask = torch.as_tensor(d["policy_loss_mask"], device=device, dtype=torch.float32)
    else:
        policy_loss_mask = None

    comp_obs = CompressedObservationBuffer(
        token_meta=torch.as_tensor(d["obs_comp_token_meta"], device=device, dtype=torch.int16),
        owner_idx=torch.as_tensor(d["obs_comp_owner_idx"], device=device, dtype=torch.int16),
        production=torch.as_tensor(d["obs_comp_production"], device=device, dtype=torch.int16),
        ships=torch.as_tensor(d["obs_comp_ships"], device=device, dtype=torch.int16),
        velocity=torch.as_tensor(d["obs_comp_velocity"], device=device, dtype=torch.float16),
        xy=torch.as_tensor(d["obs_comp_xy"], device=device, dtype=torch.float16),
        turn_progress=torch.as_tensor(d["obs_comp_turn_progress"], device=device, dtype=torch.float16),
        incoming_net=torch.as_tensor(d["obs_comp_incoming_net"], device=device, dtype=torch.int16),
        incoming_survivor=torch.as_tensor(d["obs_comp_incoming_survivor"], device=device, dtype=torch.int16),
        origin_frac_blocked=origin_frac_blocked,
    )
    adv = torch.as_tensor(d["advantages"], device=device, dtype=torch.float32)
    ret_t = torch.as_tensor(d["returns"], device=device, dtype=torch.float32)
    old_logp_t = torch.as_tensor(d["old_logprob"], device=device, dtype=torch.float32)
    old_v_t = torch.as_tensor(d["old_value"], device=device, dtype=torch.float32)

    p = MAX_PLANETS
    o_idx = pair_flat // p
    d_idx = pair_flat % p
    n_idx = torch.arange(n, device=device)
    ships = features[:, 1 : 1 + MAX_PLANETS, 1] * 1000.0
    fleet_size = torch.floor(features.new_tensor(FRACTIONS)[frac_idx] * ships[n_idx, o_idx])
    origin_frac_flat = o_idx * len(FRACTIONS) + frac_idx

    @torch.no_grad()
    def total_logp_from_out(out: dict[str, torch.Tensor], target_logits: torch.Tensor) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        halt_lp = torch.log_softmax(out["halt_logits"].float(), dim=-1)
        halt_logp = halt_lp.gather(1, halt_action[:, None]).squeeze(-1)

        ofl = out["origin_frac_logits"].flatten(start_dim=1)
        ofm = out["origin_frac_mask"].flatten(start_dim=1)
        oflp = torch.log_softmax(ofl.float().masked_fill(~ofm, -1e4), dim=-1)
        origin_frac_logp = oflp.gather(1, origin_frac_flat[:, None]).squeeze(-1)

        target_mask = out["pair_mask"][n_idx, o_idx, :] & target_valid
        abort_logits = out["abort_logits"][n_idx, o_idx, frac_idx]
        combined = torch.cat(
            [target_logits.float().masked_fill(~target_mask, -1e4), abort_logits.float()[:, None]],
            dim=-1,
        )
        target_lp = torch.log_softmax(combined, dim=-1)
        target_choice = torch.where(target_abort, torch.full_like(d_idx, MAX_PLANETS), d_idx)
        target_logp = target_lp.gather(1, target_choice[:, None]).squeeze(-1)

        origin_frac_used = (halt_action == 0) & ~no_valid_fracs
        target_used = origin_frac_used  # target_abort_enabled path
        total = halt_logp + origin_frac_used.float() * origin_frac_logp + target_used.float() * target_logp
        parts = {
            "halt": halt_logp.float().cpu().numpy(),
            "origin": origin_frac_logp.float().cpu().numpy(),
            "target": target_logp.float().cpu().numpy(),
            "origin_used": origin_frac_used.cpu().numpy(),
        }
        return total.float().cpu().numpy(), parts

    @torch.no_grad()
    def run_path(kind: str, amp: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if amp and device.type == "cuda"
            else nullcontext()
        )
        with amp_ctx:
            if kind == "dense":
                out = policy.forward_dense_rollout(
                    entity_type,
                    owner_idx,
                    features,
                    rope_pos,
                    entity_mask,
                    planet_mask,
                    origin_frac_blocked=origin_frac_blocked,
                    population_idx=population_idx,
                    value_head_idx=value_head_idx,
                )
            elif kind == "packed":
                out = policy(
                    entity_type,
                    owner_idx,
                    features,
                    rope_pos,
                    entity_mask,
                    planet_mask,
                    origin_frac_blocked=origin_frac_blocked,
                    population_idx=population_idx,
                    value_head_idx=value_head_idx,
                )
            else:
                raise ValueError(kind)
            target_logits = policy.target_logits_for_origin_fraction(
                out["planet_hidden"],
                o_idx,
                frac_idx,
                fleet_size=fleet_size,
                target_eta=target_hit_tick,
                target_ships=ships,
                population_idx=population_idx,
            )
        return total_logp_from_out(out, target_logits)

    @torch.no_grad()
    def run_ppo_loss(amp: bool, loss_fn: Callable[..., Any]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Same wiring as ``_torch_ppo_loss_from_replay`` for compressed host replay."""
        amp_dtype: Optional[torch.dtype] = (
            torch.bfloat16 if amp and device.type == "cuda" else None
        )
        target_overflow = torch.zeros((n,), dtype=torch.bool, device=device)
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if amp_dtype is not None
            else nullcontext()
        )
        with amp_ctx:
            _, mb_stats = loss_fn(
                policy,
                comp_obs.token_meta,
                comp_obs.owner_idx,
                comp_obs.production,
                comp_obs.ships,
                comp_obs.velocity,
                comp_obs.xy,
                comp_obs.turn_progress,
                comp_obs.incoming_net,
                comp_obs.incoming_survivor,
                comp_obs.origin_frac_blocked,
                int(fd),
                target_valid,
                target_overflow,
                target_hit_tick,
                halt_action,
                target_abort,
                pair_flat,
                frac_idx,
                no_valid_pairs,
                no_valid_fracs,
                must_halt_no_ships,
                adv,
                ret_t,
                old_logp_t,
                old_v_t,
                float(args.clip_eps),
                float(args.vf_coef),
                float(args.entropy_coef),
                population_idx,
                None,
                value_head_idx,
                policy_loss_mask,
                True,
            )
        lp = mb_stats["rollout_logp_new"].numpy().astype(np.float64)
        parts = {
            "halt": mb_stats["rollout_logp_halt"].numpy(),
            "origin": mb_stats["rollout_logp_origin_frac"].numpy(),
            "target": mb_stats["rollout_logp_target"].numpy(),
            "origin_used": mb_stats["rollout_logp_origin_frac_used"].numpy(),
        }
        return lp, parts

    results: dict[str, np.ndarray] = {}
    parts_by_key: dict[str, dict[str, np.ndarray]] = {}

    if args.manual_replay:
        _ = run_path("dense", amp=True)[0]
        if device.type == "cuda":
            torch.cuda.synchronize()

        for kind in ("dense", "packed"):
            for amp in (False, True):
                if amp and device.type != "cuda":
                    continue
                key = f"{kind}_{'bf16' if amp else 'fp32'}"
                lp, parts = run_path(kind, amp=amp)
                results[key] = lp
                parts_by_key[key] = parts
                if device.type == "cuda":
                    torch.cuda.synchronize()
                report(key, lp, old_np, dump_new_np)

        for a, b in (("dense_bf16", "packed_bf16"), ("dense_fp32", "packed_fp32")):
            if a in results and b in results:
                diff = np.abs(results[a] - results[b])
                print(f"{a} vs {b}: max {diff.max():.6g} mean {diff.mean():.6g}")

    if args.ppo_loss:
        if args.manual_replay:
            print()
        eager_fn: Callable[..., Any] = compute_ppo_loss_compressed_torch
        for amp in (False, True):
            if amp and device.type != "cuda":
                continue
            for label, loss_fn in (
                ("ppo_loss_eager", eager_fn),
                ("ppo_loss_compiled", compressed_loss_fn),
            ):
                if label == "ppo_loss_compiled" and not compile_loss:
                    continue
                key = f"{label}_{'bf16' if amp else 'fp32'}"
                lp, parts = run_ppo_loss(amp=amp, loss_fn=loss_fn)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                results[key] = lp
                parts_by_key[key] = parts
                report(key, lp, old_np, dump_new_np)

        if compile_loss:
            for a, b in (
                ("ppo_loss_eager_bf16", "ppo_loss_compiled_bf16"),
                ("ppo_loss_eager_fp32", "ppo_loss_compiled_fp32"),
            ):
                if a in results and b in results:
                    diff = np.abs(results[a] - results[b])
                    print(f"{a} vs {b}: max {diff.max():.6g} mean {diff.mean():.6g}")
            if "ppo_loss_compiled_bf16" in results and "packed_bf16" in results:
                diff = np.abs(results["ppo_loss_compiled_bf16"] - results["packed_bf16"])
                print(
                    "ppo_loss_compiled_bf16 vs packed_bf16 manual: "
                    f"max {diff.max():.6g} mean {diff.mean():.6g}"
                )

    if not results:
        raise SystemExit("nothing to run: enable --manual-replay and/or --ppo-loss")

    spotlight_candidates = (
        "ppo_loss_compiled_bf16",
        "ppo_loss_eager_bf16",
        "dense_bf16",
        "dense_fp32",
    )
    spotlight_key = next((k for k in spotlight_candidates if k in results), next(iter(results)))

    lp = results[spotlight_key]
    parts = parts_by_key[spotlight_key]
    worst_old = int(np.argmax(np.abs(lp - old_np)))
    worst_new = int(np.argmax(np.abs(lp - dump_new_np)))
    halt_only = int(np.flatnonzero(~parts["origin_used"])[0])

    print()
    print(f"spotlight key: {spotlight_key}")
    for tag, i in (("worst_vs_old", worst_old), ("worst_vs_dump_new", worst_new), ("halt_only", halt_only)):
        print(f"{tag} mb_row={int(row_idx_np[i])}")
        line = (
            f"  old={old_np[i]:.6f} dump_new={dump_new_np[i]:.6f} "
            f"{spotlight_key}={lp[i]:.6f}"
        )
        if "packed_bf16" in results and spotlight_key != "packed_bf16":
            line += f" packed_bf16={results['packed_bf16'][i]:.6f}"
        if "ppo_loss_compiled_bf16" in results and spotlight_key != "ppo_loss_compiled_bf16":
            line += f" ppo_loss_compiled_bf16={results['ppo_loss_compiled_bf16'][i]:.6f}"
        print(line)
        print(
            f"  replay halt/orig/tgt={parts['halt'][i]:.6f}/{parts['origin'][i]:.6f}/{parts['target'][i]:.6f} "
            f"dump halt/orig/tgt={d['halt_logprob'][i]:.6f}/{d['origin_frac_logprob'][i]:.6f}/{d['target_logprob'][i]:.6f}"
        )


if __name__ == "__main__":
    main()

"""Self-play PPO training with GAE — shared policy for both seats."""

from __future__ import annotations

import argparse
import gc
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

import numpy as np

import jax
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Must run before any orbit_wars_pt import that transitively `import jax` (e.g. parallel_rollout).
import orbit_wars_pt.xla_env  # noqa: F401

import jax.numpy as jnp

from orbit_wars_pt.batched_env import heal_terminal_env_slices
from orbit_wars_pt.env_wrapper import OrbitWarsEnvConfig
from orbit_wars_pt.gpu_mem import (
    log_cuda_mem,
    print_cuda_memory_summary,
    reset_peak_stats,
    torch_param_bytes,
)
from orbit_wars_pt.jax_setup import configure_jax_for_training
from orbit_wars_pt.model import OrbitWarsPolicy
from orbit_wars_pt.parallel_rollout import (
    RolloutCarry,
    RolloutGameStats,
    RolloutSegment,
    RolloutTiming,
    collect_parallel_micro_rollouts,
)
from orbit_wars_pt.reset_prefetch import RolloutResetPrefetch
from orbit_wars_pt.ppo_replay import compute_ppo_loss_torch, replay_ppo_loss
from orbit_wars_pt.transition_buffer import (
    TransitionBuffer,
    gather_minibatch,
    gather_minibatch_selected,
)

from jax_orbit_wars import OrbitWarsState

CHECKPOINT_VERSION = 1


@dataclass
class HostRolloutChunk:
    """A rollout segment spilled to host RAM for minibatch-sized replay transfers."""

    segment: RolloutSegment
    samples: dict


def _sanitize_experiment_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("experiment name must be non-empty")
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
    if safe != name:
        print(f"[orbit_wars_pt] sanitized experiment name -> {safe!r}", flush=True)
    if ".." in name or "/" in name or "\\" in name:
        raise ValueError("experiment name must not contain path separators or '..'")
    return safe


def experiment_dirs(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    """Paths under experiment root.

    TensorBoard logs live at ``<root>/tensorboard/<experiment>/`` so a single
    ``tensorboard --logdir <root>/tensorboard`` shows every experiment as its own run.
    Checkpoints stay under ``<root>/<experiment>/checkpoints/``.
    """

    root = Path(args.experiment_root)
    if not root.is_absolute():
        root = Path(ROOT) / root
    name = _sanitize_experiment_name(args.experiment)
    exp = root / name
    tb_dir = root / "tensorboard" / name
    return exp, tb_dir, exp / "checkpoints"


def _serialize_rollout_carry(carry: RolloutCarry) -> Dict[str, Any]:
    state_np = {
        field: np.asarray(jax.device_get(getattr(carry.state_b, field)))
        for field in OrbitWarsState._fields
    }
    return {
        "state_b": state_np,
        "cfg": asdict(carry.cfg),
        "episode_turns": list(carry.episode_turns),
    }


def _deserialize_rollout_carry(obj: Dict[str, Any]) -> RolloutCarry:
    state_b = OrbitWarsState(
        **{k: jnp.asarray(v) for k, v in obj["state_b"].items()}
    )
    cfg_d = dict(obj["cfg"])
    cfg = OrbitWarsEnvConfig(**cfg_d)
    ne = int(state_b.planets.shape[0])
    episode_turns = list(obj.get("episode_turns", [0] * ne))
    if len(episode_turns) != ne:
        episode_turns = [0] * ne
    return RolloutCarry(
        state_b=state_b,
        cfg=cfg,
        episode_turns=episode_turns,
    )


def find_latest_checkpoint(checkpoints_dir: Path) -> Optional[Path]:
    if not checkpoints_dir.is_dir():
        return None
    best: Optional[Path] = None
    best_it = -1
    for p in checkpoints_dir.glob("iter_*.pt"):
        try:
            it = int(p.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if it > best_it:
            best_it = it
            best = p
    return best


def _checkpoint_training_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Scalars needed to verify resume compatibility."""

    keys = (
        "seed",
        "num_envs",
        "max_micro_steps",
        "rollout_micro_horizon",
        "rollout_storage",
        "rollout_host_chunks",
        "lr",
        "gamma",
        "lam",
        "vf_coef",
        "entropy_coef",
        "clip_eps",
        "ppo_epochs",
        "minibatch_size",
        "ship_speed",
        "first_hit_n_rays",
        "first_hit_ray_chunk_size",
        "max_grad_norm",
        "d_model",
        "n_heads",
        "n_layers",
        "max_fleets",
        "activation_checkpointing",
        "device",
        "compile",
        "compile_mode",
        "matmul_precision",
        "amp",
        "experiment",
        "experiment_root",
    )
    return {k: getattr(args, k) for k in keys}


# Saved in checkpoints for logging / inspection but safe to change when resuming.
_RESUME_ARG_MISMATCH_IGNORE = frozenset({"activation_checkpointing"})


def _validate_checkpoint_args(saved: Dict[str, Any], args: argparse.Namespace) -> None:
    cur = _checkpoint_training_args(args)
    mismatches = [
        k
        for k in cur
        if k not in _RESUME_ARG_MISMATCH_IGNORE and k in saved and saved[k] != cur[k]
    ]
    if mismatches:
        raise RuntimeError(
            "Checkpoint training args mismatch current CLI — refusing to resume. "
            f"Differing keys: {mismatches}. Use matching flags or a fresh experiment name."
        )


def save_checkpoint(
    path: Path,
    *,
    next_iteration: int,
    policy: OrbitWarsPolicy,
    opt: torch.optim.Optimizer,
    rng: torch.Generator,
    rnd: np.random.Generator,
    rollout_env_seed: int,
    rollout_carry: Optional[RolloutCarry],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "iteration": next_iteration,
        "policy": policy.state_dict(),
        "optimizer": opt.state_dict(),
        "torch_rng": rng.get_state(),
        "numpy_rng_state": rnd.bit_generator.state,
        "rollout_env_seed": rollout_env_seed,
        "rollout_carry": _serialize_rollout_carry(rollout_carry)
        if rollout_carry is not None
        else None,
        "training_args": _checkpoint_training_args(args),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _restore_torch_generator_from_checkpoint(saved: Any, rng: torch.Generator) -> None:
    """Checkpoint tensors load cleanly as CPU uint8; dtype must stay uint8.

    CUDA ``torch.Generator`` still expects **CPU** byte state for ``set_state`` — moving
    the tensor to CUDA raises ``RNG state must be a torch.ByteTensor``."""

    if not isinstance(saved, torch.Tensor):
        raise TypeError(f"checkpoint torch_rng must be a torch.Tensor, got {type(saved)}")
    s = saved.detach().to(dtype=torch.uint8).cpu().contiguous()
    rng.set_state(s)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
    *,
    bootstrap: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    if T == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    if bootstrap is not None and not bool(dones[-1]):
        next_val = float(bootstrap)
    else:
        next_val = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_val * mask - float(values[t])
        last_gae = delta + gamma * lam * mask * last_gae
        adv[t] = last_gae
        next_val = float(values[t])
    ret = adv + values
    return adv, ret


def build_ppo_samples(
    segment: RolloutSegment,
    gamma: float,
    lam: float,
) -> Optional[dict]:
    """Compute per-trajectory GAE on host metadata; return flat sample arrays.

    Each sample's location in the device buffers is given by ``(player, t, n)``;
    PPO replay gathers state / action records via these indices. All arrays
    have shape ``[M]`` where ``M`` is the total number of valid transitions.
    """

    advs, rets, players, t_idx, n_idx, old_lp, old_v = [], [], [], [], [], [], []
    num_envs = int(segment.valid_p0.shape[1])

    for player in (0, 1):
        if player == 0:
            old_value = segment.old_value_p0
            old_logprob = segment.old_logprob_p0
            rewards = segment.reward_p0
            dones = segment.done_p0
            bootstrap = segment.bootstrap_p0
            bootstrap_valid = segment.bootstrap_valid_p0
            write_idx = segment.write_idx_p0
        else:
            old_value = segment.old_value_p1
            old_logprob = segment.old_logprob_p1
            rewards = segment.reward_p1
            dones = segment.done_p1
            bootstrap = segment.bootstrap_p1
            bootstrap_valid = segment.bootstrap_valid_p1
            write_idx = segment.write_idx_p1

        for n in range(num_envs):
            T = int(write_idx[n])
            if T == 0:
                continue
            v = old_value[:T, n]
            r = rewards[:T, n]
            d = dones[:T, n]
            bs = float(bootstrap[n]) if bool(bootstrap_valid[n]) else None
            adv, ret = compute_gae(r, v, d, gamma, lam, bootstrap=bs)
            advs.append(adv)
            rets.append(ret)
            players.append(np.full((T,), player, dtype=np.int32))
            t_idx.append(np.arange(T, dtype=np.int32))
            n_idx.append(np.full((T,), n, dtype=np.int32))
            old_lp.append(old_logprob[:T, n])
            old_v.append(v)

    if not advs:
        return None
    return {
        "advantages": np.concatenate(advs).astype(np.float32),
        "returns": np.concatenate(rets).astype(np.float32),
        "players": np.concatenate(players).astype(np.int32),
        "t_idx": np.concatenate(t_idx).astype(np.int32),
        "n_idx": np.concatenate(n_idx).astype(np.int32),
        "old_logprob": np.concatenate(old_lp).astype(np.float32),
        "old_value": np.concatenate(old_v).astype(np.float32),
    }


@dataclass
class PPOTiming:
    """Per-phase wall times accumulated across all PPO minibatches in one iter.

    Times are host-side ``perf_counter`` without explicit ``cuda.synchronize``
    (matches ``RolloutTiming``'s convention). Async PyTorch / JAX kernel
    work falls into the next sync point — typically ``sync_s`` (the
    ``loss.item()`` call), which therefore captures most of the GPU
    compute time. The other phases reflect CPU-side dispatch costs.

    Phase 5 + step-3 consolidation: forward + masking + logp + entropy +
    value + PPO surrogate are one ``torch.compile`` target, so the
    previously-separate ``replay_forward_s`` / ``replay_logp_s`` /
    ``loss_s`` collapse into a single ``compiled_loss_s``.
    """

    gather_s: float = 0.0
    replay_jax_s: float = 0.0
    replay_dlpack_s: float = 0.0
    compiled_loss_s: float = 0.0
    backward_s: float = 0.0
    optim_s: float = 0.0
    sync_s: float = 0.0
    total_s: float = 0.0
    n_minibatches: int = 0

    def accounted_s(self) -> float:
        return (
            self.gather_s
            + self.replay_jax_s
            + self.replay_dlpack_s
            + self.compiled_loss_s
            + self.backward_s
            + self.optim_s
            + self.sync_s
        )


@dataclass
class PPOStats:
    """Per-iter aggregator for PPO diagnostics across all minibatches.

    The compiled loss returns scalar tensors; we ``.item()`` them at the
    end of each minibatch (already a sync point for ``loss.item()``) and
    accumulate as Python floats. Means are unweighted across minibatches
    (each MB gets equal weight regardless of size — fine since MB sizes
    are nearly uniform). Explained variance uses pooled sufficient
    statistics so it's exact across the whole epoch.
    """

    loss_pi_sum: float = 0.0
    loss_vf_sum: float = 0.0
    entropy_sum: float = 0.0
    entropy_halt_sum: float = 0.0
    entropy_halt_n: int = 0
    entropy_origin_frac_sum: float = 0.0
    entropy_origin_frac_n: int = 0
    entropy_target_sum: float = 0.0
    entropy_target_n: int = 0
    approx_kl_sum: float = 0.0
    approx_kl_k3_sum: float = 0.0
    clip_frac_sum: float = 0.0
    value_mean_sum: float = 0.0
    grad_norm_sum: float = 0.0
    diff_sq_total: float = 0.0
    ret_sum_total: float = 0.0
    ret_sq_sum_total: float = 0.0
    count_total: float = 0.0
    n_mb: int = 0

    def update(self, stats: dict, grad_norm: float) -> None:
        self.loss_pi_sum += float(stats["loss_pi"].item())
        self.loss_vf_sum += float(stats["loss_vf"].item())
        self.entropy_sum += float(stats["entropy"].item())
        # Per-head conditional entropies may be NaN when a head's
        # conditioning set is empty in a given minibatch (e.g. every
        # row triggered ``no_valid_fracs``). Skip those minibatches
        # in the per-head running mean instead of contaminating the
        # iter summary with NaN.
        ent_h = float(stats["entropy_halt"].item())
        if ent_h == ent_h:
            self.entropy_halt_sum += ent_h
            self.entropy_halt_n += 1
        ent_of = float(stats["entropy_origin_frac"].item())
        if ent_of == ent_of:
            self.entropy_origin_frac_sum += ent_of
            self.entropy_origin_frac_n += 1
        ent_t = float(stats["entropy_target"].item())
        if ent_t == ent_t:
            self.entropy_target_sum += ent_t
            self.entropy_target_n += 1
        self.approx_kl_sum += float(stats["approx_kl"].item())
        self.approx_kl_k3_sum += float(stats["approx_kl_k3"].item())
        self.clip_frac_sum += float(stats["clip_frac"].item())
        self.value_mean_sum += float(stats["value_mean"].item())
        self.diff_sq_total += float(stats["diff_sq_sum"].item())
        self.ret_sum_total += float(stats["ret_sum"].item())
        self.ret_sq_sum_total += float(stats["ret_sq_sum"].item())
        self.count_total += float(stats["count"].item())
        self.grad_norm_sum += grad_norm
        self.n_mb += 1

    def explained_variance(self) -> float:
        # Pooled total sum-of-squares of returns from their mean.
        # Returns are revisited across PPO epochs; both numerator and
        # denominator scale by the same epoch count, so the ratio is the
        # mean (across epochs) of per-epoch EV — a fine diagnostic.
        n = max(1.0, self.count_total)
        ss_total = self.ret_sq_sum_total - (self.ret_sum_total ** 2) / n
        if ss_total <= 1e-12:
            return float("nan")
        return 1.0 - self.diff_sq_total / ss_total

    def summary(self) -> dict:
        n = max(1, self.n_mb)
        ent_h = (
            self.entropy_halt_sum / self.entropy_halt_n
            if self.entropy_halt_n > 0
            else float("nan")
        )
        ent_of = (
            self.entropy_origin_frac_sum / self.entropy_origin_frac_n
            if self.entropy_origin_frac_n > 0
            else float("nan")
        )
        ent_t = (
            self.entropy_target_sum / self.entropy_target_n
            if self.entropy_target_n > 0
            else float("nan")
        )
        return {
            "loss_pi": self.loss_pi_sum / n,
            "loss_vf": self.loss_vf_sum / n,
            "entropy": self.entropy_sum / n,
            "entropy_halt": ent_h,
            "entropy_origin_frac": ent_of,
            "entropy_target": ent_t,
            "approx_kl": self.approx_kl_sum / n,
            "approx_kl_k3": self.approx_kl_k3_sum / n,
            "clip_frac": self.clip_frac_sum / n,
            "value_mean": self.value_mean_sum / n,
            "grad_norm": self.grad_norm_sum / n,
            "explained_var": self.explained_variance(),
        }


def _ppo_stats_str(s: dict) -> str:
    return (
        f"loss_pi {s['loss_pi']:+.4f} loss_vf {s['loss_vf']:.4f} "
        f"ent {s['entropy']:.3f} "
        f"ent_h {s['entropy_halt']:.3f} ent_of {s['entropy_origin_frac']:.3f} "
        f"ent_t {s['entropy_target']:.3f} "
        f"kl {s['approx_kl']:+.4f} kl_k3 {s['approx_kl_k3']:.4f} "
        f"clip {s['clip_frac']:.3f} ev {s['explained_var']:+.3f} "
        f"v_mean {s['value_mean']:+.4f} g_norm {s['grad_norm']:.3f}"
    )


def normalize_advantages(samples: dict) -> None:
    a = samples["advantages"]
    if a.size == 0:
        return
    samples["advantages"] = ((a - a.mean()) / (a.std() + 1e-8)).astype(np.float32)


def _tree_to_host(x: Any) -> Any:
    return jax.tree.map(lambda leaf: np.asarray(jax.device_get(leaf)), x)


def _rollout_segment_to_host(segment: RolloutSegment) -> RolloutSegment:
    """Move device-resident rollout payload to CPU RAM.

    Host metadata arrays are already NumPy arrays; ``np.asarray`` keeps them as
    views.  The heavyweight JAX buffers and turn cache become CPU arrays so
    the accelerator only sees minibatch-sized slices during PPO.
    """

    return RolloutSegment(
        buf0=_tree_to_host(segment.buf0),
        buf1=_tree_to_host(segment.buf1),
        turn_state_cache=_tree_to_host(segment.turn_state_cache),
        turn_tag_p0=np.asarray(jax.device_get(segment.turn_tag_p0)),
        turn_tag_p1=np.asarray(jax.device_get(segment.turn_tag_p1)),
        write_idx_p0=np.asarray(segment.write_idx_p0),
        write_idx_p1=np.asarray(segment.write_idx_p1),
        valid_p0=np.asarray(segment.valid_p0),
        valid_p1=np.asarray(segment.valid_p1),
        old_logprob_p0=np.asarray(segment.old_logprob_p0),
        old_logprob_p1=np.asarray(segment.old_logprob_p1),
        old_value_p0=np.asarray(segment.old_value_p0),
        old_value_p1=np.asarray(segment.old_value_p1),
        reward_p0=np.asarray(segment.reward_p0),
        reward_p1=np.asarray(segment.reward_p1),
        done_p0=np.asarray(segment.done_p0),
        done_p1=np.asarray(segment.done_p1),
        bootstrap_p0=np.asarray(segment.bootstrap_p0),
        bootstrap_p1=np.asarray(segment.bootstrap_p1),
        bootstrap_valid_p0=np.asarray(segment.bootstrap_valid_p0),
        bootstrap_valid_p1=np.asarray(segment.bootstrap_valid_p1),
        env_steps_per_env=np.asarray(segment.env_steps_per_env),
    )


def _release_rollout_device_refs(device: torch.device) -> None:
    """Drop Python-held accelerator refs before learner-side allocations.

    JAX/XLA may keep its allocator pool reserved, so this is not guaranteed to
    reduce process-level ``nvidia-smi`` usage.  It does make dead arrays
    collectable/reusable before PPO builds its backward graph.
    """

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _concat_sample_dicts(sample_dicts: list[dict]) -> dict:
    keys = sample_dicts[0].keys()
    return {k: np.concatenate([s[k] for s in sample_dicts]).astype(sample_dicts[0][k].dtype) for k in keys}


def _normalize_chunk_advantages(chunks: list[HostRolloutChunk]) -> None:
    adv = np.concatenate([c.samples["advantages"] for c in chunks])
    mean = float(adv.mean())
    std = float(adv.std()) + 1e-8
    for chunk in chunks:
        chunk.samples["advantages"] = ((chunk.samples["advantages"] - mean) / std).astype(np.float32)


def _build_host_chunk_samples(
    segments: list[RolloutSegment],
    gamma: float,
    lam: float,
) -> Optional[list[dict]]:
    """Compute GAE across all host-staged chunks, then shard samples by chunk.

    Chunk boundaries are storage boundaries only.  For each player/env stream,
    rewards/values/dones are concatenated across every collected chunk and GAE
    uses only the final available bootstrap.
    """

    if not segments:
        return None

    keys = ("advantages", "returns", "players", "t_idx", "n_idx", "old_logprob", "old_value")
    per_chunk = [{k: [] for k in keys} for _ in segments]
    num_envs = int(segments[0].valid_p0.shape[1])

    for player in (0, 1):
        for n in range(num_envs):
            values_parts, rewards_parts, dones_parts, old_lp_parts = [], [], [], []
            t_parts: list[np.ndarray] = []
            lengths: list[int] = []
            last_nonempty_i: Optional[int] = None

            for ci, segment in enumerate(segments):
                if player == 0:
                    old_value = segment.old_value_p0
                    old_logprob = segment.old_logprob_p0
                    rewards = segment.reward_p0
                    dones = segment.done_p0
                    write_idx = segment.write_idx_p0
                else:
                    old_value = segment.old_value_p1
                    old_logprob = segment.old_logprob_p1
                    rewards = segment.reward_p1
                    dones = segment.done_p1
                    write_idx = segment.write_idx_p1

                T = int(write_idx[n])
                lengths.append(T)
                if T == 0:
                    continue
                last_nonempty_i = ci
                values_parts.append(old_value[:T, n])
                rewards_parts.append(rewards[:T, n])
                dones_parts.append(dones[:T, n])
                old_lp_parts.append(old_logprob[:T, n])
                t_parts.append(np.arange(T, dtype=np.int32))

            if last_nonempty_i is None:
                continue

            values = np.concatenate(values_parts).astype(np.float32)
            rewards = np.concatenate(rewards_parts).astype(np.float32)
            dones = np.concatenate(dones_parts).astype(np.bool_)
            last_segment = segments[last_nonempty_i]
            if player == 0:
                bootstrap = (
                    float(last_segment.bootstrap_p0[n])
                    if bool(last_segment.bootstrap_valid_p0[n])
                    else None
                )
            else:
                bootstrap = (
                    float(last_segment.bootstrap_p1[n])
                    if bool(last_segment.bootstrap_valid_p1[n])
                    else None
                )
            adv, ret = compute_gae(rewards, values, dones, gamma, lam, bootstrap=bootstrap)

            offset = 0
            part_i = 0
            for ci, T in enumerate(lengths):
                if T == 0:
                    continue
                old_lp = old_lp_parts[part_i]
                t_local = t_parts[part_i]
                dst = per_chunk[ci]
                dst["advantages"].append(adv[offset : offset + T])
                dst["returns"].append(ret[offset : offset + T])
                dst["players"].append(np.full((T,), player, dtype=np.int32))
                dst["t_idx"].append(t_local)
                dst["n_idx"].append(np.full((T,), n, dtype=np.int32))
                dst["old_logprob"].append(old_lp)
                dst["old_value"].append(values[offset : offset + T])
                offset += T
                part_i += 1

    out: list[dict] = []
    any_samples = False
    dtypes = {
        "advantages": np.float32,
        "returns": np.float32,
        "players": np.int32,
        "t_idx": np.int32,
        "n_idx": np.int32,
        "old_logprob": np.float32,
        "old_value": np.float32,
    }
    for chunk in per_chunk:
        if not chunk["advantages"]:
            out.append({})
            continue
        any_samples = True
        out.append({k: np.concatenate(chunk[k]).astype(dtypes[k]) for k in keys})

    return out if any_samples else None


def _combine_rollout_timing(items: list[RolloutTiming]) -> RolloutTiming:
    out = RolloutTiming()
    for rt in items:
        out.init_s += rt.init_s
        out.env_step_s += rt.env_step_s
        out.micro_cap_s += rt.micro_cap_s
        out.obs_build_s += rt.obs_build_s
        out.policy_batch_s += rt.policy_batch_s
        out.policy_forward_s += rt.policy_forward_s
        out.micro_apply_s += rt.micro_apply_s
        out.state_unstack_s += rt.state_unstack_s
        out.loop_s += rt.loop_s
        out.wall_s += rt.wall_s
        out.outer_iters += rt.outer_iters
    return out


def _combine_game_stats(items: list[RolloutGameStats]) -> RolloutGameStats:
    out = RolloutGameStats()
    for gs in items:
        out.n_completed += gs.n_completed
        out.n_step_limit += gs.n_step_limit
        out.n_decisive += gs.n_decisive
        out.n_p0_positive_reward += gs.n_p0_positive_reward
        out.n_p1_positive_reward += gs.n_p1_positive_reward
        out.sum_final_ships_p0 += gs.sum_final_ships_p0
        out.sum_final_ships_p1 += gs.sum_final_ships_p1
        out.sum_episode_turns += gs.sum_episode_turns
    return out


def _combine_segments_for_stats(segments: list[RolloutSegment]) -> RolloutSegment:
    """Small synthetic segment for logging aggregate rollout counts."""

    first = segments[0]
    return RolloutSegment(
        buf0=first.buf0,
        buf1=first.buf1,
        turn_state_cache=first.turn_state_cache,
        turn_tag_p0=first.turn_tag_p0,
        turn_tag_p1=first.turn_tag_p1,
        write_idx_p0=sum((s.write_idx_p0 for s in segments), np.zeros_like(first.write_idx_p0)),
        write_idx_p1=sum((s.write_idx_p1 for s in segments), np.zeros_like(first.write_idx_p1)),
        valid_p0=first.valid_p0,
        valid_p1=first.valid_p1,
        old_logprob_p0=first.old_logprob_p0,
        old_logprob_p1=first.old_logprob_p1,
        old_value_p0=first.old_value_p0,
        old_value_p1=first.old_value_p1,
        reward_p0=np.concatenate([s.reward_p0 for s in segments], axis=0),
        reward_p1=np.concatenate([s.reward_p1 for s in segments], axis=0),
        done_p0=first.done_p0,
        done_p1=first.done_p1,
        bootstrap_p0=first.bootstrap_p0,
        bootstrap_p1=first.bootstrap_p1,
        bootstrap_valid_p0=first.bootstrap_valid_p0,
        bootstrap_valid_p1=first.bootstrap_valid_p1,
        env_steps_per_env=sum((s.env_steps_per_env for s in segments), np.zeros_like(first.env_steps_per_env)),
    )


def _rollout_timing_str(rt: RolloutTiming) -> str:
    unacc_loop = max(0.0, rt.loop_s - rt.accounted_loop_s())
    return (
        f"rollout_wall {rt.wall_s:.3f}s loop {rt.loop_s:.3f}s "
        f"env_step {rt.env_step_s:.3f}s micro_cap {rt.micro_cap_s:.3f}s obs {rt.obs_build_s:.3f}s "
        f"pt_batch {rt.policy_batch_s:.3f}s pt_fwd {rt.policy_forward_s:.3f}s micro_apply {rt.micro_apply_s:.3f}s "
        f"unstack {rt.state_unstack_s:.3f}s init {rt.init_s:.3f}s unaccounted_loop {unacc_loop:.3f}s outer {rt.outer_iters}"
    )


def _ppo_timing_str(pt: PPOTiming) -> str:
    unacc = max(0.0, pt.total_s - pt.accounted_s())
    return (
        f"ppo_wall {pt.total_s:.3f}s mb {pt.n_minibatches} "
        f"gather {pt.gather_s:.3f}s replay_jax {pt.replay_jax_s:.3f}s "
        f"replay_dlpack {pt.replay_dlpack_s:.3f}s compiled_loss {pt.compiled_loss_s:.3f}s "
        f"backward {pt.backward_s:.3f}s optim {pt.optim_s:.3f}s sync {pt.sync_s:.3f}s "
        f"unaccounted {unacc:.3f}s"
    )


def _segment_rollout_counts(segment: RolloutSegment) -> Tuple[int, int, int, float]:
    """``micro_p0``, total micro transitions, env_steps, mean_r0."""

    total_p0 = int(segment.write_idx_p0.sum())
    total_p1 = int(segment.write_idx_p1.sum())
    total_env_steps = int(segment.env_steps_per_env.sum())
    if total_env_steps > 0:
        mean_r0 = float(segment.reward_p0.sum() / total_env_steps)
    else:
        mean_r0 = 0.0
    return total_p0, total_p0 + total_p1, total_env_steps, mean_r0


def _fleet_counts_from_state(state_b: OrbitWarsState) -> Tuple[int, float]:
    """Return total and per-env mean active fleets from the batched state."""

    active = np.asarray(jax.device_get(state_b.fleet_active))
    total_fleets = int(active.sum())
    mean_fleets_per_env = float(total_fleets / max(1, active.shape[0]))
    return total_fleets, mean_fleets_per_env


def _rollout_game_stats_str(gs: RolloutGameStats) -> str:
    """Human-readable episode outcomes completed inside this rollout segment."""

    if gs.n_completed == 0:
        return "games_done 0"
    nc = float(gs.n_completed)
    return (
        f"games_done {gs.n_completed} p_timeout {gs.n_step_limit / nc:.2f} "
        f"p_decisive {gs.n_decisive / nc:.2f} avg_env_turns {gs.sum_episode_turns / nc:.1f} "
        f"avg_final_ships_p01 ({gs.sum_final_ships_p0 / nc:.1f},{gs.sum_final_ships_p1 / nc:.1f}) "
        f"p_win_reward_p01 ({gs.n_p0_positive_reward / nc:.2f},{gs.n_p1_positive_reward / nc:.2f})"
    )


def _print_rollout_pre_ppo(
    it: int,
    num_envs: int,
    cfg_max_fleets: int,
    segment: RolloutSegment,
    rt: RolloutTiming,
    game_stats: RolloutGameStats,
) -> None:
    """Rollout summary + phase timings before GAE / PPO (so long PPO runs do not hide rollout metrics)."""

    total_p0, total_micro, total_env_steps, mean_r0 = _segment_rollout_counts(segment)
    rw = max(1e-9, rt.wall_s)
    print(
        f"iter {it:4d} rollout (pre-PPO) envs {num_envs} micro_p0 {total_p0:5d} mean_r0 {mean_r0:.6f} "
        f"max_fleets {cfg_max_fleets} micro_steps {total_micro} micro/s {total_micro / rw:.1f} "
        f"env_steps {total_env_steps} env/s {total_env_steps / rw:.1f} | {_rollout_game_stats_str(game_stats)} "
        f"| {_rollout_timing_str(rt)}",
        flush=True,
    )


def ppo_iteration(
    policy: OrbitWarsPolicy,
    opt: torch.optim.Optimizer,
    segment: RolloutSegment,
    samples: dict,
    device: torch.device,
    minibatch_size: int,
    ppo_epochs: int,
    clip_eps: float,
    vf_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    ship_speed: float,
    first_hit_n_rays: int,
    first_hit_ray_chunk_size: int,
    *,
    rnd: np.random.Generator,
    loss_fn: Optional[Any] = None,
    amp_dtype: Optional[torch.dtype] = None,
) -> Tuple[float, PPOTiming, PPOStats]:
    """Multiple epochs of clipped PPO surrogate with minibatches.

    Minibatches are gathered directly from device-resident
    ``TransitionBuffer`` via ``gather_minibatch`` — no host stacking.
    Returns ``(mean_total_loss, ppo_timing, ppo_stats)``.
    """

    advantages = samples["advantages"]
    returns = samples["returns"]
    players = samples["players"]
    t_idx = samples["t_idx"]
    n_idx = samples["n_idx"]
    old_logprob = samples["old_logprob"]
    old_value = samples["old_value"]

    n = advantages.shape[0]
    idx = np.arange(n)
    total_loss_sum = 0.0
    n_mb = 0
    timing = PPOTiming()
    stats = PPOStats()
    t_total0 = perf_counter()

    for _ in range(ppo_epochs):
        rnd.shuffle(idx)
        for start in range(0, n, minibatch_size):
            mb_idx = idx[start : start + minibatch_size]

            t0 = perf_counter()
            mb_player = players[mb_idx]
            mb_t = t_idx[mb_idx]
            mb_n = n_idx[mb_idx]

            mb_player_j = jnp.asarray(mb_player, dtype=jnp.int32)
            mb_t_j = jnp.asarray(mb_t, dtype=jnp.int32)
            mb_n_j = jnp.asarray(mb_n, dtype=jnp.int32)

            max_m = int(segment.buf0.micro_halt_now.shape[2])
            (
                state_b,
                halt_action_j,
                pair_flat_j,
                frac_idx_j,
                no_valid_pairs_j,
                no_valid_fracs_j,
                must_halt_no_ships_j,
                target_planet_reachable_j,
            ) = gather_minibatch(
                segment.buf0,
                segment.buf1,
                mb_player_j,
                mb_t_j,
                mb_n_j,
                segment.turn_state_cache,
                segment.turn_tag_p0,
                segment.turn_tag_p1,
                max_m,
            )

            ego_b_j = mb_player_j

            adv = torch.as_tensor(advantages[mb_idx], device=device, dtype=torch.float32)
            ret_t = torch.as_tensor(returns[mb_idx], device=device, dtype=torch.float32)
            old_logp = torch.as_tensor(old_logprob[mb_idx], device=device, dtype=torch.float32)
            old_v = torch.as_tensor(old_value[mb_idx], device=device, dtype=torch.float32)
            timing.gather_s += perf_counter() - t0

            loss, mb_stats = replay_ppo_loss(
                state_b=state_b,
                halt_action=halt_action_j,
                pair_flat=pair_flat_j,
                frac_idx=frac_idx_j,
                no_valid_pairs=no_valid_pairs_j,
                no_valid_fracs=no_valid_fracs_j,
                must_halt_no_ships=must_halt_no_ships_j,
                ego_b=ego_b_j,
                adv=adv,
                returns=ret_t,
                old_logp=old_logp,
                old_v=old_v,
                policy=policy,
                ship_speed=ship_speed,
                first_hit_n_rays=first_hit_n_rays,
                first_hit_ray_chunk_size=first_hit_ray_chunk_size,
                clip_eps=clip_eps,
                vf_coef=vf_coef,
                entropy_coef=entropy_coef,
                loss_fn=loss_fn,
                timing=timing,
                amp_dtype=amp_dtype,
                target_planet_reachable=target_planet_reachable_j,
            )

            t0 = perf_counter()
            opt.zero_grad()
            loss.backward()
            # ``clip_grad_norm_`` returns the total parameter grad L2 norm
            # *pre-clip*, which is what we want as a diagnostic.
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            )
            timing.backward_s += perf_counter() - t0

            t0 = perf_counter()
            opt.step()
            timing.optim_s += perf_counter() - t0

            t0 = perf_counter()
            total_loss_sum += float(loss.item())
            stats.update(mb_stats, grad_norm)
            timing.sync_s += perf_counter() - t0
            n_mb += 1

    timing.n_minibatches = n_mb
    timing.total_s = perf_counter() - t_total0
    return total_loss_sum / max(1, n_mb), timing, stats


def _host_selected_minibatch(segment: RolloutSegment, mb_player: np.ndarray, mb_t: np.ndarray, mb_n: np.ndarray):
    """Select minibatch rows from a CPU-resident rollout chunk and place them on device."""

    is_p0 = mb_player == 0
    turn_idx = np.where(is_p0, segment.turn_tag_p0[mb_t, mb_n], segment.turn_tag_p1[mb_t, mb_n])
    state_host = jax.tree.map(lambda leaf: leaf[turn_idx, mb_n], segment.turn_state_cache)

    buf0: TransitionBuffer = segment.buf0
    buf1: TransitionBuffer = segment.buf1

    def _sel_plane(field0: np.ndarray, field1: np.ndarray) -> np.ndarray:
        a = field0[mb_t, mb_n, :]
        b = field1[mb_t, mb_n, :]
        return np.where(is_p0[:, None], a, b)

    def _sel_scalar(field0: np.ndarray, field1: np.ndarray) -> np.ndarray:
        a = field0[mb_t, mb_n]
        b = field1[mb_t, mb_n]
        return np.where(is_p0, a, b)

    phase_micro_idx = _sel_scalar(buf0.phase_micro_idx, buf1.phase_micro_idx).astype(np.int32)
    max_m = int(buf0.micro_halt_now.shape[2])

    return gather_minibatch_selected(
        jax.tree.map(jnp.asarray, state_host),
        jnp.asarray(mb_player, dtype=jnp.int32),
        max_m,
        jnp.asarray(_sel_plane(buf0.micro_halt_now, buf1.micro_halt_now), dtype=jnp.bool_),
        jnp.asarray(_sel_plane(buf0.send, buf1.send), dtype=jnp.float32),
        jnp.asarray(_sel_plane(buf0.angle, buf1.angle), dtype=jnp.float32),
        jnp.asarray(_sel_plane(buf0.fleet_eta, buf1.fleet_eta), dtype=jnp.float32),
        jnp.asarray(_sel_plane(buf0.slot, buf1.slot), dtype=jnp.int32),
        jnp.asarray(_sel_scalar(buf0.halt_action, buf1.halt_action), dtype=jnp.int32),
        jnp.asarray(_sel_plane(buf0.pair_flat, buf1.pair_flat), dtype=jnp.int32),
        jnp.asarray(_sel_plane(buf0.frac_idx, buf1.frac_idx), dtype=jnp.int32),
        jnp.asarray(_sel_scalar(buf0.no_valid_pairs, buf1.no_valid_pairs), dtype=jnp.bool_),
        jnp.asarray(_sel_scalar(buf0.no_valid_fracs, buf1.no_valid_fracs), dtype=jnp.bool_),
        jnp.asarray(_sel_scalar(buf0.must_halt_no_ships, buf1.must_halt_no_ships), dtype=jnp.bool_),
        jnp.asarray(
            np.where(
                is_p0[:, None],
                buf0.target_planet_reachable[mb_t, mb_n, :],
                buf1.target_planet_reachable[mb_t, mb_n, :],
            ),
            dtype=jnp.bool_,
        ),
        jnp.asarray(phase_micro_idx, dtype=jnp.int32),
    )


def ppo_iteration_host_staged(
    policy: OrbitWarsPolicy,
    opt: torch.optim.Optimizer,
    chunks: list[HostRolloutChunk],
    device: torch.device,
    minibatch_size: int,
    ppo_epochs: int,
    clip_eps: float,
    vf_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    ship_speed: float,
    first_hit_n_rays: int,
    first_hit_ray_chunk_size: int,
    *,
    rnd: np.random.Generator,
    loss_fn: Optional[Any] = None,
    amp_dtype: Optional[torch.dtype] = None,
) -> Tuple[float, PPOTiming, PPOStats]:
    """PPO over CPU-resident rollout chunks, staging only minibatches to device."""

    total_loss_sum = 0.0
    n_mb = 0
    timing = PPOTiming()
    stats = PPOStats()
    t_total0 = perf_counter()

    for _ in range(ppo_epochs):
        chunk_order = np.arange(len(chunks))
        rnd.shuffle(chunk_order)
        for chunk_i in chunk_order:
            chunk = chunks[int(chunk_i)]
            samples = chunk.samples
            n = int(samples["advantages"].shape[0])
            idx = np.arange(n)
            rnd.shuffle(idx)
            for start in range(0, n, minibatch_size):
                mb_idx = idx[start : start + minibatch_size]

                t0 = perf_counter()
                mb_player = samples["players"][mb_idx]
                mb_t = samples["t_idx"][mb_idx]
                mb_n = samples["n_idx"][mb_idx]
                (
                    state_b,
                    halt_action_j,
                    pair_flat_j,
                    frac_idx_j,
                    no_valid_pairs_j,
                    no_valid_fracs_j,
                    must_halt_no_ships_j,
                    target_planet_reachable_j,
                ) = _host_selected_minibatch(chunk.segment, mb_player, mb_t, mb_n)

                ego_b_j = jnp.asarray(mb_player, dtype=jnp.int32)
                adv = torch.as_tensor(samples["advantages"][mb_idx], device=device, dtype=torch.float32)
                ret_t = torch.as_tensor(samples["returns"][mb_idx], device=device, dtype=torch.float32)
                old_logp = torch.as_tensor(samples["old_logprob"][mb_idx], device=device, dtype=torch.float32)
                old_v = torch.as_tensor(samples["old_value"][mb_idx], device=device, dtype=torch.float32)
                timing.gather_s += perf_counter() - t0

                loss, mb_stats = replay_ppo_loss(
                    state_b=state_b,
                    halt_action=halt_action_j,
                    pair_flat=pair_flat_j,
                    frac_idx=frac_idx_j,
                    no_valid_pairs=no_valid_pairs_j,
                    no_valid_fracs=no_valid_fracs_j,
                    must_halt_no_ships=must_halt_no_ships_j,
                    ego_b=ego_b_j,
                    adv=adv,
                    returns=ret_t,
                    old_logp=old_logp,
                    old_v=old_v,
                    policy=policy,
                    ship_speed=ship_speed,
                    first_hit_n_rays=first_hit_n_rays,
                    first_hit_ray_chunk_size=first_hit_ray_chunk_size,
                    clip_eps=clip_eps,
                    vf_coef=vf_coef,
                    entropy_coef=entropy_coef,
                    loss_fn=loss_fn,
                    timing=timing,
                    amp_dtype=amp_dtype,
                    target_planet_reachable=target_planet_reachable_j,
                )

                t0 = perf_counter()
                opt.zero_grad()
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm))
                timing.backward_s += perf_counter() - t0

                t0 = perf_counter()
                opt.step()
                timing.optim_s += perf_counter() - t0

                t0 = perf_counter()
                total_loss_sum += float(loss.item())
                stats.update(mb_stats, grad_norm)
                timing.sync_s += perf_counter() - t0
                n_mb += 1

    timing.n_minibatches = n_mb
    timing.total_s = perf_counter() - t_total0
    return total_loss_sum / max(1, n_mb), timing, stats


def _log_iter_tensorboard(
    writer: SummaryWriter,
    it: int,
    *,
    skipped: bool,
    iter_dt: float,
    total_env_steps: int,
    mean_r0: float,
    num_fleets: int,
    mean_fleets_per_env: float,
    cfg_max_fleets: int,
    game_stats: Optional[RolloutGameStats] = None,
    samples_s: float = 0.0,
    loss_mb: float = 0.0,
    ppo_s: float = 0.0,
    ppo_summary: Optional[dict] = None,
    rt: Optional[RolloutTiming] = None,
    ppo_t: Optional[PPOTiming] = None,
) -> None:
    writer.add_scalar("time/iter_seconds", iter_dt, it)
    writer.add_scalar("rollout/env_steps", total_env_steps, it)
    writer.add_scalar("rollout/env_steps_per_sec", total_env_steps / max(1e-9, iter_dt), it)
    writer.add_scalar("rollout/mean_r0", mean_r0, it)
    writer.add_scalar("rollout/number_of_fleets", float(num_fleets), it)
    writer.add_scalar("rollout/number_of_fleets_per_env", mean_fleets_per_env, it)
    writer.add_scalar("env/max_fleets", cfg_max_fleets, it)
    if game_stats is not None:
        writer.add_scalar("rollout/episodes_completed", float(game_stats.n_completed), it)
        if game_stats.n_completed > 0:
            nc = float(game_stats.n_completed)
            writer.add_scalar("rollout/p_episode_timeout", float(game_stats.n_step_limit) / nc, it)
            writer.add_scalar("rollout/p_episode_decisive", float(game_stats.n_decisive) / nc, it)
            writer.add_scalar(
                "rollout/avg_final_ships_p0", float(game_stats.sum_final_ships_p0) / nc, it
            )
            writer.add_scalar(
                "rollout/avg_final_ships_p1", float(game_stats.sum_final_ships_p1) / nc, it
            )
            writer.add_scalar(
                "rollout/avg_episode_env_turns", float(game_stats.sum_episode_turns) / nc, it
            )
            writer.add_scalar(
                "rollout/p_win_reward_p0", float(game_stats.n_p0_positive_reward) / nc, it
            )
            writer.add_scalar(
                "rollout/p_win_reward_p1", float(game_stats.n_p1_positive_reward) / nc, it
            )
    writer.add_scalar("train/skipped_empty_rollout", 1.0 if skipped else 0.0, it)
    if skipped:
        writer.flush()
        return
    writer.add_scalar("time/samples_gae_seconds", samples_s, it)
    writer.add_scalar("time/ppo_seconds", ppo_s, it)
    writer.add_scalar("ppo/loss_mb", loss_mb, it)
    assert ppo_summary is not None
    for k, v in ppo_summary.items():
        if isinstance(v, float) and v == v:  # skip nan
            writer.add_scalar(f"ppo/{k}", v, it)
    if rt is not None:
        writer.add_scalar("timing/rollout_wall_s", rt.wall_s, it)
        writer.add_scalar("timing/rollout_loop_s", rt.loop_s, it)
    if ppo_t is not None:
        writer.add_scalar("timing/ppo_total_s", ppo_t.total_s, it)


def train(args: argparse.Namespace) -> None:
    configure_jax_for_training(prefer_gpu=True, verbose=True)

    if args.checkpoint_every < 1:
        raise SystemExit("--checkpoint-every must be >= 1")
    if args.rollout_host_chunks < 1:
        raise SystemExit("--rollout-host-chunks must be >= 1")

    exp_dir, tb_dir, ckpt_dir = experiment_dirs(args)
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[orbit_wars_pt] experiment dir {exp_dir}", flush=True)
    print(f"[orbit_wars_pt] tensorboard run dir {tb_dir}", flush=True)
    print(
        f"[orbit_wars_pt] view all experiments: tensorboard --logdir {tb_dir.parent}",
        flush=True,
    )

    # ``OrbitWarsPolicy.forward`` packs active tokens to the front and
    # slices the batch to ``L_packed = counts.max().item()``; that scalar
    # extract introduces an unbacked SymInt under ``torch.compile``.
    # Enabling ``capture_scalar_outputs`` lets Dynamo carry the SymInt
    # through the rest of the graph instead of breaking on it. Safe in
    # PyTorch 2.4+; the only price is one host-sync per forward call to
    # materialize the int.
    import torch._dynamo  # noqa: F401  (registers config namespace)
    torch._dynamo.config.capture_scalar_outputs = True

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    # TF32 (and friends) for FP32 matmul. ``"high"`` enables Tensor Core
    # acceleration with a 10-bit mantissa multiply + FP32 accumulate; the
    # numerical impact is well under PPO's stochastic-gradient noise floor
    # and is the standard production setting for transformer training.
    # ``"medium"`` allows BF16 internally for matmul (a bit faster, slightly
    # less accurate). ``"highest"`` matches PyTorch's default (no Tensor Core).
    torch.set_float32_matmul_precision(args.matmul_precision)

    # BF16 autocast: matmul / linear / SDPA run in bf16 on Tensor Cores;
    # softmax / log_softmax / layernorm / reductions stay fp32. BF16 has
    # fp32's exponent range, so no GradScaler is needed. Restricted to
    # CUDA — CPU autocast doesn't help our bottleneck.
    amp_dtype: Optional[torch.dtype] = (
        torch.bfloat16 if (args.amp and device.type == "cuda") else None
    )
    if args.amp and amp_dtype is None:
        print(
            "[orbit_wars_pt] BF16 autocast skipped (device is not CUDA). "
            "Pass --no-amp to silence this notice.",
            flush=True,
        )
    elif amp_dtype is not None:
        print("[orbit_wars_pt] BF16 autocast enabled.", flush=True)

    mem_dbg = args.mem_debug

    resume_path: Optional[Path] = None
    if args.resume:
        resume_path = find_latest_checkpoint(ckpt_dir)
        if resume_path is not None:
            print(f"[orbit_wars_pt] found checkpoint {resume_path}, resuming", flush=True)
        else:
            print("[orbit_wars_pt] no checkpoint in experiment dir — starting fresh", flush=True)

    cfg = OrbitWarsEnvConfig(num_agents=2, max_fleets=args.max_fleets, episode_seed=args.seed)

    policy = OrbitWarsPolicy(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        activation_checkpointing=args.activation_checkpointing,
    ).to(device)
    opt = optim.Adam(policy.parameters(), lr=args.lr)

    # Compile targets:
    #   * ``compiled_loss_fn`` — the PPO consolidated function (encoder
    #     forward + masking + logp + entropy + value + clipped surrogate
    #     in a single trace). Large minibatch ⇒ AOT-autograd + Inductor
    #     fusion is a clear win.
    #   * ``policy.forward`` / ``policy.fraction_logits`` — the rollout
    #     hot loop. With per-sample-front packing the encoder is plain
    #     dense ops (no NestedTensor, no graph breaks), so wrapping the
    #     module's ``forward`` with ``torch.compile`` stays cheap and
    #     amortizes well even at small per-call batches.
    compiled_loss_fn: Optional[Any] = None
    if args.compile:
        compile_mode = args.compile_mode
        print(
            f"[orbit_wars_pt] torch.compile enabled (mode='{compile_mode}', dynamic=True): "
            f"policy.forward, policy.fraction_logits, and PPO loss.",
            flush=True,
        )
        compiled_loss_fn = torch.compile(
            compute_ppo_loss_torch, mode=compile_mode, dynamic=True
        )
        policy.forward = torch.compile(  # type: ignore[assignment]
            policy.forward, mode=compile_mode, dynamic=True
        )
        policy.fraction_logits = torch.compile(  # type: ignore[assignment]
            policy.fraction_logits, mode=compile_mode, dynamic=True
        )

    if mem_dbg:
        n_params = sum(p.numel() for p in policy.parameters())
        param_b = torch_param_bytes(policy)
        print(
            f"[mem] policy params {n_params:,} (~{param_b / (1024 * 1024):.2f} MiB weights @ {next(policy.parameters()).dtype})"
        )
        if device.type == "cuda":
            log_cuda_mem("after policy + Adam on device (incl. optimizer state views)", device)

    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    rnd = np.random.default_rng(args.seed)

    start_iter = 0
    rollout_carry: Optional[RolloutCarry] = None
    rollout_env_seed = args.seed

    if resume_path is not None:
        try:
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(resume_path, map_location="cpu")
        if int(ckpt.get("version", 0)) != CHECKPOINT_VERSION:
            raise RuntimeError(
                f"Unsupported checkpoint version {ckpt.get('version')!r} (expected {CHECKPOINT_VERSION})"
            )
        _validate_checkpoint_args(ckpt["training_args"], args)
        policy.load_state_dict(ckpt["policy"])
        opt.load_state_dict(ckpt["optimizer"])
        _restore_torch_generator_from_checkpoint(ckpt["torch_rng"], rng)
        rnd.bit_generator.state = ckpt["numpy_rng_state"]
        rollout_env_seed = int(ckpt["rollout_env_seed"])
        rc = ckpt["rollout_carry"]
        rollout_carry = _deserialize_rollout_carry(rc) if rc is not None else None
        start_iter = int(ckpt["iteration"])
        if rollout_carry is not None:
            cfg = rollout_carry.cfg
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
            )
            rollout_env_seed += heal_seeds
        print(f"[orbit_wars_pt] resumed at iteration {start_iter}", flush=True)

    reset_prefetch: Optional[RolloutResetPrefetch] = None
    if int(args.reset_prefetch_depth) > 0:
        reset_prefetch = RolloutResetPrefetch(
            int(args.reset_prefetch_workers), int(args.reset_prefetch_depth)
        )
        reset_prefetch.start()
        print(
            f"[orbit_wars_pt] reset prefetch on: workers={args.reset_prefetch_workers} "
            f"lookahead={args.reset_prefetch_depth}",
            flush=True,
        )

    writer = SummaryWriter(log_dir=str(tb_dir))
    try:
        _train_loop(
            args,
            device,
            amp_dtype,
            mem_dbg,
            cfg,
            policy,
            opt,
            compiled_loss_fn,
            rng,
            rnd,
            rollout_carry,
            rollout_env_seed,
            start_iter,
            writer,
            ckpt_dir,
            reset_prefetch,
        )
    finally:
        if reset_prefetch is not None:
            reset_prefetch.stop()
        writer.close()


def _train_loop(
    args: argparse.Namespace,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    mem_dbg: int,
    cfg: OrbitWarsEnvConfig,
    policy: OrbitWarsPolicy,
    opt: torch.optim.Optimizer,
    compiled_loss_fn: Optional[Any],
    rng: torch.Generator,
    rnd: np.random.Generator,
    rollout_carry: Optional[RolloutCarry],
    rollout_env_seed: int,
    start_iter: int,
    writer: SummaryWriter,
    ckpt_dir: Path,
    reset_prefetch: Optional[RolloutResetPrefetch],
) -> None:
    for it in range(start_iter, args.iterations):
        iter_start = time.perf_counter()
        if mem_dbg and device.type == "cuda":
            reset_peak_stats(device)
            log_cuda_mem(f"iter {it} start (peak reset)", device)

        host_chunks: Optional[list[HostRolloutChunk]] = None
        if args.rollout_storage == "host":
            chunk_segments: list[RolloutSegment] = []
            chunk_timings: list[RolloutTiming] = []
            chunk_stats: list[RolloutGameStats] = []
            host_chunks = []
            samples_t0 = time.perf_counter()
            samples_s = 0.0
            for chunk_i in range(int(args.rollout_host_chunks)):
                segment_i, rt_i, rollout_carry, seeds_used, game_stats_i = collect_parallel_micro_rollouts(
                    policy,
                    cfg,
                    args.num_envs,
                    device,
                    seed_base=rollout_env_seed,
                    rng=rng,
                    greedy=False,
                    ship_speed=args.ship_speed,
                    max_micro_steps_per_player=args.max_micro_steps,
                    rollout_micro_horizon=args.rollout_micro_horizon,
                    carry_in=rollout_carry,
                    mem_debug=mem_dbg if chunk_i == 0 else 0,
                    train_iter=it,
                    amp_dtype=amp_dtype,
                    min_max_fleets=args.max_fleets,
                    reset_prefetch=reset_prefetch,
                    first_hit_n_rays=max(8, int(args.first_hit_n_rays)),
                    first_hit_ray_chunk_size=max(0, int(args.first_hit_ray_chunk_size)),
                )
                rollout_env_seed += seeds_used
                cfg.max_fleets = rollout_carry.cfg.max_fleets
                chunk_sample_count = int(segment_i.write_idx_p0.sum() + segment_i.write_idx_p1.sum())
                t_host0 = time.perf_counter()
                host_segment_i = _rollout_segment_to_host(segment_i)
                host_transfer_s = time.perf_counter() - t_host0
                del segment_i
                _release_rollout_device_refs(device)
                chunk_segments.append(host_segment_i)
                chunk_timings.append(rt_i)
                chunk_stats.append(game_stats_i)
                print(
                    f"iter {it:4d} host rollout chunk {chunk_i + 1}/{args.rollout_host_chunks} "
                    f"raw_samples {chunk_sample_count} host_transfer {host_transfer_s:.3f}s "
                    f"| {_rollout_timing_str(rt_i)}",
                    flush=True,
                )
            chunk_sample_dicts = _build_host_chunk_samples(chunk_segments, args.gamma, args.lam)
            if chunk_sample_dicts is not None:
                host_chunks = [
                    HostRolloutChunk(segment=segment_i, samples=samples_i)
                    for segment_i, samples_i in zip(chunk_segments, chunk_sample_dicts)
                    if samples_i
                ]
                _normalize_chunk_advantages(host_chunks)
                segment = _combine_segments_for_stats(chunk_segments)
                rt = _combine_rollout_timing(chunk_timings)
                game_stats = _combine_game_stats(chunk_stats)
                samples = _concat_sample_dicts([c.samples for c in host_chunks])
                samples_s = time.perf_counter() - samples_t0
                _release_rollout_device_refs(device)
            else:
                # Keep a tiny empty segment around for the existing skipped-iteration logging.
                segment = _combine_segments_for_stats(chunk_segments)
                rt = _combine_rollout_timing(chunk_timings)
                game_stats = _combine_game_stats(chunk_stats)
                samples = None
                samples_s = time.perf_counter() - samples_t0
                _release_rollout_device_refs(device)
        else:
            seed_base = rollout_env_seed
            segment, rt, rollout_carry, seeds_used, game_stats = collect_parallel_micro_rollouts(
                policy,
                cfg,
                args.num_envs,
                device,
                seed_base=seed_base,
                rng=rng,
                greedy=False,
                ship_speed=args.ship_speed,
                max_micro_steps_per_player=args.max_micro_steps,
                rollout_micro_horizon=args.rollout_micro_horizon,
                carry_in=rollout_carry,
                mem_debug=mem_dbg,
                train_iter=it,
                amp_dtype=amp_dtype,
                min_max_fleets=args.max_fleets,
                reset_prefetch=reset_prefetch,
                first_hit_n_rays=max(8, int(args.first_hit_n_rays)),
                first_hit_ray_chunk_size=max(0, int(args.first_hit_ray_chunk_size)),
            )
            rollout_env_seed += seeds_used
            t_samples0 = time.perf_counter()
            samples = build_ppo_samples(segment, args.gamma, args.lam)
            if samples is not None:
                normalize_advantages(samples)
            samples_s = time.perf_counter() - t_samples0
        cfg.max_fleets = rollout_carry.cfg.max_fleets
        num_fleets, mean_fleets_per_env = _fleet_counts_from_state(rollout_carry.state_b)

        if mem_dbg and device.type == "cuda":
            log_cuda_mem(f"iter {it} after rollouts", device)

        _print_rollout_pre_ppo(it, args.num_envs, cfg.max_fleets, segment, rt, game_stats)

        if samples is None:
            iter_dt = max(1e-9, time.perf_counter() - iter_start)
            total_env_steps = int(segment.env_steps_per_env.sum())
            env_per_sec = total_env_steps / iter_dt
            _, _, _, mean_r0 = _segment_rollout_counts(segment)
            print(
                f"iter {it:4d} skipped (empty rollout) iter_s {iter_dt:.3f} "
                f"env_steps {total_env_steps} env/s {env_per_sec:.1f} | samples {samples_s:.3f}s | {_rollout_timing_str(rt)}"
            )
            _log_iter_tensorboard(
                writer,
                it,
                skipped=True,
                iter_dt=iter_dt,
                total_env_steps=total_env_steps,
                mean_r0=mean_r0,
                num_fleets=num_fleets,
                mean_fleets_per_env=mean_fleets_per_env,
                cfg_max_fleets=cfg.max_fleets,
                game_stats=game_stats,
            )
        else:
            if mem_dbg and device.type == "cuda":
                log_cuda_mem(
                    f"iter {it} before PPO minibatches ({samples['advantages'].shape[0]} transitions)",
                    device,
                )

            t_ppo0 = time.perf_counter()
            if host_chunks is not None:
                loss_mb, ppo_t, ppo_stats = ppo_iteration_host_staged(
                    policy,
                    opt,
                    host_chunks,
                    device,
                    args.minibatch_size,
                    args.ppo_epochs,
                    args.clip_eps,
                    args.vf_coef,
                    args.entropy_coef,
                    args.max_grad_norm,
                    args.ship_speed,
                    max(8, int(args.first_hit_n_rays)),
                    max(0, int(args.first_hit_ray_chunk_size)),
                    rnd=rnd,
                    loss_fn=compiled_loss_fn,
                    amp_dtype=amp_dtype,
                )
            else:
                loss_mb, ppo_t, ppo_stats = ppo_iteration(
                    policy,
                    opt,
                    segment,
                    samples,
                    device,
                    args.minibatch_size,
                    args.ppo_epochs,
                    args.clip_eps,
                    args.vf_coef,
                    args.entropy_coef,
                    args.max_grad_norm,
                    args.ship_speed,
                    max(8, int(args.first_hit_n_rays)),
                    max(0, int(args.first_hit_ray_chunk_size)),
                    rnd=rnd,
                    loss_fn=compiled_loss_fn,
                    amp_dtype=amp_dtype,
                )
            ppo_s = time.perf_counter() - t_ppo0

            if mem_dbg and device.type == "cuda":
                log_cuda_mem(f"iter {it} after PPO epochs", device)
                if mem_dbg >= 2 and it == start_iter:
                    print_cuda_memory_summary(device, abbreviated=True)

            total_p0, total_micro, total_env_steps, mean_r0 = _segment_rollout_counts(segment)
            iter_dt = max(1e-9, time.perf_counter() - iter_start)
            micro_per_sec = total_micro / iter_dt
            env_per_sec = total_env_steps / iter_dt
            print(
                f"iter {it:4d} envs {args.num_envs} micro_p0 {total_p0:5d} loss_mb {loss_mb:.4f} "
                f"mean_r0 {mean_r0:.6f} num_fleets {num_fleets} max_fleets {cfg.max_fleets} iter_s {iter_dt:.3f} "
                f"micro_steps {total_micro} micro/s {micro_per_sec:.1f} "
                f"env_steps {total_env_steps} env/s {env_per_sec:.1f} "
                f"| {_rollout_game_stats_str(game_stats)} "
                f"| samples+gae {samples_s:.3f}s ppo {ppo_s:.3f}s "
                f"| ppo_stats {_ppo_stats_str(ppo_stats.summary())} "
                f"| {_rollout_timing_str(rt)} | {_ppo_timing_str(ppo_t)}"
            )
            _log_iter_tensorboard(
                writer,
                it,
                skipped=False,
                iter_dt=iter_dt,
                total_env_steps=total_env_steps,
                mean_r0=mean_r0,
                num_fleets=num_fleets,
                mean_fleets_per_env=mean_fleets_per_env,
                cfg_max_fleets=cfg.max_fleets,
                game_stats=game_stats,
                samples_s=samples_s,
                loss_mb=loss_mb,
                ppo_s=ppo_s,
                ppo_summary=ppo_stats.summary(),
                rt=rt,
                ppo_t=ppo_t,
            )

        if (it + 1) % args.checkpoint_every == 0:
            ckpt_path = ckpt_dir / f"iter_{it + 1:08d}.pt"
            save_checkpoint(
                ckpt_path,
                next_iteration=it + 1,
                policy=policy,
                opt=opt,
                rng=rng,
                rnd=rnd,
                rollout_env_seed=rollout_env_seed,
                rollout_carry=rollout_carry,
                args=args,
            )
            print(f"[orbit_wars_pt] saved checkpoint {ckpt_path}", flush=True)

        writer.flush()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name; artifacts go under <experiment-root>/<experiment>/.",
    )
    p.add_argument(
        "--experiment-root",
        type=str,
        default="experiments",
        help="Directory (relative to repo root if not absolute) holding each experiment and "
        "a shared tensorboard/ tree: tensorboard --logdir <experiment-root>/tensorboard shows all runs.",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=20,
        help="Save a full resume checkpoint every N completed iterations.",
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, load latest iter_*.pt under the experiment checkpoints dir when present.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iterations", type=int, default=100000)
    p.add_argument("--num-envs", type=int, default=256, help="Parallel env rollouts per iteration.")
    p.add_argument(
        "--max-micro-steps",
        type=int,
        default=16,
        help="Max fleet launches per player per turn (within one env-step).",
    )
    p.add_argument(
        "--rollout-micro-horizon",
        type=int,
        default=128,
        help=(
            "Stop a rollout segment after a full env-step when any env's player-0 or player-1 "
            "micro-step count in this segment reaches this value (end-of-turn cut). "
            "State carries to the next iteration; GAE bootstraps truncated trajectories."
        ),
    )
    p.add_argument(
        "--rollout-storage",
        type=str,
        default="device",
        choices=("device", "host"),
        help=(
            "Where completed rollout chunks live during PPO. 'device' is the existing fastest "
            "path. 'host' spills each rollout chunk to CPU RAM and stages only PPO minibatches "
            "back to the accelerator, allowing longer effective rollouts at higher transfer cost."
        ),
    )
    p.add_argument(
        "--rollout-host-chunks",
        type=int,
        default=1,
        help=(
            "Number of device-sized rollout chunks to collect per PPO iteration when "
            "--rollout-storage=host. Effective rollout length is roughly "
            "rollout_micro_horizon * rollout_host_chunks."
        ),
    )
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--clip-eps", type=float, default=0.2, help="PPO ratio + value clipping ε.")
    p.add_argument("--ppo-epochs", type=int, default=4, help="Passes over rollout data per iteration.")
    p.add_argument(
        "--minibatch-size",
        type=int,
        default=512,
        help="Transitions per minibatch (capped automatically if buffer smaller).",
    )
    p.add_argument("--ship-speed", type=float, default=6.0)
    p.add_argument(
        "--first-hit-n-rays",
        type=int,
        default=256,
        help="Virtual launch directions for discrete first-hit target geometry (rollout + PPO replay). "
        "Lower values reduce JAX GPU memory (e.g. 512); minimum 8.",
    )
    p.add_argument(
        "--first-hit-ray-chunk-size",
        type=int,
        default=64,
        help=(
            "If >0 and smaller than --first-hit-n-rays, process discrete first-hit geometry "
            "in JAX ray chunks of this size to lower peak XLA temporary memory. 0 keeps "
            "the current full-ray implementation."
        ),
    )
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recompute transformer block activations during backward to reduce GPU memory usage.",
    )
    p.add_argument("--max-fleets", type=int, default=128)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    p.add_argument("--device", type=str, default=default_device)
    p.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile policy.forward, policy.fraction_logits, and the PPO consolidated "
        "loss with torch.compile (default on).",
    )
    p.add_argument(
        "--compile-mode",
        type=str,
        default="default",
        choices=("default", "reduce-overhead", "max-autotune"),
        help="torch.compile mode. 'reduce-overhead' uses CUDA Graphs (requires static shapes; "
        "may not work with our dynamic batch dim). 'default' is the safe choice; 'max-autotune' "
        "does extra search for kernel autotuning.",
    )
    p.add_argument(
        "--matmul-precision",
        type=str,
        default="high",
        choices=("highest", "high", "medium"),
        help="torch.set_float32_matmul_precision. 'high' enables TF32 Tensor Cores for FP32 "
        "matmul (recommended for transformer training; the standard production setting). "
        "'medium' uses BF16 internally for matmul (a bit faster, slightly less accurate). "
        "'highest' is PyTorch's default (no Tensor Core acceleration).",
    )
    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wrap policy forward and PPO loss in BF16 autocast (default on; CUDA only "
        "— auto-disabled on CPU with a notice). Enables FlashAttention 2 inside SDPA "
        "and runs matmuls / linears in BF16 on Tensor Cores; softmax / norms stay FP32. "
        "No GradScaler needed since BF16 has FP32's exponent range.",
    )
    p.add_argument(
        "--mem-debug",
        type=int,
        default=0,
        choices=(0, 1, 2),
        help="GPU memory instrumentation: 0=off, 1=segment deltas + rollout pins on iter 0, 2=also torch memory_summary after iter 0 step.",
    )
    p.add_argument(
        "--reset-prefetch-depth",
        type=int,
        default=256,
        help="If >0, spawn CPU worker processes to run Kaggle-backed resets ahead of the "
        "rollout (overlaps with policy/env work). 0 disables (default).",
    )
    p.add_argument(
        "--reset-prefetch-workers",
        type=int,
        default=4,
        help="Number of spawn workers for --reset-prefetch-depth (each uses JAX on CPU).",
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())

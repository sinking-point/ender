"""Phase 5 PPO replay: pure device path.

Inputs are minibatch JAX arrays (already gathered from the device-resident
``TransitionBuffer`` via ``gather_minibatch``). No host stacking, no host
upload. Computes log-probs / value / entropy of the stored actions under
the current policy.

Two entry points for the torch side:

* ``_compute_logp_value_entropy_torch``: the diagnostic-friendly variant
  used by ``check_micro_jax.py`` to inspect ``new_logp`` / ``new_value``
  / ``new_entropy`` directly.
* ``compute_ppo_loss_torch``: the full PPO scalar loss (forward + masking
  + logp + entropy + value + clipped surrogate + value clip + entropy
  bonus) in *one* function, designed as a single ``torch.compile`` target
  so Inductor can fuse across the whole forward+loss chain. The
  corresponding backward graph is captured by AOT-autograd when the
  compiled function is called inside an autograd-enabled context.
"""

from __future__ import annotations

from contextlib import nullcontext
from time import perf_counter
from typing import Any, Callable, Dict, Optional, Tuple

import jax.numpy as jnp
import torch

from orbit_wars_pt.batched_env import obs_jax_to_torch
from orbit_wars_pt.constants import FRACTIONS, MAX_PLANETS
from orbit_wars_pt.micro_jax import selected_origin_fraction_targets_batched
from orbit_wars_pt.model import OrbitWarsPolicy
from orbit_wars_pt.observation_jax import build_observation_batched_jax_per_ego


def _compute_logp_value_entropy_torch(
    policy: OrbitWarsPolicy,
    entity_type: torch.Tensor,
    owner_idx: torch.Tensor,
    features: torch.Tensor,
    rope_pos: torch.Tensor,
    entity_mask: torch.Tensor,
    planet_mask: torch.Tensor,
    target_valid: torch.Tensor,
    target_overflow: torch.Tensor,
    halt_action: torch.Tensor,
    pair_flat: torch.Tensor,
    frac_idx: torch.Tensor,
    no_valid_pairs: torch.Tensor,
    no_valid_fracs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward + masking + log-prob/entropy/value (pure torch).

    Used both as a diagnostic entry (``replay_logprob_value_entropy_jax``)
    and as the body of the consolidated ``compute_ppo_loss_torch``.

    Manual log-prob / entropy (no ``torch.distributions.Categorical``).
    Categorical's Python ``_validate_args`` and lazy ``probs`` / ``logits``
    caching trigger graph breaks under ``torch.compile``, splitting an
    otherwise-fusable region. The math is identical: for masked variants
    (origin/fraction, target) the masked positions get ``logits = -1e4`` before
    ``log_softmax``, so their probabilities underflow to 0 and contribute
    0 to the entropy ``-sum(p * lp)`` (no nan / -inf hazard).
    """

    out = policy(
        entity_type=entity_type,
        owner_idx=owner_idx,
        features=features,
        rope_pos=rope_pos,
        entity_mask=entity_mask,
        planet_mask=planet_mask,
    )

    halt_logits = out["halt_logits"]
    halt_lp = torch.log_softmax(halt_logits, dim=-1)
    new_halt_logp = halt_lp.gather(1, halt_action[:, None]).squeeze(-1)
    new_halt_entropy = -(halt_lp.exp() * halt_lp).sum(dim=-1)

    P = MAX_PLANETS
    o_idx = pair_flat // P
    d_idx = pair_flat % P
    origin_frac_flat = o_idx * len(FRACTIONS) + frac_idx

    origin_frac_mask = out["origin_frac_mask"]
    origin_frac_logits = out["origin_frac_logits"].flatten(start_dim=1)
    origin_frac_mask_flat = origin_frac_mask.flatten(start_dim=1)
    masked_origin_frac = origin_frac_logits.masked_fill(~origin_frac_mask_flat, -1e4)
    origin_frac_lp = torch.log_softmax(masked_origin_frac, dim=-1)
    new_origin_frac_logp = origin_frac_lp.gather(1, origin_frac_flat[:, None]).squeeze(-1)
    new_origin_frac_entropy = -(origin_frac_lp.exp() * origin_frac_lp).sum(dim=-1)

    mb = halt_action.shape[0]
    n_idx = torch.arange(mb, device=halt_action.device)
    ph = out["planet_hidden"]
    target_logits = policy.target_logits_for_origin_fraction(ph, o_idx, frac_idx)
    target_mask = (
        out["pair_mask"][n_idx, o_idx, :]
        & target_valid
        & ~target_overflow[:, None].to(dtype=torch.bool)
    )
    masked_target = target_logits.masked_fill(~target_mask, -1e4)
    target_lp = torch.log_softmax(masked_target, dim=-1)
    new_target_logp = target_lp.gather(1, d_idx[:, None]).squeeze(-1)
    new_target_entropy = -(target_lp.exp() * target_lp).sum(dim=-1)

    origin_frac_used = (halt_action == 0) & ~no_valid_fracs
    target_used = origin_frac_used & ~no_valid_pairs

    new_logp = (
        new_halt_logp
        + origin_frac_used.float() * new_origin_frac_logp
        + target_used.float() * new_target_logp
    )
    new_entropy = (
        new_halt_entropy
        + origin_frac_used.float() * new_origin_frac_entropy
        + target_used.float() * new_target_entropy
    )
    # Cast value back to fp32 at the autocast boundary: the value-clip
    # baseline (``old_v``) and target (``returns``) are fp32, and we want
    # the squared-error / clip math to run in fp32 to avoid bf16 rounding
    # in the loss scalar.
    return new_logp, out["value"].float(), new_entropy


def compute_ppo_loss_torch(
    policy: OrbitWarsPolicy,
    entity_type: torch.Tensor,
    owner_idx: torch.Tensor,
    features: torch.Tensor,
    rope_pos: torch.Tensor,
    entity_mask: torch.Tensor,
    planet_mask: torch.Tensor,
    target_valid: torch.Tensor,
    target_overflow: torch.Tensor,
    halt_action: torch.Tensor,
    pair_flat: torch.Tensor,
    frac_idx: torch.Tensor,
    no_valid_pairs: torch.Tensor,
    no_valid_fracs: torch.Tensor,
    must_halt_no_ships: torch.Tensor,
    adv: torch.Tensor,
    returns: torch.Tensor,
    old_logp: torch.Tensor,
    old_v: torch.Tensor,
    clip_eps: float,
    vf_coef: float,
    entropy_coef: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Full PPO scalar loss + diagnostics on one minibatch.

    Bundles forward + masking + logp + entropy + value + clipped surrogate
    + value clip + entropy bonus into one Python function. The whole
    chain becomes one Dynamo trace and one fused Inductor graph; AOT
    autograd captures the corresponding backward when ``loss.backward()``
    is called by the caller.

    Returns ``(loss, stats)`` where ``loss`` is the scalar with-grad
    objective and ``stats`` is a fixed-key dict of detached scalar
    tensors for logging:

    * ``loss_pi`` — clipped surrogate (mean over samples that are not
      ``must_halt_no_ships``).
    * ``loss_vf`` — clipped value MSE (mean, x0.5 not applied — matches
      what the optimizer sees).
    * ``entropy`` — mean policy entropy over the same non-forced subset.
    * ``approx_kl`` — ``((old_logp - new_logp) * w_pi).sum() / w_sum`` over
      non-forced rows.
    * ``approx_kl_k3`` — masked mean of ``ratio - 1 - log_ratio``.
    * ``clip_frac`` — masked mean fraction where ``|ratio - 1| > clip_eps``
      (i.e. the clip actually bit); excludes ``must_halt_no_ships`` rows.
    * ``value_mean`` — mean of predicted values (sanity).
    * ``diff_sq_sum`` / ``ret_sum`` / ``ret_sq_sum`` / ``count`` —
      sufficient statistics for pooled explained-variance over all
      minibatches and PPO epochs.

    Diagnostics live inside ``torch.no_grad()`` and are detached so they
    don't extend the autograd graph; under ``torch.compile`` they are
    just extra return values in the same trace.
    """

    new_logp, new_value, new_entropy = _compute_logp_value_entropy_torch(
        policy,
        entity_type,
        owner_idx,
        features,
        rope_pos,
        entity_mask,
        planet_mask,
        target_valid,
        target_overflow,
        halt_action,
        pair_flat,
        frac_idx,
        no_valid_pairs,
        no_valid_fracs,
    )

    log_ratio = new_logp - old_logp
    ratio = torch.exp(log_ratio)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    # No policy gradient through halt when the env forced halt (no owned ships).
    w_pi = (~must_halt_no_ships).to(dtype=torch.float32)
    w_sum = w_pi.sum().clamp(min=1.0)
    loss_pi = -(torch.min(surr1, surr2) * w_pi).sum() / w_sum

    v_clipped = old_v + (new_value - old_v).clamp(-clip_eps, clip_eps)
    loss_vf = torch.max((new_value - returns).pow(2), (v_clipped - returns).pow(2)).mean()

    entropy = (new_entropy * w_pi).sum() / w_sum
    loss_ent = -entropy_coef * entropy

    loss = loss_pi + vf_coef * loss_vf + loss_ent

    with torch.no_grad():
        approx_kl = (-log_ratio * w_pi).sum() / w_sum
        approx_kl_k3 = ((ratio - 1.0 - log_ratio) * w_pi).sum() / w_sum
        clip_frac = (((ratio - 1.0).abs() > clip_eps).float() * w_pi).sum() / w_sum
        value_mean = new_value.mean()
        diff_sq_sum = (returns - new_value).pow(2).sum()
        ret_sum = returns.sum()
        ret_sq_sum = returns.pow(2).sum()
        count = torch.full((), float(returns.numel()), device=returns.device, dtype=torch.float32)

    stats: Dict[str, torch.Tensor] = {
        "loss_pi": loss_pi.detach(),
        "loss_vf": loss_vf.detach(),
        "entropy": entropy.detach(),
        "approx_kl": approx_kl,
        "approx_kl_k3": approx_kl_k3,
        "clip_frac": clip_frac,
        "value_mean": value_mean,
        "diff_sq_sum": diff_sq_sum,
        "ret_sum": ret_sum,
        "ret_sq_sum": ret_sq_sum,
        "count": count,
    }
    return loss, stats


def _jax_preamble_to_torch(
    *,
    state_b,
    halt_action: jnp.ndarray,
    pair_flat: jnp.ndarray,
    frac_idx: jnp.ndarray,
    no_valid_pairs: jnp.ndarray,
    no_valid_fracs: jnp.ndarray,
    ego_b: jnp.ndarray,
    ship_speed: float,
    timing: Optional[Any],
):
    """Run the JAX side (obs builder + pair geom) and dlpack-import results.

    Returns a dict of torch tensors plus the converted action / mask tensors.
    Any per-phase wall times accumulate into ``timing`` if supplied.
    """

    t0 = perf_counter()
    obs_jax = build_observation_batched_jax_per_ego(state_b, ego_b, ship_speed)
    origin_idx = pair_flat // MAX_PLANETS
    _angle_j, _width_j, target_valid_j, overflow_j = selected_origin_fraction_targets_batched(
        state_b,
        origin_idx.astype(jnp.int32),
        frac_idx.astype(jnp.int32),
        horizon=24,
        ship_speed=ship_speed,
        samples_per_span=17,
    )
    if timing is not None:
        timing.replay_jax_s += perf_counter() - t0

    t0 = perf_counter()
    obs_torch = obs_jax_to_torch(obs_jax)
    target_valid_t = torch.from_dlpack(target_valid_j)
    target_overflow_t = torch.from_dlpack(overflow_j)
    halt_action_t = torch.from_dlpack(halt_action).to(torch.long)
    pair_flat_t = torch.from_dlpack(pair_flat).to(torch.long)
    frac_idx_t = torch.from_dlpack(frac_idx).to(torch.long)
    no_valid_pairs_t = torch.from_dlpack(no_valid_pairs)
    no_valid_fracs_t = torch.from_dlpack(no_valid_fracs)
    if timing is not None:
        timing.replay_dlpack_s += perf_counter() - t0

    return (
        obs_torch,
        target_valid_t,
        target_overflow_t,
        halt_action_t,
        pair_flat_t,
        frac_idx_t,
        no_valid_pairs_t,
        no_valid_fracs_t,
    )


def replay_logprob_value_entropy_jax(
    *,
    state_b,
    halt_action: jnp.ndarray,
    pair_flat: jnp.ndarray,
    frac_idx: jnp.ndarray,
    no_valid_pairs: jnp.ndarray,
    no_valid_fracs: jnp.ndarray,
    ego_b: jnp.ndarray,
    policy: OrbitWarsPolicy,
    device: torch.device,
    ship_speed: float = 6.0,
    timing: Optional[Any] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Diagnostic entry: returns ``(new_logp, new_value, new_entropy)``.

    Used by ``check_micro_jax.py``; the live training loop uses
    ``replay_ppo_loss`` instead so the whole forward+loss chain can be a
    single ``torch.compile`` target.
    """

    (
        obs_torch,
        target_valid_t,
        target_overflow_t,
        halt_action_t,
        pair_flat_t,
        frac_idx_t,
        no_valid_pairs_t,
        no_valid_fracs_t,
    ) = _jax_preamble_to_torch(
        state_b=state_b,
        halt_action=halt_action,
        pair_flat=pair_flat,
        frac_idx=frac_idx,
        no_valid_pairs=no_valid_pairs,
        no_valid_fracs=no_valid_fracs,
        ego_b=ego_b,
        ship_speed=ship_speed,
        timing=timing,
    )

    return _compute_logp_value_entropy_torch(
        policy,
        obs_torch["entity_type"],
        obs_torch["owner_idx"],
        obs_torch["features"],
        obs_torch["rope_pos"],
        obs_torch["entity_mask"],
        obs_torch["planet_mask"],
        target_valid_t,
        target_overflow_t,
        halt_action_t,
        pair_flat_t,
        frac_idx_t,
        no_valid_pairs_t,
        no_valid_fracs_t,
    )


def replay_ppo_loss(
    *,
    state_b,
    halt_action: jnp.ndarray,
    pair_flat: jnp.ndarray,
    frac_idx: jnp.ndarray,
    no_valid_pairs: jnp.ndarray,
    no_valid_fracs: jnp.ndarray,
    must_halt_no_ships: jnp.ndarray,
    ego_b: jnp.ndarray,
    adv: torch.Tensor,
    returns: torch.Tensor,
    old_logp: torch.Tensor,
    old_v: torch.Tensor,
    policy: OrbitWarsPolicy,
    ship_speed: float,
    clip_eps: float,
    vf_coef: float,
    entropy_coef: float,
    loss_fn: Optional[Callable] = None,
    timing: Optional[Any] = None,
    amp_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """JAX preamble + dlpack + (optionally compiled) PPO loss.

    The torch-side loss computation is delegated to ``loss_fn`` which is
    either the eager ``compute_ppo_loss_torch`` or its ``torch.compile``'d
    counterpart (provided by the training loop). Returns
    ``(loss, stats)``; ``backward()`` / ``opt.step()`` happen in the
    caller.

    ``amp_dtype`` (typically ``torch.bfloat16``) wraps the loss call in
    ``torch.autocast("cuda", dtype=amp_dtype)`` so matmuls run on Tensor
    Cores in reduced precision. BF16 has fp32's exponent range so no
    GradScaler is needed; numerically-sensitive ops (softmax, log_softmax,
    layernorm, norms / reductions) are kept in fp32 by autocast's op list.
    """

    fn = loss_fn if loss_fn is not None else compute_ppo_loss_torch

    (
        obs_torch,
        target_valid_t,
        target_overflow_t,
        halt_action_t,
        pair_flat_t,
        frac_idx_t,
        no_valid_pairs_t,
        no_valid_fracs_t,
    ) = _jax_preamble_to_torch(
        state_b=state_b,
        halt_action=halt_action,
        pair_flat=pair_flat,
        frac_idx=frac_idx,
        no_valid_pairs=no_valid_pairs,
        no_valid_fracs=no_valid_fracs,
        ego_b=ego_b,
        ship_speed=ship_speed,
        timing=timing,
    )
    must_halt_no_ships_t = torch.from_dlpack(must_halt_no_ships).to(
        device=adv.device, dtype=torch.bool
    )

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if amp_dtype is not None
        else nullcontext()
    )

    t0 = perf_counter()
    with amp_ctx:
        loss, stats = fn(
            policy,
            obs_torch["entity_type"],
            obs_torch["owner_idx"],
            obs_torch["features"],
            obs_torch["rope_pos"],
            obs_torch["entity_mask"],
            obs_torch["planet_mask"],
            target_valid_t,
            target_overflow_t,
            halt_action_t,
            pair_flat_t,
            frac_idx_t,
            no_valid_pairs_t,
            no_valid_fracs_t,
            must_halt_no_ships_t,
            adv,
            returns,
            old_logp,
            old_v,
            clip_eps,
            vf_coef,
            entropy_coef,
        )
    if timing is not None:
        timing.compiled_loss_s += perf_counter() - t0
    return loss, stats

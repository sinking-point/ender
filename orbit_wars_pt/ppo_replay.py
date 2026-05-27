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
from orbit_wars_pt.compressed_observation import CompressedObservationBuffer, decode_observation
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
    target_hit_tick: torch.Tensor,
    halt_action: torch.Tensor,
    pair_flat: torch.Tensor,
    frac_idx: torch.Tensor,
    no_valid_pairs: torch.Tensor,
    no_valid_fracs: torch.Tensor,
    population_idx: Optional[torch.Tensor] = None,
    member_counts: Optional[torch.Tensor] = None,
    value_head_idx: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
]:
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

    Returns
    -------
    ``(new_logp, new_value, new_entropy, halt_entropy, origin_frac_entropy,
    target_entropy, origin_frac_used, target_used)``. The first three are the
    quantities the diagnostic / loss path consumes; the latter five are kept
    so the loss path can report a per-head entropy breakdown (halt / origin+
    fraction / target) conditioned on the rows where each head is sampled.
    """

    P = MAX_PLANETS
    o_idx = pair_flat // P
    d_idx = pair_flat % P
    origin_frac_flat = o_idx * len(FRACTIONS) + frac_idx

    mb = halt_action.shape[0]
    n_idx = torch.arange(mb, device=halt_action.device)
    ships = features[:, 1 : 1 + MAX_PLANETS, 1] * 1000.0
    fleet_size = torch.floor(features.new_tensor(FRACTIONS)[frac_idx] * ships[n_idx, o_idx])
    if member_counts is not None and population_idx is not None and int(getattr(policy, "population_size", 1)) > 1:
        out = policy.forward_ppo_sorted_population(
            entity_type=entity_type,
            owner_idx=owner_idx,
            features=features,
            rope_pos=rope_pos,
            entity_mask=entity_mask,
            planet_mask=planet_mask,
            population_idx=population_idx,
            origin_idx=o_idx,
            frac_idx=frac_idx,
            fleet_size=fleet_size,
            target_eta=target_hit_tick,
            target_ships=ships,
            value_head_idx=value_head_idx,
        )
    else:
        out = policy(
            entity_type=entity_type,
            owner_idx=owner_idx,
            features=features,
            rope_pos=rope_pos,
            entity_mask=entity_mask,
            planet_mask=planet_mask,
            population_idx=population_idx,
            value_head_idx=value_head_idx,
        )
        ph = out["planet_hidden"]
        target_logits = policy.target_logits_for_origin_fraction(
            ph,
            o_idx,
            frac_idx,
            fleet_size=fleet_size,
            target_eta=target_hit_tick,
            target_ships=ships,
            population_idx=population_idx,
        )

    halt_logits = out["halt_logits"]
    halt_lp = torch.log_softmax(halt_logits, dim=-1)
    new_halt_logp = halt_lp.gather(1, halt_action[:, None]).squeeze(-1)
    new_halt_entropy = -(halt_lp.exp() * halt_lp).sum(dim=-1)

    origin_frac_mask = out["origin_frac_mask"]
    origin_frac_logits = out["origin_frac_logits"].flatten(start_dim=1)
    origin_frac_mask_flat = origin_frac_mask.flatten(start_dim=1)
    masked_origin_frac = origin_frac_logits.masked_fill(~origin_frac_mask_flat, -1e4)
    origin_frac_lp = torch.log_softmax(masked_origin_frac, dim=-1)
    new_origin_frac_logp = origin_frac_lp.gather(1, origin_frac_flat[:, None]).squeeze(-1)
    new_origin_frac_entropy = -(origin_frac_lp.exp() * origin_frac_lp).sum(dim=-1)

    if "target_logits" in out:
        target_logits = out["target_logits"]
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
    return (
        new_logp,
        out["value"].float(),
        new_entropy,
        new_halt_entropy,
        new_origin_frac_entropy,
        new_target_entropy,
        origin_frac_used,
        target_used,
    )


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
    target_hit_tick: torch.Tensor,
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
    population_idx: Optional[torch.Tensor] = None,
    member_counts: Optional[torch.Tensor] = None,
    value_head_idx: Optional[torch.Tensor] = None,
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
    * ``entropy_halt`` / ``entropy_origin_frac`` / ``entropy_target`` —
      conditional mean entropy of each head over the (non-forced) rows
      where that head was actually sampled. The halt head is always
      sampled when not forced; origin+fraction is sampled when
      ``halt_action == 0`` and at least one fraction is valid; target is
      sampled when origin+fraction sampled and at least one pair is valid.
      Each is set to ``nan`` when its conditioning set is empty for the
      minibatch (rare but possible early in training).
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

    (
        new_logp,
        new_value,
        new_entropy,
        new_halt_entropy,
        new_origin_frac_entropy,
        new_target_entropy,
        origin_frac_used,
        target_used,
    ) = _compute_logp_value_entropy_torch(
        policy,
        entity_type,
        owner_idx,
        features,
        rope_pos,
        entity_mask,
        planet_mask,
        target_valid,
        target_overflow,
        target_hit_tick,
        halt_action,
        pair_flat,
        frac_idx,
        no_valid_pairs,
        no_valid_fracs,
        population_idx,
        member_counts,
        value_head_idx,
    )

    log_ratio = new_logp - old_logp
    ratio = torch.exp(log_ratio)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    # No policy gradient through halt when the env forced halt (no owned ships).
    w_pi = (~must_halt_no_ships).to(dtype=torch.float32)
    v_clipped = old_v + (new_value - old_v).clamp(-clip_eps, clip_eps)
    value_err = torch.max((new_value - returns).pow(2), (v_clipped - returns).pow(2))

    if member_counts is None or population_idx is None or int(getattr(policy, "population_size", 1)) <= 1:
        w_sum = w_pi.sum().clamp(min=1.0)
        loss_pi = -(torch.min(surr1, surr2) * w_pi).sum() / w_sum
        loss_vf = value_err.mean()
        entropy = (new_entropy * w_pi).sum() / w_sum
    else:
        loss_pi_parts = []
        loss_vf_parts = []
        entropy_parts = []
        start = 0
        for count_t in member_counts:
            count = int(count_t.item())
            if count <= 0:
                continue
            stop = start + count
            w_pi_m = w_pi[start:stop]
            w_sum_m = w_pi_m.sum().clamp(min=1.0)
            loss_pi_parts.append(-(torch.min(surr1[start:stop], surr2[start:stop]) * w_pi_m).sum() / w_sum_m)
            loss_vf_parts.append(value_err[start:stop].mean())
            entropy_parts.append((new_entropy[start:stop] * w_pi_m).sum() / w_sum_m)
            start = stop
        loss_pi = torch.stack(loss_pi_parts).mean()
        loss_vf = torch.stack(loss_vf_parts).mean()
        entropy = torch.stack(entropy_parts).mean()

    loss_ent = -entropy_coef * entropy
    loss = loss_pi + vf_coef * loss_vf + loss_ent

    with torch.no_grad():
        if member_counts is None or population_idx is None or int(getattr(policy, "population_size", 1)) <= 1:
            w_sum = w_pi.sum().clamp(min=1.0)
            approx_kl = (-log_ratio * w_pi).sum() / w_sum
            approx_kl_k3 = ((ratio - 1.0 - log_ratio) * w_pi).sum() / w_sum
            clip_frac = (((ratio - 1.0).abs() > clip_eps).float() * w_pi).sum() / w_sum
            value_mean = new_value.mean()
        else:
            approx_kl_parts = []
            approx_kl_k3_parts = []
            clip_frac_parts = []
            value_mean_parts = []
            start = 0
            for count_t in member_counts:
                count = int(count_t.item())
                if count <= 0:
                    continue
                stop = start + count
                w_pi_m = w_pi[start:stop]
                w_sum_m = w_pi_m.sum().clamp(min=1.0)
                approx_kl_parts.append(((-log_ratio[start:stop]) * w_pi_m).sum() / w_sum_m)
                approx_kl_k3_parts.append((((ratio[start:stop] - 1.0 - log_ratio[start:stop]) * w_pi_m).sum()) / w_sum_m)
                clip_frac_parts.append(((((ratio[start:stop] - 1.0).abs() > clip_eps).float() * w_pi_m).sum()) / w_sum_m)
                value_mean_parts.append(new_value[start:stop].mean())
                start = stop
            approx_kl = torch.stack(approx_kl_parts).mean()
            approx_kl_k3 = torch.stack(approx_kl_k3_parts).mean()
            clip_frac = torch.stack(clip_frac_parts).mean()
            value_mean = torch.stack(value_mean_parts).mean()
        diff_sq_sum = (returns - new_value).pow(2).sum()
        ret_sum = returns.sum()
        ret_sq_sum = returns.pow(2).sum()
        count = torch.full((), float(returns.numel()), device=returns.device, dtype=torch.float32)

        # Per-head conditional entropies. Each head's denominator is the
        # number of non-forced rows in which that head is actually
        # sampled, so the reported value is the *average entropy of that
        # head when it is used* — directly comparable across heads even
        # though their action spaces have different sizes.
        w_pi_bool = ~must_halt_no_ships
        w_halt = w_pi  # halt is always sampled when not forced.
        w_origin_frac = (w_pi_bool & origin_frac_used).float()
        w_target = (w_pi_bool & target_used).float()
        w_halt_sum = w_halt.sum()
        w_origin_frac_sum = w_origin_frac.sum()
        w_target_sum = w_target.sum()
        nan = torch.full((), float("nan"), device=returns.device, dtype=torch.float32)
        entropy_halt = torch.where(
            w_halt_sum > 0,
            (new_halt_entropy * w_halt).sum() / w_halt_sum.clamp(min=1.0),
            nan,
        )
        entropy_origin_frac = torch.where(
            w_origin_frac_sum > 0,
            (new_origin_frac_entropy * w_origin_frac).sum() / w_origin_frac_sum.clamp(min=1.0),
            nan,
        )
        entropy_target = torch.where(
            w_target_sum > 0,
            (new_target_entropy * w_target).sum() / w_target_sum.clamp(min=1.0),
            nan,
        )

    stats: Dict[str, torch.Tensor] = {
        "loss_pi": loss_pi.detach(),
        "loss_vf": loss_vf.detach(),
        "entropy": entropy.detach(),
        "entropy_halt": entropy_halt,
        "entropy_origin_frac": entropy_origin_frac,
        "entropy_target": entropy_target,
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


def compute_ppo_loss_compressed_torch(
    policy: OrbitWarsPolicy,
    token_meta: torch.Tensor,
    owner_idx_comp: torch.Tensor,
    production: torch.Tensor,
    ships_comp: torch.Tensor,
    velocity: torch.Tensor,
    xy: torch.Tensor,
    turn_progress: torch.Tensor,
    incoming_net: torch.Tensor,
    incoming_survivor: torch.Tensor,
    feature_dim: int,
    target_valid: torch.Tensor,
    target_overflow: torch.Tensor,
    target_hit_tick: torch.Tensor,
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
    population_idx: Optional[torch.Tensor] = None,
    member_counts: Optional[torch.Tensor] = None,
    value_head_idx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    comp = CompressedObservationBuffer(
        token_meta=token_meta,
        owner_idx=owner_idx_comp,
        production=production,
        ships=ships_comp,
        velocity=velocity,
        xy=xy,
        turn_progress=turn_progress,
        incoming_net=incoming_net,
        incoming_survivor=incoming_survivor,
    )
    obs = decode_observation(comp, feature_dim=int(feature_dim))
    return compute_ppo_loss_torch(
        policy,
        obs["entity_type"],
        obs["owner_idx"],
        obs["features"],
        obs["rope_pos"],
        obs["entity_mask"],
        obs["planet_mask"],
        target_valid,
        target_overflow,
        target_hit_tick,
        halt_action,
        pair_flat,
        frac_idx,
        no_valid_pairs,
        no_valid_fracs,
        must_halt_no_ships,
        adv,
        returns,
        old_logp,
        old_v,
        clip_eps,
        vf_coef,
        entropy_coef,
        population_idx,
        member_counts,
        value_head_idx,
    )


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
    first_hit_n_rays: int = 2048,
    first_hit_ray_chunk_size: int = 0,
    first_hit_method: str = "category-rays",
    target_planet_reachable: Optional[jnp.ndarray] = None,
    target_hit_tick: Optional[jnp.ndarray] = None,
):
    """Run the JAX side (obs builder + optional first-hit) and dlpack-import results.

    When rollout snapshots are set (shape ``[B, MAX_PLANETS]``), skips
    ``selected_origin_fraction_targets_batched`` and reuses both
    ``target_valid & ~overflow`` and the matching per-target ETA feature.
    """

    t0 = perf_counter()
    obs_jax = build_observation_batched_jax_per_ego(state_b, ego_b, ship_speed)
    if target_planet_reachable is not None:
        target_valid_j = target_planet_reachable.astype(jnp.bool_)
        overflow_j = jnp.zeros((target_planet_reachable.shape[0],), dtype=jnp.bool_)
        if target_hit_tick is None:
            hit_tick_j = jnp.zeros(target_planet_reachable.shape, dtype=jnp.float32)
        else:
            hit_tick_j = target_hit_tick.astype(jnp.float32)
    else:
        origin_idx = pair_flat // MAX_PLANETS
        (
            _angle_j,
            _width_j,
            target_valid_j,
            overflow_j,
            hit_tick_j,
            _true_planet_j,
            _true_hit_tick_j,
        ) = selected_origin_fraction_targets_batched(
            state_b,
            origin_idx.astype(jnp.int32),
            frac_idx.astype(jnp.int32),
            horizon=24,
            ship_speed=ship_speed,
            samples_per_span=17,
            n_rays=first_hit_n_rays,
            ray_chunk_size=first_hit_ray_chunk_size,
            first_hit_method=first_hit_method,
        )
    if timing is not None:
        timing.replay_jax_s += perf_counter() - t0

    t0 = perf_counter()
    obs_torch = obs_jax_to_torch(obs_jax)
    target_valid_t = torch.from_dlpack(target_valid_j)
    target_overflow_t = torch.from_dlpack(overflow_j)
    target_hit_tick_t = torch.from_dlpack(hit_tick_j)
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
        target_hit_tick_t,
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
    first_hit_n_rays: int = 2048,
    first_hit_ray_chunk_size: int = 0,
    first_hit_method: str = "category-rays",
    target_planet_reachable: Optional[jnp.ndarray] = None,
    target_hit_tick: Optional[jnp.ndarray] = None,
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
        target_hit_tick_t,
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
        first_hit_n_rays=first_hit_n_rays,
        first_hit_ray_chunk_size=first_hit_ray_chunk_size,
        first_hit_method=first_hit_method,
        target_planet_reachable=target_planet_reachable,
        target_hit_tick=target_hit_tick,
    )

    new_logp, new_value, new_entropy, *_ = _compute_logp_value_entropy_torch(
        policy,
        obs_torch["entity_type"],
        obs_torch["owner_idx"],
        obs_torch["features"],
        obs_torch["rope_pos"],
        obs_torch["entity_mask"],
        obs_torch["planet_mask"],
        target_valid_t,
        target_overflow_t,
        target_hit_tick_t,
        halt_action_t,
        pair_flat_t,
        frac_idx_t,
        no_valid_pairs_t,
        no_valid_fracs_t,
    )
    return new_logp, new_value, new_entropy


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
    target_planet_reachable: Optional[jnp.ndarray] = None,
    target_hit_tick: Optional[jnp.ndarray] = None,
    first_hit_n_rays: int = 2048,
    first_hit_ray_chunk_size: int = 0,
    first_hit_method: str = "category-rays",
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
        target_hit_tick_t,
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
        first_hit_n_rays=first_hit_n_rays,
        first_hit_ray_chunk_size=first_hit_ray_chunk_size,
        first_hit_method=first_hit_method,
        target_planet_reachable=target_planet_reachable,
        target_hit_tick=target_hit_tick,
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
            target_hit_tick_t,
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

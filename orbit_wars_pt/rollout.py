"""Intra-turn action chaining until halt; converts planet pairs to env launch tuples."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from jax_orbit_wars import (
    DEFAULT_MAX_ACTIONS,
    PLANET_OWNER,
    PLANET_RADIUS,
    PLANET_SHIPS,
    PLANET_X,
    PLANET_Y,
    FLEET_ANGLE,
    FLEET_FROM_PLANET,
    FLEET_ETA,
    FLEET_OWNER,
    FLEET_SHIPS,
    FLEET_TARGET_PLANET,
    FLEET_X,
    FLEET_Y,
)

from orbit_wars_pt.constants import FRACTIONS, MAX_PLANETS, NUM_FRACTIONS
from orbit_wars_pt.geometry import estimate_time_to_hit, launch_point
from orbit_wars_pt.micro_jax import path_hits_brute_host
from orbit_wars_pt.model import OrbitWarsPolicy
from orbit_wars_pt.observation import ObservationBatch, build_observation, jax_state_to_numpy


def _masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    logits = logits.masked_fill(~mask, -1e4)
    return F.log_softmax(logits, dim=dim)


def _copy_state_np(state_np: Any) -> Any:
    new_planets = np.array(state_np.planets, copy=True)
    new_fleets = np.array(state_np.fleets, copy=True)
    new_fleet_active = np.array(state_np.fleet_active, copy=True)
    return state_np._replace(planets=new_planets, fleets=new_fleets, fleet_active=new_fleet_active)


def _first_free_fleet_slot(fleet_active: np.ndarray) -> int:
    idx = np.where(~fleet_active)[0]
    return int(idx[0]) if idx.size > 0 else -1


def _pad_observations_to_common_length(observations: List[ObservationBatch]) -> List[ObservationBatch]:
    """Pad entity sequences so batch stacking sees uniform `[L, …]` lengths."""

    max_len = max(len(o.entity_type) for o in observations)
    out: List[ObservationBatch] = []
    for o in observations:
        L = len(o.entity_type)
        if L == max_len:
            out.append(o)
            continue
        pad = max_len - L
        entity_type = np.pad(o.entity_type, (0, pad), constant_values=0)
        owner_idx = np.pad(o.owner_idx, (0, pad), constant_values=0)
        features = np.pad(o.features, ((0, pad), (0, 0)), constant_values=0.0)
        rope_pos = np.pad(o.rope_pos, ((0, pad), (0, 0)), constant_values=0.0)
        entity_mask = np.pad(o.entity_mask, (0, pad), constant_values=False)
        planet_mask = np.pad(o.planet_mask, (0, pad), constant_values=False)
        out.append(
            ObservationBatch(
                entity_type=entity_type,
                owner_idx=owner_idx,
                features=features,
                rope_pos=rope_pos,
                entity_mask=entity_mask,
                planet_mask=planet_mask,
                cls_index=o.cls_index,
                planet_positions=o.planet_positions,
                planet_ids=o.planet_ids,
                planet_idx_by_id=o.planet_idx_by_id,
                ego_player=o.ego_player,
                num_planets=o.num_planets,
            )
        )
    return out


def observation_batch_to_tensors(observations: List[ObservationBatch], device: torch.device) -> Dict[str, torch.Tensor]:
    """Stack host observations along batch dim `B` for a single policy forward."""

    observations = _pad_observations_to_common_length(observations)

    use_cuda = device.type == "cuda"

    def _stack_bool(arrs: List[np.ndarray]) -> torch.Tensor:
        x = np.stack(arrs, axis=0)
        t_cpu = torch.from_numpy(np.ascontiguousarray(x))
        if use_cuda:
            return t_cpu.pin_memory().to(device, non_blocking=True)
        return t_cpu.to(device)

    def _stack_long(arrs: List[np.ndarray]) -> torch.Tensor:
        x = np.stack(arrs, axis=0)
        t_cpu = torch.from_numpy(np.ascontiguousarray(x))
        if use_cuda:
            return t_cpu.pin_memory().to(device, non_blocking=True)
        return t_cpu.to(device)

    def _stack_float(arrs: List[np.ndarray]) -> torch.Tensor:
        x = np.stack(arrs, axis=0)
        t_cpu = torch.from_numpy(np.ascontiguousarray(x)).to(torch.float32)
        if use_cuda:
            return t_cpu.pin_memory().to(device, non_blocking=True)
        return t_cpu.to(device)

    return {
        "entity_type": _stack_long([o.entity_type for o in observations]),
        "owner_idx": _stack_long([o.owner_idx for o in observations]),
        "features": _stack_float([o.features for o in observations]),
        "rope_pos": _stack_float([o.rope_pos for o in observations]),
        "entity_mask": _stack_bool([o.entity_mask for o in observations]),
        "planet_mask": _stack_bool([o.planet_mask for o in observations]),
    }


def slice_policy_output(out: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Take batch row `index` from policy forward outputs (batch dim preserved as 1)."""

    sliced: Dict[str, Any] = {}
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            sliced[k] = v[index : index + 1]
        else:
            sliced[k] = v
    return sliced


@dataclass
class MicroStepResult:
    virt: Any
    logprob: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    halted: bool
    dispatch: Optional[Tuple[float, float, float]]


@dataclass
class MicroActionRecord:
    """Discrete choices sampled during a micro-step; used to recompute log π(a|s) under PPO."""

    halt_action: int  # 0 continue, 1 halt (meaningful whenever not forced no-ships halt)
    pair_flat: int  # o_idx * MAX_PLANETS + d_idx after final resampling; -1 if never chosen
    frac_idx: int  # selected fraction index, or -1
    halted_before_pair: bool  # True if halt_action==1 (normal) or forced no-ships
    no_valid_pairs: bool  # continue chosen but pair_mask all False
    no_valid_fracs: bool  # pair chosen but no geometry-valid fraction


@dataclass
class MicroTransition:
    """One stored timestep for on-policy replay (obs + pre-step virt + action record)."""

    obs: ObservationBatch
    virt: Any
    record: MicroActionRecord
    old_logprob: float
    old_value: float


def _must_halt_no_owned_ships(virt: Any, ego_player: int) -> bool:
    owned_ships = 0.0
    for i in range(MAX_PLANETS):
        if not bool(np.asarray(virt.planet_active)[i]):
            continue
        if int(np.asarray(virt.planets)[i, PLANET_OWNER]) != ego_player:
            continue
        owned_ships += float(np.asarray(virt.planets)[i, PLANET_SHIPS])
    return owned_ships < 1.0 - 1e-6


def replay_micro_step_logprob_entropy(
    *,
    ego_player: int,
    policy: OrbitWarsPolicy,
    out: Dict[str, Any],
    virt: Any,
    record: MicroActionRecord,
    device: torch.device,
    ship_speed: float = 6.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Recompute total log π(a|s) and entropy (same decomposition as rollout) under the current policy."""

    planet_xy = np.asarray(virt.planets)[:MAX_PLANETS, 2:4].astype(np.float64)
    planet_r = np.asarray(virt.planets)[:MAX_PLANETS, 4].astype(np.float64)
    planet_active = np.asarray(virt.planet_active)[:MAX_PLANETS].astype(bool)

    hl_b = out["halt_logits"]
    halt_logits = hl_b[0]

    must_halt = _must_halt_no_owned_ships(virt, ego_player)
    lp_parts: List[torch.Tensor] = []
    ent_parts: List[torch.Tensor] = []

    if must_halt:
        dist_h = torch.distributions.Categorical(logits=halt_logits.unsqueeze(0))
        lp_parts.append(dist_h.log_prob(torch.ones(1, dtype=torch.long, device=device))[0])
        ent_parts.append(dist_h.entropy()[0])
        logp = torch.stack(lp_parts).sum()
        ent = torch.stack(ent_parts).sum()
        return logp, ent

    dist_h = torch.distributions.Categorical(logits=halt_logits.unsqueeze(0))
    ha = int(record.halt_action)
    lp_parts.append(dist_h.log_prob(torch.tensor([ha], device=device, dtype=torch.long))[0])
    ent_parts.append(dist_h.entropy()[0])

    if ha == 1:
        return torch.stack(lp_parts).sum(), torch.stack(ent_parts).sum()

    pair_logits = out["pair_logits"][0]
    pair_mask = out["pair_mask"][0]
    if record.no_valid_pairs:
        return torch.stack(lp_parts).sum(), torch.stack(ent_parts).sum()
    if not pair_mask.any():
        return torch.stack(lp_parts).sum(), torch.stack(ent_parts).sum()

    flat = pair_logits.flatten()
    flat_mask = pair_mask.flatten()
    flat_lp = _masked_log_softmax(flat, flat_mask, dim=0)
    idx_flat = int(record.pair_flat)
    lp_parts.append(flat_lp[idx_flat])

    sub = torch.distributions.Categorical(logits=torch.where(flat_mask, flat, torch.tensor(-1e4, device=device)))
    ent_parts.append(sub.entropy())

    o_idx = idx_flat // MAX_PLANETS
    d_idx = idx_flat % MAX_PLANETS

    ph = out["planet_hidden"][0]

    ox = float(planet_xy[o_idx, 0])
    oy = float(planet_xy[o_idx, 1])
    dx = float(planet_xy[d_idx, 0])
    dy = float(planet_xy[d_idx, 1])

    ships_avail = float(np.asarray(virt.planets)[o_idx, PLANET_SHIPS])
    masks = np.zeros((NUM_FRACTIONS,), dtype=np.bool_)
    etas = np.zeros((NUM_FRACTIONS,), dtype=np.float32)
    for fi, frac in enumerate(FRACTIONS):
        send = math.floor(frac * ships_avail)
        if send < 1:
            masks[fi] = False
            continue
        bad = path_hits_brute_host(
            virt,
            o_idx,
            d_idx,
            float(send),
            ship_speed=ship_speed,
            horizon=24,
        )
        masks[fi] = not bad
        eta = estimate_time_to_hit(
            ox,
            oy,
            float(planet_r[o_idx]),
            dx,
            dy,
            float(planet_r[d_idx]),
            float(send),
            max_speed=ship_speed,
        )
        etas[fi] = float(min(eta, 500.0)) if math.isfinite(eta) else 500.0

    if record.no_valid_fracs or not masks.any():
        return torch.stack(lp_parts).sum(), torch.stack(ent_parts).sum()

    times_norm = torch.tensor(
        [[float(etas[i]) / 500.0 for i in range(NUM_FRACTIONS)]],
        device=device,
        dtype=torch.float32,
    )
    frac_logits_b = policy.fraction_logits(
        ph.unsqueeze(0),
        torch.tensor([o_idx], device=device, dtype=torch.long),
        torch.tensor([d_idx], device=device, dtype=torch.long),
        times_norm=times_norm,
    )
    frac_logits = frac_logits_b[0]

    m = torch.tensor(masks, device=device, dtype=torch.bool)
    frac_lp = _masked_log_softmax(frac_logits.unsqueeze(0), m.unsqueeze(0), dim=-1).squeeze(0)
    fi = int(record.frac_idx)
    lp_parts.append(frac_lp[fi])

    sub_f = torch.distributions.Categorical(
        logits=torch.where(m, frac_logits, torch.tensor(-1e4, device=device))
    )
    ent_parts.append(sub_f.entropy())

    return torch.stack(lp_parts).sum(), torch.stack(ent_parts).sum()


def micro_step_apply(
    virt: Any,
    ego_player: int,
    policy: OrbitWarsPolicy,
    out: Dict[str, Any],
    device: torch.device,
    *,
    rng: Optional[torch.Generator] = None,
    greedy: bool = False,
    ship_speed: float = 6.0,
) -> Tuple[MicroStepResult, MicroActionRecord]:
    """One MDP timestep: halt vs dispatch. Policy outputs must have batch size 1."""

    planet_xy = np.asarray(virt.planets)[:MAX_PLANETS, 2:4].astype(np.float64)
    planet_r = np.asarray(virt.planets)[:MAX_PLANETS, 4].astype(np.float64)
    planet_active = np.asarray(virt.planet_active)[:MAX_PLANETS].astype(bool)
    planet_ids = np.asarray(virt.planets)[:MAX_PLANETS, 0].astype(np.float64)

    entropy_parts: List[torch.Tensor] = []
    logprob_parts: List[torch.Tensor] = []

    hl_b = out["halt_logits"]
    value = out["value"][0]
    halt_logits = hl_b[0]

    must_halt = _must_halt_no_owned_ships(virt, ego_player)
    if must_halt:
        dist_h = torch.distributions.Categorical(logits=halt_logits.unsqueeze(0))
        lp_h = dist_h.log_prob(torch.ones(1, dtype=torch.long, device=device))
        logprob_parts.append(lp_h[0])
        entropy_parts.append(dist_h.entropy()[0])
        lp = torch.stack(logprob_parts).sum()
        ent = torch.stack(entropy_parts).sum()
        rec = MicroActionRecord(
            halt_action=1,
            pair_flat=-1,
            frac_idx=-1,
            halted_before_pair=True,
            no_valid_pairs=False,
            no_valid_fracs=False,
        )
        return MicroStepResult(virt=virt, logprob=lp, value=value, entropy=ent, halted=True, dispatch=None), rec

    dist_h = torch.distributions.Categorical(logits=halt_logits.unsqueeze(0))
    if greedy:
        halt_action = int(torch.argmax(halt_logits, dim=-1).item())
    else:
        probs = F.softmax(halt_logits, dim=-1)
        halt_action = int(torch.multinomial(probs, 1, generator=rng).squeeze().item())
    logprob_parts.append(dist_h.log_prob(torch.tensor([halt_action], device=device))[0])
    entropy_parts.append(dist_h.entropy()[0])

    if halt_action == 1:
        lp = torch.stack(logprob_parts).sum()
        ent = torch.stack(entropy_parts).sum()
        rec = MicroActionRecord(
            halt_action=1,
            pair_flat=-1,
            frac_idx=-1,
            halted_before_pair=True,
            no_valid_pairs=False,
            no_valid_fracs=False,
        )
        return MicroStepResult(virt=virt, logprob=lp, value=value, entropy=ent, halted=True, dispatch=None), rec

    pair_logits = out["pair_logits"][0]
    pair_mask = out["pair_mask"][0]
    if not pair_mask.any():
        lp = torch.stack(logprob_parts).sum()
        ent = torch.stack(entropy_parts).sum()
        rec = MicroActionRecord(
            halt_action=0,
            pair_flat=-1,
            frac_idx=-1,
            halted_before_pair=False,
            no_valid_pairs=True,
            no_valid_fracs=False,
        )
        return MicroStepResult(virt=virt, logprob=lp, value=value, entropy=ent, halted=True, dispatch=None), rec

    flat = pair_logits.flatten()
    flat_mask = pair_mask.flatten()
    flat_lp = _masked_log_softmax(flat, flat_mask, dim=0)
    if greedy:
        idx_flat = int(torch.argmax(torch.where(flat_mask, flat, torch.tensor(-1e9, device=device))).item())
    else:
        probs = torch.exp(flat_lp)
        idx_flat = int(torch.multinomial(probs, 1, generator=rng).item())
    logprob_parts.append(flat_lp[idx_flat])
    entropy_parts.append(
        torch.distributions.Categorical(
            logits=torch.where(flat_mask, flat, torch.tensor(-1e4, device=device))
        ).entropy()
    )

    o_idx = idx_flat // MAX_PLANETS
    d_idx = idx_flat % MAX_PLANETS

    ph = out["planet_hidden"][0]

    ox = float(planet_xy[o_idx, 0])
    oy = float(planet_xy[o_idx, 1])
    dx = float(planet_xy[d_idx, 0])
    dy = float(planet_xy[d_idx, 1])
    oid = float(planet_ids[o_idx])
    angle = math.atan2(dy - oy, dx - ox)

    ships_avail = float(np.asarray(virt.planets)[o_idx, PLANET_SHIPS])
    masks = np.zeros((NUM_FRACTIONS,), dtype=np.bool_)
    etas = np.zeros((NUM_FRACTIONS,), dtype=np.float32)
    for fi, frac in enumerate(FRACTIONS):
        send = math.floor(frac * ships_avail)
        if send < 1:
            masks[fi] = False
            continue
        bad = path_hits_brute_host(
            virt,
            o_idx,
            d_idx,
            float(send),
            ship_speed=ship_speed,
            horizon=24,
        )
        masks[fi] = not bad
        eta = estimate_time_to_hit(
            ox,
            oy,
            float(planet_r[o_idx]),
            dx,
            dy,
            float(planet_r[d_idx]),
            float(send),
            max_speed=ship_speed,
        )
        etas[fi] = float(min(eta, 500.0)) if math.isfinite(eta) else 500.0

    retries = 0
    while (not masks.any()) and retries < 32:
        retries += 1
        if greedy:
            idx_flat = int(torch.argmax(torch.where(flat_mask, flat, torch.tensor(-1e9, device=device))).item())
        else:
            probs = torch.exp(flat_lp)
            idx_flat = int(torch.multinomial(probs, 1, generator=rng).item())
        logprob_parts[-1] = flat_lp[idx_flat]
        o_idx = idx_flat // MAX_PLANETS
        d_idx = idx_flat % MAX_PLANETS
        ox = float(planet_xy[o_idx, 0])
        oy = float(planet_xy[o_idx, 1])
        dx = float(planet_xy[d_idx, 0])
        dy = float(planet_xy[d_idx, 1])
        oid = float(planet_ids[o_idx])
        angle = math.atan2(dy - oy, dx - ox)
        ships_avail = float(np.asarray(virt.planets)[o_idx, PLANET_SHIPS])
        masks = np.zeros((NUM_FRACTIONS,), dtype=np.bool_)
        etas = np.zeros((NUM_FRACTIONS,), dtype=np.float32)
        for fi, frac in enumerate(FRACTIONS):
            send = math.floor(frac * ships_avail)
            if send < 1:
                continue
            bad = path_hits_brute_host(
                virt,
                o_idx,
                d_idx,
                float(send),
                ship_speed=ship_speed,
                horizon=24,
            )
            masks[fi] = not bad
            eta = estimate_time_to_hit(
                ox,
                oy,
                float(planet_r[o_idx]),
                dx,
                dy,
                float(planet_r[d_idx]),
                float(send),
                max_speed=ship_speed,
            )
            etas[fi] = float(min(eta, 500.0)) if math.isfinite(eta) else 500.0

    if not masks.any():
        lp = torch.stack(logprob_parts).sum()
        ent = torch.stack(entropy_parts).sum()
        rec = MicroActionRecord(
            halt_action=0,
            pair_flat=int(idx_flat),
            frac_idx=-1,
            halted_before_pair=False,
            no_valid_pairs=False,
            no_valid_fracs=True,
        )
        return MicroStepResult(virt=virt, logprob=lp, value=value, entropy=ent, halted=True, dispatch=None), rec

    times_norm = torch.tensor(
        [[float(etas[i]) / 500.0 for i in range(NUM_FRACTIONS)]],
        device=device,
        dtype=torch.float32,
    )
    frac_logits_b = policy.fraction_logits(
        ph.unsqueeze(0),
        torch.tensor([o_idx], device=device, dtype=torch.long),
        torch.tensor([d_idx], device=device, dtype=torch.long),
        times_norm=times_norm,
    )
    frac_logits = frac_logits_b[0]

    m = torch.tensor(masks, device=device, dtype=torch.bool)
    frac_lp = _masked_log_softmax(frac_logits.unsqueeze(0), m.unsqueeze(0), dim=-1).squeeze(0)
    if greedy:
        fi = int(torch.argmax(torch.where(m, frac_logits, torch.tensor(-1e9, device=device))).item())
    else:
        probs = torch.exp(frac_lp)
        fi = int(torch.multinomial(probs, 1, generator=rng).item())
    logprob_parts.append(frac_lp[fi])
    entropy_parts.append(
        torch.distributions.Categorical(logits=torch.where(m, frac_logits, torch.tensor(-1e4, device=device))).entropy()
    )

    send = max(1, math.floor(FRACTIONS[fi] * ships_avail))
    send = min(send, int(ships_avail))

    dispatch = (oid, float(angle), float(send))

    p = np.asarray(virt.planets)
    p[o_idx, PLANET_SHIPS] -= float(send)
    virt = virt._replace(planets=p)

    slot = _first_free_fleet_slot(np.asarray(virt.fleet_active))
    if slot >= 0:
        fl = np.asarray(virt.fleets)
        fa = np.asarray(virt.fleet_active)
        sx, sy = launch_point(ox, oy, float(planet_r[o_idx]), dx, dy)
        row = np.zeros((fl.shape[1],), dtype=np.float32)
        row[FLEET_OWNER] = float(ego_player)
        row[FLEET_X] = sx
        row[FLEET_Y] = sy
        row[FLEET_ANGLE] = float(angle)
        row[FLEET_FROM_PLANET] = oid
        row[FLEET_SHIPS] = float(send)
        if row.shape[0] > FLEET_ETA:
            row[FLEET_TARGET_PLANET] = float(d_idx)
            row[FLEET_ETA] = float(etas[fi])
        fl[slot] = row
        fa[slot] = True
        virt = virt._replace(fleets=fl, fleet_active=fa)

    lp = torch.stack(logprob_parts).sum()
    ent = torch.stack(entropy_parts).sum()
    rec = MicroActionRecord(
        halt_action=0,
        pair_flat=int(idx_flat),
        frac_idx=int(fi),
        halted_before_pair=False,
        no_valid_pairs=False,
        no_valid_fracs=False,
    )
    return MicroStepResult(virt=virt, logprob=lp, value=value, entropy=ent, halted=False, dispatch=dispatch), rec


def _obs_to_tensors(obs: ObservationBatch, device: torch.device) -> Dict[str, torch.Tensor]:
    """Host NumPy observation → PyTorch; uses pinned CPU memory + async H2D when CUDA is enabled."""

    use_cuda = device.type == "cuda"

    def _t(x: np.ndarray, dtype_torch):
        t_cpu = torch.from_numpy(np.ascontiguousarray(x))
        t_cpu = t_cpu.to(dtype_torch)
        if use_cuda:
            t_cpu = t_cpu.pin_memory()
            return t_cpu.unsqueeze(0).to(device, non_blocking=True)
        return t_cpu.unsqueeze(0).to(device)

    return {
        "entity_type": _t(obs.entity_type, torch.long),
        "owner_idx": _t(obs.owner_idx, torch.long),
        "features": _t(obs.features, torch.float32),
        "rope_pos": _t(obs.rope_pos, torch.float32),
        "entity_mask": _t(obs.entity_mask, torch.bool),
        "planet_mask": _t(obs.planet_mask, torch.bool),
    }


def build_turn_actions(
    policy: OrbitWarsPolicy,
    state_np: Any,
    ego_player: int,
    device: torch.device,
    *,
    ship_speed: float = 6.0,
    max_micro_steps: int = 64,
    rng: Optional[torch.Generator] = None,
    greedy: bool = False,
    detach_outputs: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Queues launches until halt; returns `[max_actions, 3]` with zeros padded.

    Legacy helper: aggregates intra-turn log-probs and uses first-step value (old behavior).
    """

    virt = _copy_state_np(jax_state_to_numpy(state_np))
    actions: List[Tuple[float, float, float]] = []
    logprob_parts: List[torch.Tensor] = []
    entropy_parts: List[torch.Tensor] = []
    micro_steps = 0
    value_start: Optional[torch.Tensor] = None

    while micro_steps < max_micro_steps:
        obs: ObservationBatch = build_observation(virt, ego_player, ship_speed=ship_speed)
        batch = _obs_to_tensors(obs, device)

        out = policy(**batch)
        if value_start is None:
            value_start = out["value"][0]
        res, _ = micro_step_apply(
            virt,
            ego_player,
            policy,
            out,
            device,
            rng=rng,
            greedy=greedy,
            ship_speed=ship_speed,
        )
        virt = res.virt
        logprob_parts.append(res.logprob)
        entropy_parts.append(res.entropy)
        if res.dispatch is not None:
            actions.append(res.dispatch)
        micro_steps += 1
        if res.halted:
            break

    arr = np.zeros((DEFAULT_MAX_ACTIONS, 3), dtype=np.float32)
    for i, tup in enumerate(actions[:DEFAULT_MAX_ACTIONS]):
        arr[i, 0], arr[i, 1], arr[i, 2] = tup

    lp_sum = torch.stack(logprob_parts).sum() if logprob_parts else torch.tensor(0.0, device=device)
    if detach_outputs:
        lp_sum = lp_sum.detach()
    info = {
        "logprob": lp_sum,
        "entropy": torch.stack(entropy_parts).mean().detach() if entropy_parts else torch.tensor(0.0, device=device),
        "micro_steps": micro_steps,
        "value": (
            (value_start if value_start is not None else torch.tensor(0.0, device=device)).detach()
            if detach_outputs
            else (value_start if value_start is not None else torch.tensor(0.0, device=device, requires_grad=False))
        ),
    }
    return arr, info

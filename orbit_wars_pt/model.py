"""Transformer policy with configurable spatial RoPE and Orbit Wars action heads."""

from __future__ import annotations

from collections import OrderedDict
import math
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from orbit_wars_pt.compressed_observation import CompressedObservationBuffer, decode_observation
from orbit_wars_pt.constants import (
    BLOCKED_FRAC_FEATURES,
    ENTITY_CLS,
    FEATURE_DIM,
    FEATURE_DIM_ABORT,
    FEATURE_DIM_MULTI,
    FEATURE_DIM_MULTI_ABORT,
    FRACTIONS,
    FUTURE_PLANET_FEATURE_DIM,
    FUTURE_PLANET_FEATURES_PER_TICK,
    INCOMING_SURVIVOR_FLAT,
    INCOMING_TA_BINS,
    MAX_PLANETS,
    NUM_ENTITY_TYPES,
    NUM_FRACTIONS,
    NUM_OWNER_SLOTS,
)


def apply_rope_1d(x: torch.Tensor, pos: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Applies 1D RoPE along the last dimension of `x` (even head_dim)."""

    dtype = x.dtype
    device = x.device
    d = x.shape[-1]
    half = d // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / float(half)))
    angles = pos[..., None].to(torch.float32) * inv_freq
    cos = angles.cos().to(dtype)
    sin = angles.sin().to(dtype)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    out = torch.stack([y1, y2], dim=-1).flatten(-2)
    return out


def apply_rope_nd(
    q: torch.Tensor,
    k: torch.Tensor,
    rope_pos: torch.Tensor,
    *,
    rope_dims: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split head dims across ``rope_dims`` coordinates and apply 1D RoPE per chunk."""

    d_h = q.shape[-1]
    if rope_dims not in (2, 3):
        raise ValueError(f"rope_dims must be 2 or 3, got {rope_dims}")
    chunk = d_h // rope_dims
    if d_h % rope_dims != 0 or chunk % 2 != 0:
        raise AssertionError(
            f"head_dim must be divisible by {rope_dims} and each chunk must be even; "
            f"got head_dim={d_h}, rope_dims={rope_dims}"
        )
    if rope_pos.shape[-1] < rope_dims:
        raise ValueError(
            f"rope_pos last dim {rope_pos.shape[-1]} is smaller than rope_dims={rope_dims}"
        )

    q_chunks = q.split(chunk, dim=-1)
    k_chunks = k.split(chunk, dim=-1)
    q_out = []
    k_out = []
    for i in range(rope_dims):
        pos_i = rope_pos[..., i]
        q_out.append(apply_rope_1d(q_chunks[i], pos_i))
        k_out.append(apply_rope_1d(k_chunks[i], pos_i))
    return torch.cat(q_out, dim=-1), torch.cat(k_out, dim=-1)


class RoPEAttention(nn.Module):
    """Pre-norm RoPE attention on dense ``[B, L_packed, d]``.

    ``L_packed`` is the per-batch max active token count after
    ``OrbitWarsPolicy.forward`` packs each sample's active tokens to the
    front. ``key_padding_mask`` (``True = masked``) marks the rows beyond
    each sample's count. SDPA dispatches to FlashAttention 2 (BF16/FP16
    on Ampere+) when no mask is needed, or to the memory-efficient kernel
    when a key_padding_mask is supplied.
    """

    def __init__(self, d_model: int, n_heads: int, *, rope_dims: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.rope_dims = int(rope_dims)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout_p = float(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_pos: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """``x`` ``[B, L_packed, d]``, ``rope_pos`` ``[B, L_packed, 3]``,
        ``key_padding_mask`` ``[B, L_packed]`` bool (True = masked key).
        """

        b, l, _ = x.shape
        qkv = self.qkv(x).reshape(b, l, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        r = rope_pos[:, None, :, : self.rope_dims].expand(b, self.n_heads, l, self.rope_dims)
        q, k = apply_rope_nd(q, k, r, rope_dims=self.rope_dims)

        # SDPA mask convention is ``True = participate``; ours is
        # ``True = masked``, so invert. Shape: ``[B, 1, 1, L_packed]``
        # broadcasts across heads and query positions.
        attn_mask: Optional[torch.Tensor] = None
        if key_padding_mask is not None:
            attn_mask = (~key_padding_mask)[:, None, None, :]

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).reshape(b, l, -1)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        rope_dims: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPEAttention(d_model, n_heads, rope_dims=rope_dims, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        rope_pos: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_pos, key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


def _make_target_pick_head(d_model: int, *, extra_dim: int = 0) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model + 3 + int(extra_dim), d_model),
        nn.GELU(),
        nn.Linear(d_model, d_model // 2),
        nn.GELU(),
        nn.Linear(d_model // 2, 1),
    )


def _extract_future_forecast_inputs(
    owner_idx: torch.Tensor,
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    planet_owner = owner_idx[:, 1 : 1 + MAX_PLANETS].to(torch.long).clamp(0, NUM_OWNER_SLOTS - 1)
    planet_features = features[:, 1 : 1 + MAX_PLANETS, :]
    garrison = (planet_features[..., 1].float() * 1000.0).clamp_min(0.0)
    production = torch.expm1(planet_features[..., 0].float()).clamp_min(0.0)
    incoming_net = planet_features[..., 8 : 8 + INCOMING_TA_BINS].float() * 1000.0
    arrival_ships = incoming_net.abs()

    multi_min = 8 + INCOMING_TA_BINS + INCOMING_SURVIVOR_FLAT
    if int(planet_features.shape[-1]) >= multi_min:
        survivor = planet_features[..., 8 + INCOMING_TA_BINS : multi_min].reshape(
            planet_features.shape[0], MAX_PLANETS, INCOMING_TA_BINS, NUM_OWNER_SLOTS
        )
        arrival_owner = survivor.argmax(dim=-1).to(torch.long)
        no_survivor = arrival_ships <= 0.5
        arrival_owner = torch.where(no_survivor, torch.zeros_like(arrival_owner), arrival_owner)
    else:
        arrival_owner = torch.where(
            incoming_net > 0.5,
            torch.ones_like(incoming_net, dtype=torch.long),
            torch.where(
                incoming_net < -0.5,
                torch.full_like(incoming_net, 2, dtype=torch.long),
                torch.zeros_like(incoming_net, dtype=torch.long),
            ),
        )
    return planet_owner, garrison, production, arrival_owner, arrival_ships


def _simulate_future_planet_features(
    current_owner: torch.Tensor,
    current_garrison: torch.Tensor,
    production: torch.Tensor,
    arrival_owner: torch.Tensor,
    arrival_ships: torch.Tensor,
    *,
    active_mask: Optional[torch.Tensor] = None,
    launch_ships: Optional[torch.Tensor] = None,
    launch_eta: Optional[torch.Tensor] = None,
    launch_owner: int = 1,
) -> torch.Tensor:
    owner = current_owner.to(torch.long).clamp(0, NUM_OWNER_SLOTS - 1)
    garrison = current_garrison.float().clamp_min(0.0)
    prod = production.float().clamp_min(0.0)
    arr_owner = arrival_owner.to(torch.long).clamp(0, NUM_OWNER_SLOTS - 1)
    arr_ships = arrival_ships.float().clamp_min(0.0)
    active = None if active_mask is None else active_mask.to(dtype=torch.bool, device=garrison.device)
    extra = None if launch_ships is None else launch_ships.to(device=garrison.device, dtype=garrison.dtype).clamp_min(0.0)
    eta = None if launch_eta is None else launch_eta.to(device=garrison.device, dtype=torch.long)

    out: list[torch.Tensor] = []
    launch_owner_t = torch.full_like(owner, int(launch_owner))
    for t in range(INCOMING_TA_BINS):
        owner_non_neutral = owner > 0
        garrison = torch.where(owner_non_neutral, garrison + prod, garrison)

        arr_owner_t = arr_owner[..., t]
        arr_ships_t = arr_ships[..., t]
        if extra is not None and eta is not None:
            hit = eta == t
            same = arr_owner_t == launch_owner_t
            none = arr_owner_t == 0
            oppose = hit & (~same) & (~none)
            bigger = extra > arr_ships_t
            equal = extra == arr_ships_t
            arr_owner_t = torch.where(hit & none, launch_owner_t, arr_owner_t)
            arr_ships_t = torch.where(hit & none, extra, arr_ships_t)
            arr_ships_t = torch.where(hit & same, arr_ships_t + extra, arr_ships_t)
            arr_owner_t = torch.where(oppose & bigger, launch_owner_t, arr_owner_t)
            arr_ships_t = torch.where(oppose & bigger, extra - arr_ships_t, arr_ships_t)
            arr_owner_t = torch.where(oppose & equal, torch.zeros_like(arr_owner_t), arr_owner_t)
            arr_ships_t = torch.where(oppose & equal, torch.zeros_like(arr_ships_t), arr_ships_t)
            arr_ships_t = torch.where(oppose & (~bigger) & (~equal), arr_ships_t - extra, arr_ships_t)

        has_arrival = (arr_owner_t > 0) & (arr_ships_t > 0.5)
        same_owner = arr_owner_t == owner
        add_mask = has_arrival & same_owner
        garrison = torch.where(add_mask, garrison + arr_ships_t, garrison)

        fight_mask = has_arrival & (~same_owner)
        rem = garrison - arr_ships_t
        captured = fight_mask & (rem < 0.0)
        neutralized = fight_mask & (rem == 0.0)
        held = fight_mask & (rem > 0.0)
        owner = torch.where(captured, arr_owner_t, owner)
        owner = torch.where(neutralized, torch.zeros_like(owner), owner)
        garrison = torch.where(captured, -rem, garrison)
        garrison = torch.where(neutralized, torch.zeros_like(garrison), garrison)
        garrison = torch.where(held, rem, garrison)

        owner_oh = torch.nn.functional.one_hot(owner, NUM_OWNER_SLOTS).to(dtype=garrison.dtype)
        tick_feat = torch.cat([owner_oh, (garrison / 1000.0).unsqueeze(-1)], dim=-1)
        if active is not None:
            tick_feat = torch.where(active.unsqueeze(-1), tick_feat, torch.zeros_like(tick_feat))
        out.append(tick_feat)

    return torch.cat(out, dim=-1)


def build_future_planet_features(owner_idx: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    current_owner, current_garrison, production, arrival_owner, arrival_ships = _extract_future_forecast_inputs(
        owner_idx, features
    )
    active = features[:, 1 : 1 + MAX_PLANETS, 4] > 0.5
    return _simulate_future_planet_features(
        current_owner,
        current_garrison,
        production,
        arrival_owner,
        arrival_ships,
        active_mask=active,
    )


class OrbitWarsPopulationTail(nn.Module):
    """One population member's private final block + output heads."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        rope_dims: int,
        dropout: float = 0.0,
        value_head_count: int = 1,
        target_abort_enabled: bool = False,
        future_feature_enabled: bool = False,
        halt_init_prob: Optional[float] = None,
        fraction_init_weights: Optional[Tuple[float, ...]] = None,
    ):
        super().__init__()
        self.block = TransformerBlock(d_model, n_heads, rope_dims=rope_dims, dropout=dropout)
        self.norm_f = nn.LayerNorm(d_model)
        self.origin_frac_head = nn.Linear(d_model, NUM_FRACTIONS)
        extra_dim = 2 * FUTURE_PLANET_FEATURE_DIM if future_feature_enabled else 0
        self.target_pick_head = _make_target_pick_head(d_model, extra_dim=extra_dim)
        self.target_abort_enabled = bool(target_abort_enabled)
        if self.target_abort_enabled:
            self.abort_head = nn.Linear(d_model, NUM_FRACTIONS)
        self.time_proj = nn.Linear(1, 32)
        self.frac_heads = nn.ModuleList([nn.Linear(d_model * 2 + 32, 1) for _ in range(NUM_FRACTIONS)])
        self.halt_head = nn.Linear(d_model, 2)
        self.value_head = nn.Linear(d_model, int(value_head_count))
        _apply_action_head_init_biases(
            halt_head=self.halt_head,
            origin_frac_head=self.origin_frac_head,
            frac_heads=self.frac_heads,
            halt_init_prob=halt_init_prob,
            fraction_init_weights=fraction_init_weights,
        )


class OrbitWarsCriticPopulationTail(nn.Module):
    """One population member's private critic final block + value head."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        rope_dims: int,
        dropout: float = 0.0,
        value_head_count: int = 1,
    ):
        super().__init__()
        self.block = TransformerBlock(d_model, n_heads, rope_dims=rope_dims, dropout=dropout)
        self.norm_f = nn.LayerNorm(d_model)
        self.value_head = nn.Linear(d_model, int(value_head_count))


def _halt_logit_for_prob(prob: float) -> float:
    p = float(prob)
    if not math.isfinite(p) or not (0.0 < p < 1.0):
        raise ValueError(f"halt_init_prob must be between 0 and 1, got {prob!r}")
    return math.log(p / (1.0 - p))


def _normalize_fraction_init_weights(weights: Tuple[float, ...]) -> Tuple[float, ...]:
    if len(weights) != NUM_FRACTIONS:
        raise ValueError(f"fraction_init_weights must have length {NUM_FRACTIONS}, got {len(weights)}")
    out = []
    for idx, val in enumerate(weights):
        w = float(val)
        if not math.isfinite(w) or w <= 0.0:
            raise ValueError(f"fraction_init_weights[{idx}] must be finite and > 0, got {val!r}")
        out.append(w)
    return tuple(out)


def _apply_action_head_init_biases(
    *,
    halt_head: nn.Linear,
    origin_frac_head: nn.Linear,
    frac_heads: nn.ModuleList,
    halt_init_prob: Optional[float],
    fraction_init_weights: Optional[Tuple[float, ...]],
) -> None:
    with torch.no_grad():
        if halt_init_prob is not None:
            halt_bias = _halt_logit_for_prob(float(halt_init_prob))
            halt_head.bias.zero_()
            halt_head.bias[1] = halt_bias
        if fraction_init_weights is not None:
            weights = _normalize_fraction_init_weights(fraction_init_weights)
            frac_bias = torch.log(torch.tensor(weights, dtype=origin_frac_head.bias.dtype, device=origin_frac_head.bias.device))
            origin_frac_head.bias.copy_(frac_bias)
            for idx, head in enumerate(frac_heads):
                head.bias.fill_(float(frac_bias[idx].item()))


def infer_value_head_count_from_state_dict(state: Mapping[str, Any]) -> int:
    """Infer critic output width from the first value-head weight in ``state``."""

    for key, value in state.items():
        if not str(key).endswith("value_head.weight"):
            continue
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 2 and int(shape[0]) >= 1:
            return int(shape[0])
    return 1


def adapt_legacy_value_heads_for_model(
    state: Mapping[str, Any],
    model: nn.Module,
) -> tuple[OrderedDict[str, Any], bool]:
    """Expand legacy single-output critic heads to match ``model`` critic widths."""

    target_state = model.state_dict()
    out = OrderedDict(state.items())
    migrated = False
    for key, target_tensor in target_state.items():
        if not str(key).endswith("value_head.weight") and not str(key).endswith("value_head.bias"):
            continue
        if key not in out:
            continue
        src = out[key]
        src_shape = tuple(int(x) for x in getattr(src, "shape", ()))
        tgt_shape = tuple(int(x) for x in getattr(target_tensor, "shape", ()))
        if src_shape == tgt_shape:
            continue
        if str(key).endswith("value_head.weight") and src_shape == (1, tgt_shape[1]) and len(tgt_shape) == 2 and tgt_shape[0] > 1:
            out[key] = src.repeat(tgt_shape[0], 1)
            migrated = True
            continue
        if str(key).endswith("value_head.bias") and src_shape == (1,) and tgt_shape[0] > 1:
            out[key] = src.repeat(tgt_shape[0])
            migrated = True
    return out, migrated


def adapt_checkpoint_state_for_model(
    state: Mapping[str, Any],
    model: nn.Module,
) -> tuple[OrderedDict[str, Any], bool]:
    """Map saved weights onto ``model``, cloning actor weights into critic-only params when needed."""

    out, migrated = adapt_legacy_value_heads_for_model(state, model)
    target_state = model.state_dict()
    current = OrderedDict((key, value) for key, value in out.items() if key in target_state)

    def _clone_from_actor_name(name: str) -> Optional[str]:
        if name.startswith("critic_shared_blocks."):
            return "shared_blocks." + name[len("critic_shared_blocks."):]
        if name.startswith("critic_population_tails."):
            return "population_tails." + name[len("critic_population_tails."):]
        if name.startswith("critic_blocks."):
            return "blocks." + name[len("critic_blocks."):]
        if name.startswith("critic_norm_f."):
            return "norm_f." + name[len("critic_norm_f."):]
        if name.startswith("critic_value_head."):
            return "value_head." + name[len("critic_value_head."):]
        return None

    for key in target_state:
        if key in current:
            continue
        actor_key = _clone_from_actor_name(str(key))
        if actor_key is None or actor_key not in out:
            continue
        current[key] = out[actor_key].clone()
        migrated = True
    return current, migrated


class OrbitWarsPolicy(nn.Module):
    """Encoder-only transformer; optional population-specific final block + heads."""

    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 8,
        n_layers: int = 4,
        feature_dim: int = FEATURE_DIM,
        dropout: float = 0.0,
        activation_checkpointing: bool = False,
        population_size: int = 1,
        rope_dims: int = 2,
        value_head_count: int = 1,
        disjoint_actor_critic: bool = False,
        target_abort_enabled: bool = False,
        future_feature_enabled: bool = False,
        halt_init_prob: Optional[float] = None,
        fraction_init_weights: Optional[Tuple[float, ...]] = None,
    ):
        super().__init__()
        head_dim = d_model // n_heads
        if d_model % n_heads != 0:
            raise AssertionError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        rope_dims = int(rope_dims)
        if rope_dims not in (2, 3):
            raise ValueError(f"rope_dims must be 2 or 3, got {rope_dims}")
        if head_dim % (2 * rope_dims) != 0:
            raise AssertionError(
                f"head_dim {head_dim} must be divisible by {2 * rope_dims} for rope_dims={rope_dims}"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.rope_dims = rope_dims
        self.activation_checkpointing = activation_checkpointing
        self.population_size = int(population_size)
        self.value_head_count = int(value_head_count)
        self.disjoint_actor_critic = bool(disjoint_actor_critic)
        self.target_abort_enabled = bool(target_abort_enabled)
        self.future_feature_enabled = bool(future_feature_enabled)
        self.halt_init_prob = None if halt_init_prob is None else float(halt_init_prob)
        self.fraction_init_weights = (
            None if fraction_init_weights is None else _normalize_fraction_init_weights(tuple(fraction_init_weights))
        )
        if self.population_size < 1:
            raise ValueError(f"population_size must be >= 1, got {self.population_size}")
        if self.value_head_count < 1:
            raise ValueError(f"value_head_count must be >= 1, got {self.value_head_count}")
        self.type_emb = nn.Embedding(NUM_ENTITY_TYPES, d_model)
        self.owner_emb = nn.Embedding(NUM_OWNER_SLOTS, d_model)
        self.feat_proj = nn.Linear(feature_dim, d_model)
        if self.future_feature_enabled:
            self.future_feat_proj = nn.Linear(FUTURE_PLANET_FEATURE_DIM, d_model, bias=False)
        self.cls_type_idx = ENTITY_CLS

        if self.population_size == 1:
            self.blocks = nn.ModuleList(
                [TransformerBlock(d_model, n_heads, rope_dims=rope_dims, dropout=dropout) for _ in range(n_layers)]
            )
            self.norm_f = nn.LayerNorm(d_model)
            self.origin_frac_head = nn.Linear(d_model, NUM_FRACTIONS)
            extra_dim = 2 * FUTURE_PLANET_FEATURE_DIM if self.future_feature_enabled else 0
            self.target_pick_head = _make_target_pick_head(d_model, extra_dim=extra_dim)
            if self.target_abort_enabled:
                self.abort_head = nn.Linear(d_model, NUM_FRACTIONS)
            self.time_proj = nn.Linear(1, 32)
            self.frac_heads = nn.ModuleList([nn.Linear(d_model * 2 + 32, 1) for _ in range(NUM_FRACTIONS)])
            self.halt_head = nn.Linear(d_model, 2)
            self.value_head = nn.Linear(d_model, self.value_head_count)
            _apply_action_head_init_biases(
                halt_head=self.halt_head,
                origin_frac_head=self.origin_frac_head,
                frac_heads=self.frac_heads,
                halt_init_prob=self.halt_init_prob,
                fraction_init_weights=self.fraction_init_weights,
            )
            if self.disjoint_actor_critic:
                self.critic_blocks = nn.ModuleList(
                    [TransformerBlock(d_model, n_heads, rope_dims=rope_dims, dropout=dropout) for _ in range(n_layers)]
                )
                self.critic_norm_f = nn.LayerNorm(d_model)
                self.critic_value_head = nn.Linear(d_model, self.value_head_count)
        else:
            shared_layers = max(0, int(n_layers) - 1)
            self.shared_blocks = nn.ModuleList(
                [TransformerBlock(d_model, n_heads, rope_dims=rope_dims, dropout=dropout) for _ in range(shared_layers)]
            )
            self.population_tails = nn.ModuleList(
                [
                    OrbitWarsPopulationTail(
                        d_model,
                        n_heads,
                        rope_dims=rope_dims,
                        dropout=dropout,
                        value_head_count=self.value_head_count,
                        target_abort_enabled=self.target_abort_enabled,
                        future_feature_enabled=self.future_feature_enabled,
                        halt_init_prob=self.halt_init_prob,
                        fraction_init_weights=self.fraction_init_weights,
                    )
                    for _ in range(self.population_size)
                ]
            )
            if self.disjoint_actor_critic:
                self.critic_shared_blocks = nn.ModuleList(
                    [TransformerBlock(d_model, n_heads, rope_dims=rope_dims, dropout=dropout) for _ in range(shared_layers)]
                )
                self.critic_population_tails = nn.ModuleList(
                    [
                        OrbitWarsCriticPopulationTail(
                            d_model,
                            n_heads,
                            rope_dims=rope_dims,
                            dropout=dropout,
                            value_head_count=self.value_head_count,
                        )
                        for _ in range(self.population_size)
                    ]
                )

        self.register_buffer("_frac_const", torch.tensor(FRACTIONS, dtype=torch.float32))

    def actor_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [(name, param) for name, param in self.named_parameters() if not str(name).startswith("critic_")]

    def critic_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [(name, param) for name, param in self.named_parameters() if str(name).startswith("critic_")]

    def actor_parameters(self) -> list[nn.Parameter]:
        return [param for _, param in self.actor_named_parameters()]

    def critic_parameters(self) -> list[nn.Parameter]:
        return [param for _, param in self.critic_named_parameters()]

    def embed(
        self,
        entity_type: torch.Tensor,
        owner_idx: torch.Tensor,
        features: torch.Tensor,
        future_planet_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t_ok = entity_type.clamp(0, NUM_ENTITY_TYPES - 1)
        o_ok = owner_idx.clamp(0, NUM_OWNER_SLOTS - 1)
        x = self.type_emb(t_ok) + self.owner_emb(o_ok) + self.feat_proj(features)
        if self.future_feature_enabled:
            if future_planet_features is None:
                future_planet_features = build_future_planet_features(owner_idx, features)
            future_tokens = torch.zeros(
                features.shape[0],
                features.shape[1],
                FUTURE_PLANET_FEATURE_DIM,
                dtype=features.dtype,
                device=features.device,
            )
            future_tokens[:, 1 : 1 + MAX_PLANETS, :] = future_planet_features.to(
                device=features.device, dtype=features.dtype
            )
            x = x + self.future_feat_proj(future_tokens)
        return x

    def _run_block(
        self,
        block: TransformerBlock,
        x: torch.Tensor,
        rope_pos: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.activation_checkpointing and torch.is_grad_enabled():
            return checkpoint(block, x, rope_pos, key_padding_mask, use_reentrant=False)
        return block(x, rope_pos, key_padding_mask)

    def _normalize_population_idx(
        self,
        population_idx: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if population_idx is None:
            return torch.zeros((batch_size,), device=device, dtype=torch.long)
        pop = population_idx.to(device=device, dtype=torch.long).reshape(-1)
        if pop.shape[0] != batch_size:
            raise ValueError(f"population_idx batch {pop.shape[0]} != expected {batch_size}")
        if self.population_size == 1:
            return torch.zeros_like(pop)
        return pop

    def _apply_encoder(
        self,
        x: torch.Tensor,
        rope_pos: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        population_idx: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(x.shape[0])
        pop = self._normalize_population_idx(population_idx, batch_size, x.device)
        if self.population_size == 1:
            for blk in self.blocks:
                x = self._run_block(blk, x, rope_pos, key_padding_mask)
            return self.norm_f(x), pop

        for blk in self.shared_blocks:
            x = self._run_block(blk, x, rope_pos, key_padding_mask)

        h = torch.empty_like(x)
        for member_idx, tail in enumerate(self.population_tails):
            member_rows = torch.nonzero(pop == member_idx, as_tuple=False).squeeze(-1)
            if member_rows.numel() == 0:
                continue
            x_m = x.index_select(0, member_rows)
            rope_m = rope_pos.index_select(0, member_rows)
            mask_m = None if key_padding_mask is None else key_padding_mask.index_select(0, member_rows)
            x_m = self._run_block(tail.block, x_m, rope_m, mask_m)
            h.index_copy_(0, member_rows, tail.norm_f(x_m))
        return h, pop

    def _apply_critic_encoder(
        self,
        x: torch.Tensor,
        rope_pos: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        population_idx: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.disjoint_actor_critic:
            return self._apply_encoder(x, rope_pos, key_padding_mask, population_idx)

        batch_size = int(x.shape[0])
        pop = self._normalize_population_idx(population_idx, batch_size, x.device)
        if self.population_size == 1:
            for blk in self.critic_blocks:
                x = self._run_block(blk, x, rope_pos, key_padding_mask)
            return self.critic_norm_f(x), pop

        for blk in self.critic_shared_blocks:
            x = self._run_block(blk, x, rope_pos, key_padding_mask)

        h = torch.empty_like(x)
        for member_idx, tail in enumerate(self.critic_population_tails):
            member_rows = torch.nonzero(pop == member_idx, as_tuple=False).squeeze(-1)
            if member_rows.numel() == 0:
                continue
            x_m = x.index_select(0, member_rows)
            rope_m = rope_pos.index_select(0, member_rows)
            mask_m = None if key_padding_mask is None else key_padding_mask.index_select(0, member_rows)
            x_m = self._run_block(tail.block, x_m, rope_m, mask_m)
            h.index_copy_(0, member_rows, tail.norm_f(x_m))
        return h, pop

    def _population_group_size(self, batch_size: int) -> int:
        if self.population_size <= 1:
            return int(batch_size)
        if batch_size % self.population_size != 0:
            raise ValueError(
                f"grouped population batch {batch_size} must be divisible by population_size {self.population_size}"
            )
        return batch_size // self.population_size

    def _apply_encoder_grouped_population(
        self,
        x: torch.Tensor,
        rope_pos: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.population_size == 1:
            for blk in self.blocks:
                x = self._run_block(blk, x, rope_pos, key_padding_mask)
            return self.norm_f(x)

        for blk in self.shared_blocks:
            x = self._run_block(blk, x, rope_pos, key_padding_mask)

        group_size = self._population_group_size(int(x.shape[0]))
        chunks = []
        for member_idx, tail in enumerate(self.population_tails):
            start = member_idx * group_size
            stop = start + group_size
            x_m = x[start:stop]
            rope_m = rope_pos[start:stop]
            mask_m = None if key_padding_mask is None else key_padding_mask[start:stop]
            x_m = self._run_block(tail.block, x_m, rope_m, mask_m)
            chunks.append(tail.norm_f(x_m))
        return torch.cat(chunks, dim=0)

    def _apply_critic_encoder_grouped_population(
        self,
        x: torch.Tensor,
        rope_pos: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.disjoint_actor_critic:
            return self._apply_encoder_grouped_population(x, rope_pos, key_padding_mask)
        if self.population_size == 1:
            for blk in self.critic_blocks:
                x = self._run_block(blk, x, rope_pos, key_padding_mask)
            return self.critic_norm_f(x)

        for blk in self.critic_shared_blocks:
            x = self._run_block(blk, x, rope_pos, key_padding_mask)

        group_size = self._population_group_size(int(x.shape[0]))
        chunks = []
        for member_idx, tail in enumerate(self.critic_population_tails):
            start = member_idx * group_size
            stop = start + group_size
            x_m = x[start:stop]
            rope_m = rope_pos[start:stop]
            mask_m = None if key_padding_mask is None else key_padding_mask[start:stop]
            x_m = self._run_block(tail.block, x_m, rope_m, mask_m)
            chunks.append(tail.norm_f(x_m))
        return torch.cat(chunks, dim=0)

    def _population_member_counts(
        self,
        population_idx: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        pop = self._normalize_population_idx(population_idx, batch_size, device)
        if self.population_size == 1:
            return torch.full((1,), batch_size, device=device, dtype=torch.long)
        return torch.bincount(pop, minlength=self.population_size).to(device=device, dtype=torch.long)

    def _normalize_origin_frac_blocked(
        self,
        origin_frac_blocked: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if not self.target_abort_enabled or origin_frac_blocked is None:
            return None
        blocked = origin_frac_blocked.to(device=device, dtype=torch.bool)
        expected = (batch_size, MAX_PLANETS, NUM_FRACTIONS)
        if tuple(blocked.shape) != expected:
            raise ValueError(f"origin_frac_blocked shape {tuple(blocked.shape)} != {expected}")
        return blocked

    def _heads_autocast_disabled(self, device: torch.device) -> torch.autocast:
        """Disable autocast for post-trunk head math so logits/value stay in fp32."""

        return torch.autocast(device_type=device.type, enabled=False)

    def _fp32_head_input(self, t: torch.Tensor) -> torch.Tensor:
        return t.float() if t.is_floating_point() else t

    def _target_future_context(
        self,
        owner_idx: torch.Tensor,
        features: torch.Tensor,
        origin_idx: torch.Tensor,
        fleet_size: torch.Tensor,
        target_eta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_owner, current_garrison, production, arrival_owner, arrival_ships = _extract_future_forecast_inputs(
            owner_idx, features
        )
        active = features[:, 1 : 1 + MAX_PLANETS, 4] > 0.5
        batch_idx = torch.arange(origin_idx.shape[0], device=features.device)
        origin_long = origin_idx.to(device=features.device, dtype=torch.long)
        fleet = fleet_size.to(device=features.device, dtype=torch.float32).clamp_min(0.0)

        origin_owner = current_owner[batch_idx, origin_long].unsqueeze(1)
        origin_garrison = (current_garrison[batch_idx, origin_long] - fleet).clamp_min(0.0).unsqueeze(1)
        origin_prod = production[batch_idx, origin_long].unsqueeze(1)
        origin_arr_owner = arrival_owner[batch_idx, origin_long].unsqueeze(1)
        origin_arr_ships = arrival_ships[batch_idx, origin_long].unsqueeze(1)
        origin_active = active[batch_idx, origin_long].unsqueeze(1)
        origin_future = _simulate_future_planet_features(
            origin_owner,
            origin_garrison,
            origin_prod,
            origin_arr_owner,
            origin_arr_ships,
            active_mask=origin_active,
        ).expand(-1, MAX_PLANETS, -1)

        target_future = _simulate_future_planet_features(
            current_owner,
            current_garrison,
            production,
            arrival_owner,
            arrival_ships,
            active_mask=active,
            launch_ships=fleet.unsqueeze(1),
            launch_eta=target_eta.to(device=features.device, dtype=torch.long),
            launch_owner=1,
        )
        return origin_future, target_future

    def _apply_encoder_sorted_population(
        self,
        x: torch.Tensor,
        rope_pos: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        member_counts: torch.Tensor,
    ) -> torch.Tensor:
        if self.population_size == 1:
            for blk in self.blocks:
                x = self._run_block(blk, x, rope_pos, key_padding_mask)
            return self.norm_f(x)

        for blk in self.shared_blocks:
            x = self._run_block(blk, x, rope_pos, key_padding_mask)

        chunks = []
        start = 0
        for member_idx, tail in enumerate(self.population_tails):
            count = int(member_counts[member_idx].item())
            if count <= 0:
                continue
            stop = start + count
            x_m = x[start:stop]
            rope_m = rope_pos[start:stop]
            mask_m = None if key_padding_mask is None else key_padding_mask[start:stop]
            x_m = self._run_block(tail.block, x_m, rope_m, mask_m)
            chunks.append(tail.norm_f(x_m))
            start = stop
        return torch.cat(chunks, dim=0) if chunks else x.new_empty((0,) + x.shape[1:])

    def _compute_outputs_single(
        self,
        h: torch.Tensor,
        value_h: torch.Tensor,
        owner_idx: torch.Tensor,
        features: torch.Tensor,
        entity_mask: torch.Tensor,
        planet_mask: torch.Tensor,
        origin_frac_blocked: Optional[torch.Tensor] = None,
        *,
        origin_frac_head: nn.Linear,
        halt_head: nn.Linear,
        value_head: nn.Linear,
        abort_head: Optional[nn.Linear] = None,
        value_head_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        with self._heads_autocast_disabled(h.device):
            h = self._fp32_head_input(h)
            value_h = self._fp32_head_input(value_h)
            features = self._fp32_head_input(features)
            b = h.shape[0]
            cls_h = h[:, 0, :]
            value_cls_h = value_h[:, 0, :]
            halt_logits = halt_head(cls_h)
            value_all = value_head(value_cls_h)
            if int(value_all.shape[-1]) == 1:
                value = value_all.squeeze(-1)
            else:
                if value_head_idx is None:
                    value = value_all[:, 0]
                else:
                    value = value_all.gather(
                        1,
                        value_head_idx.to(device=value_all.device, dtype=torch.long).reshape(-1, 1),
                    ).squeeze(-1)

            planet_h = h[:, 1 : 1 + MAX_PLANETS, :]
            blocked_mask = None
            if self.target_abort_enabled and int(features.shape[-1]) in (FEATURE_DIM_ABORT, FEATURE_DIM_MULTI_ABORT):
                blocked_mask = features[:, 1 : 1 + MAX_PLANETS, -BLOCKED_FRAC_FEATURES:] > 0.5
            elif origin_frac_blocked is not None:
                blocked_mask = self._normalize_origin_frac_blocked(origin_frac_blocked, int(h.shape[0]), h.device)

            pm = planet_mask[:, 1 : 1 + MAX_PLANETS]
            em = entity_mask[:, 1 : 1 + MAX_PLANETS]
            active_planet = pm & em
            eye = torch.eye(MAX_PLANETS, device=h.device, dtype=torch.bool).unsqueeze(0).expand(b, -1, -1)
            owned_self = owner_idx[:, 1 : 1 + MAX_PLANETS] == 1
            ships = features[:, 1 : 1 + MAX_PLANETS, 1] * 1000.0
            has_ships = ships > 0.5
            origin_ok = active_planet & owned_self & has_ships
            dest_ok = active_planet
            pair_mask = origin_ok[:, :, None] & dest_ok[:, None, :] & ~eye
            sends = torch.floor(self._frac_const.to(features.dtype)[None, None, :] * ships[:, :, None])
            origin_frac_mask = origin_ok[:, :, None] & (sends >= 1.0)
            if blocked_mask is not None:
                origin_frac_mask = origin_frac_mask & (~blocked_mask)
            origin_frac_logits = origin_frac_head(planet_h)

            return {
                "hidden": h,
                "halt_logits": halt_logits,
                "value": value,
                "pair_mask": pair_mask,
                "origin_frac_logits": origin_frac_logits,
                "origin_frac_mask": origin_frac_mask,
                "planet_hidden": planet_h,
                **(
                    {"abort_logits": abort_head(planet_h)}
                    if self.target_abort_enabled and abort_head is not None
                    else {}
                ),
            }

    def _compute_ppo_outputs_single(
        self,
        h: torch.Tensor,
        value_h: torch.Tensor,
        owner_idx: torch.Tensor,
        features: torch.Tensor,
        entity_mask: torch.Tensor,
        planet_mask: torch.Tensor,
        origin_idx: torch.Tensor,
        frac_idx: torch.Tensor,
        fleet_size: torch.Tensor,
        target_eta: torch.Tensor,
        target_ships: torch.Tensor,
        origin_frac_blocked: Optional[torch.Tensor] = None,
        *,
        origin_frac_head: nn.Linear,
        halt_head: nn.Linear,
        value_head: nn.Linear,
        target_pick_head: nn.Module,
        abort_head: Optional[nn.Linear] = None,
        value_head_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        with self._heads_autocast_disabled(h.device):
            out = self._compute_outputs_single(
                h,
                value_h,
                owner_idx,
                features,
                entity_mask,
                planet_mask,
                origin_frac_blocked=origin_frac_blocked,
                origin_frac_head=origin_frac_head,
                halt_head=halt_head,
                value_head=value_head,
                abort_head=abort_head,
                value_head_idx=value_head_idx,
            )
            planet_hidden = out["planet_hidden"]
            fleet_scalar = (fleet_size.to(device=planet_hidden.device, dtype=planet_hidden.dtype) / 1000.0).reshape(-1, 1, 1)
            fleet_feat = fleet_scalar.expand(-1, planet_hidden.shape[1], -1)
            eta_feat = (target_eta.to(device=planet_hidden.device, dtype=planet_hidden.dtype) / 500.0).unsqueeze(-1)
            target_ships_t = target_ships.to(device=planet_hidden.device, dtype=planet_hidden.dtype)
            is_bigger = (fleet_scalar > target_ships_t.unsqueeze(-1)).to(dtype=planet_hidden.dtype)
            target_parts = [planet_hidden, fleet_feat, eta_feat, is_bigger]
            if self.future_feature_enabled:
                origin_future, target_future = self._target_future_context(
                    owner_idx,
                    features,
                    origin_idx,
                    fleet_size,
                    target_eta,
                )
                target_parts.extend(
                    [
                        origin_future.to(device=planet_hidden.device, dtype=planet_hidden.dtype),
                        target_future.to(device=planet_hidden.device, dtype=planet_hidden.dtype),
                    ]
                )
            target_in = torch.cat(target_parts, dim=-1)
            target_logits = target_pick_head(target_in).squeeze(-1)
            out["target_logits"] = target_logits
            if self.target_abort_enabled and "abort_logits" in out:
                batch_idx = torch.arange(origin_idx.shape[0], device=planet_hidden.device)
                out["abort_logit"] = out["abort_logits"][
                    batch_idx,
                    origin_idx.to(device=planet_hidden.device, dtype=torch.long),
                    frac_idx.to(device=planet_hidden.device, dtype=torch.long),
                ]
            return out

    def _value_head_for_population_member(self, member_idx: int) -> nn.Linear:
        if not self.disjoint_actor_critic:
            return self.population_tails[member_idx].value_head
        return self.critic_population_tails[member_idx].value_head

    def _compute_outputs_population(
        self,
        h: torch.Tensor,
        value_h: torch.Tensor,
        owner_idx: torch.Tensor,
        features: torch.Tensor,
        entity_mask: torch.Tensor,
        planet_mask: torch.Tensor,
        population_idx: torch.Tensor,
        origin_frac_blocked: Optional[torch.Tensor] = None,
        value_head_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        batch_size = int(h.shape[0])
        outputs: Optional[Dict[str, Any]] = None

        for member_idx, tail in enumerate(self.population_tails):
            member_rows = torch.nonzero(population_idx == member_idx, as_tuple=False).squeeze(-1)
            if member_rows.numel() == 0:
                continue
            value_head = self._value_head_for_population_member(member_idx)
            out_m = self._compute_outputs_single(
                h.index_select(0, member_rows),
                value_h.index_select(0, member_rows),
                owner_idx.index_select(0, member_rows),
                features.index_select(0, member_rows),
                entity_mask.index_select(0, member_rows),
                planet_mask.index_select(0, member_rows),
                origin_frac_blocked=None if origin_frac_blocked is None else origin_frac_blocked.index_select(0, member_rows),
                origin_frac_head=tail.origin_frac_head,
                halt_head=tail.halt_head,
                value_head=value_head,
                abort_head=getattr(tail, "abort_head", None),
                value_head_idx=None if value_head_idx is None else value_head_idx.index_select(0, member_rows),
            )
            if outputs is None:
                outputs = {
                    "hidden": h,
                    "halt_logits": out_m["halt_logits"].new_empty((batch_size, 2)),
                    "value": out_m["value"].new_empty((batch_size,)),
                    "pair_mask": out_m["pair_mask"].new_zeros((batch_size, MAX_PLANETS, MAX_PLANETS)),
                    "origin_frac_logits": out_m["origin_frac_logits"].new_empty(
                        (batch_size, MAX_PLANETS, NUM_FRACTIONS)
                    ),
                    "origin_frac_mask": out_m["origin_frac_mask"].new_zeros(
                        (batch_size, MAX_PLANETS, NUM_FRACTIONS)
                    ),
                    "planet_hidden": out_m["planet_hidden"].new_empty((batch_size, MAX_PLANETS, self.d_model)),
                }
                if self.target_abort_enabled and "abort_logits" in out_m:
                    outputs["abort_logits"] = out_m["abort_logits"].new_empty(
                        (batch_size, MAX_PLANETS, NUM_FRACTIONS)
                    )
            outputs["halt_logits"].index_copy_(0, member_rows, out_m["halt_logits"])
            outputs["value"].index_copy_(0, member_rows, out_m["value"])
            outputs["pair_mask"].index_copy_(0, member_rows, out_m["pair_mask"])
            outputs["origin_frac_logits"].index_copy_(0, member_rows, out_m["origin_frac_logits"])
            outputs["origin_frac_mask"].index_copy_(0, member_rows, out_m["origin_frac_mask"])
            outputs["planet_hidden"].index_copy_(0, member_rows, out_m["planet_hidden"])
            if self.target_abort_enabled and "abort_logits" in out_m:
                outputs["abort_logits"].index_copy_(0, member_rows, out_m["abort_logits"])

        assert outputs is not None
        return outputs

    def _compute_outputs_grouped_population(
        self,
        h: torch.Tensor,
        value_h: torch.Tensor,
        owner_idx: torch.Tensor,
        features: torch.Tensor,
        entity_mask: torch.Tensor,
        planet_mask: torch.Tensor,
        origin_frac_blocked: Optional[torch.Tensor] = None,
        value_head_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if self.population_size == 1:
            return self._compute_outputs_single(
                h,
                value_h,
                owner_idx,
                features,
                entity_mask,
                planet_mask,
                origin_frac_blocked=origin_frac_blocked,
                origin_frac_head=self.origin_frac_head,
                halt_head=self.halt_head,
                value_head=(self.critic_value_head if self.disjoint_actor_critic else self.value_head),
                abort_head=getattr(self, "abort_head", None),
                value_head_idx=value_head_idx,
            )

        group_size = self._population_group_size(int(h.shape[0]))
        outputs: Dict[str, Any] = {"hidden": h}
        halt_logits = []
        value = []
        pair_mask = []
        origin_frac_logits = []
        origin_frac_mask = []
        planet_hidden = []
        abort_logits = []
        for member_idx, tail in enumerate(self.population_tails):
            start = member_idx * group_size
            stop = start + group_size
            value_head = self._value_head_for_population_member(member_idx)
            out_m = self._compute_outputs_single(
                h[start:stop],
                value_h[start:stop],
                owner_idx[start:stop],
                features[start:stop],
                entity_mask[start:stop],
                planet_mask[start:stop],
                origin_frac_blocked=None if origin_frac_blocked is None else origin_frac_blocked[start:stop],
                origin_frac_head=tail.origin_frac_head,
                halt_head=tail.halt_head,
                value_head=value_head,
                abort_head=getattr(tail, "abort_head", None),
                value_head_idx=None if value_head_idx is None else value_head_idx[start:stop],
            )
            halt_logits.append(out_m["halt_logits"])
            value.append(out_m["value"])
            pair_mask.append(out_m["pair_mask"])
            origin_frac_logits.append(out_m["origin_frac_logits"])
            origin_frac_mask.append(out_m["origin_frac_mask"])
            planet_hidden.append(out_m["planet_hidden"])
            if self.target_abort_enabled and "abort_logits" in out_m:
                abort_logits.append(out_m["abort_logits"])
        outputs["halt_logits"] = torch.cat(halt_logits, dim=0)
        outputs["value"] = torch.cat(value, dim=0)
        outputs["pair_mask"] = torch.cat(pair_mask, dim=0)
        outputs["origin_frac_logits"] = torch.cat(origin_frac_logits, dim=0)
        outputs["origin_frac_mask"] = torch.cat(origin_frac_mask, dim=0)
        outputs["planet_hidden"] = torch.cat(planet_hidden, dim=0)
        if self.target_abort_enabled and abort_logits:
            outputs["abort_logits"] = torch.cat(abort_logits, dim=0)
        return outputs

    def _compute_outputs_sorted_population(
        self,
        h: torch.Tensor,
        value_h: torch.Tensor,
        owner_idx: torch.Tensor,
        features: torch.Tensor,
        entity_mask: torch.Tensor,
        planet_mask: torch.Tensor,
        member_counts: torch.Tensor,
        origin_frac_blocked: Optional[torch.Tensor] = None,
        value_head_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if self.population_size == 1:
            return self._compute_outputs_single(
                h,
                value_h,
                owner_idx,
                features,
                entity_mask,
                planet_mask,
                origin_frac_blocked=origin_frac_blocked,
                origin_frac_head=self.origin_frac_head,
                halt_head=self.halt_head,
                value_head=(self.critic_value_head if self.disjoint_actor_critic else self.value_head),
                abort_head=getattr(self, "abort_head", None),
                value_head_idx=value_head_idx,
            )

        outputs: Dict[str, Any] = {"hidden": h}
        halt_logits = []
        value = []
        pair_mask = []
        origin_frac_logits = []
        origin_frac_mask = []
        planet_hidden = []
        abort_logits = []
        start = 0
        for member_idx, tail in enumerate(self.population_tails):
            count = int(member_counts[member_idx].item())
            if count <= 0:
                continue
            stop = start + count
            value_head = self._value_head_for_population_member(member_idx)
            out_m = self._compute_outputs_single(
                h[start:stop],
                value_h[start:stop],
                owner_idx[start:stop],
                features[start:stop],
                entity_mask[start:stop],
                planet_mask[start:stop],
                origin_frac_blocked=None if origin_frac_blocked is None else origin_frac_blocked[start:stop],
                origin_frac_head=tail.origin_frac_head,
                halt_head=tail.halt_head,
                value_head=value_head,
                abort_head=getattr(tail, "abort_head", None),
                value_head_idx=None if value_head_idx is None else value_head_idx[start:stop],
            )
            halt_logits.append(out_m["halt_logits"])
            value.append(out_m["value"])
            pair_mask.append(out_m["pair_mask"])
            origin_frac_logits.append(out_m["origin_frac_logits"])
            origin_frac_mask.append(out_m["origin_frac_mask"])
            planet_hidden.append(out_m["planet_hidden"])
            if self.target_abort_enabled and "abort_logits" in out_m:
                abort_logits.append(out_m["abort_logits"])
            start = stop
        outputs["halt_logits"] = torch.cat(halt_logits, dim=0)
        outputs["value"] = torch.cat(value, dim=0)
        outputs["pair_mask"] = torch.cat(pair_mask, dim=0)
        outputs["origin_frac_logits"] = torch.cat(origin_frac_logits, dim=0)
        outputs["origin_frac_mask"] = torch.cat(origin_frac_mask, dim=0)
        outputs["planet_hidden"] = torch.cat(planet_hidden, dim=0)
        if self.target_abort_enabled and abort_logits:
            outputs["abort_logits"] = torch.cat(abort_logits, dim=0)
        return outputs

    def _compute_ppo_outputs_sorted_population(
        self,
        h: torch.Tensor,
        value_h: torch.Tensor,
        owner_idx: torch.Tensor,
        features: torch.Tensor,
        entity_mask: torch.Tensor,
        planet_mask: torch.Tensor,
        origin_idx: torch.Tensor,
        frac_idx: torch.Tensor,
        fleet_size: torch.Tensor,
        target_eta: torch.Tensor,
        target_ships: torch.Tensor,
        member_counts: torch.Tensor,
        origin_frac_blocked: Optional[torch.Tensor] = None,
        value_head_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if self.population_size == 1:
            return self._compute_ppo_outputs_single(
                h,
                value_h,
                owner_idx,
                features,
                entity_mask,
                planet_mask,
                origin_idx,
                frac_idx,
                fleet_size,
                target_eta,
                target_ships,
                origin_frac_blocked=origin_frac_blocked,
                origin_frac_head=self.origin_frac_head,
                halt_head=self.halt_head,
                value_head=(self.critic_value_head if self.disjoint_actor_critic else self.value_head),
                target_pick_head=self.target_pick_head,
                abort_head=getattr(self, "abort_head", None),
                value_head_idx=value_head_idx,
            )

        outputs: Dict[str, Any] = {"hidden": h}
        halt_logits = []
        value = []
        pair_mask = []
        origin_frac_logits = []
        origin_frac_mask = []
        planet_hidden = []
        target_logits = []
        abort_logits = []
        start = 0
        for member_idx, tail in enumerate(self.population_tails):
            count = int(member_counts[member_idx].item())
            if count <= 0:
                continue
            stop = start + count
            value_head = self._value_head_for_population_member(member_idx)
            out_m = self._compute_ppo_outputs_single(
                h[start:stop],
                value_h[start:stop],
                owner_idx[start:stop],
                features[start:stop],
                entity_mask[start:stop],
                planet_mask[start:stop],
                origin_idx[start:stop],
                frac_idx[start:stop],
                fleet_size[start:stop],
                target_eta[start:stop],
                target_ships[start:stop],
                origin_frac_blocked=None if origin_frac_blocked is None else origin_frac_blocked[start:stop],
                origin_frac_head=tail.origin_frac_head,
                halt_head=tail.halt_head,
                value_head=value_head,
                target_pick_head=tail.target_pick_head,
                abort_head=getattr(tail, "abort_head", None),
                value_head_idx=None if value_head_idx is None else value_head_idx[start:stop],
            )
            halt_logits.append(out_m["halt_logits"])
            value.append(out_m["value"])
            pair_mask.append(out_m["pair_mask"])
            origin_frac_logits.append(out_m["origin_frac_logits"])
            origin_frac_mask.append(out_m["origin_frac_mask"])
            planet_hidden.append(out_m["planet_hidden"])
            target_logits.append(out_m["target_logits"])
            if self.target_abort_enabled:
                abort_logits.append(out_m["abort_logit"])
            start = stop
        outputs["halt_logits"] = torch.cat(halt_logits, dim=0)
        outputs["value"] = torch.cat(value, dim=0)
        outputs["pair_mask"] = torch.cat(pair_mask, dim=0)
        outputs["origin_frac_logits"] = torch.cat(origin_frac_logits, dim=0)
        outputs["origin_frac_mask"] = torch.cat(origin_frac_mask, dim=0)
        outputs["planet_hidden"] = torch.cat(planet_hidden, dim=0)
        outputs["target_logits"] = torch.cat(target_logits, dim=0)
        if self.target_abort_enabled:
            outputs["abort_logit"] = torch.cat(abort_logits, dim=0)
        return outputs

    def forward(
        self,
        entity_type: torch.Tensor,
        owner_idx: torch.Tensor,
        features: torch.Tensor,
        rope_pos: torch.Tensor,
        entity_mask: torch.Tensor,
        planet_mask: torch.Tensor,
        origin_frac_blocked: Optional[torch.Tensor] = None,
        population_idx: Optional[torch.Tensor] = None,
        value_head_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Returns logits and masks for action decoding.

        The encoder runs on a **per-sample-front packed** view: each
        sample's active tokens are stable-sorted to the front, the batch
        is sliced to ``L_packed = max_b count_b``, and the dense encoder
        runs on ``[B, L_packed, d]`` with a standard ``key_padding_mask``.
        After the encoder we scatter back to dense ``[B, L, d]`` so the
        action heads can read fixed positions (CLS at 0, planets at
        ``[1, 1+MAX_PLANETS)``).

        FLOP reduction vs unpacked dense ≈ ``(L / L_packed)**2`` on
        attention (typical 4-6× for our token budgets). All ops are
        plain dense torch — no NestedTensor, no graph breaks — so
        ``torch.compile`` traces straight through. The single host sync
        is ``counts.max().item()`` for ``L_packed`` (compile users must
        enable ``torch._dynamo.config.capture_scalar_outputs = True``).
        """

        b, l, _ = features.shape
        future_planet_features = build_future_planet_features(owner_idx, features) if self.future_feature_enabled else None
        x_dense = self.embed(entity_type, owner_idx, features, future_planet_features)  # [B, L, d]

        # Stable sort (active first, original order preserved) — 0 for
        # active, 1 for inactive ⇒ stable argsort puts active before
        # inactive while keeping each group's original order. CLS is at
        # original index 0 and is always active, so it ends up at pack
        # position 0; planet slots that are active stay in slot order.
        counts = entity_mask.sum(dim=-1).to(torch.int64)  # [B]
        L_packed = int(counts.max().item())  # host sync (Python int)
        sort_keys = (~entity_mask).to(torch.int32)  # [B, L]
        pack_idx_full = sort_keys.argsort(dim=-1, stable=True)  # [B, L]
        pack_idx = pack_idx_full[:, :L_packed]  # [B, L_packed]

        pack_idx_d = pack_idx.unsqueeze(-1).expand(b, L_packed, self.d_model)
        x_packed = torch.gather(x_dense, 1, pack_idx_d)
        pack_idx_r = pack_idx.unsqueeze(-1).expand(b, L_packed, rope_pos.shape[-1])
        rope_packed = torch.gather(rope_pos, 1, pack_idx_r)

        # ``True = pad/masked`` for keys past each sample's active count.
        arange = torch.arange(L_packed, device=counts.device)
        padding_mask = arange[None, :] >= counts[:, None]  # [B, L_packed]

        pop = self._normalize_population_idx(population_idx, b, features.device)
        h_packed, _ = self._apply_encoder(x_packed, rope_packed, padding_mask, pop)
        value_h_packed = (
            self._apply_critic_encoder(x_packed, rope_packed, padding_mask, pop)[0]
            if self.disjoint_actor_critic
            else h_packed
        )

        # Scatter back to dense [B, L, d]. Padding rows of ``h_packed``
        # contain garbage (masked-attention output), and they get scattered
        # into the original positions of inactive tokens. Action heads
        # never read those positions: halt/value read CLS at index 0
        # (always active), and pair / fraction heads gate on
        # ``pair_mask`` / explicit indices, so the garbage is harmless.
        h = torch.zeros(b, l, self.d_model, dtype=h_packed.dtype, device=h_packed.device)
        h = h.scatter(1, pack_idx_d, h_packed)
        if self.disjoint_actor_critic:
            value_h = torch.zeros(b, l, self.d_model, dtype=value_h_packed.dtype, device=value_h_packed.device)
            value_h = value_h.scatter(1, pack_idx_d, value_h_packed)
        else:
            value_h = h

        if self.population_size == 1:
            return self._compute_outputs_single(
                h,
                value_h,
                owner_idx,
                features,
                entity_mask,
                planet_mask,
                origin_frac_blocked=origin_frac_blocked,
                origin_frac_head=self.origin_frac_head,
                halt_head=self.halt_head,
                value_head=(self.critic_value_head if self.disjoint_actor_critic else self.value_head),
                abort_head=getattr(self, "abort_head", None),
                value_head_idx=value_head_idx,
            )
        return self._compute_outputs_population(
            h,
            value_h,
            owner_idx,
            features,
            entity_mask,
            planet_mask,
            pop,
            origin_frac_blocked=origin_frac_blocked,
            value_head_idx=value_head_idx,
        )

    def _forward_dense_fixed(
        self,
        entity_type: torch.Tensor,
        owner_idx: torch.Tensor,
        features: torch.Tensor,
        rope_pos: torch.Tensor,
        entity_mask: torch.Tensor,
        planet_mask: torch.Tensor,
        origin_frac_blocked: Optional[torch.Tensor] = None,
        population_idx: Optional[torch.Tensor] = None,
        value_head_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Fixed-length dense path shared by rollout and PPO."""

        future_planet_features = build_future_planet_features(owner_idx, features) if self.future_feature_enabled else None
        x = self.embed(entity_type, owner_idx, features, future_planet_features)
        padding_mask = ~entity_mask

        h, pop = self._apply_encoder(x, rope_pos, padding_mask, population_idx)
        value_h = self._apply_critic_encoder(x, rope_pos, padding_mask, population_idx)[0] if self.disjoint_actor_critic else h
        if self.population_size == 1:
            return self._compute_outputs_single(
                h,
                value_h,
                owner_idx,
                features,
                entity_mask,
                planet_mask,
                origin_frac_blocked=origin_frac_blocked,
                origin_frac_head=self.origin_frac_head,
                halt_head=self.halt_head,
                value_head=(self.critic_value_head if self.disjoint_actor_critic else self.value_head),
                abort_head=getattr(self, "abort_head", None),
                value_head_idx=value_head_idx,
            )
        return self._compute_outputs_population(
            h,
            value_h,
            owner_idx,
            features,
            entity_mask,
            planet_mask,
            pop,
            origin_frac_blocked=origin_frac_blocked,
            value_head_idx=value_head_idx,
        )

    def forward_dense_rollout(
        self,
        entity_type: torch.Tensor,
        owner_idx: torch.Tensor,
        features: torch.Tensor,
        rope_pos: torch.Tensor,
        entity_mask: torch.Tensor,
        planet_mask: torch.Tensor,
        origin_frac_blocked: Optional[torch.Tensor] = None,
        population_idx: Optional[torch.Tensor] = None,
        value_head_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Fixed-length rollout forward path.

        This entry point is compiled separately for rollout, whose active
        batch size changes across micro-steps.
        """

        return self._forward_dense_fixed(
            entity_type=entity_type,
            owner_idx=owner_idx,
            features=features,
            rope_pos=rope_pos,
            entity_mask=entity_mask,
            planet_mask=planet_mask,
            origin_frac_blocked=origin_frac_blocked,
            population_idx=population_idx,
            value_head_idx=value_head_idx,
        )

    def forward_dense_rollout_grouped_population(
        self,
        entity_type: torch.Tensor,
        owner_idx: torch.Tensor,
        features: torch.Tensor,
        rope_pos: torch.Tensor,
        entity_mask: torch.Tensor,
        planet_mask: torch.Tensor,
        origin_frac_blocked: Optional[torch.Tensor] = None,
        value_head_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Fixed-length rollout forward path for contiguous per-member batch chunks."""

        future_planet_features = build_future_planet_features(owner_idx, features) if self.future_feature_enabled else None
        x = self.embed(entity_type, owner_idx, features, future_planet_features)
        padding_mask = ~entity_mask
        h = self._apply_encoder_grouped_population(x, rope_pos, padding_mask)
        value_h = self._apply_critic_encoder_grouped_population(x, rope_pos, padding_mask) if self.disjoint_actor_critic else h
        return self._compute_outputs_grouped_population(
            h,
            value_h,
            owner_idx,
            features,
            entity_mask,
            planet_mask,
            origin_frac_blocked=origin_frac_blocked,
            value_head_idx=value_head_idx,
        )

    def forward_dense_rollout_compressed(
        self,
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
        origin_frac_blocked: Optional[torch.Tensor] = None,
        population_idx: Optional[torch.Tensor] = None,
        value_head_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
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
            origin_frac_blocked=torch.zeros(
                (token_meta.shape[0], MAX_PLANETS, NUM_FRACTIONS),
                dtype=torch.bool,
                device=token_meta.device,
            )
            if origin_frac_blocked is None
            else origin_frac_blocked.to(device=token_meta.device, dtype=torch.bool),
        )
        obs = decode_observation(comp, feature_dim=int(feature_dim))
        return self._forward_dense_fixed(
            entity_type=obs["entity_type"],
            owner_idx=obs["owner_idx"],
            features=obs["features"],
            rope_pos=obs["rope_pos"],
            entity_mask=obs["entity_mask"],
            planet_mask=obs["planet_mask"],
            origin_frac_blocked=origin_frac_blocked,
            population_idx=population_idx,
            value_head_idx=value_head_idx,
        )

    def forward_dense_rollout_grouped_population_compressed(
        self,
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
        origin_frac_blocked: Optional[torch.Tensor] = None,
        value_head_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
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
            origin_frac_blocked=torch.zeros(
                (token_meta.shape[0], MAX_PLANETS, NUM_FRACTIONS),
                dtype=torch.bool,
                device=token_meta.device,
            )
            if origin_frac_blocked is None
            else origin_frac_blocked.to(device=token_meta.device, dtype=torch.bool),
        )
        obs = decode_observation(comp, feature_dim=int(feature_dim))
        future_planet_features = (
            build_future_planet_features(obs["owner_idx"], obs["features"]) if self.future_feature_enabled else None
        )
        x = self.embed(obs["entity_type"], obs["owner_idx"], obs["features"], future_planet_features)
        padding_mask = ~obs["entity_mask"]
        h = self._apply_encoder_grouped_population(x, obs["rope_pos"], padding_mask)
        value_h = self._apply_critic_encoder_grouped_population(x, obs["rope_pos"], padding_mask) if self.disjoint_actor_critic else h
        return self._compute_outputs_grouped_population(
            h,
            value_h,
            obs["owner_idx"],
            obs["features"],
            obs["entity_mask"],
            obs["planet_mask"],
            origin_frac_blocked=origin_frac_blocked,
            value_head_idx=value_head_idx,
        )

    def forward_sorted_population(
        self,
        entity_type: torch.Tensor,
        owner_idx: torch.Tensor,
        features: torch.Tensor,
        rope_pos: torch.Tensor,
        entity_mask: torch.Tensor,
        planet_mask: torch.Tensor,
        population_idx: torch.Tensor,
        origin_frac_blocked: Optional[torch.Tensor] = None,
        value_head_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Packed forward assuming rows are already contiguous by population member."""

        b, l, _ = features.shape
        future_planet_features = build_future_planet_features(owner_idx, features) if self.future_feature_enabled else None
        x_dense = self.embed(entity_type, owner_idx, features, future_planet_features)
        counts = entity_mask.sum(dim=-1).to(torch.int64)
        L_packed = int(counts.max().item())
        sort_keys = (~entity_mask).to(torch.int32)
        pack_idx_full = sort_keys.argsort(dim=-1, stable=True)
        pack_idx = pack_idx_full[:, :L_packed]
        pack_idx_d = pack_idx.unsqueeze(-1).expand(b, L_packed, self.d_model)
        x_packed = torch.gather(x_dense, 1, pack_idx_d)
        pack_idx_r = pack_idx.unsqueeze(-1).expand(b, L_packed, rope_pos.shape[-1])
        rope_packed = torch.gather(rope_pos, 1, pack_idx_r)
        arange = torch.arange(L_packed, device=counts.device)
        padding_mask = arange[None, :] >= counts[:, None]
        member_counts = self._population_member_counts(population_idx, b, features.device)
        h_packed = self._apply_encoder_sorted_population(x_packed, rope_packed, padding_mask, member_counts)
        value_h_packed = (
            self._apply_critic_encoder_sorted_population(x_packed, rope_packed, padding_mask, member_counts)
            if self.disjoint_actor_critic
            else h_packed
        )
        h = torch.zeros(b, l, self.d_model, dtype=h_packed.dtype, device=h_packed.device)
        h = h.scatter(1, pack_idx_d, h_packed)
        if self.disjoint_actor_critic:
            value_h = torch.zeros(b, l, self.d_model, dtype=value_h_packed.dtype, device=value_h_packed.device)
            value_h = value_h.scatter(1, pack_idx_d, value_h_packed)
        else:
            value_h = h
        return self._compute_outputs_sorted_population(
            h,
            value_h,
            owner_idx,
            features,
            entity_mask,
            planet_mask,
            member_counts,
            origin_frac_blocked=origin_frac_blocked,
            value_head_idx=value_head_idx,
        )

    def forward_ppo_sorted_population(
        self,
        entity_type: torch.Tensor,
        owner_idx: torch.Tensor,
        features: torch.Tensor,
        rope_pos: torch.Tensor,
        entity_mask: torch.Tensor,
        planet_mask: torch.Tensor,
        population_idx: torch.Tensor,
        origin_idx: torch.Tensor,
        frac_idx: torch.Tensor,
        fleet_size: torch.Tensor,
        target_eta: torch.Tensor,
        target_ships: torch.Tensor,
        origin_frac_blocked: Optional[torch.Tensor] = None,
        value_head_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """PPO-specific sorted forward that computes all private outputs in one member loop."""

        b, l, _ = features.shape
        future_planet_features = build_future_planet_features(owner_idx, features) if self.future_feature_enabled else None
        x_dense = self.embed(entity_type, owner_idx, features, future_planet_features)
        counts = entity_mask.sum(dim=-1).to(torch.int64)
        L_packed = int(counts.max().item())
        sort_keys = (~entity_mask).to(torch.int32)
        pack_idx_full = sort_keys.argsort(dim=-1, stable=True)
        pack_idx = pack_idx_full[:, :L_packed]
        pack_idx_d = pack_idx.unsqueeze(-1).expand(b, L_packed, self.d_model)
        x_packed = torch.gather(x_dense, 1, pack_idx_d)
        pack_idx_r = pack_idx.unsqueeze(-1).expand(b, L_packed, rope_pos.shape[-1])
        rope_packed = torch.gather(rope_pos, 1, pack_idx_r)
        arange = torch.arange(L_packed, device=counts.device)
        padding_mask = arange[None, :] >= counts[:, None]
        member_counts = self._population_member_counts(population_idx, b, features.device)
        h_packed = self._apply_encoder_sorted_population(x_packed, rope_packed, padding_mask, member_counts)
        value_h_packed = (
            self._apply_critic_encoder_sorted_population(x_packed, rope_packed, padding_mask, member_counts)
            if self.disjoint_actor_critic
            else h_packed
        )
        h = torch.zeros(b, l, self.d_model, dtype=h_packed.dtype, device=h_packed.device)
        h = h.scatter(1, pack_idx_d, h_packed)
        if self.disjoint_actor_critic:
            value_h = torch.zeros(b, l, self.d_model, dtype=value_h_packed.dtype, device=value_h_packed.device)
            value_h = value_h.scatter(1, pack_idx_d, value_h_packed)
        else:
            value_h = h
        return self._compute_ppo_outputs_sorted_population(
            h,
            value_h,
            owner_idx,
            features,
            entity_mask,
            planet_mask,
            origin_idx,
            frac_idx,
            fleet_size,
            target_eta,
            target_ships,
            member_counts,
            origin_frac_blocked=origin_frac_blocked,
            value_head_idx=value_head_idx,
        )

    def _apply_critic_encoder_sorted_population(
        self,
        x: torch.Tensor,
        rope_pos: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        member_counts: torch.Tensor,
    ) -> torch.Tensor:
        if not self.disjoint_actor_critic:
            return self._apply_encoder_sorted_population(x, rope_pos, key_padding_mask, member_counts)
        if self.population_size == 1:
            for blk in self.critic_blocks:
                x = self._run_block(blk, x, rope_pos, key_padding_mask)
            return self.critic_norm_f(x)

        for blk in self.critic_shared_blocks:
            x = self._run_block(blk, x, rope_pos, key_padding_mask)

        chunks = []
        start = 0
        for member_idx, tail in enumerate(self.critic_population_tails):
            count = int(member_counts[member_idx].item())
            if count <= 0:
                continue
            stop = start + count
            x_m = x[start:stop]
            rope_m = rope_pos[start:stop]
            mask_m = None if key_padding_mask is None else key_padding_mask[start:stop]
            x_m = self._run_block(tail.block, x_m, rope_m, mask_m)
            chunks.append(tail.norm_f(x_m))
            start = stop
        return torch.cat(chunks, dim=0) if chunks else x.new_empty((0,) + x.shape[1:])

    def target_logits_for_origin_fraction(
        self,
        planet_hidden: torch.Tensor,
        owner_idx: torch.Tensor,
        features: torch.Tensor,
        origin_idx: torch.Tensor,
        frac_idx: torch.Tensor,
        fleet_size: Optional[torch.Tensor] = None,
        target_eta: Optional[torch.Tensor] = None,
        target_ships: Optional[torch.Tensor] = None,
        population_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-target pick logits after sampled launch size/reachability are known.

        ``origin_idx`` and ``frac_idx`` are accepted for call-site stability, but
        target scoring intentionally ignores origin identity: once fleet size is
        fixed, the launch angle is chosen by first-hit geometry.
        """

        with self._heads_autocast_disabled(planet_hidden.device):
            planet_hidden = self._fp32_head_input(planet_hidden)
            b = origin_idx.shape[0]
            device = planet_hidden.device
            del frac_idx
            if fleet_size is None:
                fleet_scalar = torch.zeros((b, 1, 1), device=device, dtype=planet_hidden.dtype)
            else:
                fleet_scalar = (fleet_size.to(device=device, dtype=planet_hidden.dtype) / 1000.0).reshape(b, 1, 1)
            fleet_feat = fleet_scalar.expand(-1, planet_hidden.shape[1], -1)
            if target_eta is None:
                eta_feat = torch.zeros(
                    (b, planet_hidden.shape[1], 1), device=device, dtype=planet_hidden.dtype
                )
            else:
                eta_feat = (target_eta.to(device=device, dtype=planet_hidden.dtype) / 500.0).unsqueeze(-1)
            if target_ships is None:
                is_bigger = torch.zeros((b, planet_hidden.shape[1], 1), device=device, dtype=planet_hidden.dtype)
            else:
                target_ships_t = target_ships.to(device=device, dtype=planet_hidden.dtype)
                is_bigger = (fleet_scalar > target_ships_t.unsqueeze(-1)).to(dtype=planet_hidden.dtype)
            target_parts = [planet_hidden, fleet_feat, eta_feat, is_bigger]
            if self.future_feature_enabled:
                if target_eta is None or fleet_size is None:
                    raise ValueError("future_feature_enabled target head requires fleet_size and target_eta")
                origin_future, target_future = self._target_future_context(
                    owner_idx,
                    features,
                    origin_idx,
                    fleet_size,
                    target_eta,
                )
                target_parts.extend(
                    [
                        origin_future.to(device=device, dtype=planet_hidden.dtype),
                        target_future.to(device=device, dtype=planet_hidden.dtype),
                    ]
                )
            target_in = torch.cat(target_parts, dim=-1)
            if self.population_size == 1:
                return self.target_pick_head(target_in).squeeze(-1)

            pop = self._normalize_population_idx(population_idx, b, planet_hidden.device)
            logits: Optional[torch.Tensor] = None
            for member_idx, tail in enumerate(self.population_tails):
                member_rows = torch.nonzero(pop == member_idx, as_tuple=False).squeeze(-1)
                if member_rows.numel() == 0:
                    continue
                logits_m = tail.target_pick_head(target_in.index_select(0, member_rows)).squeeze(-1)
                if logits is None:
                    logits = logits_m.new_empty((b, planet_hidden.shape[1]))
                logits.index_copy_(0, member_rows, logits_m)
            assert logits is not None
            return logits
    def target_logits_for_origin_fraction_grouped_population(
        self,
        planet_hidden: torch.Tensor,
        owner_idx: torch.Tensor,
        features: torch.Tensor,
        origin_idx: torch.Tensor,
        frac_idx: torch.Tensor,
        fleet_size: Optional[torch.Tensor] = None,
        target_eta: Optional[torch.Tensor] = None,
        target_ships: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Grouped rollout target logits for contiguous per-member batch chunks."""

        with self._heads_autocast_disabled(planet_hidden.device):
            planet_hidden = self._fp32_head_input(planet_hidden)
            b = origin_idx.shape[0]
            device = planet_hidden.device
            del frac_idx
            if fleet_size is None:
                fleet_scalar = torch.zeros((b, 1, 1), device=device, dtype=planet_hidden.dtype)
            else:
                fleet_scalar = (fleet_size.to(device=device, dtype=planet_hidden.dtype) / 1000.0).reshape(b, 1, 1)
            fleet_feat = fleet_scalar.expand(-1, planet_hidden.shape[1], -1)
            if target_eta is None:
                eta_feat = torch.zeros(
                    (b, planet_hidden.shape[1], 1), device=device, dtype=planet_hidden.dtype
                )
            else:
                eta_feat = (target_eta.to(device=device, dtype=planet_hidden.dtype) / 500.0).unsqueeze(-1)
            if target_ships is None:
                is_bigger = torch.zeros((b, planet_hidden.shape[1], 1), device=device, dtype=planet_hidden.dtype)
            else:
                target_ships_t = target_ships.to(device=device, dtype=planet_hidden.dtype)
                is_bigger = (fleet_scalar > target_ships_t.unsqueeze(-1)).to(dtype=planet_hidden.dtype)
            target_parts = [planet_hidden, fleet_feat, eta_feat, is_bigger]
            if self.future_feature_enabled:
                if target_eta is None or fleet_size is None:
                    raise ValueError("future_feature_enabled target head requires fleet_size and target_eta")
                origin_future, target_future = self._target_future_context(
                    owner_idx,
                    features,
                    origin_idx,
                    fleet_size,
                    target_eta,
                )
                target_parts.extend(
                    [
                        origin_future.to(device=device, dtype=planet_hidden.dtype),
                        target_future.to(device=device, dtype=planet_hidden.dtype),
                    ]
                )
            target_in = torch.cat(target_parts, dim=-1)
            if self.population_size == 1:
                return self.target_pick_head(target_in).squeeze(-1)

            group_size = self._population_group_size(b)
            logits = []
            for member_idx, tail in enumerate(self.population_tails):
                start = member_idx * group_size
                stop = start + group_size
                logits.append(tail.target_pick_head(target_in[start:stop]).squeeze(-1))
            return torch.cat(logits, dim=0)
    def target_logits_for_origin_fraction_sorted_population(
        self,
        planet_hidden: torch.Tensor,
        owner_idx: torch.Tensor,
        features: torch.Tensor,
        origin_idx: torch.Tensor,
        frac_idx: torch.Tensor,
        population_idx: torch.Tensor,
        fleet_size: Optional[torch.Tensor] = None,
        target_eta: Optional[torch.Tensor] = None,
        target_ships: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Target logits assuming rows are already contiguous by population member."""

        with self._heads_autocast_disabled(planet_hidden.device):
            planet_hidden = self._fp32_head_input(planet_hidden)
            b = origin_idx.shape[0]
            device = planet_hidden.device
            del frac_idx
            if fleet_size is None:
                fleet_scalar = torch.zeros((b, 1, 1), device=device, dtype=planet_hidden.dtype)
            else:
                fleet_scalar = (fleet_size.to(device=device, dtype=planet_hidden.dtype) / 1000.0).reshape(b, 1, 1)
            fleet_feat = fleet_scalar.expand(-1, planet_hidden.shape[1], -1)
            if target_eta is None:
                eta_feat = torch.zeros(
                    (b, planet_hidden.shape[1], 1), device=device, dtype=planet_hidden.dtype
                )
            else:
                eta_feat = (target_eta.to(device=device, dtype=planet_hidden.dtype) / 500.0).unsqueeze(-1)
            if target_ships is None:
                is_bigger = torch.zeros((b, planet_hidden.shape[1], 1), device=device, dtype=planet_hidden.dtype)
            else:
                target_ships_t = target_ships.to(device=device, dtype=planet_hidden.dtype)
                is_bigger = (fleet_scalar > target_ships_t.unsqueeze(-1)).to(dtype=planet_hidden.dtype)
            target_parts = [planet_hidden, fleet_feat, eta_feat, is_bigger]
            if self.future_feature_enabled:
                if target_eta is None or fleet_size is None:
                    raise ValueError("future_feature_enabled target head requires fleet_size and target_eta")
                origin_future, target_future = self._target_future_context(
                    owner_idx,
                    features,
                    origin_idx,
                    fleet_size,
                    target_eta,
                )
                target_parts.extend(
                    [
                        origin_future.to(device=device, dtype=planet_hidden.dtype),
                        target_future.to(device=device, dtype=planet_hidden.dtype),
                    ]
                )
            target_in = torch.cat(target_parts, dim=-1)
            if self.population_size == 1:
                return self.target_pick_head(target_in).squeeze(-1)

            member_counts = self._population_member_counts(population_idx, b, planet_hidden.device)
            logits = []
            start = 0
            for member_idx, tail in enumerate(self.population_tails):
                count = int(member_counts[member_idx].item())
                if count <= 0:
                    continue
                stop = start + count
                logits.append(tail.target_pick_head(target_in[start:stop]).squeeze(-1))
                start = stop
            return torch.cat(logits, dim=0) if logits else target_in.new_empty((0, target_in.shape[1]))
    def fraction_logits(
        self,
        planet_hidden: torch.Tensor,
        origin_idx: torch.Tensor,
        dest_idx: torch.Tensor,
        times_norm: torch.Tensor,
        population_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """times_norm: [B, NUM_FRACTIONS] — eta_k / 500 per fraction head."""

        with self._heads_autocast_disabled(planet_hidden.device):
            planet_hidden = self._fp32_head_input(planet_hidden)
            times_norm = self._fp32_head_input(times_norm)
            b = origin_idx.shape[0]
            device = planet_hidden.device
            ho = planet_hidden[torch.arange(b, device=device), origin_idx]
            hd = planet_hidden[torch.arange(b, device=device), dest_idx]
            if self.population_size == 1:
                logits = []
                for k in range(NUM_FRACTIONS):
                    tt = times_norm[:, k : k + 1]
                    te = self.time_proj(tt)
                    z = torch.cat([ho, hd, te], dim=-1)
                    logits.append(self.frac_heads[k](z).squeeze(-1))
                return torch.stack(logits, dim=-1)

            pop = self._normalize_population_idx(population_idx, b, planet_hidden.device)
            logits: Optional[torch.Tensor] = None
            for member_idx, tail in enumerate(self.population_tails):
                member_rows = torch.nonzero(pop == member_idx, as_tuple=False).squeeze(-1)
                if member_rows.numel() == 0:
                    continue
                ho_m = ho.index_select(0, member_rows)
                hd_m = hd.index_select(0, member_rows)
                logits_m = []
                for k in range(NUM_FRACTIONS):
                    tt = times_norm.index_select(0, member_rows)[:, k : k + 1]
                    te = tail.time_proj(tt)
                    z = torch.cat([ho_m, hd_m, te], dim=-1)
                    logits_m.append(tail.frac_heads[k](z).squeeze(-1))
                logits_m_t = torch.stack(logits_m, dim=-1)
                if logits is None:
                    logits = logits_m_t.new_empty((b, NUM_FRACTIONS))
                logits.index_copy_(0, member_rows, logits_m_t)
            assert logits is not None
            return logits

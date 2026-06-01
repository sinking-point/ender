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
    FEATURE_DIM_MULTI_ABORT,
    FRACTIONS,
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


def _make_target_pick_head(d_model: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model + 3, d_model),
        nn.GELU(),
        nn.Linear(d_model, d_model // 2),
        nn.GELU(),
        nn.Linear(d_model // 2, 1),
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
        halt_init_prob: Optional[float] = None,
        fraction_init_weights: Optional[Tuple[float, ...]] = None,
    ):
        super().__init__()
        self.block = TransformerBlock(d_model, n_heads, rope_dims=rope_dims, dropout=dropout)
        self.norm_f = nn.LayerNorm(d_model)
        self.pair_q = nn.Linear(d_model, d_model // 2, bias=False)
        self.pair_k = nn.Linear(d_model, d_model // 2, bias=False)
        self.origin_frac_head = nn.Linear(d_model, NUM_FRACTIONS)
        self.target_pick_head = _make_target_pick_head(d_model)
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
        target_abort_enabled: bool = False,
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
        self.target_abort_enabled = bool(target_abort_enabled)
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
        self.cls_type_idx = ENTITY_CLS

        if self.population_size == 1:
            self.blocks = nn.ModuleList(
                [TransformerBlock(d_model, n_heads, rope_dims=rope_dims, dropout=dropout) for _ in range(n_layers)]
            )
            self.norm_f = nn.LayerNorm(d_model)
            self.pair_q = nn.Linear(d_model, d_model // 2, bias=False)
            self.pair_k = nn.Linear(d_model, d_model // 2, bias=False)
            self.origin_frac_head = nn.Linear(d_model, NUM_FRACTIONS)
            self.target_pick_head = _make_target_pick_head(d_model)
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
                        halt_init_prob=self.halt_init_prob,
                        fraction_init_weights=self.fraction_init_weights,
                    )
                    for _ in range(self.population_size)
                ]
            )

        self.register_buffer("_frac_const", torch.tensor(FRACTIONS, dtype=torch.float32))

    def embed(self, entity_type: torch.Tensor, owner_idx: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        t_ok = entity_type.clamp(0, NUM_ENTITY_TYPES - 1)
        o_ok = owner_idx.clamp(0, NUM_OWNER_SLOTS - 1)
        return self.type_emb(t_ok) + self.owner_emb(o_ok) + self.feat_proj(features)

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
        owner_idx: torch.Tensor,
        features: torch.Tensor,
        entity_mask: torch.Tensor,
        planet_mask: torch.Tensor,
        origin_frac_blocked: Optional[torch.Tensor] = None,
        *,
        pair_q: nn.Linear,
        pair_k: nn.Linear,
        origin_frac_head: nn.Linear,
        halt_head: nn.Linear,
        value_head: nn.Linear,
        abort_head: Optional[nn.Linear] = None,
        value_head_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        b = h.shape[0]
        cls_h = h[:, 0, :]
        halt_logits = halt_head(cls_h)
        value_all = value_head(cls_h)
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
        pq = pair_q(planet_h)
        pk = pair_k(planet_h)
        pair_logits = torch.matmul(pq, pk.transpose(-2, -1)) * (pq.shape[-1] ** -0.5)

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
            "pair_logits": pair_logits,
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
        pair_q: nn.Linear,
        pair_k: nn.Linear,
        origin_frac_head: nn.Linear,
        halt_head: nn.Linear,
        value_head: nn.Linear,
        target_pick_head: nn.Module,
        abort_head: Optional[nn.Linear] = None,
        value_head_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        out = self._compute_outputs_single(
            h,
            owner_idx,
            features,
            entity_mask,
            planet_mask,
            origin_frac_blocked=origin_frac_blocked,
            pair_q=pair_q,
            pair_k=pair_k,
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
        target_in = torch.cat([planet_hidden, fleet_feat, eta_feat, is_bigger], dim=-1)
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

    def _compute_outputs_population(
        self,
        h: torch.Tensor,
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
            out_m = self._compute_outputs_single(
                h.index_select(0, member_rows),
                owner_idx.index_select(0, member_rows),
                features.index_select(0, member_rows),
                entity_mask.index_select(0, member_rows),
                planet_mask.index_select(0, member_rows),
                origin_frac_blocked=None if origin_frac_blocked is None else origin_frac_blocked.index_select(0, member_rows),
                pair_q=tail.pair_q,
                pair_k=tail.pair_k,
                origin_frac_head=tail.origin_frac_head,
                halt_head=tail.halt_head,
                value_head=tail.value_head,
                abort_head=getattr(tail, "abort_head", None),
                value_head_idx=None if value_head_idx is None else value_head_idx.index_select(0, member_rows),
            )
            if outputs is None:
                outputs = {
                    "hidden": h,
                    "halt_logits": out_m["halt_logits"].new_empty((batch_size, 2)),
                    "value": out_m["value"].new_empty((batch_size,)),
                    "pair_logits": out_m["pair_logits"].new_empty((batch_size, MAX_PLANETS, MAX_PLANETS)),
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
            outputs["pair_logits"].index_copy_(0, member_rows, out_m["pair_logits"])
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
                owner_idx,
                features,
                entity_mask,
                planet_mask,
                origin_frac_blocked=origin_frac_blocked,
                pair_q=self.pair_q,
                pair_k=self.pair_k,
                origin_frac_head=self.origin_frac_head,
                halt_head=self.halt_head,
                value_head=self.value_head,
                abort_head=getattr(self, "abort_head", None),
                value_head_idx=value_head_idx,
            )

        group_size = self._population_group_size(int(h.shape[0]))
        outputs: Dict[str, Any] = {"hidden": h}
        halt_logits = []
        value = []
        pair_logits = []
        pair_mask = []
        origin_frac_logits = []
        origin_frac_mask = []
        planet_hidden = []
        abort_logits = []
        for member_idx, tail in enumerate(self.population_tails):
            start = member_idx * group_size
            stop = start + group_size
            out_m = self._compute_outputs_single(
                h[start:stop],
                owner_idx[start:stop],
                features[start:stop],
                entity_mask[start:stop],
                planet_mask[start:stop],
                origin_frac_blocked=None if origin_frac_blocked is None else origin_frac_blocked[start:stop],
                pair_q=tail.pair_q,
                pair_k=tail.pair_k,
                origin_frac_head=tail.origin_frac_head,
                halt_head=tail.halt_head,
                value_head=tail.value_head,
                abort_head=getattr(tail, "abort_head", None),
                value_head_idx=None if value_head_idx is None else value_head_idx[start:stop],
            )
            halt_logits.append(out_m["halt_logits"])
            value.append(out_m["value"])
            pair_logits.append(out_m["pair_logits"])
            pair_mask.append(out_m["pair_mask"])
            origin_frac_logits.append(out_m["origin_frac_logits"])
            origin_frac_mask.append(out_m["origin_frac_mask"])
            planet_hidden.append(out_m["planet_hidden"])
            if self.target_abort_enabled and "abort_logits" in out_m:
                abort_logits.append(out_m["abort_logits"])
        outputs["halt_logits"] = torch.cat(halt_logits, dim=0)
        outputs["value"] = torch.cat(value, dim=0)
        outputs["pair_logits"] = torch.cat(pair_logits, dim=0)
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
                owner_idx,
                features,
                entity_mask,
                planet_mask,
                origin_frac_blocked=origin_frac_blocked,
                pair_q=self.pair_q,
                pair_k=self.pair_k,
                origin_frac_head=self.origin_frac_head,
                halt_head=self.halt_head,
                value_head=self.value_head,
                abort_head=getattr(self, "abort_head", None),
                value_head_idx=value_head_idx,
            )

        outputs: Dict[str, Any] = {"hidden": h}
        halt_logits = []
        value = []
        pair_logits = []
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
            out_m = self._compute_outputs_single(
                h[start:stop],
                owner_idx[start:stop],
                features[start:stop],
                entity_mask[start:stop],
                planet_mask[start:stop],
                origin_frac_blocked=None if origin_frac_blocked is None else origin_frac_blocked[start:stop],
                pair_q=tail.pair_q,
                pair_k=tail.pair_k,
                origin_frac_head=tail.origin_frac_head,
                halt_head=tail.halt_head,
                value_head=tail.value_head,
                abort_head=getattr(tail, "abort_head", None),
                value_head_idx=None if value_head_idx is None else value_head_idx[start:stop],
            )
            halt_logits.append(out_m["halt_logits"])
            value.append(out_m["value"])
            pair_logits.append(out_m["pair_logits"])
            pair_mask.append(out_m["pair_mask"])
            origin_frac_logits.append(out_m["origin_frac_logits"])
            origin_frac_mask.append(out_m["origin_frac_mask"])
            planet_hidden.append(out_m["planet_hidden"])
            if self.target_abort_enabled and "abort_logits" in out_m:
                abort_logits.append(out_m["abort_logits"])
            start = stop
        outputs["halt_logits"] = torch.cat(halt_logits, dim=0)
        outputs["value"] = torch.cat(value, dim=0)
        outputs["pair_logits"] = torch.cat(pair_logits, dim=0)
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
                pair_q=self.pair_q,
                pair_k=self.pair_k,
                origin_frac_head=self.origin_frac_head,
                halt_head=self.halt_head,
                value_head=self.value_head,
                target_pick_head=self.target_pick_head,
                abort_head=getattr(self, "abort_head", None),
                value_head_idx=value_head_idx,
            )

        outputs: Dict[str, Any] = {"hidden": h}
        halt_logits = []
        value = []
        pair_logits = []
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
            out_m = self._compute_ppo_outputs_single(
                h[start:stop],
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
                pair_q=tail.pair_q,
                pair_k=tail.pair_k,
                origin_frac_head=tail.origin_frac_head,
                halt_head=tail.halt_head,
                value_head=tail.value_head,
                target_pick_head=tail.target_pick_head,
                abort_head=getattr(tail, "abort_head", None),
                value_head_idx=None if value_head_idx is None else value_head_idx[start:stop],
            )
            halt_logits.append(out_m["halt_logits"])
            value.append(out_m["value"])
            pair_logits.append(out_m["pair_logits"])
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
        outputs["pair_logits"] = torch.cat(pair_logits, dim=0)
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
        x_dense = self.embed(entity_type, owner_idx, features)  # [B, L, d]

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

        # Scatter back to dense [B, L, d]. Padding rows of ``h_packed``
        # contain garbage (masked-attention output), and they get scattered
        # into the original positions of inactive tokens. Action heads
        # never read those positions: halt/value read CLS at index 0
        # (always active), and pair / fraction heads gate on
        # ``pair_mask`` / explicit indices, so the garbage is harmless.
        h = torch.zeros(b, l, self.d_model, dtype=h_packed.dtype, device=h_packed.device)
        h = h.scatter(1, pack_idx_d, h_packed)

        if self.population_size == 1:
            return self._compute_outputs_single(
                h,
                owner_idx,
                features,
                entity_mask,
                planet_mask,
                origin_frac_blocked=origin_frac_blocked,
                pair_q=self.pair_q,
                pair_k=self.pair_k,
                origin_frac_head=self.origin_frac_head,
                halt_head=self.halt_head,
                value_head=self.value_head,
                abort_head=getattr(self, "abort_head", None),
                value_head_idx=value_head_idx,
            )
        return self._compute_outputs_population(
            h,
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

        x = self.embed(entity_type, owner_idx, features)
        padding_mask = ~entity_mask

        h, pop = self._apply_encoder(x, rope_pos, padding_mask, population_idx)
        if self.population_size == 1:
            return self._compute_outputs_single(
                h,
                owner_idx,
                features,
                entity_mask,
                planet_mask,
                origin_frac_blocked=origin_frac_blocked,
                pair_q=self.pair_q,
                pair_k=self.pair_k,
                origin_frac_head=self.origin_frac_head,
                halt_head=self.halt_head,
                value_head=self.value_head,
                abort_head=getattr(self, "abort_head", None),
                value_head_idx=value_head_idx,
            )
        return self._compute_outputs_population(
            h,
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

        x = self.embed(entity_type, owner_idx, features)
        padding_mask = ~entity_mask
        h = self._apply_encoder_grouped_population(x, rope_pos, padding_mask)
        return self._compute_outputs_grouped_population(
            h,
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
        x = self.embed(obs["entity_type"], obs["owner_idx"], obs["features"])
        padding_mask = ~obs["entity_mask"]
        h = self._apply_encoder_grouped_population(x, obs["rope_pos"], padding_mask)
        return self._compute_outputs_grouped_population(
            h,
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
        x_dense = self.embed(entity_type, owner_idx, features)
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
        h = torch.zeros(b, l, self.d_model, dtype=h_packed.dtype, device=h_packed.device)
        h = h.scatter(1, pack_idx_d, h_packed)
        return self._compute_outputs_sorted_population(
            h,
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
        x_dense = self.embed(entity_type, owner_idx, features)
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
        h = torch.zeros(b, l, self.d_model, dtype=h_packed.dtype, device=h_packed.device)
        h = h.scatter(1, pack_idx_d, h_packed)
        return self._compute_ppo_outputs_sorted_population(
            h,
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

    def target_logits_for_origin_fraction(
        self,
        planet_hidden: torch.Tensor,
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
        target_in = torch.cat([planet_hidden, fleet_feat, eta_feat, is_bigger], dim=-1)
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
        origin_idx: torch.Tensor,
        frac_idx: torch.Tensor,
        fleet_size: Optional[torch.Tensor] = None,
        target_eta: Optional[torch.Tensor] = None,
        target_ships: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Grouped rollout target logits for contiguous per-member batch chunks."""

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
        target_in = torch.cat([planet_hidden, fleet_feat, eta_feat, is_bigger], dim=-1)
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
        origin_idx: torch.Tensor,
        frac_idx: torch.Tensor,
        population_idx: torch.Tensor,
        fleet_size: Optional[torch.Tensor] = None,
        target_eta: Optional[torch.Tensor] = None,
        target_ships: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Target logits assuming rows are already contiguous by population member."""

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
        target_in = torch.cat([planet_hidden, fleet_feat, eta_feat, is_bigger], dim=-1)
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
            times_m = times_norm.index_select(0, member_rows)
            out_m = []
            for k in range(NUM_FRACTIONS):
                tt = times_m[:, k : k + 1]
                te = tail.time_proj(tt)
                z = torch.cat([ho_m, hd_m, te], dim=-1)
                out_m.append(tail.frac_heads[k](z).squeeze(-1))
            logits_m = torch.stack(out_m, dim=-1)
            if logits is None:
                logits = logits_m.new_empty((b, NUM_FRACTIONS))
            logits.index_copy_(0, member_rows, logits_m)
        assert logits is not None
        return logits

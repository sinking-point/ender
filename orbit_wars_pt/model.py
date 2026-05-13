"""Transformer policy with 3D RoPE (x, y, time) and Orbit Wars action heads."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from orbit_wars_pt.constants import (
    ENTITY_CLS,
    FEATURE_DIM,
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


def apply_rope_3d(q: torch.Tensor, k: torch.Tensor, rope_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split head dimensions into three chunks for spatial x,y and temporal t."""

    d_h = q.shape[-1]
    assert d_h % 6 == 0, "head_dim must be divisible by 6 for 3D RoPE chunks"
    c = d_h // 3
    qx, qy, qt = q[..., :c], q[..., c : 2 * c], q[..., 2 * c :]
    kx, ky, kt = k[..., :c], k[..., c : 2 * c], k[..., 2 * c :]
    px = rope_pos[..., 0]
    py = rope_pos[..., 1]
    pt = rope_pos[..., 2]
    qx = apply_rope_1d(qx, px)
    kx = apply_rope_1d(kx, px)
    qy = apply_rope_1d(qy, py)
    ky = apply_rope_1d(ky, py)
    qt = apply_rope_1d(qt, pt)
    kt = apply_rope_1d(kt, pt)
    q_out = torch.cat([qx, qy, qt], dim=-1)
    k_out = torch.cat([kx, ky, kt], dim=-1)
    return q_out, k_out


class RoPEAttention(nn.Module):
    """Pre-norm RoPE attention on dense ``[B, L_packed, d]``.

    ``L_packed`` is the per-batch max active token count after
    ``OrbitWarsPolicy.forward`` packs each sample's active tokens to the
    front. ``key_padding_mask`` (``True = masked``) marks the rows beyond
    each sample's count. SDPA dispatches to FlashAttention 2 (BF16/FP16
    on Ampere+) when no mask is needed, or to the memory-efficient kernel
    when a key_padding_mask is supplied.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
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
        r = rope_pos[:, None, :, :].expand(b, self.n_heads, l, 3)
        q, k = apply_rope_3d(q, k, r)

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
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPEAttention(d_model, n_heads, dropout=dropout)
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


class OrbitWarsPolicy(nn.Module):
    """Encoder-only transformer; heads for halt, planet pair, dispatch fractions, value."""

    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 8,
        n_layers: int = 4,
        feature_dim: int = FEATURE_DIM,
        dropout: float = 0.0,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        assert (d_model // n_heads) % 6 == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.activation_checkpointing = activation_checkpointing
        self.type_emb = nn.Embedding(NUM_ENTITY_TYPES, d_model)
        self.owner_emb = nn.Embedding(NUM_OWNER_SLOTS, d_model)
        self.feat_proj = nn.Linear(feature_dim, d_model)
        self.cls_type_idx = ENTITY_CLS

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.norm_f = nn.LayerNorm(d_model)

        self.pair_q = nn.Linear(d_model, d_model // 2, bias=False)
        self.pair_k = nn.Linear(d_model, d_model // 2, bias=False)
        self.target_q = nn.Linear(d_model, d_model // 2, bias=False)
        self.frac_emb = nn.Embedding(NUM_FRACTIONS, d_model)
        self.origin_frac_head = nn.Linear(d_model, NUM_FRACTIONS)

        self.time_proj = nn.Linear(1, 32)
        self.frac_heads = nn.ModuleList([nn.Linear(d_model * 2 + 32, 1) for _ in range(NUM_FRACTIONS)])

        self.halt_head = nn.Linear(d_model, 2)
        self.value_head = nn.Linear(d_model, 1)

        self.register_buffer("_frac_const", torch.tensor(FRACTIONS, dtype=torch.float32))

    def embed(self, entity_type: torch.Tensor, owner_idx: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        t_ok = entity_type.clamp(0, NUM_ENTITY_TYPES - 1)
        o_ok = owner_idx.clamp(0, NUM_OWNER_SLOTS - 1)
        return self.type_emb(t_ok) + self.owner_emb(o_ok) + self.feat_proj(features)

    def forward(
        self,
        entity_type: torch.Tensor,
        owner_idx: torch.Tensor,
        features: torch.Tensor,
        rope_pos: torch.Tensor,
        entity_mask: torch.Tensor,
        planet_mask: torch.Tensor,
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
        pack_idx_r = pack_idx.unsqueeze(-1).expand(b, L_packed, 3)
        rope_packed = torch.gather(rope_pos, 1, pack_idx_r)

        # ``True = pad/masked`` for keys past each sample's active count.
        arange = torch.arange(L_packed, device=counts.device)
        padding_mask = arange[None, :] >= counts[:, None]  # [B, L_packed]

        for blk in self.blocks:
            if self.activation_checkpointing and torch.is_grad_enabled():
                x_packed = checkpoint(
                    blk,
                    x_packed,
                    rope_packed,
                    padding_mask,
                    use_reentrant=False,
                )
            else:
                x_packed = blk(x_packed, rope_packed, padding_mask)
        h_packed = self.norm_f(x_packed)

        # Scatter back to dense [B, L, d]. Padding rows of ``h_packed``
        # contain garbage (masked-attention output), and they get scattered
        # into the original positions of inactive tokens. Action heads
        # never read those positions: halt/value read CLS at index 0
        # (always active), and pair / fraction heads gate on
        # ``pair_mask`` / explicit indices, so the garbage is harmless.
        h = torch.zeros(b, l, self.d_model, dtype=h_packed.dtype, device=h_packed.device)
        h = h.scatter(1, pack_idx_d, h_packed)

        cls_h = h[:, 0, :]
        halt_logits = self.halt_head(cls_h)
        value = self.value_head(cls_h).squeeze(-1)

        planet_h = h[:, 1 : 1 + MAX_PLANETS, :]
        pq = self.pair_q(planet_h)
        pk = self.pair_k(planet_h)
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
        origin_frac_logits = self.origin_frac_head(planet_h)

        return {
            "hidden": h,
            "halt_logits": halt_logits,
            "value": value,
            "pair_logits": pair_logits,
            "pair_mask": pair_mask,
            "origin_frac_logits": origin_frac_logits,
            "origin_frac_mask": origin_frac_mask,
            "planet_hidden": planet_h,
        }

    def _forward_dense_fixed(
        self,
        entity_type: torch.Tensor,
        owner_idx: torch.Tensor,
        features: torch.Tensor,
        rope_pos: torch.Tensor,
        entity_mask: torch.Tensor,
        planet_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        """Fixed-length dense path shared by rollout and PPO."""

        x = self.embed(entity_type, owner_idx, features)
        padding_mask = ~entity_mask

        for blk in self.blocks:
            if self.activation_checkpointing and torch.is_grad_enabled():
                x = checkpoint(
                    blk,
                    x,
                    rope_pos,
                    padding_mask,
                    use_reentrant=False,
                )
            else:
                x = blk(x, rope_pos, padding_mask)
        h = self.norm_f(x)

        cls_h = h[:, 0, :]
        halt_logits = self.halt_head(cls_h)
        value = self.value_head(cls_h).squeeze(-1)

        planet_h = h[:, 1 : 1 + MAX_PLANETS, :]
        pq = self.pair_q(planet_h)
        pk = self.pair_k(planet_h)
        pair_logits = torch.matmul(pq, pk.transpose(-2, -1)) * (pq.shape[-1] ** -0.5)

        pm = planet_mask[:, 1 : 1 + MAX_PLANETS]
        em = entity_mask[:, 1 : 1 + MAX_PLANETS]
        active_planet = pm & em
        eye = torch.eye(MAX_PLANETS, device=h.device, dtype=torch.bool).unsqueeze(0).expand(h.shape[0], -1, -1)
        owned_self = owner_idx[:, 1 : 1 + MAX_PLANETS] == 1
        ships = features[:, 1 : 1 + MAX_PLANETS, 1] * 1000.0
        has_ships = ships > 0.5
        origin_ok = active_planet & owned_self & has_ships
        dest_ok = active_planet
        pair_mask = origin_ok[:, :, None] & dest_ok[:, None, :] & ~eye
        sends = torch.floor(self._frac_const.to(features.dtype)[None, None, :] * ships[:, :, None])
        origin_frac_mask = origin_ok[:, :, None] & (sends >= 1.0)
        origin_frac_logits = self.origin_frac_head(planet_h)

        return {
            "hidden": h,
            "halt_logits": halt_logits,
            "value": value,
            "pair_logits": pair_logits,
            "pair_mask": pair_mask,
            "origin_frac_logits": origin_frac_logits,
            "origin_frac_mask": origin_frac_mask,
            "planet_hidden": planet_h,
        }

    def forward_dense_rollout(
        self,
        entity_type: torch.Tensor,
        owner_idx: torch.Tensor,
        features: torch.Tensor,
        rope_pos: torch.Tensor,
        entity_mask: torch.Tensor,
        planet_mask: torch.Tensor,
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
        )

    def target_logits_for_origin_fraction(
        self,
        planet_hidden: torch.Tensor,
        origin_idx: torch.Tensor,
        frac_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Target logits ``[B, MAX_PLANETS]`` conditioned on sampled origin/fraction."""

        b = origin_idx.shape[0]
        device = planet_hidden.device
        ho = planet_hidden[torch.arange(b, device=device), origin_idx]
        hf = ho + self.frac_emb(frac_idx)
        q = self.target_q(hf)
        k = self.pair_k(planet_hidden)
        return torch.einsum("bd,bpd->bp", q, k) * (q.shape[-1] ** -0.5)

    def fraction_logits(
        self,
        planet_hidden: torch.Tensor,
        origin_idx: torch.Tensor,
        dest_idx: torch.Tensor,
        times_norm: torch.Tensor,
    ) -> torch.Tensor:
        """times_norm: [B, NUM_FRACTIONS] — eta_k / 500 per fraction head."""

        b = origin_idx.shape[0]
        device = planet_hidden.device
        ho = planet_hidden[torch.arange(b, device=device), origin_idx]
        hd = planet_hidden[torch.arange(b, device=device), dest_idx]
        logits = []
        for k in range(NUM_FRACTIONS):
            tt = times_norm[:, k : k + 1]
            te = self.time_proj(tt)
            z = torch.cat([ho, hd, te], dim=-1)
            logits.append(self.frac_heads[k](z).squeeze(-1))
        return torch.stack(logits, dim=-1)

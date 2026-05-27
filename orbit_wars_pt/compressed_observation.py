"""Compressed rollout observation storage.

The policy still receives the standard observation dict.  Rollout storage keeps
a compact, lossy representation and decodes it immediately before forward, so
rollout and PPO replay use the same observation path.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from orbit_wars_pt.constants import (
    BOARD_SIZE,
    BLOCKED_FRAC_FEATURES,
    CENTER,
    FEATURE_DIM,
    FEATURE_DIM_ABORT,
    FEATURE_DIM_MULTI,
    FEATURE_DIM_MULTI_ABORT,
    INCOMING_SURVIVOR_FLAT,
    INCOMING_TA_BINS,
    MAX_PLANETS,
    NUM_FRACTIONS,
    NUM_OWNER_SLOTS,
)


SEQ_LEN = 1 + MAX_PLANETS
_META_ENTITY_MASK = 1 << 4
_META_PLANET_MASK = 1 << 5
_I16_MIN = -32768
_I16_MAX = 32767


class CompressedObservationBuffer(NamedTuple):
    """Per-player compressed observations with leading ``[H, N]`` axes."""

    token_meta: torch.Tensor
    owner_idx: torch.Tensor
    production: torch.Tensor
    ships: torch.Tensor
    velocity: torch.Tensor
    xy: torch.Tensor
    turn_progress: torch.Tensor
    incoming_net: torch.Tensor
    incoming_survivor: torch.Tensor
    origin_frac_blocked: torch.Tensor


def _u16_to_i16(x: torch.Tensor) -> torch.Tensor:
    xi = x.to(torch.int32).clamp(0, 65535)
    return torch.where(xi > 32767, xi - 65536, xi).to(torch.int16)


def _i16_to_u16_float(x: torch.Tensor) -> torch.Tensor:
    xi = x.to(torch.int32)
    return torch.where(xi < 0, xi + 65536, xi).to(torch.float32)


def _i16_to_u16_int(x: torch.Tensor) -> torch.Tensor:
    xi = x.to(torch.int32)
    return torch.where(xi < 0, xi + 65536, xi)


def init_compressed_observation_buffer(
    num_envs: int,
    H_buf: int,
    *,
    device: torch.device,
) -> CompressedObservationBuffer:
    return CompressedObservationBuffer(
        token_meta=torch.zeros((H_buf, num_envs, SEQ_LEN), dtype=torch.int16, device=device),
        owner_idx=torch.zeros((H_buf, num_envs, SEQ_LEN), dtype=torch.int16, device=device),
        production=torch.zeros((H_buf, num_envs, MAX_PLANETS), dtype=torch.int16, device=device),
        ships=torch.zeros((H_buf, num_envs, MAX_PLANETS), dtype=torch.int16, device=device),
        velocity=torch.zeros((H_buf, num_envs, MAX_PLANETS, 2), dtype=torch.float16, device=device),
        xy=torch.zeros((H_buf, num_envs, MAX_PLANETS, 2), dtype=torch.float16, device=device),
        turn_progress=torch.zeros((H_buf, num_envs), dtype=torch.float16, device=device),
        incoming_net=torch.zeros(
            (H_buf, num_envs, MAX_PLANETS, INCOMING_TA_BINS),
            dtype=torch.int16,
            device=device,
        ),
        incoming_survivor=torch.zeros(
            (H_buf, num_envs, MAX_PLANETS, INCOMING_TA_BINS),
            dtype=torch.int16,
            device=device,
        ),
        origin_frac_blocked=torch.zeros(
            (H_buf, num_envs, MAX_PLANETS, NUM_FRACTIONS),
            dtype=torch.bool,
            device=device,
        ),
    )


def _obs_to_compressed_planes(obs: dict[str, torch.Tensor]) -> CompressedObservationBuffer:
    entity_type = obs["entity_type"].to(torch.int32)
    owner_idx = obs["owner_idx"].clamp(_I16_MIN, _I16_MAX).to(torch.int16)
    entity_mask = obs["entity_mask"].to(torch.int32)
    planet_mask = obs["planet_mask"].to(torch.int32)
    token_meta_u16 = (
        entity_type.clamp(0, 15).to(torch.int32)
        | (entity_mask << 4)
        | (planet_mask << 5)
    )

    features = obs["features"].to(torch.float32)
    planet_f = features[:, 1 : 1 + MAX_PLANETS, :]
    production = _u16_to_i16(torch.expm1(planet_f[..., 0]).round())
    ships = _u16_to_i16((planet_f[..., 1] * 1000.0).round())
    velocity = planet_f[..., 2:4].to(torch.float16)
    incoming_net = (planet_f[..., 8 : 8 + INCOMING_TA_BINS] * 1000.0).round()
    incoming_net = incoming_net.clamp(_I16_MIN, _I16_MAX).to(torch.int16)
    incoming_survivor = torch.zeros_like(incoming_net)
    multi_min_f = 8 + INCOMING_TA_BINS + INCOMING_SURVIVOR_FLAT
    if planet_f.shape[-1] >= multi_min_f:
        oh = planet_f[..., 8 + INCOMING_TA_BINS : multi_min_f].reshape(
            planet_f.shape[0], MAX_PLANETS, INCOMING_TA_BINS, NUM_OWNER_SLOTS
        )
        incoming_survivor = oh.argmax(dim=-1).to(torch.int16)
    xy = obs["rope_pos"][:, 1 : 1 + MAX_PLANETS, 0:2].to(torch.float16)
    turn_progress = features[:, 0, 6].to(torch.float16)

    return CompressedObservationBuffer(
        token_meta=_u16_to_i16(token_meta_u16),
        owner_idx=owner_idx,
        production=production,
        ships=ships,
        velocity=velocity,
        xy=xy,
        turn_progress=turn_progress,
        incoming_net=incoming_net,
        incoming_survivor=incoming_survivor,
        origin_frac_blocked=torch.zeros(
            (obs["features"].shape[0], MAX_PLANETS, NUM_FRACTIONS),
            dtype=torch.bool,
            device=obs["features"].device,
        ),
    )


@torch.no_grad()
def compress_observation(obs: dict[str, torch.Tensor]) -> CompressedObservationBuffer:
    return _obs_to_compressed_planes(obs)


def decode_observation(
    comp: CompressedObservationBuffer, *, feature_dim: int = FEATURE_DIM
) -> dict[str, torch.Tensor]:
    token_meta = _i16_to_u16_int(comp.token_meta)
    entity_type = (token_meta & 0xF).to(torch.long)
    entity_mask = (token_meta & _META_ENTITY_MASK) != 0
    planet_mask = (token_meta & _META_PLANET_MASK) != 0
    owner_idx = comp.owner_idx.to(torch.long)

    valid_dims = (FEATURE_DIM, FEATURE_DIM_ABORT, FEATURE_DIM_MULTI, FEATURE_DIM_MULTI_ABORT)
    if feature_dim not in valid_dims:
        raise ValueError(f"feature_dim must be one of {valid_dims}, got {feature_dim}")
    prefix_shape = comp.token_meta.shape[:-1]
    device = comp.token_meta.device
    features = torch.zeros((*prefix_shape, SEQ_LEN, feature_dim), dtype=torch.float32, device=device)
    rope_pos = torch.zeros((*prefix_shape, SEQ_LEN, 3), dtype=torch.float32, device=device)

    prod = _i16_to_u16_float(comp.production)
    prod_for_radius = prod.clamp_min(1.0)
    planet_f = features[..., 1 : 1 + MAX_PLANETS, :]
    planet_f[..., 0] = torch.log1p(prod)
    planet_f[..., 1] = _i16_to_u16_float(comp.ships) / 1000.0
    planet_f[..., 2:4] = comp.velocity.to(torch.float32)
    planet_f[..., 4] = entity_mask[..., 1 : 1 + MAX_PLANETS].to(torch.float32)
    planet_f[..., 5] = (1.0 + torch.log(prod_for_radius)) / 10.0
    planet_f[..., 8 : 8 + INCOMING_TA_BINS] = comp.incoming_net.to(torch.float32) / 1000.0
    is_multi = feature_dim in (FEATURE_DIM_MULTI, FEATURE_DIM_MULTI_ABORT)
    if is_multi:
        idx = comp.incoming_survivor.to(torch.long).clamp(0, NUM_OWNER_SLOTS - 1)
        oh = torch.nn.functional.one_hot(idx, NUM_OWNER_SLOTS).to(dtype=torch.float32, device=device)
        planet_f[
            ...,
            8 + INCOMING_TA_BINS : 8 + INCOMING_TA_BINS + INCOMING_SURVIVOR_FLAT,
        ] = oh.reshape(*prefix_shape, MAX_PLANETS, INCOMING_SURVIVOR_FLAT)
    if feature_dim in (FEATURE_DIM_ABORT, FEATURE_DIM_MULTI_ABORT):
        planet_f[..., -BLOCKED_FRAC_FEATURES:] = comp.origin_frac_blocked.to(torch.float32)
    features[..., 0, 6] = comp.turn_progress.to(torch.float32)

    rope_pos[..., 0, 0] = CENTER / BOARD_SIZE
    rope_pos[..., 0, 1] = CENTER / BOARD_SIZE
    rope_pos[..., 1 : 1 + MAX_PLANETS, 0:2] = comp.xy.to(torch.float32)

    return {
        "entity_type": entity_type,
        "owner_idx": owner_idx,
        "features": features,
        "rope_pos": rope_pos,
        "entity_mask": entity_mask,
        "planet_mask": planet_mask,
    }


@torch.no_grad()
def store_compressed_observation_rows(
    dst: CompressedObservationBuffer,
    row: torch.Tensor,
    env: torch.Tensor,
    obs: dict[str, torch.Tensor],
) -> CompressedObservationBuffer:
    comp = compress_observation(obs)
    r = row.to(device=dst.token_meta.device, dtype=torch.long)
    e = env.to(device=dst.token_meta.device, dtype=torch.long)
    dst.token_meta[r, e, :] = comp.token_meta.to(dst.token_meta.device)
    dst.owner_idx[r, e, :] = comp.owner_idx.to(dst.owner_idx.device)
    dst.production[r, e, :] = comp.production.to(dst.production.device)
    dst.ships[r, e, :] = comp.ships.to(dst.ships.device)
    dst.velocity[r, e, :, :] = comp.velocity.to(dst.velocity.device)
    dst.xy[r, e, :, :] = comp.xy.to(dst.xy.device)
    dst.turn_progress[r, e] = comp.turn_progress.to(dst.turn_progress.device)
    dst.incoming_net[r, e, :, :] = comp.incoming_net.to(dst.incoming_net.device)
    dst.incoming_survivor[r, e, :, :] = comp.incoming_survivor.to(dst.incoming_survivor.device)
    dst.origin_frac_blocked[r, e, :, :] = comp.origin_frac_blocked.to(dst.origin_frac_blocked.device)
    return dst


@torch.no_grad()
def store_precompressed_observation_rows(
    dst: CompressedObservationBuffer,
    row: torch.Tensor,
    env: torch.Tensor,
    comp: CompressedObservationBuffer,
) -> CompressedObservationBuffer:
    r = row.to(device=dst.token_meta.device, dtype=torch.long)
    e = env.to(device=dst.token_meta.device, dtype=torch.long)
    dst.token_meta[r, e, :] = comp.token_meta.to(dst.token_meta.device)
    dst.owner_idx[r, e, :] = comp.owner_idx.to(dst.owner_idx.device)
    dst.production[r, e, :] = comp.production.to(dst.production.device)
    dst.ships[r, e, :] = comp.ships.to(dst.ships.device)
    dst.velocity[r, e, :, :] = comp.velocity.to(dst.velocity.device)
    dst.xy[r, e, :, :] = comp.xy.to(dst.xy.device)
    dst.turn_progress[r, e] = comp.turn_progress.to(dst.turn_progress.device)
    dst.incoming_net[r, e, :, :] = comp.incoming_net.to(dst.incoming_net.device)
    dst.incoming_survivor[r, e, :, :] = comp.incoming_survivor.to(dst.incoming_survivor.device)
    dst.origin_frac_blocked[r, e, :, :] = comp.origin_frac_blocked.to(dst.origin_frac_blocked.device)
    return dst


def index_compressed_observation_rows(
    comp: CompressedObservationBuffer,
    idx: torch.Tensor,
) -> CompressedObservationBuffer:
    ii = idx.to(device=comp.token_meta.device, dtype=torch.long)
    return CompressedObservationBuffer(
        token_meta=comp.token_meta.index_select(0, ii),
        owner_idx=comp.owner_idx.index_select(0, ii),
        production=comp.production.index_select(0, ii),
        ships=comp.ships.index_select(0, ii),
        velocity=comp.velocity.index_select(0, ii),
        xy=comp.xy.index_select(0, ii),
        turn_progress=comp.turn_progress.index_select(0, ii),
        incoming_net=comp.incoming_net.index_select(0, ii),
        incoming_survivor=comp.incoming_survivor.index_select(0, ii),
        origin_frac_blocked=comp.origin_frac_blocked.index_select(0, ii),
    )


def select_compressed_observation(
    src: CompressedObservationBuffer,
    t: torch.Tensor,
    n: torch.Tensor,
    *,
    device: torch.device,
) -> CompressedObservationBuffer:
    tt = t.to(device=src.token_meta.device, dtype=torch.long)
    nn = n.to(device=src.token_meta.device, dtype=torch.long)
    return CompressedObservationBuffer(
        token_meta=src.token_meta[tt, nn].to(device),
        owner_idx=src.owner_idx[tt, nn].to(device),
        production=src.production[tt, nn].to(device),
        ships=src.ships[tt, nn].to(device),
        velocity=src.velocity[tt, nn].to(device),
        xy=src.xy[tt, nn].to(device),
        turn_progress=src.turn_progress[tt, nn].to(device),
        incoming_net=src.incoming_net[tt, nn].to(device),
        incoming_survivor=src.incoming_survivor[tt, nn].to(device),
        origin_frac_blocked=src.origin_frac_blocked[tt, nn].to(device),
    )


def compressed_observation_to_host(obs: CompressedObservationBuffer) -> CompressedObservationBuffer:
    return CompressedObservationBuffer(
        **{field: getattr(obs, field).detach().cpu().contiguous() for field in obs._fields}
    )

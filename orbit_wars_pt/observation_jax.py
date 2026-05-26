"""Vmap'd JAX observation builder + dlpack handoff to PyTorch.

Phase 2 of the JAX-exploitation rework. ``build_observation_batched_jax``
constructs the policy input directly from a batched ``OrbitWarsState`` on
device — no host loop, no per-env padding, no NumPy stacking.

Output layout matches the original ``ObservationBatch`` semantics:

* ``L = 1 (CLS) + MAX_PLANETS = 61``.
* CLS token feature index 6 = turn progress in ``[0,1]`` (``step_count / (episode_steps-2)``).
* ``entity_type[..., 0] = ENTITY_CLS``;
  ``[..., 1:1+MAX_PLANETS]`` are planet/comet tokens (fixed slots).
* Incoming fleets: per-planet signed **post–inter-fleet** mass for each TA bin
  (``features[..., 8 : 8 + INCOMING_TA_BINS]``). With ``FEATURE_DIM_MULTI``, the following
  ``INCOMING_TA_BINS * NUM_OWNER_SLOTS`` dimensions are a **one-hot per TA bin** for egocentric
  survivor owner after the interfleet step (TA-major: bin0 slots 0..4, bin1 slots 0..4, ...).
"""

from __future__ import annotations

from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

import jax_orbit_wars as jow
from jax_orbit_wars import OrbitWarsState

from orbit_wars_pt.constants import (
    BOARD_SIZE,
    CENTER,
    ENTITY_CLS,
    ENTITY_COMET,
    ENTITY_PLANET,
    FEATURE_DIM,
    FEATURE_DIM_MULTI,
    INCOMING_SURVIVOR_FLAT,
    INCOMING_TA_BINS,
    MAX_PLANETS,
    NUM_OWNER_SLOTS,
    ROTATION_RADIUS_LIMIT,
)


# Fixed sequence length: [CLS, planet tokens].
SEQ_LEN = 1 + MAX_PLANETS

_SEAT_QTURNS_TO_P0_2P = jnp.asarray([0, 2], dtype=jnp.int32)
_SEAT_QTURNS_TO_P0_4P = jnp.asarray([0, 3, 1, 2], dtype=jnp.int32)
_NORMALIZED_OWNER_SLOT_4P = jnp.asarray(
    [
        [1, 3, 4, 2],
        [4, 1, 2, 3],
        [3, 2, 1, 4],
        [2, 4, 3, 1],
    ],
    dtype=jnp.int32,
)


def _u16_to_i16_jax(x: jnp.ndarray) -> jnp.ndarray:
    xi = jnp.asarray(x, dtype=jnp.int32)
    xi = jnp.clip(xi, 0, 65535)
    return jnp.where(xi > 32767, xi - 65536, xi).astype(jnp.int16)


# ---- Owner remap (matches host ``observation._remap_owner``). ----


def _obs_qturns_to_p0_jax(ego: int, num_agents: jnp.ndarray, normalize_to_p0: bool) -> jnp.ndarray:
    ego_j = jnp.asarray(ego, dtype=jnp.int32)
    na = jnp.asarray(num_agents, dtype=jnp.int32)
    qturns_2p = jnp.take(_SEAT_QTURNS_TO_P0_2P, jnp.clip(ego_j, 0, 1))
    qturns_4p = jnp.take(_SEAT_QTURNS_TO_P0_4P, jnp.clip(ego_j, 0, 3))
    qturns = jnp.where(na <= 2, qturns_2p, qturns_4p)
    return jnp.where(jnp.asarray(normalize_to_p0, dtype=jnp.bool_), qturns, 0)


def _rotate_vec_jax(xy: jnp.ndarray, qturns: jnp.ndarray) -> jnp.ndarray:
    q = jnp.asarray(qturns, dtype=jnp.int32) & 3
    x = xy[..., 0]
    y = xy[..., 1]
    rx = jnp.where(q == 0, x, jnp.where(q == 1, -y, jnp.where(q == 2, -x, y)))
    ry = jnp.where(q == 0, y, jnp.where(q == 1, x, jnp.where(q == 2, -y, -x)))
    return jnp.stack([rx, ry], axis=-1)


def _rotate_xy_about_center_jax(xy: jnp.ndarray, qturns: jnp.ndarray) -> jnp.ndarray:
    centered = xy - jnp.asarray([CENTER, CENTER], dtype=xy.dtype)
    return _rotate_vec_jax(centered, qturns) + jnp.asarray([CENTER, CENTER], dtype=xy.dtype)


def _opponent_slot_4p_jax(owner: jnp.ndarray, ego: int, normalize_to_p0: bool) -> jnp.ndarray:
    o = owner.astype(jnp.int32)
    ego_j = jnp.asarray(ego, dtype=jnp.int32)
    canonical = jnp.where(o < ego_j, 2 + o, 2 + (o - 1))
    row = jnp.take(_NORMALIZED_OWNER_SLOT_4P, jnp.clip(ego_j, 0, 3), axis=0)
    normalized = jnp.take(row, jnp.clip(o, 0, 3), axis=0)
    return jnp.where(jnp.asarray(normalize_to_p0, dtype=jnp.bool_), normalized, canonical)


def _remap_owner_jax(
    owner: jnp.ndarray, ego: int, num_agents: jnp.ndarray, normalize_to_p0: bool = False
) -> jnp.ndarray:
    """Egocentric owner bucket per planet: 0 neutral, 1 self, 2–4 opponents (4p) or 2 (2p)."""

    o = owner.astype(jnp.int32)
    ego_j = jnp.asarray(ego, dtype=jnp.int32)
    na = jnp.asarray(num_agents, dtype=jnp.int32)
    is_neutral = o < 0
    is_self = o == ego_j
    slot_2p = jnp.full_like(o, 2)
    slot_4p = _opponent_slot_4p_jax(o, ego, normalize_to_p0)
    opponent_slot = jnp.where(na <= 2, slot_2p, slot_4p)
    out = jnp.where(is_neutral, 0, jnp.where(is_self, 1, opponent_slot))
    return jnp.minimum(out, NUM_OWNER_SLOTS - 1).astype(jnp.int32)

# ---- Per-env observation builder (called via vmap over num_envs). ----


def _planet_velocities_one_env(
    initial_planets: jnp.ndarray,
    planets: jnp.ndarray,
    planet_active: jnp.ndarray,
    initial_active: jnp.ndarray,
    angular_velocity: jnp.ndarray,
    step_count: jnp.ndarray,
    is_comet: jnp.ndarray,
    comet_planet_ids: jnp.ndarray,
    comet_paths: jnp.ndarray,
    comet_path_lengths: jnp.ndarray,
    comet_path_index: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns ``(vx[MAX_PLANETS], vy[MAX_PLANETS])`` matching host ``planet_pred_velocity``.

    For comet planets we use the path-based ``paths[g, k, idx+1] - paths[g, k, idx]``
    velocity; for ordinary planets the rotation prediction; for inactive
    planets, zero.
    """

    init_xy = initial_planets[:, 2:4]  # [P, 2]
    cur_xy = planets[:, 2:4]
    radius = planets[:, 4]
    pid = planets[:, 0].astype(jnp.int32)

    # Rotation-based velocity (matches `planet_pred_velocity` for non-comets).
    dx0 = init_xy[:, 0] - CENTER
    dy0 = init_xy[:, 1] - CENTER
    orbital_r = jnp.hypot(dx0, dy0)
    rotates = (orbital_r >= 1e-6) & ((orbital_r + radius) < ROTATION_RADIUS_LIMIT) & initial_active & planet_active
    th0 = jnp.arctan2(dy0, dx0)
    th_next = th0 + angular_velocity * (step_count.astype(jnp.float32) + 1.0)
    nx = CENTER + orbital_r * jnp.cos(th_next)
    ny = CENTER + orbital_r * jnp.sin(th_next)
    rot_vx = jnp.where(rotates, nx - cur_xy[:, 0], 0.0)
    rot_vy = jnp.where(rotates, ny - cur_xy[:, 1], 0.0)

    # Comet velocity from path table. Each comet pid appears in at most one (g, k).
    # comet_paths: [G, K, T, 2]; comet_path_index: [G]; comet_path_lengths: [G, K].
    path_idx_g = comet_path_index  # [G]
    path_len_gk = comet_path_lengths  # [G, K]
    safe_idx_gk = jnp.clip(path_idx_g[:, None], 0, comet_paths.shape[2] - 2)  # [G, K]
    G, K, _, _ = comet_paths.shape
    g_grid = jnp.arange(G)[:, None]  # [G, 1]
    k_grid = jnp.arange(K)[None, :]  # [1, K]
    p0 = comet_paths[g_grid, k_grid, safe_idx_gk]  # [G, K, 2]
    p1 = comet_paths[g_grid, k_grid, safe_idx_gk + 1]
    diff_gk = p1 - p0  # [G, K, 2]
    # Validity: per (g, k), the comet has at least one extra path step.
    gk_valid = (path_len_gk > 1) & (path_idx_g[:, None] >= 0) & (path_idx_g[:, None] < path_len_gk - 1)

    # For each planet slot, find the matching (g, k) and pull its diff (zero otherwise).
    def comet_velocity_for_planet(pid_i: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        match = (comet_planet_ids == pid_i)  # [G, K]
        weight = (match & gk_valid).astype(jnp.float32)
        vx = jnp.sum(diff_gk[..., 0] * weight)
        vy = jnp.sum(diff_gk[..., 1] * weight)
        return vx, vy

    comet_vx, comet_vy = jax.vmap(comet_velocity_for_planet)(pid)

    use_comet = is_comet & planet_active
    vx = jnp.where(use_comet, comet_vx, rot_vx)
    vy = jnp.where(use_comet, comet_vy, rot_vy)
    # Inactive planets have zero velocity.
    vx = jnp.where(planet_active, vx, 0.0)
    vy = jnp.where(planet_active, vy, 0.0)
    return vx, vy


def _turn_fraction_jax(step_count: jnp.ndarray) -> jnp.ndarray:
    """Matches host ``_cls_turn_fraction``: progress toward ``episode_steps - 2`` timeout."""

    ep = jow.OrbitWarsConfig().episode_steps.astype(jnp.float32)
    denom = jnp.maximum(ep - 2.0, 1.0)
    return jnp.clip(step_count.astype(jnp.float32) / denom, 0.0, 1.0)


def _incoming_interfleet_ego_features(
    incoming_apt: jnp.ndarray,
    ego: int,
    num_agents: jnp.ndarray,
    normalize_to_p0: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Match ``jax_orbit_wars._resolve_combats`` interfleet step (largest vs 2nd largest per slot).

    Returns
    -------
    signed_net
        ``[P, T]`` egocentric signed surviving attacker ships / 1000 (positive if self wins duel).
    survivor_slot
        ``[P, T] int32`` egocentric owner bucket after remap (0 = none/tie, 1 = self, 2–4 = opponents).
    """

    a = incoming_apt.shape[0]
    pad = 4 - a
    incoming_f = incoming_apt.astype(jnp.float32)
    padded = jnp.pad(incoming_f, ((0, pad), (0, 0), (0, 0)))
    ships = jnp.transpose(padded, (1, 2, 0))
    order = jnp.argsort(-ships, axis=-1)
    top_s = jnp.take_along_axis(ships, order[..., :1], axis=-1)[..., 0]
    second_s = jnp.take_along_axis(ships, order[..., 1:2], axis=-1)[..., 0]
    top_p = order[..., 0].astype(jnp.int32)
    survivor = jnp.where(top_s == second_s, 0.0, top_s - second_s)
    ego_j = jnp.asarray(ego, dtype=jnp.int32)
    signed = jnp.where(survivor <= 0.0, 0.0, jnp.where(top_p == ego_j, survivor, -survivor))

    na = jnp.asarray(num_agents, dtype=jnp.int32)
    is_self = top_p == ego_j
    slot_le2 = jnp.where(survivor <= 0.0, 0, jnp.where(is_self, 1, 2))
    slot_gt2 = jnp.where(
        survivor <= 0.0,
        0,
        jnp.where(is_self, 1, _opponent_slot_4p_jax(top_p, ego, normalize_to_p0)),
    )
    survivor_slot = jnp.where(na <= 2, slot_le2, slot_gt2)
    survivor_slot = jnp.clip(survivor_slot, 0, NUM_OWNER_SLOTS - 1).astype(jnp.int32)
    return signed / 1000.0, survivor_slot


def _build_observation_one_env(
    state: OrbitWarsState,
    ego: int,
    ship_speed: float,
    obs_feature_dim: int,
    normalize_to_p0: bool,
) -> Dict[str, jnp.ndarray]:
    """Single-env observation (the function that we vmap over the num_envs axis)."""

    planets = state.planets
    planet_active = state.planet_active

    pid = planets[:, 0].astype(jnp.int32)  # [P]
    planet_xy = planets[:, 2:4]
    planet_r = planets[:, 4]
    planet_owner = planets[:, 1]
    planet_ships = planets[:, 5]
    planet_prod = planets[:, 6]

    incoming = state.incoming_fleets.astype(jnp.float32)  # [A, P, T]
    incoming_net, survivor_slot = _incoming_interfleet_ego_features(
        incoming, ego, state.num_agents, normalize_to_p0
    )
    obs_qturns = _obs_qturns_to_p0_jax(ego, state.num_agents, normalize_to_p0)

    # is_comet[i] = True iff pid[i] appears anywhere in comet_planet_ids (which holds
    # only currently spawned-and-alive comet pids; expired entries are -1).
    is_comet_per_planet = jnp.any(state.comet_planet_ids == pid[:, None, None], axis=(1, 2))  # [P]

    vx, vy = _planet_velocities_one_env(
        state.initial_planets,
        planets,
        planet_active,
        state.initial_active,
        state.angular_velocity,
        state.step_count,
        is_comet_per_planet,
        state.comet_planet_ids,
        state.comet_paths,
        state.comet_path_lengths,
        state.comet_path_index,
    )

    # ---- Planet token tensors (slot 1 .. 1+P). ----
    planet_etype = jnp.where(is_comet_per_planet, ENTITY_COMET, ENTITY_PLANET).astype(jnp.int32)
    planet_owner_idx = _remap_owner_jax(planet_owner, ego, state.num_agents, normalize_to_p0)  # [P], int32
    vel_xy = _rotate_vec_jax(jnp.stack([vx, vy], axis=-1), obs_qturns)
    rope_xy = _rotate_xy_about_center_jax(planet_xy, obs_qturns)

    assert obs_feature_dim in (FEATURE_DIM, FEATURE_DIM_MULTI)
    planet_features = jnp.zeros((MAX_PLANETS, obs_feature_dim), dtype=jnp.float32)
    planet_features = planet_features.at[:, 0].set(jnp.log1p(jnp.maximum(planet_prod, 0.0)))
    planet_features = planet_features.at[:, 1].set(planet_ships / 1000.0)
    planet_features = planet_features.at[:, 2].set(vel_xy[:, 0] / 5.0)
    planet_features = planet_features.at[:, 3].set(vel_xy[:, 1] / 5.0)
    planet_features = planet_features.at[:, 4].set(planet_active.astype(jnp.float32))
    planet_features = planet_features.at[:, 5].set(planet_r / 10.0)
    planet_features = planet_features.at[:, 8 : 8 + INCOMING_TA_BINS].set(incoming_net)
    na_i = jnp.asarray(state.num_agents, dtype=jnp.int32)
    if obs_feature_dim == FEATURE_DIM_MULTI:
        oh = jax.nn.one_hot(survivor_slot, NUM_OWNER_SLOTS, axis=-1)
        oh_flat = oh.reshape(MAX_PLANETS, INCOMING_SURVIVOR_FLAT).astype(jnp.float32)
        surv_write = jnp.where(na_i > 2, oh_flat, jnp.zeros_like(oh_flat))
        planet_features = planet_features.at[
            :, 8 + INCOMING_TA_BINS : 8 + INCOMING_TA_BINS + INCOMING_SURVIVOR_FLAT
        ].set(surv_write)

    planet_xy_for_rope = jnp.where(planet_active[:, None], rope_xy, 0.0)
    planet_rope = jnp.zeros((MAX_PLANETS, 3), dtype=jnp.float32)
    planet_rope = planet_rope.at[:, 0].set(planet_xy_for_rope[:, 0] / BOARD_SIZE)
    planet_rope = planet_rope.at[:, 1].set(planet_xy_for_rope[:, 1] / BOARD_SIZE)

    planet_entity_mask = planet_active

    # ---- Assemble [CLS, planets]. ----
    cls_etype = jnp.asarray([ENTITY_CLS], dtype=jnp.int32)
    cls_owner = jnp.asarray([1], dtype=jnp.int32)
    cls_features = jnp.zeros((1, obs_feature_dim), dtype=jnp.float32)
    cls_features = cls_features.at[0, 6].set(_turn_fraction_jax(state.step_count))
    cls_rope = jnp.asarray([[CENTER / BOARD_SIZE, CENTER / BOARD_SIZE, 0.0]], dtype=jnp.float32)
    cls_entity_mask = jnp.asarray([True], dtype=jnp.bool_)
    cls_planet_mask = jnp.asarray([False], dtype=jnp.bool_)

    entity_type = jnp.concatenate([cls_etype, planet_etype], axis=0)
    owner_idx = jnp.concatenate([cls_owner, planet_owner_idx], axis=0)
    features = jnp.concatenate([cls_features, planet_features], axis=0)
    rope_pos = jnp.concatenate([cls_rope, planet_rope], axis=0)
    entity_mask = jnp.concatenate([cls_entity_mask, planet_entity_mask], axis=0)
    planet_mask = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=jnp.bool_),
            jnp.ones((MAX_PLANETS,), dtype=jnp.bool_),
        ],
        axis=0,
    )

    return {
        "entity_type": entity_type,
        "owner_idx": owner_idx,
        "features": features,
        "rope_pos": rope_pos,
        "entity_mask": entity_mask,
        "planet_mask": planet_mask,
    }


def _build_compressed_observation_one_env(
    state: OrbitWarsState,
    ego: int,
    ship_speed: float,
    obs_feature_dim: int,
    normalize_to_p0: bool,
) -> Dict[str, jnp.ndarray]:
    del ship_speed

    planets = state.planets
    planet_active = state.planet_active

    pid = planets[:, 0].astype(jnp.int32)
    planet_xy = planets[:, 2:4]
    planet_owner = planets[:, 1]
    planet_ships = planets[:, 5]
    planet_prod = planets[:, 6]

    incoming = state.incoming_fleets.astype(jnp.float32)
    incoming_net_f, survivor_slot = _incoming_interfleet_ego_features(
        incoming, ego, state.num_agents, normalize_to_p0
    )
    obs_qturns = _obs_qturns_to_p0_jax(ego, state.num_agents, normalize_to_p0)

    is_comet_per_planet = jnp.any(state.comet_planet_ids == pid[:, None, None], axis=(1, 2))
    vx, vy = _planet_velocities_one_env(
        state.initial_planets,
        planets,
        planet_active,
        state.initial_active,
        state.angular_velocity,
        state.step_count,
        is_comet_per_planet,
        state.comet_planet_ids,
        state.comet_paths,
        state.comet_path_lengths,
        state.comet_path_index,
    )

    planet_etype = jnp.where(is_comet_per_planet, ENTITY_COMET, ENTITY_PLANET).astype(jnp.int32)
    planet_owner_idx = _remap_owner_jax(planet_owner, ego, state.num_agents, normalize_to_p0)
    vel_xy = _rotate_vec_jax(jnp.stack([vx, vy], axis=-1), obs_qturns) / 5.0
    rope_xy = _rotate_xy_about_center_jax(planet_xy, obs_qturns)
    rope_xy = jnp.where(planet_active[:, None], rope_xy / BOARD_SIZE, 0.0)

    cls_type = jnp.asarray([ENTITY_CLS], dtype=jnp.int32)
    cls_owner = jnp.asarray([1], dtype=jnp.int32)
    entity_type = jnp.concatenate([cls_type, planet_etype], axis=0)
    owner_idx = jnp.concatenate([cls_owner, planet_owner_idx], axis=0)
    entity_mask = jnp.concatenate(
        [jnp.asarray([True], dtype=jnp.bool_), planet_active],
        axis=0,
    )
    planet_mask = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=jnp.bool_),
            jnp.ones((MAX_PLANETS,), dtype=jnp.bool_),
        ],
        axis=0,
    )
    token_meta_u16 = (
        jnp.clip(entity_type, 0, 15).astype(jnp.int32)
        | (entity_mask.astype(jnp.int32) << 4)
        | (planet_mask.astype(jnp.int32) << 5)
    )

    incoming_net = jnp.clip(jnp.round(incoming_net_f * 1000.0), -32768, 32767).astype(jnp.int16)
    na_i = jnp.asarray(state.num_agents, dtype=jnp.int32)
    survivor_store = jnp.where(
        (na_i > 2) & (jnp.asarray(obs_feature_dim, dtype=jnp.int32) == FEATURE_DIM_MULTI),
        survivor_slot,
        jnp.zeros_like(survivor_slot),
    ).astype(jnp.int16)

    return {
        "token_meta": _u16_to_i16_jax(token_meta_u16),
        "owner_idx": owner_idx.astype(jnp.int16),
        "production": _u16_to_i16_jax(jnp.round(jnp.maximum(planet_prod, 0.0))),
        "ships": _u16_to_i16_jax(jnp.round(planet_ships)),
        "velocity": vel_xy.astype(jnp.float16),
        "xy": rope_xy.astype(jnp.float16),
        "turn_progress": _turn_fraction_jax(state.step_count).astype(jnp.float16),
        "incoming_net": incoming_net,
        "incoming_survivor": survivor_store,
    }


@partial(jax.jit, static_argnames=("ego", "ship_speed", "obs_feature_dim", "normalize_to_p0"))
def build_observation_batched_jax(
    state_b: OrbitWarsState,
    ego: int,
    ship_speed: float = 6.0,
    obs_feature_dim: int = FEATURE_DIM,
    normalize_to_p0: bool = False,
) -> Dict[str, jnp.ndarray]:
    """Batched observation: ``state_b`` has leading ``num_envs`` axis on every leaf.

    Returns a dict of JAX arrays:
      ``entity_type[N, L] int32``,  ``owner_idx[N, L] int32``,
      ``features[N, L, obs_feature_dim] float32``, ``rope_pos[N, L, 3] float32``,
      ``entity_mask[N, L] bool``,   ``planet_mask[N, L] bool``.

    The two integer fields are cast to ``torch.long`` inside
    ``obs_jax_to_torch`` (``nn.Embedding`` requires long indices).
    """

    return jax.vmap(
        lambda s: _build_observation_one_env(s, ego, ship_speed, obs_feature_dim, normalize_to_p0)
    )(state_b)


@partial(jax.jit, static_argnames=("ship_speed", "obs_feature_dim", "normalize_to_p0"))
def build_observation_batched_jax_per_ego(
    state_b: OrbitWarsState,
    ego_b: jnp.ndarray,
    ship_speed: float = 6.0,
    obs_feature_dim: int = FEATURE_DIM,
    normalize_to_p0: bool = False,
) -> Dict[str, jnp.ndarray]:
    """Per-element ego variant used at PPO replay.

    ``ego_b`` is ``[N] int32`` — one ego per stacked transition. Self-play
    minibatches mix ``ego=0`` and ``ego=1`` samples, so we vmap over both the
    state and the ego.
    """

    return jax.vmap(_build_observation_one_env, in_axes=(0, 0, None, None, None))(
        state_b, ego_b, ship_speed, obs_feature_dim, normalize_to_p0
    )


@partial(jax.jit, static_argnames=("ego", "ship_speed", "obs_feature_dim", "normalize_to_p0"))
def build_compressed_observation_batched_jax(
    state_b: OrbitWarsState,
    ego: int,
    ship_speed: float = 6.0,
    obs_feature_dim: int = FEATURE_DIM,
    normalize_to_p0: bool = False,
) -> Dict[str, jnp.ndarray]:
    return jax.vmap(
        lambda s: _build_compressed_observation_one_env(
            s, ego, ship_speed, obs_feature_dim, normalize_to_p0
        )
    )(state_b)


@partial(jax.jit, static_argnames=("ship_speed", "obs_feature_dim", "normalize_to_p0"))
def build_compressed_observation_batched_jax_per_ego(
    state_b: OrbitWarsState,
    ego_b: jnp.ndarray,
    ship_speed: float = 6.0,
    obs_feature_dim: int = FEATURE_DIM,
    normalize_to_p0: bool = False,
) -> Dict[str, jnp.ndarray]:
    return jax.vmap(_build_compressed_observation_one_env, in_axes=(0, 0, None, None, None))(
        state_b, ego_b, ship_speed, obs_feature_dim, normalize_to_p0
    )

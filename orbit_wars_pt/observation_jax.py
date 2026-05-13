"""Vmap'd JAX observation builder + dlpack handoff to PyTorch.

Phase 2 of the JAX-exploitation rework. ``build_observation_batched_jax``
constructs the policy input directly from a batched ``OrbitWarsState`` on
device — no host loop, no per-env padding, no NumPy stacking.

Output layout matches the original ``ObservationBatch`` semantics:

* ``L = 1 (CLS) + MAX_PLANETS = 61``.
* CLS token feature index 6 = turn progress in ``[0,1]`` (``step_count / (episode_steps-2)``).
* ``entity_type[..., 0] = ENTITY_CLS``;
  ``[..., 1:1+MAX_PLANETS]`` are planet/comet tokens (fixed slots).
* Incoming fleets are collapsed into per-planet signed countdown bins appended
  to the planet feature vector.
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
    INCOMING_TA_BINS,
    MAX_PLANETS,
    NUM_OWNER_SLOTS,
    ROTATION_RADIUS_LIMIT,
)


# Fixed sequence length: [CLS, planet tokens].
SEQ_LEN = 1 + MAX_PLANETS


# ---- Owner remap (2-agent specialization; matches host `_remap_owner` for num_agents <= 2). ----


def _remap_owner_2p(owner: jnp.ndarray, ego: int) -> jnp.ndarray:
    """0 = neutral, 1 = self, 2 = enemy. Result clipped to ``NUM_OWNER_SLOTS - 1``.

    Returns ``int32`` (JAX's default integer width); the dlpack handoff casts to
    ``torch.long`` for the embedding lookups.
    """

    o = owner.astype(jnp.int32)
    is_neutral = o < 0
    is_self = o == ego
    out = jnp.where(is_neutral, 0, jnp.where(is_self, 1, 2))
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


def _build_observation_one_env(state: OrbitWarsState, ego: int, ship_speed: float) -> Dict[str, jnp.ndarray]:
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
    self_incoming = incoming[jnp.asarray(ego, dtype=jnp.int32)]
    other_mask = jnp.arange(state.incoming_fleets.shape[0], dtype=jnp.int32) != jnp.asarray(ego, dtype=jnp.int32)
    enemy_incoming = jnp.sum(jnp.where(other_mask[:, None, None], incoming, 0.0), axis=0)
    incoming_net = (self_incoming - enemy_incoming) / 1000.0

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
    planet_owner_idx = _remap_owner_2p(planet_owner, ego)  # [P], int32

    planet_features = jnp.zeros((MAX_PLANETS, FEATURE_DIM), dtype=jnp.float32)
    planet_features = planet_features.at[:, 0].set(jnp.log1p(jnp.maximum(planet_prod, 0.0)))
    planet_features = planet_features.at[:, 1].set(planet_ships / 1000.0)
    planet_features = planet_features.at[:, 2].set(vx / 5.0)
    planet_features = planet_features.at[:, 3].set(vy / 5.0)
    planet_features = planet_features.at[:, 4].set(planet_active.astype(jnp.float32))
    planet_features = planet_features.at[:, 5].set(planet_r / 10.0)
    planet_features = planet_features.at[:, 8:].set(incoming_net)

    planet_xy_for_rope = jnp.where(planet_active[:, None], planet_xy, 0.0)
    planet_rope = jnp.zeros((MAX_PLANETS, 3), dtype=jnp.float32)
    planet_rope = planet_rope.at[:, 0].set(planet_xy_for_rope[:, 0] / BOARD_SIZE)
    planet_rope = planet_rope.at[:, 1].set(planet_xy_for_rope[:, 1] / BOARD_SIZE)

    planet_entity_mask = planet_active

    # ---- Assemble [CLS, planets]. ----
    cls_etype = jnp.asarray([ENTITY_CLS], dtype=jnp.int32)
    cls_owner = jnp.asarray([1], dtype=jnp.int32)
    cls_features = jnp.zeros((1, FEATURE_DIM), dtype=jnp.float32)
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


@partial(jax.jit, static_argnames=("ego", "ship_speed"))
def build_observation_batched_jax(state_b: OrbitWarsState, ego: int, ship_speed: float = 6.0) -> Dict[str, jnp.ndarray]:
    """Batched observation: ``state_b`` has leading ``num_envs`` axis on every leaf.

    Returns a dict of JAX arrays:
      ``entity_type[N, L] int32``,  ``owner_idx[N, L] int32``,
      ``features[N, L, FEATURE_DIM] float32``, ``rope_pos[N, L, 3] float32``,
      ``entity_mask[N, L] bool``,   ``planet_mask[N, L] bool``.

    The two integer fields are cast to ``torch.long`` inside
    ``obs_jax_to_torch`` (``nn.Embedding`` requires long indices).
    """

    return jax.vmap(lambda s: _build_observation_one_env(s, ego, ship_speed))(state_b)


@partial(jax.jit, static_argnames=("ship_speed",))
def build_observation_batched_jax_per_ego(
    state_b: OrbitWarsState, ego_b: jnp.ndarray, ship_speed: float = 6.0
) -> Dict[str, jnp.ndarray]:
    """Per-element ego variant used at PPO replay.

    ``ego_b`` is ``[N] int32`` — one ego per stacked transition. Self-play
    minibatches mix ``ego=0`` and ``ego=1`` samples, so we vmap over both the
    state and the ego.
    """

    return jax.vmap(_build_observation_one_env, in_axes=(0, 0, None))(state_b, ego_b, ship_speed)

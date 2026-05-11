"""Vmap'd JAX observation builder + dlpack handoff to PyTorch.

Phase 2 of the JAX-exploitation rework. ``build_observation_batched_jax``
constructs the policy input directly from a batched ``OrbitWarsState`` on
device — no host loop, no per-env padding, no NumPy stacking.

Output layout matches the original ``ObservationBatch`` semantics:

* ``L = 1 (CLS) + MAX_PLANETS + MAX_FLEET_TOKENS = 1 + 60 + 128 = 189``.
* CLS token feature index 6 = turn progress in ``[0,1]`` (``step_count / (episode_steps-2)``).
* ``entity_type[..., 0] = ENTITY_CLS``;
  ``[..., 1:1+MAX_PLANETS]`` are planet/comet tokens (fixed slots);
  ``[..., 1+MAX_PLANETS:]`` are compacted active-fleet tokens.

Approximations vs the host builder (numerically tiny; behaviorally equivalent):

* ``estimate_eta_to_planet`` is replaced by a closed-form
  ``(distance_from_launch_point - dest_radius) / fleet_speed(ships)``. The
  original integrated 1-step Euler with the same speed and the same straight
  path; the closed-form differs by at most one step.
* Fleet tokens are emitted in slot order for active fleets that have *both*
  an existing origin planet and a finite first-planet hit — same filter as
  the host builder.
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
    ENTITY_FLEET,
    ENTITY_PLANET,
    MAX_FLEET_TOKENS,
    MAX_PLANETS,
    NUM_OWNER_SLOTS,
    ROTATION_RADIUS_LIMIT,
)


# Fixed sequence length: [CLS, planet tokens, fleet tokens].
SEQ_LEN = 1 + MAX_PLANETS + MAX_FLEET_TOKENS


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


# ---- Geometry helpers (JAX). ----


def _fleet_speed(ships: jnp.ndarray, max_speed: float) -> jnp.ndarray:
    """Matches host ``fleet_speed``. ``ships <= 1`` clamps to 1.0."""

    safe = jnp.maximum(ships, 1.0)
    log_s = jnp.log(safe)
    log_1000 = jnp.log(1000.0)
    factor = (log_s / log_1000) ** 1.5
    speed = jnp.minimum(1.0 + (max_speed - 1.0) * factor, max_speed)
    return jnp.where(ships <= 1.0, 1.0, speed)


def _launch_point(ox: jnp.ndarray, oy: jnp.ndarray, radius: jnp.ndarray, dx: jnp.ndarray, dy: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Slightly offset the origin along the direction toward the destination."""

    vx = dx - ox
    vy = dy - oy
    d = jnp.hypot(vx, vy)
    safe_d = jnp.maximum(d, 1e-6)
    ux = vx / safe_d
    uy = vy / safe_d
    sx = ox + ux * (radius + 0.1)
    sy = oy + uy * (radius + 0.1)
    # Degenerate fallback if origin and destination coincide.
    sx = jnp.where(d < 1e-6, ox + radius + 0.1, sx)
    sy = jnp.where(d < 1e-6, oy, sy)
    return sx, sy


def _ray_planet_t(
    fx: jnp.ndarray,
    fy: jnp.ndarray,
    cos_a: jnp.ndarray,
    sin_a: jnp.ndarray,
    cx: jnp.ndarray,
    cy: jnp.ndarray,
    r: jnp.ndarray,
) -> jnp.ndarray:
    """Smallest non-negative ray-circle intersection ``t`` (or ``inf``).

    Mirrors host ``ray_circle_intersections``: roots of
    ``|(fx,fy) + t*(cos_a, sin_a) - (cx, cy)|^2 = r^2``.
    """

    fxc = fx - cx
    fyc = fy - cy
    b = 2.0 * (fxc * cos_a + fyc * sin_a)
    c = fxc * fxc + fyc * fyc - r * r
    disc = b * b - 4.0 * c
    sd = jnp.sqrt(jnp.maximum(disc, 0.0))
    t0 = (-b - sd) / 2.0
    t1 = (-b + sd) / 2.0
    valid_disc = disc >= 0.0
    t_min = jnp.where(
        valid_disc & (t0 >= 0.0),
        t0,
        jnp.where(valid_disc & (t1 >= 0.0), t1, jnp.inf),
    )
    return t_min


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
    fleets = state.fleets
    fleet_active = state.fleet_active

    pid = planets[:, 0].astype(jnp.int32)  # [P]
    planet_xy = planets[:, 2:4]
    planet_r = planets[:, 4]
    planet_owner = planets[:, 1]
    planet_ships = planets[:, 5]
    planet_prod = planets[:, 6]

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

    planet_features = jnp.zeros((MAX_PLANETS, 8), dtype=jnp.float32)
    planet_features = planet_features.at[:, 0].set(jnp.log1p(jnp.maximum(planet_prod, 0.0)))
    planet_features = planet_features.at[:, 1].set(planet_ships / 1000.0)
    planet_features = planet_features.at[:, 2].set(vx / 5.0)
    planet_features = planet_features.at[:, 3].set(vy / 5.0)
    planet_features = planet_features.at[:, 4].set(planet_active.astype(jnp.float32))
    planet_features = planet_features.at[:, 5].set(planet_r / 10.0)

    planet_xy_for_rope = jnp.where(planet_active[:, None], planet_xy, 0.0)
    planet_rope = jnp.zeros((MAX_PLANETS, 3), dtype=jnp.float32)
    planet_rope = planet_rope.at[:, 0].set(planet_xy_for_rope[:, 0] / BOARD_SIZE)
    planet_rope = planet_rope.at[:, 1].set(planet_xy_for_rope[:, 1] / BOARD_SIZE)

    planet_entity_mask = planet_active

    # ---- Fleet tokens. ----
    F = fleets.shape[0]
    f_owner = fleets[:, 1]
    f_x = fleets[:, 2]
    f_y = fleets[:, 3]
    f_ang = fleets[:, 4]
    f_origin_pid = fleets[:, 5].astype(jnp.int32)
    f_ships = fleets[:, 6]

    cos_a = jnp.cos(f_ang)
    sin_a = jnp.sin(f_ang)

    # Origin slot per fleet: argmax of (planet_id == origin_pid). 0 if no match;
    # mask via has_origin separately.
    origin_match = pid[None, :] == f_origin_pid[:, None]  # [F, P]
    has_origin = jnp.any(origin_match, axis=1)  # [F]
    origin_slot = jnp.argmax(origin_match.astype(jnp.int32), axis=1)  # [F]

    # Per-(fleet, planet) ray hit time, masking out the origin and inactive planets.
    # Shapes: fleets are F, planets are P; broadcast.
    t_each = _ray_planet_t(
        f_x[:, None],
        f_y[:, None],
        cos_a[:, None],
        sin_a[:, None],
        planet_xy[None, :, 0],
        planet_xy[None, :, 1],
        planet_r[None, :],
    )  # [F, P]

    planet_idx_grid = jnp.arange(MAX_PLANETS)[None, :]  # [1, P]
    is_origin = planet_idx_grid == origin_slot[:, None]  # [F, P]
    planet_mask_2d = planet_active[None, :] & ~is_origin  # [F, P]
    t_eff = jnp.where(planet_mask_2d, t_each, jnp.inf)
    hit_pi = jnp.argmin(t_eff, axis=1)  # [F]
    hit_t = jnp.take_along_axis(t_eff, hit_pi[:, None], axis=1).squeeze(-1)  # [F]
    has_hit = jnp.isfinite(hit_t)

    valid_fleet = fleet_active & has_origin & has_hit  # [F]

    # Compact valid fleet slots into the first MAX_FLEET_TOKENS positions, slot-order preserving.
    F_idx = jnp.arange(F, dtype=jnp.int32)
    sort_key = jnp.where(valid_fleet, F_idx, F + F_idx)
    order = jnp.argsort(sort_key)  # [F]

    # When the underlying fleet buffer ``F`` is smaller than ``MAX_FLEET_TOKENS``,
    # slicing produces a short array; pad to a static ``[MAX_FLEET_TOKENS]`` so
    # downstream broadcasts have a stable shape. Padded positions point at slot 0
    # (arbitrary; masked off via ``fleet_token_active``).
    if F >= MAX_FLEET_TOKENS:
        take_idx = order[:MAX_FLEET_TOKENS]
    else:
        pad_amount = MAX_FLEET_TOKENS - F
        take_idx = jnp.concatenate(
            [order, jnp.zeros((pad_amount,), dtype=order.dtype)],
            axis=0,
        )

    num_valid = jnp.sum(valid_fleet.astype(jnp.int32))
    # Compacted slots beyond the underlying buffer length can never be valid.
    cap = jnp.minimum(num_valid, MAX_FLEET_TOKENS)
    fleet_token_active = jnp.arange(MAX_FLEET_TOKENS, dtype=jnp.int32) < cap  # [MAX_FLEET_TOKENS]

    # Gather fleet attributes at compact indices.
    g_owner = f_owner[take_idx]
    g_ships = f_ships[take_idx]
    g_hit_pi = hit_pi[take_idx]

    # Destination XY = planet center of the hit planet.
    dst_xy = planet_xy[g_hit_pi]  # [MAX_FLEET_TOKENS, 2]
    dst_r = planet_r[g_hit_pi]
    g_fx = f_x[take_idx]
    g_fy = f_y[take_idx]

    # Closed-form ETA approximation matching the host straight-line stepping
    # from the launch point at radius=0.25 toward the dest planet rim.
    sx, sy = _launch_point(g_fx, g_fy, jnp.float32(0.25), dst_xy[:, 0], dst_xy[:, 1])
    dist = jnp.hypot(dst_xy[:, 0] - sx, dst_xy[:, 1] - sy) - dst_r - 0.05
    sp = _fleet_speed(g_ships, ship_speed)
    eta = jnp.clip(dist / jnp.maximum(sp, 1e-6), 0.0, 500.0)

    fleet_etype = jnp.where(fleet_token_active, ENTITY_FLEET, 0).astype(jnp.int32)
    fleet_owner_idx = _remap_owner_2p(g_owner, ego)  # uses 2p remap; matches host
    fleet_owner_idx = jnp.where(fleet_token_active, fleet_owner_idx, jnp.int32(0))

    fleet_features = jnp.zeros((MAX_FLEET_TOKENS, 8), dtype=jnp.float32)
    fleet_features = fleet_features.at[:, 0].set(g_ships / 1000.0)
    fleet_features = fleet_features.at[:, 1].set(g_hit_pi.astype(jnp.float32) / float(MAX_PLANETS))

    fleet_rope = jnp.zeros((MAX_FLEET_TOKENS, 3), dtype=jnp.float32)
    fleet_rope = fleet_rope.at[:, 0].set(dst_xy[:, 0] / BOARD_SIZE)
    fleet_rope = fleet_rope.at[:, 1].set(dst_xy[:, 1] / BOARD_SIZE)
    fleet_rope = fleet_rope.at[:, 2].set(eta / 500.0)

    # Mask features/rope for inactive token slots.
    fleet_features = jnp.where(fleet_token_active[:, None], fleet_features, 0.0)
    fleet_rope = jnp.where(fleet_token_active[:, None], fleet_rope, 0.0)

    # ---- Assemble [CLS, planets, fleets]. ----
    cls_etype = jnp.asarray([ENTITY_CLS], dtype=jnp.int32)
    cls_owner = jnp.asarray([1], dtype=jnp.int32)
    cls_features = jnp.zeros((1, 8), dtype=jnp.float32)
    cls_features = cls_features.at[0, 6].set(_turn_fraction_jax(state.step_count))
    cls_rope = jnp.asarray([[CENTER / BOARD_SIZE, CENTER / BOARD_SIZE, 0.0]], dtype=jnp.float32)
    cls_entity_mask = jnp.asarray([True], dtype=jnp.bool_)
    cls_planet_mask = jnp.asarray([False], dtype=jnp.bool_)

    entity_type = jnp.concatenate([cls_etype, planet_etype, fleet_etype], axis=0)
    owner_idx = jnp.concatenate([cls_owner, planet_owner_idx, fleet_owner_idx], axis=0)
    features = jnp.concatenate([cls_features, planet_features, fleet_features], axis=0)
    rope_pos = jnp.concatenate([cls_rope, planet_rope, fleet_rope], axis=0)
    entity_mask = jnp.concatenate([cls_entity_mask, planet_entity_mask, fleet_token_active], axis=0)
    planet_mask = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=jnp.bool_),
            jnp.ones((MAX_PLANETS,), dtype=jnp.bool_),
            jnp.zeros((MAX_FLEET_TOKENS,), dtype=jnp.bool_),
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
      ``features[N, L, 8] float32``, ``rope_pos[N, L, 3] float32``,
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

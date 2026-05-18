"""JITted ship-mass ratios on JAX device — avoids copying full state to host for reward deltas."""

from __future__ import annotations

from typing import Any, Tuple

import jax
import jax.numpy as jnp


@jax.jit
def _per_player_ship_masses_core(
    planets: jnp.ndarray,
    planet_active: jnp.ndarray,
    incoming_fleets: jnp.ndarray,
) -> jnp.ndarray:
    """Per-player ship mass on planets plus inbound fleets, shape ``(4,)`` float32."""

    owners_p = planets[:, 1].astype(jnp.int32)
    ships_p = planets[:, 5]
    mask_p = planet_active.astype(jnp.bool_) & (owners_p >= 0)
    oc = jnp.clip(owners_p, 0, 3)
    oh = jax.nn.one_hot(oc, 4)
    scores_p = jnp.sum(oh * mask_p[:, None] * ships_p[:, None], axis=0)

    scores_f_a = jnp.sum(incoming_fleets.astype(jnp.float32), axis=(1, 2))
    scores_f = jnp.pad(scores_f_a, (0, 4 - incoming_fleets.shape[0]))

    return scores_p + scores_f


@jax.jit
def _ship_ratio_scores_core(
    planets: jnp.ndarray,
    planet_active: jnp.ndarray,
    incoming_fleets: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns (r0, r1, total_denom) with total_denom = 1e-6 + sum(all players' ships)."""

    scores = _per_player_ship_masses_core(planets, planet_active, incoming_fleets)
    total = jnp.sum(scores) + jnp.asarray(1e-6, dtype=scores.dtype)
    r0 = scores[0] / total
    r1 = scores[1] / total
    return r0, r1, total


@jax.jit
def _ship_mass_ratios_four_core(
    planets: jnp.ndarray,
    planet_active: jnp.ndarray,
    incoming_fleets: jnp.ndarray,
) -> jnp.ndarray:
    """Mass shares ``own_i / sum_j own_j`` for ``i`` in ``{0..3}``, shape ``(4,)``."""

    scores = _per_player_ship_masses_core(planets, planet_active, incoming_fleets)
    total = jnp.sum(scores) + jnp.asarray(1e-6, dtype=scores.dtype)
    return scores / total


@jax.jit
def _player_alive_four_core(
    planets: jnp.ndarray,
    planet_active: jnp.ndarray,
    incoming_fleets: jnp.ndarray,
) -> jnp.ndarray:
    """Whether each player still owns any planet or has inbound fleets, shape ``(4,)``.

    This matches the official interpreter's elimination logic: a player with
    one or more owned planets is alive even if those planets currently have
    zero ships.
    """

    owners_p = planets[:, 1].astype(jnp.int32)
    safe_owners = jnp.clip(owners_p, 0, 3)
    owned_planet = planet_active.astype(jnp.bool_) & (owners_p >= 0)
    alive_from_planets = (
        jnp.zeros((4,), dtype=jnp.int32).at[safe_owners].max(owned_planet.astype(jnp.int32))
        > 0
    )
    alive_from_fleets_a = jnp.sum(incoming_fleets, axis=(1, 2)) > 0
    alive_from_fleets = jnp.pad(alive_from_fleets_a, (0, 4 - incoming_fleets.shape[0]))
    return alive_from_planets | alive_from_fleets


@jax.jit
def _ship_totals_p01_core(
    planets: jnp.ndarray,
    planet_active: jnp.ndarray,
    incoming_fleets: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Total ships (planets + fleets) owned by players 0 and 1 — matches ``_score_and_done`` mass."""

    scores = _per_player_ship_masses_core(planets, planet_active, incoming_fleets)
    return scores[0], scores[1]


def ship_ratio_scores_host(state: Any) -> Tuple[float, float, float]:
    """Ratios from a JAX `OrbitWarsState`; syncs only three scalars to the host."""

    r0, r1, tot = _ship_ratio_scores_core(
        state.planets,
        state.planet_active,
        state.incoming_fleets,
    )
    return float(jax.device_get(r0)), float(jax.device_get(r1)), float(jax.device_get(tot))

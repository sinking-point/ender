"""JITted ship-mass ratios on JAX device — avoids copying full state to host for reward deltas."""

from __future__ import annotations

from typing import Any, Tuple

import jax
import jax.numpy as jnp


@jax.jit
def _ship_ratio_scores_core(
    planets: jnp.ndarray,
    planet_active: jnp.ndarray,
    fleets: jnp.ndarray,
    fleet_active: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns (r0, r1, total_denom) with total_denom = 1e-6 + sum(ships)."""

    owners_p = planets[:, 1].astype(jnp.int32)
    ships_p = planets[:, 5]
    mask_p = planet_active.astype(jnp.bool_) & (owners_p >= 0)
    oc = jnp.clip(owners_p, 0, 3)
    oh = jax.nn.one_hot(oc, 4)
    scores_p = jnp.sum(oh * mask_p[:, None] * ships_p[:, None], axis=0)

    owners_f = fleets[:, 1].astype(jnp.int32)
    ships_f = fleets[:, 6]
    mask_f = fleet_active.astype(jnp.bool_)
    oc_f = jnp.clip(owners_f, 0, 3)
    oh_f = jax.nn.one_hot(oc_f, 4)
    scores_f = jnp.sum(oh_f * mask_f[:, None] * ships_f[:, None], axis=0)

    scores = scores_p + scores_f
    total = jnp.sum(scores) + jnp.asarray(1e-6, dtype=scores.dtype)
    r0 = scores[0] / total
    r1 = scores[1] / total
    return r0, r1, total


@jax.jit
def _ship_totals_p01_core(
    planets: jnp.ndarray,
    planet_active: jnp.ndarray,
    fleets: jnp.ndarray,
    fleet_active: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Total ships (planets + fleets) owned by players 0 and 1 — matches ``_score_and_done`` mass."""

    owners_p = planets[:, 1].astype(jnp.int32)
    ships_p = planets[:, 5]
    mask_p = planet_active.astype(jnp.bool_) & (owners_p >= 0)
    oc = jnp.clip(owners_p, 0, 3)
    oh = jax.nn.one_hot(oc, 4)
    scores_p = jnp.sum(oh * mask_p[:, None] * ships_p[:, None], axis=0)

    owners_f = fleets[:, 1].astype(jnp.int32)
    ships_f = fleets[:, 6]
    mask_f = fleet_active.astype(jnp.bool_)
    oc_f = jnp.clip(owners_f, 0, 3)
    oh_f = jax.nn.one_hot(oc_f, 4)
    scores_f = jnp.sum(oh_f * mask_f[:, None] * ships_f[:, None], axis=0)

    scores = scores_p + scores_f
    return scores[0], scores[1]


def ship_ratio_scores_host(state: Any) -> Tuple[float, float, float]:
    """Ratios from a JAX `OrbitWarsState`; syncs only three scalars to the host."""

    r0, r1, tot = _ship_ratio_scores_core(
        state.planets,
        state.planet_active,
        state.fleets,
        state.fleet_active,
    )
    return float(jax.device_get(r0)), float(jax.device_get(r1)), float(jax.device_get(tot))

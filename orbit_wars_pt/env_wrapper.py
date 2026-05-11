"""JAX environment stepping — detect fleet-table overflow so callers can raise `max_fleets`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_orbit_wars import OrbitWarsConfig, OrbitWarsState, jit_step, reset_from_reference


@dataclass
class OrbitWarsEnvConfig:
    num_agents: int = 2
    max_fleets: int = 512
    episode_seed: int = 0


def reset_env(cfg: OrbitWarsEnvConfig) -> OrbitWarsState:
    return reset_from_reference(cfg.episode_seed, cfg.num_agents, max_fleets=cfg.max_fleets)


def step_env(state: OrbitWarsState, actions_np: np.ndarray, cfg: OrbitWarsEnvConfig) -> Tuple[OrbitWarsState, bool]:
    """Returns `(next_state, overflow)` where `overflow` is True if launch slots were exhausted."""

    del cfg
    actions_jax = jnp.asarray(actions_np, dtype=jnp.float32)
    next_state = jit_step(state, actions_jax, OrbitWarsConfig())
    overflow = bool(np.asarray(jax.device_get(next_state.overflow)))
    return next_state, overflow

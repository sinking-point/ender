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
    reward_mode: str = "ship-mass-share"
    reward_ship_mass_share_coef: float = 1.0
    reward_ship_mass_share_member_coefs: list[float] | None = None
    reward_production_share_coef: float = 0.0
    reward_production_share_member_coefs: list[float] | None = None
    reward_terminal_win_loss_coef: float = 0.0
    reward_terminal_win_loss_member_coefs: list[float] | None = None
    reward_terminal_loss: float = -1.0
    reward_terminal_draw: float = 0.0
    reward_terminal_win: float = 1.0
    reward_time_bonus_coef: float = 0.0
    reward_time_bonus_member_coefs: list[float] | None = None
    normalize_obs_to_p0: bool = False


def reset_env(cfg: OrbitWarsEnvConfig) -> OrbitWarsState:
    return reset_from_reference(cfg.episode_seed, cfg.num_agents, max_fleets=cfg.max_fleets)


def step_env(state: OrbitWarsState, actions_np: np.ndarray, cfg: OrbitWarsEnvConfig) -> Tuple[OrbitWarsState, bool]:
    """Returns `(next_state, overflow)` where `overflow` is True if launch slots were exhausted."""

    actions_jax = jnp.asarray(actions_np, dtype=jnp.float32)
    next_state = jit_step(
        state,
        actions_jax,
        OrbitWarsConfig(
            reward_terminal_loss=jnp.asarray(cfg.reward_terminal_loss, dtype=jnp.float32),
            reward_terminal_draw=jnp.asarray(cfg.reward_terminal_draw, dtype=jnp.float32),
            reward_terminal_win=jnp.asarray(cfg.reward_terminal_win, dtype=jnp.float32),
        ),
    )
    overflow = bool(np.asarray(jax.device_get(next_state.overflow)))
    return next_state, overflow

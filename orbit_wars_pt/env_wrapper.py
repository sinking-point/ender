"""JAX environment stepping — detect fleet-table overflow so callers can raise `max_fleets`."""

from __future__ import annotations

from dataclasses import dataclass
from jax_orbit_wars import OrbitWarsState


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

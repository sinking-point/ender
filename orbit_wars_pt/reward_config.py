"""Shared reward coefficient resolution for training and inference."""

from __future__ import annotations

from typing import Any, Mapping


def _cfg_get(config: Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def resolve_reward_mix(config: Mapping[str, Any] | Any) -> tuple[float, float, float, float]:
    """Resolve reward coefficients, preserving legacy ``reward_mode`` presets."""

    reward_mode = _cfg_get(config, "reward_mode", "ship-mass-share")
    if reward_mode == "ship-mass-share":
        ship_coef, prod_coef, terminal_coef = 1.0, 0.0, 0.0
    elif reward_mode == "production-share":
        ship_coef, prod_coef, terminal_coef = 0.0, 1.0, 0.0
    elif reward_mode == "terminal-win-loss":
        ship_coef, prod_coef, terminal_coef = 0.0, 0.0, 1.0
    else:
        raise ValueError(f"unknown reward mode {reward_mode!r}")
    time_coef = 0.0
    ship_raw = _cfg_get(config, "reward_ship_mass_share_coef")
    prod_raw = _cfg_get(config, "reward_production_share_coef")
    terminal_raw = _cfg_get(config, "reward_terminal_win_loss_coef")
    time_raw = _cfg_get(config, "reward_time_bonus_coef")
    if ship_raw is not None:
        ship_coef = float(ship_raw)
    if prod_raw is not None:
        prod_coef = float(prod_raw)
    if terminal_raw is not None:
        terminal_coef = float(terminal_raw)
    if time_raw is not None:
        time_coef = float(time_raw)
    return float(ship_coef), float(prod_coef), float(terminal_coef), float(time_coef)

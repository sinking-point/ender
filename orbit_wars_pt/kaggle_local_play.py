"""Defaults for running Orbit Wars locally (human or long think-time).

Kaggle's ``Agent.act`` treats ``configuration.actTimeout`` as the baseline
seconds per call; time beyond that consumes ``observation.remainingOverageTime``
and can return ``DeadlineExceeded``. ``env.run`` also enforces
``configuration.runTimeout`` wall-clock for the whole episode loop.
"""

from __future__ import annotations

from typing import Any

# Per-agent ``act()`` call: think time that does not consume overage bank.
DEFAULT_LOCAL_ACT_TIMEOUT_S = 86400.0  # 24 hours

# Whole ``env.run`` loop wall clock (official episodes are up to 500 steps).
DEFAULT_LOCAL_RUN_TIMEOUT_S = 86400.0 * 14.0  # 14 days

# Floor for banked overage after ``reset`` (observation default is often 60s).
DEFAULT_LOCAL_MIN_OVERAGE_S = 86400.0 * 7.0  # 7 days


def orbit_wars_local_configuration(
    *,
    seed: int | None = None,
    act_timeout_s: float = DEFAULT_LOCAL_ACT_TIMEOUT_S,
    run_timeout_s: float = DEFAULT_LOCAL_RUN_TIMEOUT_S,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "actTimeout": float(act_timeout_s),
        "runTimeout": float(run_timeout_s),
    }
    if seed is not None:
        cfg["seed"] = int(seed)
    return cfg


def boost_local_overage_after_reset(env: Any, *, floor_s: float = DEFAULT_LOCAL_MIN_OVERAGE_S) -> None:
    """Raise ``remainingOverageTime`` on every seat if the schema default is tiny."""
    fl = float(floor_s)
    for s in env.state:
        cur = float(s.observation.remainingOverageTime)
        if cur < fl:
            s.observation.remainingOverageTime = fl

"""Count interval pressure on real Orbit Wars generated states."""

from __future__ import annotations

import os
import sys
from collections import Counter
from statistics import mean

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import jax_orbit_wars as jow
from orbit_wars_pt.constants import FRACTIONS, MAX_PLANETS
from orbit_wars_pt.geometry import (
    first_hit_angle_intervals,
    fleet_speed,
    tick_hit_angle_intervals,
    union_angle_intervals,
)


def _planet_paths_for_horizon(state: jow.OrbitWarsState, horizon: int):
    """Return p0/p1/active for planet slots over future ticks.

    Uses the JAX environment step with no actions, then records planet movement
    between consecutive states. This includes current real comet state if any.
    """

    states = [state]
    s = state
    actions = jow.empty_actions(int(np.asarray(jax.device_get(state.num_agents))))
    for _ in range(horizon):
        s = jow.step(s, actions, jow.OrbitWarsConfig())
        states.append(s)

    states_np = [jax.device_get(s0) for s0 in states]
    p0 = []
    p1 = []
    active = []
    for k in range(horizon):
        p0.append(np.asarray(states_np[k].planets)[:MAX_PLANETS, 2:4])
        p1.append(np.asarray(states_np[k + 1].planets)[:MAX_PLANETS, 2:4])
        active.append(np.asarray(states_np[k].planet_active)[:MAX_PLANETS].astype(bool))
    radii = np.asarray(states_np[0].planets)[:MAX_PLANETS, 4]
    return np.stack(p0), np.stack(p1), radii, np.stack(active)


def _count_case(state: jow.OrbitWarsState, horizon: int, max_origins: int, seed: int):
    state_np = jax.device_get(state)
    planets = np.asarray(state_np.planets)[:MAX_PLANETS]
    active0 = np.asarray(state_np.planet_active)[:MAX_PLANETS].astype(bool)
    p0, p1, radii, active_by_tick = _planet_paths_for_horizon(state, horizon)
    rng = np.random.default_rng(seed)

    owned = np.where(active0 & (planets[:, jow.PLANET_OWNER] == 0) & (planets[:, jow.PLANET_SHIPS] >= 1.0))[0]
    if owned.size > max_origins:
        owned = rng.choice(owned, size=max_origins, replace=False)
    targets = np.where(active0)[0]

    first_counts = []
    raw_counts = []
    blocker_counts = []

    for o_idx in owned:
        origin_xy = planets[o_idx, 2:4]
        origin_radius = float(planets[o_idx, 4])
        ships = float(planets[o_idx, 5])
        sends = [int(np.floor(frac * ships)) for frac in FRACTIONS]
        sends = sorted({s for s in sends if s >= 1})
        for send in sends:
            speed = fleet_speed(float(send))
            blocked = []
            for tick in range(horizon):
                for obj_idx in range(MAX_PLANETS):
                    if not active_by_tick[tick, obj_idx]:
                        continue
                    hit = tick_hit_angle_intervals(
                        origin_xy,
                        origin_radius,
                        speed,
                        tick,
                        p0[tick, obj_idx],
                        p1[tick, obj_idx],
                        float(radii[obj_idx]),
                        max_depth=8,
                    )
                    raw_counts.append(len(hit))
                    if hit:
                        blocked = union_angle_intervals([*blocked, *hit])
                    blocker_counts.append(len(blocked))

            for target_idx in targets:
                if target_idx == o_idx:
                    continue
                intervals = first_hit_angle_intervals(
                    origin_xy,
                    origin_radius,
                    speed,
                    p0,
                    p1,
                    radii,
                    active_by_tick,
                    int(target_idx),
                    max_depth=8,
                )
                first_counts.append(len(intervals))

    return first_counts, raw_counts, blocker_counts


def _summary(name: str, counts: list[int]) -> None:
    if not counts:
        print(f"{name}: no samples")
        return
    arr = np.asarray(counts)
    hist = Counter(counts)
    qs = np.percentile(arr, [50, 75, 90, 95, 99, 100])
    print(
        f"{name}: n={len(counts)} mean={mean(counts):.3f} "
        f"p50={qs[0]:.0f} p75={qs[1]:.0f} p90={qs[2]:.0f} "
        f"p95={qs[3]:.0f} p99={qs[4]:.0f} max={qs[5]:.0f}"
    )
    print(f"{name} hist <=10: " + " ".join(f"{k}:{hist[k]}" for k in range(11) if hist[k]))


def main() -> None:
    seeds = range(3)
    horizon = 10
    max_origins = 2
    first_all: list[int] = []
    raw_all: list[int] = []
    block_all: list[int] = []

    for seed in seeds:
        state = jow.reset_from_reference(seed, 2, max_fleets=128)
        first, raw, block = _count_case(state, horizon, max_origins, seed)
        first_all.extend(first)
        raw_all.extend(raw)
        block_all.extend(block)

    print(f"real generated states: seeds={len(list(seeds))} horizon={horizon} max_origins={max_origins}")
    _summary("raw tick/object hit intervals", raw_all)
    _summary("blocked union intervals during sweep", block_all)
    _summary("first-hit target intervals", first_all)


if __name__ == "__main__":
    main()

"""Targeted test: JAX raycast ghost collisions after comet path expiry.

Reproduces the fleet-459 failure mode from selfplay without a training run:
  - Launch at env step 171 from planet 25, angle ~0.589 rad, full fleet (frac 4).
  - Comet (planet id 32) expires mid-horizon in the path replay.
  - Bug (pre-fix): collision_enabled stays True after expiry → ray still hits at tick 23.
  - Kaggle: comet removed before fleet arrives → OOB.

Uses the same path as training: micro_jax._forecast_planet_paths_one_tick → jax _planet_paths.

Run:
  JAX_PLATFORMS=cpu python -m orbit_wars_pt.check_jax_comet_expiry_raycast
  JAX_PLATFORMS=cpu python -m orbit_wars_pt.check_jax_comet_expiry_raycast --record records/selfplay_011.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax_orbit_wars as jow
from orbit_wars_pt.constants import MAX_PLANETS
from orbit_wars_pt.kaggle_adapter import observation_to_state
from orbit_wars_pt.micro_jax import (
    _forecast_planet_paths_one_tick,
    selected_origin_fraction_targets_batched,
)


def add_batch_dim(state: jow.OrbitWarsState) -> jow.OrbitWarsState:
    return jax.tree.map(lambda x: jnp.asarray(x)[None], state)


def kaggle_np_state_to_jax(np_state) -> jow.OrbitWarsState:
    """Map kaggle_adapter.OrbitWarsState → jax_orbit_wars.OrbitWarsState."""

    num_agents = int(np.asarray(np_state.num_agents))
    return jow.OrbitWarsState(
        planets=jnp.asarray(np_state.planets),
        planet_active=jnp.asarray(np_state.planet_active),
        initial_planets=jnp.asarray(np_state.initial_planets),
        initial_active=jnp.asarray(np_state.initial_active),
        origin_frac_blocked=jnp.asarray(np_state.origin_frac_blocked),
        fleets=jnp.asarray(np_state.fleets),
        fleet_active=jnp.asarray(np_state.fleet_active),
        incoming_fleets=jnp.asarray(np_state.incoming_fleets),
        incoming_fake_correction=jnp.zeros_like(jnp.asarray(np_state.incoming_fleets), dtype=jnp.uint16),
        comet_paths=jnp.asarray(np_state.comet_paths),
        comet_path_lengths=jnp.asarray(np_state.comet_path_lengths),
        comet_ships=jnp.asarray(np_state.comet_ships),
        comet_group_active=jnp.asarray(np_state.comet_group_active),
        comet_path_index=jnp.asarray(np_state.comet_path_index),
        comet_planet_ids=jnp.asarray(np_state.comet_planet_ids),
        comet_slots=jnp.asarray(np_state.comet_slots),
        next_fleet_id=jnp.asarray(np_state.next_fleet_id),
        angular_velocity=jnp.asarray(np_state.angular_velocity),
        step_count=jnp.asarray(np_state.step_count),
        num_agents=jnp.asarray(np_state.num_agents),
        rewards=jnp.asarray(np_state.rewards),
        done=jnp.asarray(np_state.done),
        overflow=jnp.array(False),
    )


def forecast_raw_planet_paths_trace(jax_state: jow.OrbitWarsState, slot: int, horizon: int) -> dict[str, np.ndarray]:
    """Expire once, then replay ``_planet_paths`` only (comet slots not cleared on expiry)."""

    s0 = jow._expire_comets(jax_state)
    slot_j = jnp.asarray(slot, dtype=jnp.int32)

    def body(carry, _):
        s_before = carry
        _, new_pos, collision_enabled, cpi_next, expired = jow._planet_paths(s_before)
        pa_next = s_before.planet_active & ~expired
        tick_row = jnp.stack(
            [
                collision_enabled[slot_j],
                s_before.planet_active[slot_j],
                pa_next[slot_j],
            ]
        )
        s_after = s_before._replace(
            planets=s_before.planets.at[:, jow.PLANET_X : jow.PLANET_Y + 1].set(new_pos),
            planet_active=pa_next,
            initial_active=s_before.initial_active & ~expired,
            comet_path_index=cpi_next,
        )
        return s_after, tick_row

    final, ticks = jax.lax.scan(body, s0, None, length=horizon)
    ticks_np = np.asarray(ticks)
    return {
        "collision_enabled": ticks_np[:, 0].astype(bool),
        "planet_active_before": ticks_np[:, 1].astype(bool),
        "planet_active_after": ticks_np[:, 2].astype(bool),
        "final_active": bool(np.asarray(final.planet_active[slot])),
    }


def _next_planet_positions_old_collision(
    planets: np.ndarray,
    planet_active: np.ndarray,
    initial_planets: np.ndarray,
    initial_active: np.ndarray,
    comet_paths: np.ndarray,
    comet_path_lengths: np.ndarray,
    comet_group_active: np.ndarray,
    comet_path_index: np.ndarray,
    comet_slots: np.ndarray,
    angular_velocity: float,
    step_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pre-fix Kaggle adapter: ``check = active & ((~first_placement) | expired)`` (ghost after expiry)."""

    from orbit_wars_pt.kaggle_adapter import (
        CENTER,
        MAX_COMET_GROUPS,
        PLANET_RADIUS,
        PLANET_X,
        PLANET_Y,
        ROTATION_RADIUS_LIMIT,
    )

    old_pos = planets[:, PLANET_X : PLANET_Y + 1].copy()
    new_pos = old_pos.copy()
    init_pos = initial_planets[:, PLANET_X : PLANET_Y + 1]
    delta = init_pos - np.asarray([CENTER, CENTER], dtype=np.float32)
    orbital_r = np.linalg.norm(delta, axis=1)
    initial_angle = np.arctan2(delta[:, 1], delta[:, 0])
    rotating = planet_active & initial_active & (orbital_r + planets[:, PLANET_RADIUS] < ROTATION_RADIUS_LIMIT)
    current_angle = initial_angle + float(angular_velocity) * float(step_count)
    new_pos[rotating, 0] = CENTER + orbital_r[rotating] * np.cos(current_angle[rotating])
    new_pos[rotating, 1] = CENTER + orbital_r[rotating] * np.sin(current_angle[rotating])

    collision_enabled = planet_active.copy()
    next_path_index = comet_path_index + comet_group_active.astype(np.int32)
    expired_after_move = np.zeros_like(planet_active)

    for g in range(MAX_COMET_GROUPS):
        if not comet_group_active[g]:
            continue
        idx = int(next_path_index[g])
        for k in range(4):
            slot = int(comet_slots[g, k])
            if slot < 0:
                continue
            length = int(comet_path_lengths[g, k])
            expired = idx >= length
            in_path = idx < length
            if in_path:
                new_pos[slot] = comet_paths[g, k, max(idx, 0)]
            first_placement = planets[slot, PLANET_X] < 0.0
            collision_enabled[slot] = (not first_placement) or expired
            expired_after_move[slot] = expired_after_move[slot] or expired

    return old_pos, new_pos, collision_enabled, next_path_index, planet_active & ~expired_after_move, initial_active & ~expired_after_move


def numpy_old_collision_forecast(np_state, slot: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """NumPy path forecast with pre-fix comet collision (fleet-459 Kaggle selfplay bug)."""

    from orbit_wars_pt.kaggle_adapter import PLANET_X, PLANET_Y, _expire_comets_for_forecast

    planets = np.asarray(np_state.planets).copy()
    planet_active = np.asarray(np_state.planet_active).astype(bool).copy()
    initial_planets = np.asarray(np_state.initial_planets)
    initial_active = np.asarray(np_state.initial_active).astype(bool).copy()
    comet_paths = np.asarray(np_state.comet_paths)
    comet_path_lengths = np.asarray(np_state.comet_path_lengths)
    comet_group_active = np.asarray(np_state.comet_group_active).astype(bool).copy()
    comet_path_index = np.asarray(np_state.comet_path_index).astype(np.int32).copy()
    comet_slots = np.asarray(np_state.comet_slots).astype(np.int32).copy()
    comet_planet_ids = np.asarray(np_state.comet_planet_ids)
    av = float(np.asarray(np_state.angular_velocity))
    sc = int(np.asarray(np_state.step_count))

    planet_active, initial_active, comet_group_active, comet_planet_ids, comet_slots = (
        _expire_comets_for_forecast(
            planet_active,
            initial_active,
            comet_group_active,
            comet_path_index,
            comet_path_lengths,
            comet_slots,
            comet_planet_ids,
        )
    )

    coll_old: list[bool] = []
    pa_after: list[bool] = []
    for t in range(horizon):
        _, new_pos, collision_enabled, cpi_next, pa_next, ia_next = _next_planet_positions_old_collision(
            planets,
            planet_active,
            initial_planets,
            initial_active,
            comet_paths,
            comet_path_lengths,
            comet_group_active,
            comet_path_index,
            comet_slots,
            av,
            sc + t,
        )
        coll_old.append(bool(collision_enabled[slot]))
        pa_after.append(bool(pa_next[slot]))
        planets[:, PLANET_X : PLANET_Y + 1] = new_pos
        planet_active = pa_next
        initial_active = ia_next
        comet_path_index = cpi_next

    return np.asarray(coll_old, dtype=bool), np.asarray(pa_after, dtype=bool)


def forecast_collision_trace(jax_state: jow.OrbitWarsState, slot: int, horizon: int) -> dict[str, np.ndarray]:
    """Replay training path-only ticks; record collision mask and planet_active per tick."""

    slot_j = jnp.asarray(slot, dtype=jnp.int32)

    def body(carry, _):
        s_before = carry
        s_after, (_, _, collision_enabled) = _forecast_planet_paths_one_tick(s_before)
        tick_row = jnp.stack(
            [
                collision_enabled[slot_j],
                s_before.planet_active[slot_j],
                s_after.planet_active[slot_j],
            ]
        )
        return s_after, tick_row

    final, ticks = jax.lax.scan(body, jax_state, None, length=horizon)
    ticks_np = np.asarray(ticks)
    return {
        "collision_enabled": ticks_np[:, 0].astype(bool),
        "planet_active_before": ticks_np[:, 1].astype(bool),
        "planet_active_after": ticks_np[:, 2].astype(bool),
        "final_active": bool(np.asarray(final.planet_active[slot])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record",
        type=Path,
        default=Path("records/selfplay_011.json"),
        help="Selfplay record with fleet-459 scenario (step 171 launch).",
    )
    parser.add_argument("--launch-step", type=int, default=171)
    parser.add_argument("--origin-slot", type=int, default=25)
    parser.add_argument("--target-slot", type=int, default=32)
    parser.add_argument("--frac-idx", type=int, default=4)
    parser.add_argument("--angle", type=float, default=0.589049)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--n-rays", type=int, default=256)
    args = parser.parse_args()

    record = json.loads(args.record.expanduser().read_text())
    cfg = record["configuration"]
    steps = record["steps"]
    si = args.launch_step
    if si >= len(steps):
        raise SystemExit(f"launch-step {si} >= record length {len(steps)}")

    np_state = observation_to_state(
        steps[si][0]["observation"],
        cfg,
        step_count_override=args.launch_step,
    )
    planets = np.asarray(np_state.planets)
    o_slot = args.origin_slot
    t_slot = args.target_slot
    if not bool(np_state.planet_active[t_slot]):
        raise SystemExit(f"target slot {t_slot} not active at launch step (unexpected)")

    print(f"record={args.record.name} launch_step={args.launch_step}")
    print(
        f"origin slot {o_slot} pid={int(planets[o_slot, 0]):.0f} "
        f"target slot {t_slot} pid={int(planets[t_slot, 0]):.0f}"
    )
    print(f"comet path_index={int(np.max(np_state.comet_path_index))} "
          f"path_len={int(np_state.comet_path_lengths[0, 0])}")

    jax_state = kaggle_np_state_to_jax(np_state)
    trace = forecast_collision_trace(jax_state, t_slot, args.horizon)

    print("\n=== JAX path replay (training micro_jax tick loop) ===")
    print("tick  coll_en  pa_before  pa_after  (approx env step = 172 + tick)")
    expiry_tick = None
    ghost_ticks: list[int] = []
    for t in range(args.horizon):
        coll = trace["collision_enabled"][t]
        pb = trace["planet_active_before"][t]
        pa = trace["planet_active_after"][t]
        if pb and not pa and expiry_tick is None:
            expiry_tick = t
        if (not pa) and coll:
            ghost_ticks.append(t)
        if t <= 15 or t >= 18 or coll or (pb != pa):
            print(f" {t:3d}   {coll!s:5s}  {pb!s:9s}  {pa!s:8s}   (~{172 + t})")

    print(f"\nexpiry forecast tick (pa True→False): {expiry_tick}")
    print(f"ghost ticks (pa False but coll True): {ghost_ticks[:12]}{'...' if len(ghost_ticks) > 12 else ''}")
    print(f"count ghost ticks: {len(ghost_ticks)}")

    raw = forecast_raw_planet_paths_trace(jax_state, t_slot, args.horizon)
    raw_post = [
        t
        for t in range(args.horizon)
        if expiry_tick is not None
        and t > expiry_tick
        and (not raw["planet_active_after"][t])
        and raw["collision_enabled"][t]
    ]
    np_old_coll, np_old_pa = numpy_old_collision_forecast(np_state, t_slot, args.horizon)
    np_old_post = [
        t
        for t in range(args.horizon)
        if expiry_tick is not None
        and t > expiry_tick
        and (not np_old_pa[t])
        and np_old_coll[t]
    ]
    print("\n=== JAX raw _planet_paths (expire once, slots persist; reverted check=...) ===")
    print(f"  post-expiry ghost ticks: {len(raw_post)} {raw_post[:8]}")
    print("=== NumPy OLD collision rule (fleet-459 / Kaggle selfplay bug) ===")
    print(f"  post-expiry ghost ticks: {len(np_old_post)} {np_old_post[:8]}")

    # Same API training uses for target validity (batch size 1).
    jax_b = add_batch_dim(jax_state)
    origin_idx = jnp.asarray([o_slot], dtype=jnp.int32)
    frac_idx = jnp.asarray([args.frac_idx], dtype=jnp.int32)
    angle_j, _, valid_j, _, hit_tick_j, true_planet_j, true_hit_tick_j = (
        selected_origin_fraction_targets_batched(
            jax_b,
            origin_idx,
            frac_idx,
            horizon=min(args.horizon, 24),
            ship_speed=6.0,
            n_rays=args.n_rays,
        )
    )
    valid = np.asarray(valid_j[0])
    hit_tick = np.asarray(hit_tick_j[0])
    true_planet = np.asarray(true_planet_j[0])
    true_hit_tick = np.asarray(true_hit_tick_j[0])
    ray_angles = np.asarray(angle_j[0])

    print("\n=== JAX selected_origin_fraction_targets_batched (training raycast) ===")
    print(f"  target slot {t_slot} valid={bool(valid[t_slot])} "
          f"hit_tick={hit_tick[t_slot]:.1f} true_planet={true_planet[t_slot]} "
          f"true_hit_tick={true_hit_tick[t_slot]:.1f}")

    ang_tol = 2 * math.pi / args.n_rays + 1e-3
    matches = [
        d
        for d in range(MAX_PLANETS)
        if valid[d] and abs(((args.angle - ray_angles[d] + math.pi) % (2 * math.pi)) - math.pi) < ang_tol
    ]
    print(f"  launch angle {args.angle:.6f} matches valid slots: {matches}")

    # Failure mode: post-expiry ghost coll mask and/or late valid target (fleet 459).
    failed = []
    post_expiry_ghost = [t for t in ghost_ticks if expiry_tick is not None and t > expiry_tick]
    if post_expiry_ghost:
        failed.append(f"ghost_collision_after_expiry ticks={post_expiry_ghost[:8]}")
    if bool(valid[t_slot]) and float(hit_tick[t_slot]) >= 20:
        failed.append("target_still_valid_late_hit")

    # NumPy adapter forecast for contrast (Kaggle selfplay path).
    from orbit_wars_pt.kaggle_adapter import _forecast_planet_paths_np, _raycast_targets_np

    _, _, np_coll = _forecast_planet_paths_np(np_state, horizon=args.horizon)
    np_ghost = int(np.sum((~trace["planet_active_after"]) & trace["collision_enabled"]))
    np_ra, np_valid, np_hit, _, _ = _raycast_targets_np(
        np_state, o_slot, args.frac_idx, ship_speed=6.0, n_rays=args.n_rays
    )
    print("\n=== NumPy kaggle_adapter forecast (current tree, with expiry fix) ===")
    print(f"  ghost ticks after expiry: {np_ghost}")
    print(
        f"  target slot {t_slot} valid={bool(np_valid[t_slot])} hit_tick={np_hit[t_slot]:.1f}"
    )

    print("\n=== Verdict ===")
    jax_training_ok = not failed
    if jax_training_ok:
        print("PASS (training / micro_jax): no post-expiry ghost; target 32 not valid at late hit.")
    else:
        print("FAIL (training path):", ", ".join(failed))
        sys.exit(1)

    if len(np_old_post) >= 10:
        print(
            f"CONFIRMED NumPy old rule: {len(np_old_post)} post-expiry ghost ticks "
            f"(explains fleet-459 Kaggle selfplay)."
        )
    if len(raw_post) >= 10:
        print(f"CONFIRMED JAX reverted _planet_paths (raw loop): {len(raw_post)} post-expiry ghost ticks.")
    elif len(raw_post) == 0:
        print(
            "JAX reverted _planet_paths (raw loop): no post-expiry ghosts — "
            "training path already safe; JAX change was not required for this scenario."
        )


if __name__ == "__main__":
    main()

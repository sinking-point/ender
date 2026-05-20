"""Consistency test: legacy raycaster vs category-rays.

This compares ``selected_origin_fraction_targets_batched(..., first_hit_method="rays")``
against ``first_hit_method="category-rays"`` on a mix of:

- real reset states
- real states stepped to windows with upcoming comets
- real states stepped to windows with active comets
- constructed states designed to stress future-comet policy/true-hit behavior

Run:

    JAX_PLATFORMS=cpu ./.venv/bin/python -m orbit_wars_pt.check_category_rays_consistency
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--platform", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--num-seeds", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--n-rays", type=int, default=256)
    p.add_argument("--ship-speed", type=float, default=6.0)
    p.add_argument("--real-steps", type=int, nargs="*", default=(0, 48, 55, 148, 155))
    p.add_argument("--max-report", type=int, default=20)
    p.add_argument("--atol", type=float, default=1e-4)
    return p.parse_args()


ARGS = parse_args()
if ARGS.platform == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"
elif ARGS.platform == "cuda":
    os.environ["JAX_PLATFORMS"] = "cuda,cpu"

import jax
import jax.numpy as jnp
import numpy as np

import jax_orbit_wars as jow
from jax_orbit_wars import OrbitWarsState, PLANET_OWNER, PLANET_SHIPS, empty_actions
from orbit_wars_pt.batched_env import stack_initial_states
from orbit_wars_pt.check_jax_fake_incoming_e2e import (
    COMET_IDX,
    EGO,
    FRAC_IDX,
    P0_SLOT,
    build_constructed_state,
)
from orbit_wars_pt.constants import FRACTIONS
from orbit_wars_pt.env_wrapper import OrbitWarsEnvConfig
from orbit_wars_pt.jax_setup import configure_jax_for_training
from orbit_wars_pt.micro_jax import selected_origin_fraction_targets_batched


@dataclass(frozen=True)
class Case:
    label: str
    state: OrbitWarsState
    origin_idx: int
    frac_idx: int


def _stack_states(states: list[OrbitWarsState]) -> OrbitWarsState:
    return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *states)


def _slice_env(state_b: OrbitWarsState, env_i: int) -> OrbitWarsState:
    return jax.tree.map(lambda x: x[env_i], state_b)


def _batched_noop_actions(num_envs: int, num_agents: int) -> jnp.ndarray:
    one = empty_actions(num_agents, 1)
    return jnp.broadcast_to(one, (num_envs,) + one.shape)


def _normalize_fleet_capacity(state: OrbitWarsState, max_fleets: int) -> OrbitWarsState:
    cur = int(state.fleets.shape[0])
    if cur == max_fleets:
        return state
    if cur > max_fleets:
        return state._replace(
            fleets=state.fleets[:max_fleets],
            fleet_active=state.fleet_active[:max_fleets],
        )
    pad = max_fleets - cur
    fleets = jnp.concatenate(
        [state.fleets, jnp.zeros((pad, state.fleets.shape[-1]), dtype=state.fleets.dtype)],
        axis=0,
    )
    fleet_active = jnp.concatenate(
        [state.fleet_active, jnp.zeros((pad,), dtype=state.fleet_active.dtype)],
        axis=0,
    )
    return state._replace(fleets=fleets, fleet_active=fleet_active)


def _step_to_targets(state_b: OrbitWarsState, targets: list[int]) -> dict[int, OrbitWarsState]:
    targets = sorted(set(int(t) for t in targets))
    out: dict[int, OrbitWarsState] = {}
    cur = state_b
    num_envs = int(cur.planets.shape[0])
    num_agents = int(np.asarray(jax.device_get(cur.num_agents[0])))
    noop = _batched_noop_actions(num_envs, num_agents)
    step_fn = jax.jit(jax.vmap(lambda s, a: jow.jit_step(s, a, jow.OrbitWarsConfig())))
    max_step = max(targets) if targets else 0
    for step in range(max_step + 1):
        if step in targets:
            out[step] = cur
        if step != max_step:
            cur = step_fn(cur, noop)
    return out


def _owned_origins(state: OrbitWarsState, ego: int = EGO) -> list[int]:
    active = np.asarray(jax.device_get(state.planet_active)).astype(bool)
    owner = np.asarray(jax.device_get(state.planets[:, PLANET_OWNER])).astype(np.int32)
    return [int(i) for i in np.flatnonzero(active & (owner == ego))]


def _expand_cases_from_batched(state_b: OrbitWarsState, *, suite: str) -> list[Case]:
    cases: list[Case] = []
    num_envs = int(state_b.planets.shape[0])
    for env_i in range(num_envs):
        state = _slice_env(state_b, env_i)
        origins = _owned_origins(state)
        for origin_idx in origins:
            for frac_idx, frac in enumerate(FRACTIONS):
                cases.append(
                    Case(
                        label=f"{suite}/env={env_i}/origin={origin_idx}/frac={frac:.1f}",
                        state=state,
                        origin_idx=origin_idx,
                        frac_idx=frac_idx,
                    )
                )
    return cases


def _constructed_cases() -> list[Case]:
    step_fn = jax.jit(jow.step)
    one_noop = empty_actions(2, 1)

    def with_origin_ships(state: OrbitWarsState, ships: float) -> OrbitWarsState:
        ships_v = jnp.asarray(float(ships), dtype=state.planets.dtype)
        planets = state.planets.at[P0_SLOT, PLANET_SHIPS].set(ships_v)
        initial_planets = state.initial_planets.at[P0_SLOT, PLANET_SHIPS].set(ships_v)
        return state._replace(planets=planets, initial_planets=initial_planets)

    def case_family(label_prefix: str, ships: float) -> list[Case]:
        base = _normalize_fleet_capacity(
            with_origin_ships(
                build_constructed_state(step_count=int(jow.COMET_SPAWN_STEPS[0]) - 2),
                ships,
            ),
            max_fleets=128,
        )
        after_launch = step_fn(base, one_noop)   # step 49
        after_spawn = step_fn(after_launch, one_noop)  # step 50, active comet
        after_active = step_fn(after_spawn, one_noop)  # step 51

        comet_slots = np.asarray(jax.device_get(after_spawn.comet_slots))[0]
        true_slot = int(comet_slots[COMET_IDX])
        print(
            f"{label_prefix}: ships={int(ships)} active comet slot group0[{COMET_IDX}] -> "
            f"slot {true_slot} at step {int(np.asarray(jax.device_get(after_spawn.step_count)))}",
            flush=True,
        )
        return [
            Case(f"{label_prefix}/upcoming/ships={int(ships)}", base, P0_SLOT, FRAC_IDX),
            Case(f"{label_prefix}/pre_spawn/ships={int(ships)}", after_launch, P0_SLOT, FRAC_IDX),
            Case(f"{label_prefix}/active_spawn/ships={int(ships)}", after_spawn, P0_SLOT, FRAC_IDX),
            Case(f"{label_prefix}/active_next/ships={int(ships)}", after_active, P0_SLOT, FRAC_IDX),
        ]

    cases: list[Case] = []
    cases.extend(case_family("constructed", 20.0))
    for ships in (200.0, 300.0, 400.0, 500.0):
        cases.extend(case_family("constructed_fast", ships))
    return cases


def _run_method(
    state_b: OrbitWarsState,
    origin_idx_b: jnp.ndarray,
    frac_idx_b: jnp.ndarray,
    *,
    first_hit_method: str,
) -> tuple[np.ndarray, ...]:
    out = selected_origin_fraction_targets_batched(
        state_b,
        origin_idx_b,
        frac_idx_b,
        horizon=int(ARGS.horizon),
        ship_speed=float(ARGS.ship_speed),
        n_rays=int(ARGS.n_rays),
        ray_chunk_size=0,
        first_hit_method=first_hit_method,
    )
    return tuple(np.asarray(x) for x in jax.device_get(out))


def _collect_cases() -> list[Case]:
    cfg = OrbitWarsEnvConfig(
        num_agents=2,
        max_fleets=128,
        episode_seed=int(ARGS.seed),
    )
    init_b, _ = stack_initial_states(cfg, int(ARGS.num_seeds), int(ARGS.seed))
    real_by_step = _step_to_targets(init_b, list(ARGS.real_steps))

    cases: list[Case] = []
    for step in sorted(real_by_step):
        cases.extend(_expand_cases_from_batched(real_by_step[step], suite=f"real/step={step}"))
    cases.extend(_constructed_cases())
    return cases


def _report_mismatch(
    idx: int,
    label: str,
    legacy: tuple[np.ndarray, ...],
    category: tuple[np.ndarray, ...],
    *,
    atol: float,
) -> list[str]:
    names = (
        "angle",
        "width",
        "valid",
        "overflow",
        "policy_hit_tick",
        "true_hit_planet",
        "true_hit_tick",
    )
    lines = [f"[mismatch {idx}] {label}"]
    for name, a, b in zip(names, legacy, category):
        if a.dtype.kind in "fc":
            same = np.allclose(a, b, atol=atol, rtol=0.0, equal_nan=True)
            if not same:
                diff = np.max(np.abs(a - b))
                lines.append(f"  {name}: max_abs_diff={diff:.6g}")
        else:
            same = np.array_equal(a, b)
            if not same:
                neq = int(np.count_nonzero(a != b))
                lines.append(f"  {name}: {neq} differing entries")
    return lines


def main() -> None:
    configure_jax_for_training(prefer_gpu=(ARGS.platform != "cpu"), verbose=False)
    print(f"devices {jax.devices()}", flush=True)

    cases = _collect_cases()
    if not cases:
        raise SystemExit("no cases collected")

    state_b = _stack_states([c.state for c in cases])
    origin_idx_b = jnp.asarray([c.origin_idx for c in cases], dtype=jnp.int32)
    frac_idx_b = jnp.asarray([c.frac_idx for c in cases], dtype=jnp.int32)

    print(
        f"running consistency suite: cases={len(cases)} horizon={ARGS.horizon} "
        f"n_rays={ARGS.n_rays} fractions={len(FRACTIONS)}",
        flush=True,
    )

    legacy = _run_method(state_b, origin_idx_b, frac_idx_b, first_hit_method="rays")
    category = _run_method(state_b, origin_idx_b, frac_idx_b, first_hit_method="category-rays")

    mismatches = 0
    reported = 0
    max_angle_diff = 0.0
    max_width_diff = 0.0
    max_policy_tick_diff = 0.0
    max_true_tick_diff = 0.0

    for i, case in enumerate(cases):
        legacy_i = tuple(arr[i] for arr in legacy)
        category_i = tuple(arr[i] for arr in category)
        max_angle_diff = max(max_angle_diff, float(np.max(np.abs(legacy_i[0] - category_i[0]))))
        max_width_diff = max(max_width_diff, float(np.max(np.abs(legacy_i[1] - category_i[1]))))
        max_policy_tick_diff = max(
            max_policy_tick_diff, float(np.max(np.abs(legacy_i[4] - category_i[4])))
        )
        max_true_tick_diff = max(
            max_true_tick_diff, float(np.max(np.abs(legacy_i[6] - category_i[6])))
        )

        same = (
            np.allclose(legacy_i[0], category_i[0], atol=ARGS.atol, rtol=0.0, equal_nan=True)
            and np.allclose(legacy_i[1], category_i[1], atol=ARGS.atol, rtol=0.0, equal_nan=True)
            and np.array_equal(legacy_i[2], category_i[2])
            and np.array_equal(legacy_i[3], category_i[3])
            and np.allclose(legacy_i[4], category_i[4], atol=ARGS.atol, rtol=0.0, equal_nan=True)
            and np.array_equal(legacy_i[5], category_i[5])
            and np.allclose(legacy_i[6], category_i[6], atol=ARGS.atol, rtol=0.0, equal_nan=True)
        )
        if not same:
            mismatches += 1
            if reported < int(ARGS.max_report):
                for line in _report_mismatch(
                    mismatches, case.label, legacy_i, category_i, atol=float(ARGS.atol)
                ):
                    print(line, flush=True)
                reported += 1

    print(
        f"summary: mismatches={mismatches}/{len(cases)} "
        f"max_angle_diff={max_angle_diff:.6g} "
        f"max_width_diff={max_width_diff:.6g} "
        f"max_policy_tick_diff={max_policy_tick_diff:.6g} "
        f"max_true_tick_diff={max_true_tick_diff:.6g}",
        flush=True,
    )
    if mismatches:
        raise SystemExit(1)
    print("PASS", flush=True)


if __name__ == "__main__":
    main()

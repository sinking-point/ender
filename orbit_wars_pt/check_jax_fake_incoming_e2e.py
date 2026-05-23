"""E2E spec: fake incoming through comet spawn on a constructed two-planet map.

No seed / reference generator. Two planets, four comet paths (spawn requires all
lengths > 0). Intercept comet is index 1 → slot 3 (not first inactive slot 2).

Asserts true mass on the raycast comet slot and fake display on the policy planet,
each at its own ETA bin.

Run: JAX_PLATFORMS=cpu python -m orbit_wars_pt.check_jax_fake_incoming_e2e
"""

from __future__ import annotations

import math
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import jax_orbit_wars as jow
from jax_orbit_wars import (
    COMET_SPAWN_STEPS,
    PLANET_ID,
    PLANET_OWNER,
    PLANET_SHIPS,
    PLANET_X,
    PLANET_Y,
    empty_actions,
)
from orbit_wars_pt.constants import INCOMING_TA_BINS, MAX_PLANETS
from orbit_wars_pt.micro_jax import selected_origin_fraction_targets_batched
from orbit_wars_pt.observation_jax import build_observation_batched_jax

# Map layout
P0_SLOT = 0
P1_SLOT = 1
P0_POS = (10.0, 10.0)
P1_POS = (10.0, 50.0)
COMET_XY = (10.0, 40.0)
PLANET_R = 3.0
P0_SHIPS = 20
P1_SHIPS = 10
FRAC_IDX = 4  # 100% → 20 ships
SEND = 20
EGO = 0

FIRST_INACTIVE_SLOT = 2
COMET_IDX = 1
TRUE_SLOT = FIRST_INACTIVE_SLOT + COMET_IDX  # 3

SPAWN_STEP = int(COMET_SPAWN_STEPS[0])  # 50
# Launch at 48; one noop reaches spawn (step_count 50).
LAUNCH_STEP = SPAWN_STEP - 2


def _ta(eta: float) -> int:
    return int(math.floor(eta))


def _shifted(ta: int, n: int) -> int:
    return max(ta - n, 0)


def _plane(table: jnp.ndarray, ego: int, slot: int) -> np.ndarray:
    arr = np.asarray(jax.device_get(table))
    if arr.ndim == 4:
        return arr[0, ego, slot, :].astype(np.int64)
    return arr[ego, slot, :].astype(np.int64)


def _obs_incoming(obs: dict, slot: int) -> np.ndarray:
    """Net self incoming on planet token (features[8:]).

    JAX obs is ``[num_envs, 1 + MAX_PLANETS, FEATURE_DIM]``; planet slot ``s`` is index ``1 + s``.
    """

    feat = np.asarray(jax.device_get(obs["features"]))
    if feat.ndim == 3:
        feat = feat[0, 1 + slot, 8:]
    else:
        feat = feat[1 + slot, 8:]
    return np.rint(feat * 1000.0).astype(np.int64)


def _assert_bins(plane: np.ndarray, bins: dict[int, int], *, label: str) -> None:
    expected = np.zeros_like(plane)
    for i, mass in bins.items():
        expected[i] = mass
    assert np.array_equal(plane, expected), f"{label}: want {bins}, got {plane.tolist()}"


def _make_planet_row(pid: int, owner: float, xy: tuple[float, float], ships: float) -> list[float]:
    return [float(pid), owner, xy[0], xy[1], PLANET_R, ships, 1.0]


def build_constructed_state(*, step_count: int) -> jow.OrbitWarsState:
    """Minimal map: p0, neutral p1, four comet paths; intercept on comet index 1."""

    planets = np.zeros((jow.MAX_PLANETS, 7), dtype=np.float32)
    planets[P0_SLOT] = _make_planet_row(0, 0.0, P0_POS, P0_SHIPS)
    planets[P1_SLOT] = _make_planet_row(1, 1.0, P1_POS, P1_SHIPS)  # enemy-owned so 2 players stay alive
    for slot in range(2, 10):
        planets[slot] = [float(slot), -1.0, 200.0, 200.0, 0.0, 0.0, 0.0]

    active = np.zeros(jow.MAX_PLANETS, dtype=bool)
    active[P0_SLOT] = True
    active[P1_SLOT] = True

    paths = np.zeros((jow.MAX_COMET_GROUPS, 4, jow.MAX_COMET_PATH, 2), dtype=np.float32)
    lens = np.zeros((jow.MAX_COMET_GROUPS, 4), dtype=np.int32)
    # All four paths must be non-empty or ``_spawn_comets`` skips the group.
    decoy_coords = [(90.0, 90.0), COMET_XY, (80.0, 80.0), (70.0, 70.0)]
    for k, xy in enumerate(decoy_coords):
        paths[0, k, 0] = xy
        paths[0, k, 1] = (xy[0] + 1.0, xy[1])
        lens[0, k] = 2
    comet_path = np.tile(np.asarray(COMET_XY, dtype=np.float32), (jow.MAX_COMET_PATH, 1))
    paths[0, COMET_IDX] = comet_path
    lens[0, COMET_IDX] = jow.MAX_COMET_PATH

    return jow.OrbitWarsState(
        planets=jnp.asarray(planets),
        planet_active=jnp.asarray(active),
        initial_planets=jnp.asarray(planets),
        initial_active=jnp.asarray(active),
        fleets=jnp.zeros((jow.DEFAULT_MAX_FLEETS, jow.FLEET_ROW_WIDTH), dtype=jnp.float32),
        fleet_active=jnp.zeros((jow.DEFAULT_MAX_FLEETS,), dtype=bool),
        incoming_fleets=jnp.zeros((2, jow.MAX_PLANETS, INCOMING_TA_BINS), dtype=jnp.uint16),
        incoming_fake_correction=jnp.zeros((2, jow.MAX_PLANETS, INCOMING_TA_BINS), dtype=jnp.uint16),
        comet_paths=jnp.asarray(paths),
        comet_path_lengths=jnp.asarray(lens),
        comet_ships=jnp.asarray([5.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        comet_group_active=jnp.zeros((jow.MAX_COMET_GROUPS,), dtype=bool),
        comet_path_index=jnp.full((jow.MAX_COMET_GROUPS,), -1, dtype=jnp.int32),
        comet_planet_ids=jnp.full((jow.MAX_COMET_GROUPS, 4), -1, dtype=jnp.int32),
        comet_slots=jnp.full((jow.MAX_COMET_GROUPS, 4), -1, dtype=jnp.int32),
        next_fleet_id=jnp.int32(0),
        angular_velocity=jnp.float32(0.0),
        step_count=jnp.int32(step_count),
        num_agents=jnp.int32(2),
        rewards=jnp.zeros((4,), dtype=jnp.float32),
        done=jnp.bool_(False),
        overflow=jnp.bool_(False),
    )


def _launch_actions(angle: float, policy_eta: float, true_eta: float) -> jnp.ndarray:
    actions = jnp.zeros((2, 1, 7), dtype=jnp.float32)
    actions = actions.at[:, :, 3].set(-1.0).at[:, :, 4].set(500.0).at[:, :, 5].set(-1.0).at[:, :, 6].set(500.0)
    actions = (
        actions.at[EGO, 0, 0]
        .set(0.0)
        .at[EGO, 0, 1]
        .set(angle)
        .at[EGO, 0, 2]
        .set(float(SEND))
        .at[EGO, 0, 3]
        .set(float(TRUE_SLOT))
        .at[EGO, 0, 4]
        .set(true_eta)
        .at[EGO, 0, 5]
        .set(float(P1_SLOT))
        .at[EGO, 0, 6]
        .set(policy_eta)
    )
    return actions


def main() -> None:
    step_fn = jax.jit(jow.step)
    state = build_constructed_state(step_count=LAUNCH_STEP)
    batched = jax.tree.map(lambda x: jnp.asarray(x)[None], state)

    ang, _, valid, _, hit_tick, true_planet, true_hit = selected_origin_fraction_targets_batched(
        batched,
        jnp.asarray([P0_SLOT], dtype=jnp.int32),
        jnp.asarray([FRAC_IDX], dtype=jnp.int32),
        horizon=24,
        ship_speed=6.0,
        n_rays=256,  # 512 can pick a non-intercept ray for planet 1
    )
    valid_np = np.asarray(valid[0])
    only = np.flatnonzero(valid_np)
    assert only.tolist() == [P1_SLOT], f"expected only planet {P1_SLOT} valid, got {only.tolist()}"

    policy_eta = float(hit_tick[0, P1_SLOT])
    true_eta = float(true_hit[0, P1_SLOT])
    true_slot = int(true_planet[0, P1_SLOT])
    angle = float(ang[0, P1_SLOT])

    assert true_slot == TRUE_SLOT, f"true slot {true_slot}, want {TRUE_SLOT} (comet {COMET_IDX})"
    assert true_slot != FIRST_INACTIVE_SLOT
    assert true_eta < policy_eta, f"true_eta {true_eta} must be < policy_eta {policy_eta}"
    assert true_eta < policy_eta - 1.0, (
        f"true_eta {true_eta} should arrive before policy_eta {policy_eta}"
    )

    ta_policy = _ta(policy_eta)
    ta_true = _ta(true_eta)
    assert ta_policy != ta_true

    print(
        f"raycast: policy_eta={policy_eta:.2f} ta_policy={ta_policy} | "
        f"true_slot={true_slot} true_eta={true_eta:.2f} ta_true={ta_true} | "
        f"spawn on next step after launch (step {LAUNCH_STEP} → {SPAWN_STEP})"
    )

    # Launch (step 48 → 49)
    state = step_fn(state, _launch_actions(angle, policy_eta, true_eta))
    assert int(jax.device_get(state.step_count)) == LAUNCH_STEP + 1
    ta_p1 = _shifted(ta_policy, 1)
    ta_t1 = _shifted(ta_true, 1)

    obs = build_observation_batched_jax(jax.tree.map(lambda x: jnp.asarray(x)[None], state), EGO)
    _assert_bins(_plane(state.incoming_fleets, EGO, TRUE_SLOT), {ta_t1: SEND}, label="env true incoming on comet slot")
    assert int(_plane(state.incoming_fleets, EGO, FIRST_INACTIVE_SLOT).sum()) == 0, (
        "env must not book true mass on first_inactive"
    )
    _assert_bins(_plane(state.incoming_fake_correction, EGO, P1_SLOT), {ta_p1: SEND}, label="env fake correction at policy TA")
    _assert_bins(_obs_incoming(obs, P1_SLOT), {ta_p1: SEND}, label="obs fake incoming on policy planet at policy TA")

    # One noop: step 49 → 50; comets spawn at step entry (step_count + 1 == SPAWN_STEP)
    state = step_fn(state, empty_actions(2, 1))
    assert int(jax.device_get(state.step_count)) == SPAWN_STEP

    slots = np.asarray(jax.device_get(state.comet_slots))[0]
    assert int(slots[COMET_IDX]) == TRUE_SLOT, f"comet slots {slots.tolist()}"

    pos = np.asarray(jax.device_get(state.planets))[TRUE_SLOT, PLANET_X : PLANET_Y + 1]
    path0 = np.asarray(jax.device_get(state.comet_paths))[0, COMET_IDX, 0]
    assert np.allclose(pos, path0, atol=1e-3), f"comet at {pos}, path[0]={path0}"

    ta_t_after = _shifted(ta_true, 2)
    assert ta_t_after >= 1, "true incoming should survive spawn-step bin-0 combat"

    obs2 = build_observation_batched_jax(jax.tree.map(lambda x: jnp.asarray(x)[None], state), EGO)
    _assert_bins(_obs_incoming(obs2, P1_SLOT), {}, label="obs fake stripped: all bins zero on policy planet")
    _assert_bins(_plane(state.incoming_fake_correction, EGO, P1_SLOT), {}, label="env fake correction zeroed on policy planet")
    _assert_bins(
        _plane(state.incoming_fleets, EGO, TRUE_SLOT),
        {ta_t_after: SEND},
        label="true incoming on comet after spawn step",
    )
    _assert_bins(
        _obs_incoming(obs2, TRUE_SLOT),
        {ta_t_after: SEND},
        label="obs true incoming on comet slot",
    )

    print("PASS")


if __name__ == "__main__":
    main()

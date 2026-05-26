from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax
import jax.numpy as jnp
import numpy as np

import jax_orbit_wars as jow

from orbit_wars_pt.batched_env import obs_jax_to_torch, stack_initial_states
from orbit_wars_pt.compressed_observation import compress_observation
from orbit_wars_pt.constants import MAX_PLANETS, obs_feature_dim_for_num_agents
from orbit_wars_pt.env_wrapper import OrbitWarsEnvConfig
from orbit_wars_pt.micro_jax import apply_micro_step_batched
from orbit_wars_pt.observation_jax import (
    build_compressed_observation_batched_jax,
    build_compressed_observation_batched_jax_per_ego,
    build_observation_batched_jax,
    build_observation_batched_jax_per_ego,
)


def _find_owned_origin(state_b: jow.OrbitWarsState, ego: int) -> int:
    planets = np.asarray(jax.device_get(state_b.planets))[0]
    active = np.asarray(jax.device_get(state_b.planet_active))[0]
    owned = np.flatnonzero(
        active
        & (planets[:, jow.PLANET_OWNER].astype(np.int32) == int(ego))
        & (planets[:, jow.PLANET_SHIPS] >= 1.0)
    )
    if owned.size == 0:
        raise AssertionError(f"no owned origin found for ego={ego}")
    return int(owned[0])


def _find_shared_target(state_b: jow.OrbitWarsState, forbidden: set[int]) -> int:
    active = np.asarray(jax.device_get(state_b.planet_active))[0]
    candidates = [int(idx) for idx in np.flatnonzero(active) if int(idx) not in forbidden]
    if not candidates:
        raise AssertionError("no shared active target found")
    return candidates[0]


def _dispatch_real_fleet(
    state_b: jow.OrbitWarsState,
    *,
    ego: int,
    origin_idx: int,
    target_idx: int,
    frac_idx: int = 4,
    fleet_eta: float = 3.0,
) -> jow.OrbitWarsState:
    pair_flat = jnp.asarray([origin_idx * MAX_PLANETS + target_idx], dtype=jnp.int32)
    frac = jnp.asarray([frac_idx], dtype=jnp.int32)
    halt_now = jnp.asarray([False], dtype=jnp.bool_)
    eta = jnp.asarray([fleet_eta], dtype=jnp.float32)
    new_state, oid_j, send_j, dispatched_j, _slot_j = apply_micro_step_batched(
        state_b,
        jnp.int32(ego),
        halt_now,
        pair_flat,
        frac,
        eta,
    )
    dispatched = bool(np.asarray(jax.device_get(dispatched_j))[0])
    send = float(np.asarray(jax.device_get(send_j))[0])
    oid = float(np.asarray(jax.device_get(oid_j))[0])
    if not dispatched or send <= 0.0 or oid < 0.0:
        raise AssertionError(
            f"dispatch failed for ego={ego} origin={origin_idx} target={target_idx} send={send} oid={oid}"
        )
    return new_state


def _compressed_to_numpy_dict(comp) -> dict[str, np.ndarray]:
    return {field: getattr(comp, field).detach().cpu().numpy() for field in comp._fields}


def _jax_comp_to_numpy_dict(comp: dict[str, jnp.ndarray]) -> dict[str, np.ndarray]:
    return {field: np.asarray(jax.device_get(value)) for field, value in comp.items()}


class ObservationCompressedJaxParityTest(unittest.TestCase):
    def _make_state_with_incoming(self, num_agents: int) -> jow.OrbitWarsState:
        cfg = OrbitWarsEnvConfig(num_agents=num_agents, max_fleets=32, episode_seed=0)
        state_b, _ = stack_initial_states(cfg, num_envs=1, seed_base=7)

        origin0 = _find_owned_origin(state_b, 0)
        origin1 = _find_owned_origin(state_b, 1)
        target = _find_shared_target(state_b, {origin0, origin1})

        state_b = _dispatch_real_fleet(state_b, ego=0, origin_idx=origin0, target_idx=target, fleet_eta=3.0)
        state_b = _dispatch_real_fleet(state_b, ego=1, origin_idx=origin1, target_idx=target, fleet_eta=3.0)

        if num_agents > 2:
            origin2 = _find_owned_origin(state_b, 2)
            state_b = _dispatch_real_fleet(state_b, ego=2, origin_idx=origin2, target_idx=target, fleet_eta=5.0)

        incoming = np.asarray(jax.device_get(state_b.incoming_fleets))
        self.assertGreater(int(incoming.sum()), 0, "expected non-empty incoming bins after real dispatches")
        return state_b

    def test_direct_compressed_builder_matches_dense_then_compress_two_player(self) -> None:
        state_b = self._make_state_with_incoming(num_agents=2)
        feature_dim = obs_feature_dim_for_num_agents(2)

        for ego in (0, 1):
            obs_j = build_observation_batched_jax(
                state_b,
                ego,
                ship_speed=6.0,
                obs_feature_dim=feature_dim,
                normalize_to_p0=False,
            )
            ref = _compressed_to_numpy_dict(compress_observation(obs_jax_to_torch(obs_j)))
            direct = _jax_comp_to_numpy_dict(
                build_compressed_observation_batched_jax(
                    state_b,
                    ego,
                    ship_speed=6.0,
                    obs_feature_dim=feature_dim,
                    normalize_to_p0=False,
                )
            )
            for field in ref:
                with self.subTest(ego=ego, field=field):
                    np.testing.assert_array_equal(direct[field], ref[field])

    def test_direct_compressed_builder_per_ego_matches_dense_then_compress_four_player(self) -> None:
        state_b = self._make_state_with_incoming(num_agents=4)
        feature_dim = obs_feature_dim_for_num_agents(4)
        state_rows = jax.tree.map(lambda x: jnp.repeat(x, 4, axis=0), state_b)
        ego_b = jnp.asarray([0, 1, 2, 3], dtype=jnp.int32)

        obs_j = build_observation_batched_jax_per_ego(
            state_rows,
            ego_b,
            ship_speed=6.0,
            obs_feature_dim=feature_dim,
            normalize_to_p0=True,
        )
        ref = _compressed_to_numpy_dict(compress_observation(obs_jax_to_torch(obs_j)))
        direct = _jax_comp_to_numpy_dict(
            build_compressed_observation_batched_jax_per_ego(
                state_rows,
                ego_b,
                ship_speed=6.0,
                obs_feature_dim=feature_dim,
                normalize_to_p0=True,
            )
        )
        for field in ref:
            with self.subTest(field=field):
                np.testing.assert_array_equal(direct[field], ref[field])


if __name__ == "__main__":
    unittest.main()

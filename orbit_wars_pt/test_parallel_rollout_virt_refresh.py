from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import jax_orbit_wars as jow

from orbit_wars_pt.batched_env import stack_initial_states
from orbit_wars_pt.env_wrapper import OrbitWarsEnvConfig
from orbit_wars_pt.parallel_rollout import (
    _gather_state_rows,
    _refresh_virt_from_state_masked_grouped,
    _refresh_virt_from_state_masked_non_grouped,
)


def _mutate_planet_ships(state_b, row_idx: int, delta: float):
    planets = state_b.planets.at[row_idx, 0, jow.PLANET_SHIPS].add(jnp.asarray(delta, dtype=state_b.planets.dtype))
    return state_b._replace(planets=planets)


class TestVirtRefreshMasked(unittest.TestCase):
    def _two_env_state(self):
        cfg = OrbitWarsEnvConfig(num_agents=2, max_fleets=128, episode_seed=17)
        state_b, _ = stack_initial_states(cfg, num_envs=2, seed_base=17)
        return state_b

    def test_non_grouped_refresh_only_updates_ready_env_rows(self) -> None:
        state_b = self._two_env_state()
        num_envs = int(state_b.planets.shape[0])
        virt_b = jax.tree.map(lambda leaf: jnp.concatenate([leaf] * 2, axis=0), state_b)

        # Simulate accumulated microstep deltas on one ready row and one non-ready row.
        virt_b = _mutate_planet_ships(virt_b, row_idx=0, delta=111.0)
        virt_b = _mutate_planet_ships(virt_b, row_idx=1, delta=222.0)
        virt_b = _mutate_planet_ships(virt_b, row_idx=num_envs, delta=333.0)
        virt_b = _mutate_planet_ships(virt_b, row_idx=num_envs + 1, delta=444.0)

        ready_mask = jnp.asarray([True, False], dtype=jnp.bool_)
        refreshed = _refresh_virt_from_state_masked_non_grouped(virt_b, state_b, ready_mask)

        state_np = jax.device_get(state_b.planets)
        virt_np = jax.device_get(virt_b.planets)
        out_np = jax.device_get(refreshed.planets)

        # Ready env 0 should refresh from state_b for both ego rows.
        np.testing.assert_allclose(out_np[0], state_np[0], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(out_np[num_envs], state_np[0], rtol=0.0, atol=0.0)
        # Non-ready env 1 should preserve prior virt_b mutations for both ego rows.
        np.testing.assert_allclose(out_np[1], virt_np[1], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(out_np[num_envs + 1], virt_np[num_envs + 1], rtol=0.0, atol=0.0)
        self.assertFalse(np.array_equal(out_np[1], state_np[1]))
        self.assertFalse(np.array_equal(out_np[num_envs + 1], state_np[1]))

    def test_grouped_refresh_only_updates_rows_backed_by_ready_envs(self) -> None:
        state_b = self._two_env_state()
        row_env = np.asarray([1, 0, 1, 0], dtype=np.int32)
        virt_b = _gather_state_rows(state_b, jnp.asarray(row_env, dtype=jnp.int32))

        # Mutate rows sourced from both env 0 and env 1.
        virt_b = _mutate_planet_ships(virt_b, row_idx=0, delta=101.0)  # env 1
        virt_b = _mutate_planet_ships(virt_b, row_idx=1, delta=202.0)  # env 0
        virt_b = _mutate_planet_ships(virt_b, row_idx=2, delta=303.0)  # env 1
        virt_b = _mutate_planet_ships(virt_b, row_idx=3, delta=404.0)  # env 0

        ready_mask = jnp.asarray([True, False], dtype=jnp.bool_)
        refreshed = _refresh_virt_from_state_masked_grouped(
            virt_b,
            state_b,
            jnp.asarray(row_env, dtype=jnp.int32),
            ready_mask,
        )

        state_np = jax.device_get(state_b.planets)
        virt_np = jax.device_get(virt_b.planets)
        out_np = jax.device_get(refreshed.planets)

        # Rows backed by ready env 0 should refresh.
        np.testing.assert_allclose(out_np[1], state_np[0], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(out_np[3], state_np[0], rtol=0.0, atol=0.0)
        # Rows backed by non-ready env 1 should keep their mutated virt values.
        np.testing.assert_allclose(out_np[0], virt_np[0], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(out_np[2], virt_np[2], rtol=0.0, atol=0.0)
        self.assertFalse(np.array_equal(out_np[0], state_np[1]))
        self.assertFalse(np.array_equal(out_np[2], state_np[1]))


if __name__ == "__main__":
    unittest.main()

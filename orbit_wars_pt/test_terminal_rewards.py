from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

import jax_orbit_wars as jow
from orbit_wars_pt.batched_env import reward_delta_from_state_pair_batched


def _blank_state(num_agents: int = 2) -> jow.OrbitWarsState:
    state = jow.reset_from_reference(0, num_agents=num_agents, max_fleets=32)
    planets = jnp.zeros_like(state.planets)
    planet_active = jnp.zeros_like(state.planet_active)
    fleets = jnp.zeros_like(state.fleets)
    fleet_active = jnp.zeros_like(state.fleet_active)
    incoming = jnp.zeros_like(state.incoming_fleets)
    fake_incoming = jnp.zeros_like(state.incoming_fake_correction)
    return state._replace(
        planets=planets,
        planet_active=planet_active,
        fleets=fleets,
        fleet_active=fleet_active,
        incoming_fleets=incoming,
        incoming_fake_correction=fake_incoming,
        rewards=jnp.zeros_like(state.rewards),
        done=jnp.asarray(False),
        step_count=jnp.asarray(0, dtype=state.step_count.dtype),
    )


def _planet_row(owner: int, ships: float) -> jnp.ndarray:
    row = jnp.zeros((7,), dtype=jnp.float32)
    row = row.at[jow.PLANET_OWNER].set(float(owner))
    row = row.at[jow.PLANET_SHIPS].set(float(ships))
    return row


def _batched(state: jow.OrbitWarsState) -> jow.OrbitWarsState:
    return jax.tree.map(lambda x: x[None, ...], state)


class TestTerminalRewards(unittest.TestCase):
    def test_terminal_rewards_use_configured_loss_draw_win_values(self) -> None:
        base = _blank_state()
        draw_state = base._replace(
            planets=base.planets.at[0].set(_planet_row(0, 10.0)).at[1].set(_planet_row(1, 10.0)),
            planet_active=base.planet_active.at[0].set(True).at[1].set(True),
            step_count=jnp.asarray(498, dtype=base.step_count.dtype),
        )
        win_state = base._replace(
            planets=base.planets.at[0].set(_planet_row(0, 10.0)),
            planet_active=base.planet_active.at[0].set(True),
            step_count=jnp.asarray(498, dtype=base.step_count.dtype),
        )
        cfg = jow.OrbitWarsConfig(
            reward_terminal_loss=jnp.asarray(0.0, dtype=jnp.float32),
            reward_terminal_draw=jnp.asarray(1.0, dtype=jnp.float32),
            reward_terminal_win=jnp.asarray(2.0, dtype=jnp.float32),
        )

        draw_out = jow.step(draw_state, jow.empty_actions(num_agents=2, max_actions=1), cfg)
        win_out = jow.step(win_state, jow.empty_actions(num_agents=2, max_actions=1), cfg)

        np.testing.assert_allclose(np.asarray(draw_out.rewards[:2]), np.asarray([1.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(np.asarray(win_out.rewards[:2]), np.asarray([2.0, 0.0], dtype=np.float32))

    def test_reward_delta_uses_raw_terminal_values_and_winner_only_time_bonus(self) -> None:
        state = _blank_state()
        next_state = state._replace(
            rewards=jnp.asarray([2.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
            done=jnp.asarray(True),
        )
        draw_next_state = state._replace(
            rewards=jnp.asarray([1.0, 1.0, 0.0, 0.0], dtype=jnp.float32),
            done=jnp.asarray(True),
        )
        ratios_pre = jnp.zeros((1, 4), dtype=jnp.float32)
        coef = jnp.ones((1, 4), dtype=jnp.float32)

        win_delta = reward_delta_from_state_pair_batched(
            _batched(state),
            _batched(next_state),
            ratios_pre,
            reward_ship_mass_share_coef=jnp.zeros((1, 4), dtype=jnp.float32),
            reward_production_share_coef=jnp.zeros((1, 4), dtype=jnp.float32),
            reward_terminal_win_loss_coef=coef,
            reward_terminal_win=2.0,
            reward_time_bonus_coef=coef,
        )
        draw_delta = reward_delta_from_state_pair_batched(
            _batched(state),
            _batched(draw_next_state),
            ratios_pre,
            reward_ship_mass_share_coef=jnp.zeros((1, 4), dtype=jnp.float32),
            reward_production_share_coef=jnp.zeros((1, 4), dtype=jnp.float32),
            reward_terminal_win_loss_coef=coef,
            reward_terminal_win=2.0,
            reward_time_bonus_coef=coef,
        )

        np.testing.assert_allclose(np.asarray(jax.device_get(win_delta))[0, :2], np.asarray([3.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(np.asarray(jax.device_get(draw_delta))[0, :2], np.asarray([1.0, 1.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()

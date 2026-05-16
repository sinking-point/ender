"""Tests for 4-player observation encoding (intended vs current training path).

Reference (spec): ``observation.build_observation`` — ego-centric owner slots
0 = neutral, 1 = self, 2–4 = distinct opponents on planet ``owner_idx`` and in the
incoming survivor plane when ``num_agents > 2``.

Training (``parallel_rollout.collect_parallel_micro_rollouts``):
``observation_jax.build_observation_batched_jax_per_ego`` on each micro-step, and
``observation_jax.build_observation_batched_jax`` for horizon bootstrap values.
Both call ``_build_observation_one_env`` in ``observation_jax.py``.

These tests are expected to **fail** until the JAX builders match the reference.

Run::

    python -m orbit_wars_pt.test_observation_4p
    python -m unittest orbit_wars_pt.test_observation_4p -v
"""

from __future__ import annotations

import os
import unittest
from typing import Dict

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from orbit_wars_pt.batched_env import stack_initial_states
from orbit_wars_pt.constants import (
    FEATURE_DIM_MULTI,
    INCOMING_SURVIVOR_FLAT,
    INCOMING_TA_BINS,
    MAX_PLANETS,
    NUM_OWNER_SLOTS,
    obs_feature_dim_for_num_agents,
)
from orbit_wars_pt.env_wrapper import OrbitWarsEnvConfig
from orbit_wars_pt.observation import build_observation, jax_state_to_numpy
from orbit_wars_pt.observation_jax import build_observation_batched_jax


def _host_obs_padded(state_np, ego: int, ship_speed: float = 6.0) -> Dict[str, np.ndarray]:
    obs = build_observation(state_np, ego, ship_speed=ship_speed)
    target_L = 1 + MAX_PLANETS
    L = len(obs.entity_type)
    if L < target_L:
        pad = target_L - L
        et = np.pad(obs.entity_type, (0, pad), constant_values=0)
        oi = np.pad(obs.owner_idx, (0, pad), constant_values=0)
        ft = np.pad(obs.features, ((0, pad), (0, 0)), constant_values=0.0)
        rp = np.pad(obs.rope_pos, ((0, pad), (0, 0)), constant_values=0.0)
        em = np.pad(obs.entity_mask, (0, pad), constant_values=False)
        pm = np.pad(obs.planet_mask, (0, pad), constant_values=False)
    elif L == target_L:
        et, oi, ft, rp, em, pm = (
            obs.entity_type,
            obs.owner_idx,
            obs.features,
            obs.rope_pos,
            obs.entity_mask,
            obs.planet_mask,
        )
    else:
        raise AssertionError(f"host obs length {L} > {target_L}")
    return {
        "entity_type": np.asarray(et),
        "owner_idx": np.asarray(oi),
        "features": np.asarray(ft),
        "rope_pos": np.asarray(rp),
        "entity_mask": np.asarray(em),
        "planet_mask": np.asarray(pm),
    }


def _jax_obs_one(state_b, ego: int, ship_speed: float, obs_feature_dim: int) -> Dict[str, np.ndarray]:
    obs_jax = build_observation_batched_jax(state_b, ego, ship_speed, obs_feature_dim)
    return {k: np.asarray(jax.device_get(v[0])) for k, v in obs_jax.items()}


def _planet_owner_slots(owner_idx: np.ndarray, planet_mask: np.ndarray) -> np.ndarray:
    """Owner indices on active planet tokens (excludes CLS at index 0)."""

    active = planet_mask & (np.arange(len(planet_mask)) > 0)
    return owner_idx[active].astype(np.int64)


def _four_player_reset(seed: int = 7):
    cfg = OrbitWarsEnvConfig(num_agents=4, max_fleets=128, episode_seed=seed)
    state_b, _ = stack_initial_states(cfg, num_envs=1, seed_base=seed)
    state_np = jax_state_to_numpy(jax.tree.map(lambda x: x[0], state_b))
    return state_b, state_np


class TestFourPlayerObservationReference(unittest.TestCase):
    """Sanity: the host reference builder encodes distinct opponent slots."""

    def test_host_planets_use_distinct_opponent_slots(self) -> None:
        _, state_np = _four_player_reset()
        ego = 0
        host = _host_obs_padded(state_np, ego)
        slots = _planet_owner_slots(host["owner_idx"], host["planet_mask"])
        enemy_slots = set(int(s) for s in slots if s >= 2)
        self.assertGreaterEqual(
            len(enemy_slots),
            3,
            f"ego={ego}: expected opponent owner slots {{2,3,4}} on planets, got {sorted(enemy_slots)}",
        )
        self.assertTrue(
            enemy_slots.issubset({2, 3, 4}),
            f"unexpected opponent slots: {enemy_slots}",
        )

    def test_host_incoming_survivor_plane_uses_opponent_slots(self) -> None:
        state_b, _ = _four_player_reset()
        state_one = jax.tree.map(lambda x: x[0], state_b)
        planets = np.asarray(state_one.planets)
        active = np.asarray(state_one.planet_active)
        target = next(
            (i for i in range(MAX_PLANETS) if active[i] and int(planets[i, 1]) < 0),
            None,
        )
        self.assertIsNotNone(target, "need a neutral planet for incoming test")
        incoming = np.asarray(state_one.incoming_fleets).copy()
        incoming[3, target, 8] = 25
        state_one = state_one._replace(incoming_fleets=jnp.asarray(incoming))
        state_np2 = jax_state_to_numpy(state_one)
        host = _host_obs_padded(state_np2, 0)
        fdim = obs_feature_dim_for_num_agents(4)
        self.assertEqual(fdim, FEATURE_DIM_MULTI)
        surv = host["features"][1 + target, 8 + INCOMING_TA_BINS :].reshape(
            INCOMING_TA_BINS, NUM_OWNER_SLOTS
        )
        winner = int(np.argmax(surv[8]))
        self.assertIn(winner, (2, 3, 4), f"expected opponent slot 2–4, got {winner}")


class TestFourPlayerObservationTrainingPath(unittest.TestCase):
    """``observation_jax`` builders (PPO rollout) should match the host reference."""

    def test_jax_owner_idx_matches_host_on_fresh_4p_reset(self) -> None:
        state_b, state_np = _four_player_reset()
        fdim = obs_feature_dim_for_num_agents(4)
        mismatches = []
        for ego in range(4):
            host = _host_obs_padded(state_np, ego)
            jax_obs = _jax_obs_one(state_b, ego, 6.0, fdim)
            h_oi = host["owner_idx"]
            j_oi = jax_obs["owner_idx"]
            if not np.array_equal(h_oi, j_oi):
                diff_mask = h_oi != j_oi
                n = int(diff_mask.sum())
                host_slots = _planet_owner_slots(h_oi, host["planet_mask"])
                jax_slots = _planet_owner_slots(j_oi, jax_obs["planet_mask"])
                mismatches.append(
                    f"ego={ego}: {n} token owner_idx mismatches; "
                    f"host planet slots={sorted(set(host_slots.tolist()))} "
                    f"jax planet slots={sorted(set(jax_slots.tolist()))}"
                )
        self.assertFalse(mismatches, "JAX obs owner_idx != host reference:\n" + "\n".join(mismatches))

    def test_jax_does_not_collapse_all_enemies_to_slot_2(self) -> None:
        """4p training must not map every opponent planet to owner_idx==2."""

        state_b, _ = _four_player_reset()
        fdim = obs_feature_dim_for_num_agents(4)
        jax_obs = _jax_obs_one(state_b, 0, 6.0, fdim)
        slots = _planet_owner_slots(jax_obs["owner_idx"], jax_obs["planet_mask"])
        enemy_slots = set(int(s) for s in slots if s >= 2)
        self.assertGreater(
            len(enemy_slots),
            1,
            "JAX maps all opponent planets to a single owner slot (expected 2,3,4)",
        )

    def test_jax_incoming_survivor_plane_matches_host_with_two_attackers(self) -> None:
        """When two opponents queue fleets at the same planet/T, survivor slots must differ."""

        state_b, state_np = _four_player_reset()
        state_one = jax.tree.map(lambda x: x[0], state_b)
        planets = np.asarray(state_one.planets)
        active = np.asarray(state_one.planet_active)
        # Neutral planet owned by no one in [-1] or pick ego-0 planet as target
        target = next(
            (i for i in range(MAX_PLANETS) if active[i] and int(planets[i, 1]) < 0),
            None,
        )
        if target is None:
            self.skipTest("no neutral active planet in this seed")
        incoming = np.asarray(state_one.incoming_fleets).copy()
        t_bin = 5
        incoming[1, target, t_bin] = 40  # opponent raw id 1
        incoming[2, target, t_bin] = 10  # opponent raw id 2 (loses interfleet duel)
        state_one = state_one._replace(incoming_fleets=jnp.asarray(incoming))
        state_b2 = jax.tree.map(lambda x: x[None, ...], state_one)
        state_np2 = jax_state_to_numpy(state_one)

        ego = 0
        fdim = obs_feature_dim_for_num_agents(4)
        host = _host_obs_padded(state_np2, ego)
        jax_obs = _jax_obs_one(state_b2, ego, 6.0, fdim)

        surv_off = 8 + INCOMING_TA_BINS
        surv_len = INCOMING_SURVIVOR_FLAT
        h_surv = host["features"][1 + target, surv_off : surv_off + surv_len].reshape(
            INCOMING_TA_BINS, NUM_OWNER_SLOTS
        )
        j_surv = jax_obs["features"][1 + target, surv_off : surv_off + surv_len].reshape(
            INCOMING_TA_BINS, NUM_OWNER_SLOTS
        )
        # Host: winner of bin t_bin should be ego-remapped opponent slot for raw owner 1
        h_winner = int(np.argmax(h_surv[t_bin]))
        j_winner = int(np.argmax(j_surv[t_bin]))
        self.assertNotEqual(h_winner, 0, "host should record a surviving attacker")
        self.assertEqual(
            h_winner,
            j_winner,
            f"survivor slot at planet slot {target} bin {t_bin}: host={h_winner} jax={j_winner}",
        )


if __name__ == "__main__":
    unittest.main()

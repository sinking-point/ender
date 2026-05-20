from __future__ import annotations

import os
import unittest
from typing import Dict

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np

from orbit_wars_pt.batched_env import stack_initial_states
from orbit_wars_pt.constants import obs_feature_dim_for_num_agents
from orbit_wars_pt.env_wrapper import OrbitWarsEnvConfig
from orbit_wars_pt.kaggle_adapter import _obs_tensors_for_state
from orbit_wars_pt.observation_jax import build_observation_batched_jax


def _normalized_obs_one(state_b, ego: int, num_agents: int) -> Dict[str, np.ndarray]:
    obs = build_observation_batched_jax(
        state_b,
        ego,
        6.0,
        obs_feature_dim_for_num_agents(num_agents),
        normalize_to_p0=True,
    )
    return {k: np.asarray(jax.device_get(v[0])) for k, v in obs.items()}


def _canonical_active_input_rows(obs: Dict[str, np.ndarray]) -> np.ndarray:
    active = np.flatnonzero(obs["entity_mask"])
    rows = []
    for idx in active:
        row = np.concatenate(
            [
                np.asarray([obs["entity_type"][idx], obs["owner_idx"][idx]], dtype=np.float32),
                obs["features"][idx].astype(np.float32),
                obs["rope_pos"][idx].astype(np.float32),
            ]
        )
        rows.append(np.round(row, 6))
    out = np.stack(rows, axis=0)
    order = np.lexsort(out.T[::-1])
    return out[order]


def _canonical_active_input_rows_torch(obs: Dict[str, object]) -> np.ndarray:
    arr = {
        k: np.asarray(v.detach().cpu().numpy()[0])  # type: ignore[union-attr]
        for k, v in obs.items()
    }
    return _canonical_active_input_rows(arr)


class TestObservationNormalization(unittest.TestCase):
    def _assert_same_network_inputs(self, base: Dict[str, np.ndarray], other: Dict[str, np.ndarray], label: str) -> None:
        base_rows = _canonical_active_input_rows(base)
        other_rows = _canonical_active_input_rows(other)
        self.assertEqual(base_rows.shape, other_rows.shape, f"{label}: active token count differed")
        np.testing.assert_allclose(base_rows, other_rows, rtol=0.0, atol=1e-5, err_msg=label)

    def test_two_player_initial_inputs_match_modulo_token_order(self) -> None:
        cfg = OrbitWarsEnvConfig(num_agents=2, max_fleets=128, episode_seed=11, normalize_obs_to_p0=True)
        state_b, _ = stack_initial_states(cfg, num_envs=1, seed_base=11)
        base = _normalized_obs_one(state_b, 0, 2)
        other = _normalized_obs_one(state_b, 1, 2)
        self._assert_same_network_inputs(base, other, "2p normalized reset network inputs")

    def test_four_player_initial_inputs_match_modulo_token_order(self) -> None:
        cfg = OrbitWarsEnvConfig(num_agents=4, max_fleets=128, episode_seed=7, normalize_obs_to_p0=True)
        state_b, _ = stack_initial_states(cfg, num_envs=1, seed_base=7)
        base = _normalized_obs_one(state_b, 0, 4)
        for ego in range(1, 4):
            other = _normalized_obs_one(state_b, ego, 4)
            self._assert_same_network_inputs(base, other, f"4p normalized reset network inputs ego={ego}")

    def test_kaggle_adapter_two_player_inputs_match_modulo_token_order(self) -> None:
        cfg = OrbitWarsEnvConfig(num_agents=2, max_fleets=128, episode_seed=11, normalize_obs_to_p0=True)
        state_b, _ = stack_initial_states(cfg, num_envs=1, seed_base=11)
        state = jax.tree.map(lambda x: x[0], state_b)
        base = _canonical_active_input_rows_torch(
            _obs_tensors_for_state(state, 0, device="cpu", normalize_obs_to_p0=True)
        )
        other = _canonical_active_input_rows_torch(
            _obs_tensors_for_state(state, 1, device="cpu", normalize_obs_to_p0=True)
        )
        np.testing.assert_allclose(base, other, rtol=0.0, atol=1e-5)

    def test_kaggle_adapter_four_player_inputs_match_modulo_token_order(self) -> None:
        cfg = OrbitWarsEnvConfig(num_agents=4, max_fleets=128, episode_seed=7, normalize_obs_to_p0=True)
        state_b, _ = stack_initial_states(cfg, num_envs=1, seed_base=7)
        state = jax.tree.map(lambda x: x[0], state_b)
        base = _canonical_active_input_rows_torch(
            _obs_tensors_for_state(state, 0, device="cpu", normalize_obs_to_p0=True)
        )
        for ego in range(1, 4):
            other = _canonical_active_input_rows_torch(
                _obs_tensors_for_state(state, ego, device="cpu", normalize_obs_to_p0=True)
            )
            np.testing.assert_allclose(base, other, rtol=0.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()

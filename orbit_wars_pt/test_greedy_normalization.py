from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import torch

from orbit_wars_pt.kaggle_adapter import (
    _dual_sampling_mode_from_env,
    _dual_greedy_from_env,
    _greedy_from_env,
    _greedy_halt_action_from_logits,
    _model_search_greedy_launch_threshold_from_env,
    _normalize_greedy,
    _normalize_sampling_mode,
    _sampling_mode_from_env,
    SAMPLING_MODE_GREEDY,
    SAMPLING_MODE_MIXED,
    SAMPLING_MODE_STOCHASTIC,
)


class TestGreedyNormalization(unittest.TestCase):
    def test_bool_greedy_applies_to_all_four_seats(self) -> None:
        self.assertEqual(_normalize_greedy(True), {0: True, 1: True, 2: True, 3: True})
        self.assertEqual(_normalize_greedy(False), {0: False, 1: False, 2: False, 3: False})

    def test_mapping_keeps_per_seat_overrides(self) -> None:
        self.assertEqual(_normalize_greedy({2: True}), {0: False, 1: False, 2: True, 3: False})

    def test_sampling_mode_defaults_from_greedy_when_not_set(self) -> None:
        self.assertEqual(
            _normalize_sampling_mode(None, fallback_greedy={0: True, 1: False, 2: True, 3: False}),
            {
                0: SAMPLING_MODE_GREEDY,
                1: SAMPLING_MODE_STOCHASTIC,
                2: SAMPLING_MODE_GREEDY,
                3: SAMPLING_MODE_STOCHASTIC,
            },
        )

    def test_sampling_mode_mapping_keeps_per_seat_overrides(self) -> None:
        self.assertEqual(
            _normalize_sampling_mode({2: "mixed"}, fallback_greedy=False),
            {
                0: SAMPLING_MODE_STOCHASTIC,
                1: SAMPLING_MODE_STOCHASTIC,
                2: SAMPLING_MODE_MIXED,
                3: SAMPLING_MODE_STOCHASTIC,
            },
        )

    def test_env_per_seat_values_override_global_fallback(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ORBIT_WARS_GREEDY": "1",
                "ORBIT_WARS_GREEDY_P2": "0",
            },
            clear=False,
        ):
            self.assertEqual(_greedy_from_env(), {0: True, 1: True, 2: False, 3: True})

    def test_dual_env_can_split_4p_and_2p_modes(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ORBIT_WARS_GREEDY": "0",
                "ORBIT_WARS_GREEDY_2P": "1",
            },
            clear=False,
        ):
            greedy_4p, greedy_2p = _dual_greedy_from_env()
            self.assertEqual(greedy_4p, False)
            self.assertEqual(greedy_2p, True)

    def test_sampling_mode_env_per_seat_values_override_global_fallback(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ORBIT_WARS_SAMPLING_MODE": "stochastic",
                "ORBIT_WARS_SAMPLING_MODE_P2": "mixed",
            },
            clear=False,
        ):
            self.assertEqual(
                _sampling_mode_from_env(),
                {
                    0: SAMPLING_MODE_STOCHASTIC,
                    1: SAMPLING_MODE_STOCHASTIC,
                    2: SAMPLING_MODE_MIXED,
                    3: SAMPLING_MODE_STOCHASTIC,
                },
            )

    def test_dual_sampling_mode_can_split_4p_and_2p_modes(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ORBIT_WARS_SAMPLING_MODE": "stochastic",
                "ORBIT_WARS_SAMPLING_MODE_2P": "mixed",
            },
            clear=False,
        ):
            sampling_mode_4p, sampling_mode_2p = _dual_sampling_mode_from_env()
            self.assertEqual(sampling_mode_4p, SAMPLING_MODE_STOCHASTIC)
            self.assertEqual(sampling_mode_2p, SAMPLING_MODE_MIXED)

    def test_model_search_greedy_launch_threshold_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ORBIT_WARS_MODEL_SEARCH_GREEDY_LAUNCH_THRESHOLD": "0.75",
            },
            clear=False,
        ):
            self.assertEqual(_model_search_greedy_launch_threshold_from_env(), 0.75)

    def test_greedy_halt_action_threshold_biases_toward_halt(self) -> None:
        halt_logits = torch.tensor([0.0, 0.0], dtype=torch.float32)
        self.assertEqual(_greedy_halt_action_from_logits(halt_logits), 0)
        self.assertEqual(_greedy_halt_action_from_logits(halt_logits, launch_threshold=0.75), 1)


if __name__ == "__main__":
    unittest.main()

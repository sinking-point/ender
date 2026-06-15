from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from orbit_wars_pt.kaggle_adapter import (
    KaggleOrbitWarsAgent,
    KaggleOrbitWarsDualPolicyAgent,
    LaunchGeometryInputs,
    ModelSearchSettings,
    RewardSettings,
    _dual_sampling_mode_from_env,
    _dual_greedy_from_env,
    _greedy_from_env,
    _greedy_halt_action_from_logits,
    _launch_probability_meets_threshold,
    _model_search_greedy_launch_threshold_from_env,
    _model_search_launch_probability_threshold_from_env,
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

    def test_model_search_launch_probability_threshold_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ORBIT_WARS_MODEL_SEARCH_LAUNCH_PROB_THRESHOLD": "0.6",
            },
            clear=False,
        ):
            self.assertEqual(_model_search_launch_probability_threshold_from_env(), 0.6)

    def test_greedy_halt_action_threshold_biases_toward_halt(self) -> None:
        halt_logits = torch.tensor([0.0, 0.0], dtype=torch.float32)
        self.assertEqual(_greedy_halt_action_from_logits(halt_logits), 0)
        self.assertEqual(_greedy_halt_action_from_logits(halt_logits, launch_threshold=0.75), 1)

    def test_launch_probability_threshold_uses_launch_softmax_probability(self) -> None:
        halt_logits = torch.tensor([0.0, 0.0], dtype=torch.float32)
        self.assertTrue(_launch_probability_meets_threshold(halt_logits, threshold=0.5))
        self.assertFalse(_launch_probability_meets_threshold(halt_logits, threshold=0.75))

    def test_agent_can_load_distinct_search_checkpoint(self) -> None:
        main_policy = torch.nn.Linear(1, 1)
        search_policy = torch.nn.Linear(1, 1)
        load_calls: list[str] = []

        def fake_load_policy(path, *, device=None, policy_key="policy"):
            load_calls.append(str(path))
            if str(path) == "/resolved/main.pt":
                return (
                    main_policy,
                    torch.device("cpu"),
                    {
                        "population_size": 4,
                        "normalize_obs_to_p0": False,
                        "num_agents": 2,
                        "max_micro_steps": 7,
                    },
                )
            if str(path) == "/resolved/search.pt":
                return (
                    search_policy,
                    torch.device("cpu"),
                    {
                        "population_size": 1,
                        "normalize_obs_to_p0": True,
                        "num_agents": 2,
                        "max_micro_steps": 7,
                    },
                )
            raise AssertionError(path)

        with patch("orbit_wars_pt.kaggle_adapter.resolve_checkpoint_path", side_effect=lambda path: f"/resolved/{path}"), patch(
            "orbit_wars_pt.kaggle_adapter.load_policy",
            side_effect=fake_load_policy,
        ), patch(
            "orbit_wars_pt.kaggle_adapter._maybe_compile_policy_batched_forward_for_inference",
            side_effect=lambda policy: policy,
        ):
            agent = KaggleOrbitWarsAgent(
                "main.pt",
                search_checkpoint_path="search.pt",
                device="cpu",
                population_member=3,
            )

        self.assertEqual(load_calls, ["/resolved/main.pt", "/resolved/search.pt"])
        self.assertIs(agent.policy, main_policy)
        self.assertIs(agent.search_policy, search_policy)
        self.assertEqual(agent.search_checkpoint_path, "/resolved/search.pt")
        self.assertEqual(agent._population_member_for_player(0), 3)
        self.assertIsNone(agent._search_population_member_for_player(0))
        self.assertFalse(agent._search_compiled_forward_warmup_done)
        self.assertTrue(agent.search_normalize_obs_to_p0)

    def test_dual_agent_threads_mode_specific_search_checkpoints(self) -> None:
        with patch("orbit_wars_pt.kaggle_adapter.resolve_checkpoint_path", side_effect=lambda path: f"/resolved/{path}"):
            agent = KaggleOrbitWarsDualPolicyAgent(
                "main4.pt",
                "main2.pt",
                search_checkpoint_4p="search4.pt",
                search_checkpoint_2p="search2.pt",
            )

        with patch("orbit_wars_pt.kaggle_adapter.KaggleOrbitWarsAgent") as delegate_cls:
            agent._build_delegate("4p")
            agent._build_delegate("2p")

        first_kwargs = delegate_cls.call_args_list[0].kwargs
        second_kwargs = delegate_cls.call_args_list[1].kwargs
        self.assertEqual(first_kwargs["search_checkpoint_path"], "/resolved/search4.pt")
        self.assertEqual(second_kwargs["search_checkpoint_path"], "/resolved/search2.pt")

    def test_search_active_keeps_root_sampling_mode_greedy_even_with_launch_threshold(self) -> None:
        agent = object.__new__(KaggleOrbitWarsAgent)
        agent.policy = object()
        agent.device = torch.device("cpu")
        agent.policy_player_count = 2
        agent.normalize_obs_to_p0 = False
        agent.max_fleets = 16
        agent.max_micro_steps = 4
        agent.raycast_rays = 8
        agent.interval_samples_per_span = 4
        agent.target_method = "interval"
        agent.rng = torch.Generator()
        agent.launch_tracker = SimpleNamespace(
            sync_game=lambda *args, **kwargs: None,
            observe_fleets=lambda *args, **kwargs: None,
            check_forecast_vs_raycast=lambda *args, **kwargs: None,
        )
        agent._fleet_arrival_cache = None
        agent._compiled_forward_warmup_done = True
        agent._sanity_warnings = set()
        agent.model_search = ModelSearchSettings(
            horizon_steps=1,
            reward=RewardSettings(),
            min_overage_s=0.0,
            launch_probability_threshold=0.9,
        )
        agent._choose_launch_via_model_search_batched_single_policy = object()
        agent._last_call_timing = None
        agent._step_count_for_obs = lambda obs: 0
        agent._obs_game_key = lambda obs: "game"
        agent._num_agents_for_obs = lambda obs, config: 2
        agent._population_member_for_player = lambda player: None
        agent._sampling_mode_for_player = lambda player: SAMPLING_MODE_STOCHASTIC

        obs = {"player": 0, "fleets": [], "remainingOverageTime": 60.0}
        config = {"shipSpeed": 6.0, "actTimeout": 1.0}
        fake_state = SimpleNamespace(
            num_agents=np.array(2),
            planets=np.zeros((1, 7), dtype=np.float32),
            comet_planet_ids=np.zeros((1,), dtype=np.int32),
        )
        fake_launch_geometry = LaunchGeometryInputs(
            planets=np.zeros((1, 7), dtype=np.float32),
            planet_active=np.zeros((1,), dtype=np.bool_),
            initial_planets=np.zeros((1, 7), dtype=np.float32),
            initial_active=np.zeros((1,), dtype=np.bool_),
            comet_paths=np.zeros((1, 1, 2), dtype=np.float32),
            comet_path_lengths=np.zeros((1,), dtype=np.int32),
            comet_group_active=np.zeros((1,), dtype=np.bool_),
            comet_path_index=np.zeros((1,), dtype=np.int32),
            comet_slots=np.zeros((1,), dtype=np.int32),
            angular_velocity=0.0,
        )

        with patch("orbit_wars_pt.kaggle_adapter.observation_to_state", return_value=fake_state), patch(
            "orbit_wars_pt.kaggle_adapter._launch_geometry_from_obs",
            return_value=fake_launch_geometry,
        ), patch(
            "orbit_wars_pt.kaggle_adapter._check_4p_adapter_sanity",
            return_value=None,
        ), patch(
            "orbit_wars_pt.kaggle_adapter._public_obs_for_player",
            return_value={},
        ), patch(
            "orbit_wars_pt.kaggle_adapter._build_turn_actions_torch_only",
            return_value=[],
        ) as build_actions:
            agent(obs, config)

        self.assertEqual(build_actions.call_args.kwargs["sampling_mode"], SAMPLING_MODE_GREEDY)
        self.assertEqual(build_actions.call_args.kwargs["search_launch_probability_threshold"], 0.9)


if __name__ == "__main__":
    unittest.main()

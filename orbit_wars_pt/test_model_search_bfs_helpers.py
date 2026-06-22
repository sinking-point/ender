from __future__ import annotations

import json
import math
import os
import unittest
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

from orbit_wars_pt.kaggle_adapter import (
    FleetLaunchDebugTracker,
    ModelSearchSettings,
    PlannedLaunchAction,
    RewardSettings,
    SearchPlannedLaunchAction,
    _SearchTreeNode,
    _search_branching_enabled_for_env_step,
    _search_has_deadline,
    _search_should_advance_closed_turn,
    _search_time_scale_from_overage,
    _search_single_root_turn_path,
    _search_uses_turn_end_opponent_samples,
    _launch_geometry_from_obs,
    _refine_interval_launches_in_place,
    _search_branch_indices_from_probs,
    _search_turn_signature,
    observation_to_state,
)


_SELFPLAY_4P_RECORD = Path("records/selfplay_4p.json")


class ModelSearchBfsHelperTests(unittest.TestCase):
    def test_branch_indices_threshold_and_fallback(self) -> None:
        probs = torch.tensor([0.51, 0.34, 0.15], dtype=torch.float32)
        picks = _search_branch_indices_from_probs(
            probs,
            threshold=0.2,
            max_branching=2,
        )
        self.assertEqual(picks, [0, 1])

        fallback = _search_branch_indices_from_probs(
            probs,
            threshold=0.9,
            max_branching=2,
        )
        self.assertEqual(fallback, [0])

    def test_turn_signature_is_order_insensitive(self) -> None:
        blocked = np.zeros((3, 3), dtype=np.bool_)
        a0 = SearchPlannedLaunchAction(
            action=[3.0, 0.25, 12],
            origin_slot=0,
            frac_idx=1,
            target_slot=2,
            planned_send=12,
            policy_hit_tick=3.0,
            true_hit_tick=3.0,
            planets_snapshot=np.zeros((3, 7), dtype=np.float32),
        )
        a1 = SearchPlannedLaunchAction(
            action=[7.0, 1.25, 5],
            origin_slot=1,
            frac_idx=2,
            target_slot=0,
            planned_send=5,
            policy_hit_tick=2.0,
            true_hit_tick=2.0,
            planets_snapshot=np.zeros((3, 7), dtype=np.float32),
        )
        self.assertEqual(
            _search_turn_signature([a0, a1], blocked),
            _search_turn_signature([a1, a0], blocked),
        )

    def test_search_time_scale_from_overage(self) -> None:
        self.assertTrue(math.isclose(_search_time_scale_from_overage({}), 1.0))
        self.assertTrue(math.isclose(_search_time_scale_from_overage({"remainingOverageTime": 60.0}), 1.0))
        self.assertTrue(math.isclose(_search_time_scale_from_overage({"remainingOverageTime": 30.0}), 0.5))
        self.assertTrue(math.isclose(_search_time_scale_from_overage({"remainingOverageTime": -5.0}), 0.0))
        self.assertTrue(math.isclose(_search_time_scale_from_overage({"remainingOverageTime": 90.0}), 1.0))

    def test_branching_can_be_disabled_after_first_env_step(self) -> None:
        settings = ModelSearchSettings(
            horizon_steps=4,
            reward=RewardSettings(),
            branch_after_first_env_step=False,
        )
        self.assertTrue(_search_branching_enabled_for_env_step(settings, search_env_step_from_root=0))
        self.assertFalse(_search_branching_enabled_for_env_step(settings, search_env_step_from_root=1))
        self.assertFalse(_search_branching_enabled_for_env_step(settings, search_env_step_from_root=3))

        settings_all = ModelSearchSettings(
            horizon_steps=4,
            reward=RewardSettings(),
            branch_after_first_env_step=True,
        )
        self.assertTrue(_search_branching_enabled_for_env_step(settings_all, search_env_step_from_root=0))
        self.assertTrue(_search_branching_enabled_for_env_step(settings_all, search_env_step_from_root=1))

    def test_search_can_stop_at_turn_end(self) -> None:
        settings = ModelSearchSettings(
            horizon_steps=4,
            reward=RewardSettings(),
            stop_at_turn_end=True,
        )
        self.assertFalse(_search_should_advance_closed_turn(settings, search_env_step_from_root=0))
        self.assertFalse(_search_has_deadline(settings))
        self.assertTrue(_search_should_advance_closed_turn(settings, search_env_step_from_root=1))

        settings_rollout = ModelSearchSettings(
            horizon_steps=4,
            reward=RewardSettings(),
            stop_at_turn_end=False,
        )
        self.assertTrue(_search_should_advance_closed_turn(settings_rollout, search_env_step_from_root=0))
        self.assertTrue(_search_has_deadline(settings_rollout))

    def test_turn_end_opponent_sampling_only_applies_at_root_turn_end(self) -> None:
        settings = ModelSearchSettings(
            horizon_steps=4,
            reward=RewardSettings(),
            stop_at_turn_end=True,
            turn_end_opponent_samples=4,
        )
        self.assertTrue(_search_uses_turn_end_opponent_samples(settings, search_env_step_from_root=0))
        self.assertFalse(_search_uses_turn_end_opponent_samples(settings, search_env_step_from_root=1))

        settings_no_samples = ModelSearchSettings(
            horizon_steps=4,
            reward=RewardSettings(),
            stop_at_turn_end=True,
            turn_end_opponent_samples=0,
        )
        self.assertFalse(_search_uses_turn_end_opponent_samples(settings_no_samples, search_env_step_from_root=0))

    def test_single_root_turn_path_requires_closed_turn(self) -> None:
        state = np.array(False)
        plan = SearchPlannedLaunchAction(
            action=[5.0, 1.0, 20],
            origin_slot=1,
            frac_idx=2,
            target_slot=3,
            planned_send=20,
            policy_hit_tick=4.0,
            true_hit_tick=4.0,
            planets_snapshot=np.zeros((3, 7), dtype=np.float32),
        )
        open_node = _SearchTreeNode(
            env_public_obs={},
            env_step_start_state=state,
            current_state=state,
            step_count=12,
            search_env_step_from_root=0,
            current_turn_actions=[plan],
            current_micro_idx=1,
            turn_closed=False,
            root_turn_actions=[plan],
            root_turn_complete=False,
            discounted_reward=0.0,
            discount=1.0,
            done=False,
        )
        self.assertIsNone(_search_single_root_turn_path([open_node]))

        closed_node = _SearchTreeNode(
            env_public_obs={},
            env_step_start_state=state,
            current_state=state,
            step_count=12,
            search_env_step_from_root=0,
            current_turn_actions=[plan],
            current_micro_idx=1,
            turn_closed=True,
            root_turn_actions=[plan],
            root_turn_complete=True,
            discounted_reward=0.0,
            discount=1.0,
            done=False,
        )
        self.assertIs(_search_single_root_turn_path([closed_node]), closed_node)

        advanced_node = _SearchTreeNode(
            env_public_obs={},
            env_step_start_state=state,
            current_state=state,
            step_count=13,
            search_env_step_from_root=1,
            current_turn_actions=[],
            current_micro_idx=0,
            turn_closed=False,
            root_turn_actions=[plan],
            root_turn_complete=True,
            discounted_reward=0.0,
            discount=1.0,
            done=False,
        )
        self.assertIs(_search_single_root_turn_path([closed_node, advanced_node]), closed_node)

    def test_refine_interval_launches_preserves_search_true_hit_tick_when_out_of_time(self) -> None:
        if not _SELFPLAY_4P_RECORD.exists():
            raise unittest.SkipTest(f"missing regression record: {_SELFPLAY_4P_RECORD}")

        with patch.dict(os.environ, {"ORBIT_WARS_INTERVAL_GEOMETRY": "tangent"}, clear=False):
            record = json.loads(_SELFPLAY_4P_RECORD.read_text())
            obs = record["steps"][62][3]["observation"]
            state = observation_to_state(
                obs,
                record["configuration"],
                step_count_override=62,
            )
            launch_geometry = _launch_geometry_from_obs(obs, record["configuration"])
            angle = 5.4948489039908415

            tracker = FleetLaunchDebugTracker()
            actions = [[5.0, angle, 219]]
            planned = PlannedLaunchAction(
                action_index=0,
                micro_idx=0,
                origin_slot=5,
                frac_idx=4,
                target_slot=14,
                planned_send=219,
                policy_hit_tick=7.0,
                true_hit_tick=6.0,
                coarse_angle=angle,
                planets_snapshot=np.array(np.asarray(state.planets), copy=True),
                refine_job=None,
            )

            _refine_interval_launches_in_place(
                actions,
                [planned],
                state,
                launch_geometry,
                ship_speed=6.0,
                horizon=24,
                n_rays=256,
                samples_per_span=9,
                launch_tracker=tracker,
                game_step=62,
                ego_player=3,
                deadline_s=perf_counter() - 1.0,
            )

            self.assertEqual(len(tracker._pending), 1)
            rec = tracker._pending[0]
            self.assertTrue(math.isclose(float(rec.true_hit_tick), 6.0))
            self.assertTrue(math.isclose(float(rec.policy_hit_tick), 7.0))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import torch

from orbit_wars_pt.kaggle_adapter import (
    CachedSearchPolicyOutputs,
    CachedSearchRollout,
    KaggleOrbitWarsAgent,
    _action_sequence_match,
)


class TestSearchCacheMetadata(unittest.TestCase):
    def test_action_sequence_match_tolerates_small_float_noise(self) -> None:
        lhs = [[12.0, 1.25, 7], [4.0, -0.5, 3]]
        rhs = [[12.0 + 1e-7, 1.25 - 1e-7, 7.0], [4.0, -0.5, 3]]
        self.assertTrue(_action_sequence_match(lhs, rhs))

    def test_identify_cached_branch_matches_halt_prefix(self) -> None:
        agent = object.__new__(KaggleOrbitWarsAgent)
        cache = CachedSearchRollout(
            game_key="g",
            ego_player=0,
            root_ego_actions=[[3.0, 0.4, 11]],
            root_public_obs={},
            root_state=None,  # type: ignore[arg-type]
            root_step_count=1,
            root_policy_outputs=None,
            transitions=[],
        )
        self.assertEqual(
            agent._identify_cached_branch(
                action_prefix=[[3.0, 0.4, 11]],
                launch_action=[8.0, 1.1, 5],
                cache=cache,
            ),
            "halt",
        )

    def test_identify_cached_branch_matches_launch_prefix(self) -> None:
        agent = object.__new__(KaggleOrbitWarsAgent)
        cache = CachedSearchRollout(
            game_key="g",
            ego_player=0,
            root_ego_actions=[[3.0, 0.4, 11], [8.0, 1.1, 5], [9.0, -0.2, 4]],
            root_public_obs={},
            root_state=None,  # type: ignore[arg-type]
            root_step_count=1,
            root_policy_outputs=CachedSearchPolicyOutputs(
                players=(0, 1),
                halt_logits=torch.zeros((2, 2), dtype=torch.float32),
                value=torch.zeros((2,), dtype=torch.float32),
                pair_mask=torch.zeros((2, 60, 60), dtype=torch.bool),
                origin_frac_logits=torch.zeros((2, 60, 5), dtype=torch.float32),
                origin_frac_mask=torch.zeros((2, 60, 5), dtype=torch.bool),
                planet_hidden=torch.zeros((2, 60, 4), dtype=torch.float32),
                abort_logits=None,
            ),
            transitions=[],
        )
        self.assertEqual(
            agent._identify_cached_branch(
                action_prefix=[[3.0, 0.4, 11]],
                launch_action=[8.0, 1.1, 5],
                cache=cache,
            ),
            "launch",
        )


if __name__ == "__main__":
    unittest.main()

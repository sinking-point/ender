from __future__ import annotations

import copy
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from orbit_wars_pt.kaggle_adapter import (
    CachedSearchPolicyOutputs,
    CachedSearchRollout,
    KaggleOrbitWarsAgent,
    _action_sequence_match,
)

_CACHE_REGRESSION_RECORD = Path("records/79515266.json")
_CACHE_REGRESSION_CHECKPOINT = Path("experiments/30/checkpoints/iter_00001340.pt")
_CACHE_REGRESSION_MAX_STEP = 6


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

    def test_search_cache_matches_uncached_realistic_rollout(self) -> None:
        if not _CACHE_REGRESSION_RECORD.exists():
            raise unittest.SkipTest(f"missing regression record: {_CACHE_REGRESSION_RECORD}")
        if not _CACHE_REGRESSION_CHECKPOINT.exists():
            raise unittest.SkipTest(f"missing regression checkpoint: {_CACHE_REGRESSION_CHECKPOINT}")

        record = json.loads(_CACHE_REGRESSION_RECORD.read_text())
        config = copy.deepcopy(record["configuration"])
        config["agentCount"] = 2
        player = 0
        max_step = min(_CACHE_REGRESSION_MAX_STEP, len(record["steps"]) - 1)

        def run_case(*, disable_cache: bool) -> list[dict[str, object]]:
            with patch.dict(
                os.environ,
                {
                    "ORBIT_WARS_INTERVAL_GEOMETRY": "tangent",
                    "ORBIT_WARS_WARN_UNMATCHED_FLEET": "0",
                },
                clear=False,
            ):
                agent = KaggleOrbitWarsAgent(
                    _CACHE_REGRESSION_CHECKPOINT,
                    device="cpu",
                    sampling_mode="mixed",
                    target_method="interval",
                    seed=0,
                    model_search_adaptive_horizon=True,
                    model_search_adaptive_horizon_offset=2,
                    model_search_min_overage_s=10.0,
                )
                decisions: list[dict[str, object]] = []
                original_choose = agent._choose_launch_via_model_search_batched_single_policy
                original_eval = agent._evaluate_search_branches
                original_score_cache = agent._score_branch_from_cache

                def wrapped_choose(runtime, **kwargs):
                    cached_branch = None
                    if not disable_cache:
                        cache = agent._search_cache_match(runtime, ego_player=kwargs["ego_player"])
                        if cache is not None:
                            cached_branch = agent._identify_cached_branch(
                            action_prefix=kwargs["action_prefix"],
                            launch_action=kwargs["launch_action"],
                            cache=cache,
                        )

                    recorded_eval: dict[str, object] = {}
                    recorded_cache: dict[str, object] = {}

                    def wrapped_eval(*args, **inner_kwargs):
                        out = original_eval(*args, **inner_kwargs)
                        recorded_eval["result"] = out
                        return out

                    def wrapped_score_cache(*args, **inner_kwargs):
                        out = original_score_cache(*args, **inner_kwargs)
                        recorded_cache["result"] = out
                        return out

                    with patch.object(agent, "_evaluate_search_branches", wrapped_eval), patch.object(
                        agent, "_score_branch_from_cache", wrapped_score_cache
                    ):
                        if disable_cache:
                            with patch.object(agent, "_search_cache_match", return_value=None):
                                choice = bool(original_choose(runtime, **kwargs))
                        else:
                            choice = bool(original_choose(runtime, **kwargs))

                    if cached_branch == "halt":
                        cache_score, _cache_transitions = recorded_cache["result"]
                        eval_scores, _eval_traces, _eval_roots = recorded_eval["result"]
                        halt_score = float(cache_score)
                        launch_score = float(eval_scores[1])
                    elif cached_branch == "launch":
                        cache_score, _cache_transitions = recorded_cache["result"]
                        eval_scores, _eval_traces, _eval_roots = recorded_eval["result"]
                        halt_score = float(eval_scores[0])
                        launch_score = float(cache_score)
                    else:
                        eval_scores, _eval_traces, _eval_roots = recorded_eval["result"]
                        halt_score = float(eval_scores[0])
                        launch_score = float(eval_scores[1])

                    decisions.append(
                        {
                            "step": int(runtime.step_count),
                            "choice": choice,
                            "halt_score": halt_score,
                            "launch_score": launch_score,
                        }
                    )
                    return choice

                with patch.object(agent, "_choose_launch_via_model_search_batched_single_policy", wrapped_choose):
                    for step in range(max_step + 1):
                        obs = copy.deepcopy(record["steps"][step][player]["observation"])
                        actions = agent(obs, config)
                        if decisions and int(decisions[-1]["step"]) == step:
                            decisions[-1]["actions"] = actions
                return decisions

        cached = run_case(disable_cache=False)
        uncached = run_case(disable_cache=True)

        self.assertEqual(
            [entry["step"] for entry in cached],
            [entry["step"] for entry in uncached],
        )
        self.assertGreaterEqual(len(cached), 5)
        for cached_entry, uncached_entry in zip(cached, uncached):
            self.assertEqual(cached_entry["actions"], uncached_entry["actions"])
            self.assertEqual(cached_entry["choice"], uncached_entry["choice"])
            self.assertAlmostEqual(
                float(cached_entry["halt_score"]),
                float(uncached_entry["halt_score"]),
                places=6,
            )
            self.assertAlmostEqual(
                float(cached_entry["launch_score"]),
                float(uncached_entry["launch_score"]),
                places=6,
            )


if __name__ == "__main__":
    unittest.main()

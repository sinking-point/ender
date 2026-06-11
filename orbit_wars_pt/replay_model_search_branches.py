"""Replay one searched halt-vs-launch decision from a Kaggle record.

Writes two partial Kaggle-style records containing the exact futures explored by
the step's model-search branch evaluator, plus prints the raw leaf critic values
used at the rollout frontier.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from orbit_wars_pt.kaggle_adapter import (
    CachedSearchTransition,
    KaggleOrbitWarsAgent,
    SearchRuntime,
    _model_search_rollout_horizon,
    _public_obs_for_player,
)


@dataclass
class CapturedBranchSearch:
    choice: bool
    rollout_horizon: int
    halt_score: float
    launch_score: float
    halt_transitions: list[CachedSearchTransition]
    launch_transitions: list[CachedSearchTransition]
    halt_root_actions: list[list[float]]
    launch_root_actions: list[list[float]]
    halt_leaf_value: float | None
    launch_leaf_value: float | None
    halt_leaf_discount: float
    launch_leaf_discount: float


@dataclass
class _RecordedBranchEval:
    score: float
    transitions: list[CachedSearchTransition]
    root_actions: list[list[float]]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _leaf_value_for_branch(
    agent: KaggleOrbitWarsAgent,
    *,
    ego_player: int,
    transitions: list[CachedSearchTransition],
) -> tuple[float | None, float]:
    if not transitions:
        return None, 1.0
    discount = 1.0
    gamma = float(agent.reward_settings.gamma)
    for trans in transitions:
        if bool(trans.done):
            return None, discount
        discount *= gamma
    leaf_state = transitions[-1].state
    leaf_value = float(agent._policy_values_for_states_batched([leaf_state], [int(ego_player)])[0])
    return leaf_value, discount


def _branch_record(
    source_record: dict[str, Any],
    *,
    player: int,
    branch_name: str,
    step_index: int,
    root_actions: list[list[float]],
    transitions: list[CachedSearchTransition],
) -> dict[str, Any]:
    out = copy.deepcopy(source_record)
    num_agents = len(source_record["steps"][0])
    prefix = copy.deepcopy(source_record["steps"][: step_index + 1])
    future_steps: list[list[dict[str, Any]]] = []

    for idx, trans in enumerate(transitions):
        step_entries: list[dict[str, Any]] = []
        rewards_vec = np.asarray(trans.state.rewards).tolist()
        for seat in range(num_agents):
            obs = _public_obs_for_player(
                trans.public_obs,
                player=int(seat),
                step_count=int(trans.step_count),
            )
            entry = {
                "action": [],
                "info": {},
                "observation": obs,
                "reward": 0,
                "status": "DONE" if bool(trans.done) else "ACTIVE",
            }
            if idx == 0 and seat == int(player):
                entry["action"] = copy.deepcopy(root_actions)
                entry["info"] = {
                    "search_branch": branch_name,
                    "search_root_step": int(step_index),
                }
            if bool(trans.done):
                entry["reward"] = rewards_vec[seat] if seat < len(rewards_vec) else 0
            step_entries.append(entry)
        future_steps.append(step_entries)

    out["steps"] = prefix + future_steps
    if future_steps and future_steps[-1][0]["status"] == "DONE":
        out["rewards"] = [entry["reward"] for entry in future_steps[-1]]
    out["search_debug"] = {
        "branch": branch_name,
        "root_step": int(step_index),
        "root_player": int(player),
        "root_actions": copy.deepcopy(root_actions),
        "future_len": len(transitions),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path, help="Kaggle replay JSON to inspect.")
    parser.add_argument("--checkpoint-2p", type=Path, required=True, help="2-player checkpoint.")
    parser.add_argument("--checkpoint-4p", type=Path, default=None, help="4-player checkpoint.")
    parser.add_argument("--player", type=int, default=0, help="Seat to inspect. Default: 0.")
    parser.add_argument("--step", type=int, default=2, help="Env step to inspect. Default: 2.")
    parser.add_argument("--out-dir", type=Path, default=Path("dist/model_search_replay"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sampling-mode", choices=("stochastic", "greedy", "mixed"), default="mixed")
    parser.add_argument("--target-method", choices=("rays", "interval"), default="interval")
    parser.add_argument("--interval-geometry", choices=("sampled", "orthogonal", "tangent"), default="tangent")
    parser.add_argument("--model-search-steps", type=int, default=0)
    parser.add_argument("--model-search-adaptive-horizon", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model-search-adaptive-horizon-offset", type=int, default=2)
    parser.add_argument("--model-search-min-overage-s", type=float, default=10.0)
    parser.add_argument("--model-search-gamma", type=float, default=None)
    parser.add_argument("--model-search-greedy-launch-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0, help="Adapter RNG seed. Default: 0.")
    args = parser.parse_args()

    record = _load_json(args.record.expanduser())
    num_agents = len(record["steps"][0])
    if num_agents not in (2, 4):
        raise SystemExit(f"expected 2 or 4 agents, got {num_agents}")
    if int(args.player) < 0 or int(args.player) >= num_agents:
        raise SystemExit(f"--player must be in [0, {num_agents - 1}]")
    if int(args.step) < 0 or int(args.step) >= len(record["steps"]):
        raise SystemExit(f"--step must be in [0, {len(record['steps']) - 1}]")

    checkpoint = args.checkpoint_2p.expanduser()
    if num_agents == 4:
        if args.checkpoint_4p is None:
            raise SystemExit("--checkpoint-4p is required for 4-player records")
        checkpoint = args.checkpoint_4p.expanduser()

    os.environ["ORBIT_WARS_INTERVAL_GEOMETRY"] = str(args.interval_geometry)
    if args.model_search_greedy_launch_threshold is None:
        os.environ.pop("ORBIT_WARS_MODEL_SEARCH_GREEDY_LAUNCH_THRESHOLD", None)
    else:
        os.environ["ORBIT_WARS_MODEL_SEARCH_GREEDY_LAUNCH_THRESHOLD"] = str(
            float(args.model_search_greedy_launch_threshold)
        )

    agent = KaggleOrbitWarsAgent(
        checkpoint,
        device=str(args.device),
        sampling_mode=str(args.sampling_mode),
        target_method=str(args.target_method),
        seed=int(args.seed),
        model_search_steps=int(args.model_search_steps),
        model_search_gamma=args.model_search_gamma,
        model_search_adaptive_horizon=bool(args.model_search_adaptive_horizon),
        model_search_adaptive_horizon_offset=int(args.model_search_adaptive_horizon_offset),
        model_search_min_overage_s=float(args.model_search_min_overage_s),
    )

    obs = copy.deepcopy(record["steps"][int(args.step)][int(args.player)]["observation"])
    config = copy.deepcopy(record["configuration"])
    config["agentCount"] = num_agents
    capture: dict[str, CapturedBranchSearch] = {}
    original_choose = agent._choose_launch_via_model_search_batched_single_policy
    original_eval = agent._evaluate_search_branches
    original_score_cache = agent._score_branch_from_cache

    def wrapped_choose(
        runtime: SearchRuntime,
        *,
        ego_player: int,
        current_state: Any,
        current_micro_idx: int,
        action_prefix: list[list[float]],
        launch_action: list[float],
        launch_origin_slot: int,
        launch_send: int,
        launch_true_target_slot: int,
        launch_true_hit_tick: float,
        timing: Any = None,
    ) -> bool:
        rollout_horizon = _model_search_rollout_horizon(
            runtime.settings,
            launch_true_hit_tick=float(launch_true_hit_tick),
        )
        recorded_eval: dict[str, Any] = {}
        recorded_cache: dict[str, Any] = {}
        cache = agent._search_cache_match(runtime, ego_player=int(ego_player))
        cached_branch = None
        if cache is not None:
            cached_branch = agent._identify_cached_branch(
                action_prefix=action_prefix,
                launch_action=launch_action,
                cache=cache,
            )

        def wrapped_eval(*eval_args: Any, **eval_kwargs: Any) -> Any:
            result = original_eval(*eval_args, **eval_kwargs)
            recorded_eval["result"] = result
            return result

        def wrapped_score_cache(*cache_args: Any, **cache_kwargs: Any) -> Any:
            result = original_score_cache(*cache_args, **cache_kwargs)
            recorded_cache["result"] = result
            return result

        agent._evaluate_search_branches = wrapped_eval  # type: ignore[method-assign]
        agent._score_branch_from_cache = wrapped_score_cache  # type: ignore[method-assign]
        try:
            choice = bool(
                original_choose(
                    runtime,
                    ego_player=int(ego_player),
                    current_state=current_state,
                    current_micro_idx=int(current_micro_idx),
                    action_prefix=action_prefix,
                    launch_action=launch_action,
                    launch_origin_slot=int(launch_origin_slot),
                    launch_send=int(launch_send),
                    launch_true_target_slot=int(launch_true_target_slot),
                    launch_true_hit_tick=float(launch_true_hit_tick),
                    timing=timing,
                )
            )
        finally:
            agent._evaluate_search_branches = original_eval  # type: ignore[method-assign]
            agent._score_branch_from_cache = original_score_cache  # type: ignore[method-assign]

        halt_eval: _RecordedBranchEval | None = None
        launch_eval: _RecordedBranchEval | None = None
        if cached_branch == "halt":
            cache_score, cache_transitions = recorded_cache["result"]
            eval_scores, eval_traces, eval_roots = recorded_eval["result"]
            halt_eval = _RecordedBranchEval(
                score=float(cache_score),
                transitions=cache_transitions,
                root_actions=copy.deepcopy(cache.root_ego_actions),
            )
            launch_eval = _RecordedBranchEval(
                score=float(eval_scores[1]),
                transitions=eval_traces[1],
                root_actions=eval_roots[1],
            )
        elif cached_branch == "launch":
            cache_score, cache_transitions = recorded_cache["result"]
            eval_scores, eval_traces, eval_roots = recorded_eval["result"]
            halt_eval = _RecordedBranchEval(
                score=float(eval_scores[0]),
                transitions=eval_traces[0],
                root_actions=eval_roots[0],
            )
            launch_eval = _RecordedBranchEval(
                score=float(cache_score),
                transitions=cache_transitions,
                root_actions=copy.deepcopy(cache.root_ego_actions),
            )
        else:
            eval_scores, eval_traces, eval_roots = recorded_eval["result"]
            halt_eval = _RecordedBranchEval(
                score=float(eval_scores[0]),
                transitions=eval_traces[0],
                root_actions=eval_roots[0],
            )
            launch_eval = _RecordedBranchEval(
                score=float(eval_scores[1]),
                transitions=eval_traces[1],
                root_actions=eval_roots[1],
            )

        halt_leaf_value, halt_leaf_discount = _leaf_value_for_branch(
            agent,
            ego_player=int(ego_player),
            transitions=halt_eval.transitions,
        )
        launch_leaf_value, launch_leaf_discount = _leaf_value_for_branch(
            agent,
            ego_player=int(ego_player),
            transitions=launch_eval.transitions,
        )
        capture["result"] = CapturedBranchSearch(
            choice=choice,
            rollout_horizon=int(rollout_horizon),
            halt_score=halt_eval.score,
            launch_score=launch_eval.score,
            halt_transitions=halt_eval.transitions,
            launch_transitions=launch_eval.transitions,
            halt_root_actions=halt_eval.root_actions,
            launch_root_actions=launch_eval.root_actions,
            halt_leaf_value=halt_leaf_value,
            launch_leaf_value=launch_leaf_value,
            halt_leaf_discount=float(halt_leaf_discount),
            launch_leaf_discount=float(launch_leaf_discount),
        )
        return choice

    for warmup_step in range(int(args.step)):
        warmup_obs = copy.deepcopy(record["steps"][warmup_step][int(args.player)]["observation"])
        agent(warmup_obs, config)

    agent._choose_launch_via_model_search_batched_single_policy = wrapped_choose  # type: ignore[method-assign]
    actions = agent(obs, config)
    if "result" not in capture:
        raise SystemExit("search hook did not fire; the selected step may not have performed halt-vs-launch search")

    result = capture["result"]
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    halt_record = _branch_record(
        record,
        player=int(args.player),
        branch_name="halt",
        step_index=int(args.step),
        root_actions=result.halt_root_actions,
        transitions=result.halt_transitions,
    )
    launch_record = _branch_record(
        record,
        player=int(args.player),
        branch_name="launch",
        step_index=int(args.step),
        root_actions=result.launch_root_actions,
        transitions=result.launch_transitions,
    )
    halt_path = out_dir / f"{args.record.stem}.step{int(args.step)}.p{int(args.player)}.halt.json"
    launch_path = out_dir / f"{args.record.stem}.step{int(args.step)}.p{int(args.player)}.launch.json"
    halt_path.write_text(json.dumps(halt_record, separators=(",", ":")), encoding="utf-8")
    launch_path.write_text(json.dumps(launch_record, separators=(",", ":")), encoding="utf-8")

    print(
        json.dumps(
            {
                "step": int(args.step),
                "player": int(args.player),
                "rollout_horizon": int(result.rollout_horizon),
                "actual_actions": actions,
                "halt_root_actions": result.halt_root_actions,
                "launch_root_actions": result.launch_root_actions,
                "halt_score": result.halt_score,
                "launch_score": result.launch_score,
                "halt_leaf_value": result.halt_leaf_value,
                "launch_leaf_value": result.launch_leaf_value,
                "halt_leaf_discount": result.halt_leaf_discount,
                "launch_leaf_discount": result.launch_leaf_discount,
                "choice": "launch" if result.choice else "halt",
                "halt_record": str(halt_path),
                "launch_record": str(launch_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

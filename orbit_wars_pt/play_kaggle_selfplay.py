"""Run one official Kaggle Orbit Wars self-play episode and save the record."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "toJSON"):
        return obj.toJSON()
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return str(obj)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint.pt",
        help=(
            "Policy checkpoint path used by orbit_wars_pt.kaggle_adapter.agent. "
            "This can be a normal training checkpoint or an exploiter-mode checkpoint; "
            "the single-policy selfplay path uses the main policy."
        ),
    )
    parser.add_argument(
        "--checkpoint-p0",
        type=str,
        default=None,
        help="Checkpoint override for player 0. Default: same as --checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-p1",
        type=str,
        default=None,
        help="Checkpoint override for player 1. Default: same as --checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-p2",
        type=str,
        default=None,
        help="Checkpoint override for player 2 in 4-player mode. Default: same as --checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-p3",
        type=str,
        default=None,
        help="Checkpoint override for player 3 in 4-player mode. Default: same as --checkpoint.",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=2,
        choices=(2, 4),
        help="Run a 2-player duel or 4-player FFA self-play episode.",
    )
    parser.add_argument(
        "--main-vs-exploiter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the checkpoint's main policy for one seat and its exploiter policy for the others. "
            "In 2p this is 1 main vs 1 exploiter; in 4p it is 1 main vs 3 exploiters."
        ),
    )
    parser.add_argument(
        "--main-seat",
        type=int,
        default=None,
        help=(
            "Seat index for the main policy when --main-vs-exploiter is set. "
            "Default: sample from --seed (or --agent-seed if --seed is unset)."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default="kaggle_selfplay_record.json",
        help="Path to write the episode record JSON.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional official env seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device for the adapter. Default is cpu for local Kaggle-env runs.",
    )
    parser.add_argument(
        "--member",
        type=int,
        default=None,
        help="Population member used by all seats unless overridden by --member-p0 / --member-p1 / --member-p2 / --member-p3.",
    )
    parser.add_argument(
        "--member-p0",
        type=int,
        default=None,
        help="Population member for player 0. Default: same as --member.",
    )
    parser.add_argument(
        "--member-p1",
        type=int,
        default=None,
        help="Population member for player 1. Default: same as --member.",
    )
    parser.add_argument(
        "--member-p2",
        type=int,
        default=None,
        help="Population member for player 2 in 4-player mode. Default: same as --member.",
    )
    parser.add_argument(
        "--member-p3",
        type=int,
        default=None,
        help="Population member for player 3 in 4-player mode. Default: same as --member.",
    )
    parser.add_argument(
        "--greedy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use argmax action selection for all players unless overridden by "
            "--greedy-p0 / --greedy-p1 / --greedy-p2 / --greedy-p3."
        ),
    )
    parser.add_argument(
        "--greedy-p0",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Greedy (argmax) for player 0. Default: same as --greedy.",
    )
    parser.add_argument(
        "--greedy-p1",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Greedy (argmax) for player 1. Default: same as --greedy.",
    )
    parser.add_argument(
        "--greedy-p2",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Greedy (argmax) for player 2 in 4-player mode. Default: same as --greedy.",
    )
    parser.add_argument(
        "--greedy-p3",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Greedy (argmax) for player 3 in 4-player mode. Default: same as --greedy.",
    )
    parser.add_argument(
        "--agent-seed",
        type=int,
        default=0,
        help="Torch sampling seed used by the adapter when --no-greedy.",
    )
    parser.add_argument(
        "--raycast-rays",
        type=int,
        default=None,
        help="Override discrete ray count for the NumPy first-hit target filter. Default: checkpoint first_hit_n_rays.",
    )
    parser.add_argument(
        "--max-micro-steps",
        type=int,
        default=None,
        help="Override maximum policy micro-actions per turn. Default: checkpoint max_micro_steps.",
    )
    parser.add_argument(
        "--swap-player-view",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic: swap player ids in each agent's observation before the adapter sees it. "
            "Owner labels in planets/fleets are swapped too, so returned actions still launch "
            "from the agent's true physical planets."
        ),
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help="CPU worker threads for Torch/BLAS libraries. Use 0 to leave existing settings alone.",
    )
    parser.add_argument(
        "--timings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print per-agent-call wall-clock timings while the episode is running "
        "(includes adapter internal breakdown from orbit_wars_pt.kaggle_adapter).",
    )
    parser.add_argument(
        "--debug-launch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set ORBIT_WARS_DEBUG_LAUNCH=1: log launch/raycast bookkeeping and fleet matching to stderr.",
    )
    parser.add_argument(
        "--warn-forecast-mismatch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Set ORBIT_WARS_WARN_FORECAST_MISMATCH: warn when _forecast_incoming_fleets "
            "predicts a different planet than launch raycast for a tracked friendly fleet. "
            "Default follows ORBIT_WARS_WARN_OOB_LAUNCHES (on)."
        ),
    )
    parser.add_argument(
        "--interval-geometry",
        choices=("sampled", "orthogonal"),
        default=None,
        help=(
            "Set ORBIT_WARS_INTERVAL_GEOMETRY for interval target_method: "
            "sampled (per-tick hulls) or orthogonal (tangent growing-circle cones)."
        ),
    )
    parser.add_argument(
        "--warn-unmatched-fleet",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Set ORBIT_WARS_WARN_UNMATCHED_FLEET: warn when a friendly fleet in obs "
            "cannot be matched to a LaunchRaycastRecord. "
            "Default follows ORBIT_WARS_WARN_OOB_LAUNCHES (on)."
        ),
    )
    args = parser.parse_args()

    if args.cpu_threads > 0:
        for name in CPU_THREAD_ENV_VARS:
            os.environ[name] = str(int(args.cpu_threads))
        os.environ["ORBIT_WARS_CPU_THREADS"] = str(int(args.cpu_threads))
    else:
        os.environ["ORBIT_WARS_CPU_THREADS"] = "0"

    os.environ["ORBIT_WARS_CHECKPOINT"] = str(Path(args.checkpoint).expanduser())
    os.environ.pop("ORBIT_WARS_CHECKPOINT_4P", None)
    os.environ.pop("ORBIT_WARS_CHECKPOINT_2P", None)
    os.environ["ORBIT_WARS_DEVICE"] = str(args.device)
    if args.member is not None:
        os.environ["ORBIT_WARS_MEMBER"] = str(int(args.member))
    else:
        os.environ.pop("ORBIT_WARS_MEMBER", None)
    member_p0 = args.member if args.member_p0 is None else args.member_p0
    member_p1 = args.member if args.member_p1 is None else args.member_p1
    member_p2 = args.member if args.member_p2 is None else args.member_p2
    member_p3 = args.member if args.member_p3 is None else args.member_p3
    member_env = {
        "ORBIT_WARS_MEMBER_P0": member_p0,
        "ORBIT_WARS_MEMBER_P1": member_p1,
        "ORBIT_WARS_MEMBER_P2": member_p2,
        "ORBIT_WARS_MEMBER_P3": member_p3,
    }
    for key, value in member_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(int(value))
    greedy_p0 = args.greedy if args.greedy_p0 is None else args.greedy_p0
    greedy_p1 = args.greedy if args.greedy_p1 is None else args.greedy_p1
    greedy_p2 = args.greedy if args.greedy_p2 is None else args.greedy_p2
    greedy_p3 = args.greedy if args.greedy_p3 is None else args.greedy_p3
    os.environ["ORBIT_WARS_GREEDY"] = "1" if args.greedy else "0"
    os.environ["ORBIT_WARS_GREEDY_P0"] = "1" if greedy_p0 else "0"
    os.environ["ORBIT_WARS_GREEDY_P1"] = "1" if greedy_p1 else "0"
    os.environ["ORBIT_WARS_GREEDY_P2"] = "1" if greedy_p2 else "0"
    os.environ["ORBIT_WARS_GREEDY_P3"] = "1" if greedy_p3 else "0"
    os.environ["ORBIT_WARS_AGENT_SEED"] = str(int(args.agent_seed))
    if args.raycast_rays is not None:
        os.environ["ORBIT_WARS_RAYCAST_RAYS"] = str(int(args.raycast_rays))
    else:
        os.environ.pop("ORBIT_WARS_RAYCAST_RAYS", None)
    if args.max_micro_steps is not None:
        os.environ["ORBIT_WARS_MAX_MICRO_STEPS"] = str(int(args.max_micro_steps))
    else:
        os.environ.pop("ORBIT_WARS_MAX_MICRO_STEPS", None)
    if args.debug_launch:
        os.environ["ORBIT_WARS_DEBUG_LAUNCH"] = "1"
    if args.warn_forecast_mismatch is not None:
        os.environ["ORBIT_WARS_WARN_FORECAST_MISMATCH"] = "1" if args.warn_forecast_mismatch else "0"
    if args.warn_unmatched_fleet is not None:
        os.environ["ORBIT_WARS_WARN_UNMATCHED_FLEET"] = "1" if args.warn_unmatched_fleet else "0"
    if args.interval_geometry is not None:
        os.environ["ORBIT_WARS_INTERVAL_GEOMETRY"] = str(args.interval_geometry)

    from kaggle_environments import make

    from orbit_wars_pt.interval_geometry_np import format_interval_aim_stats, reset_interval_aim_stats
    from orbit_wars_pt.kaggle_adapter import KaggleOrbitWarsAgent

    def _agent_internal_timing_suffix(dt_wall: float, timing_obj: Any) -> str:
        t = timing_obj
        if t is None:
            return " internal=unavailable"
        micro_sum = t.micro_sum_s()
        slack = dt_wall - t.obs_to_state_s - micro_sum
        target_detail = t.micro_target.format_suffix()
        return (
            " internal["
            f"obs_to_state={t.obs_to_state_s:.4f}s "
            f"micro_iters={t.micro_iters} "
            f"micro_obs_tensors={t.micro_obs_tensors_s:.4f}s "
            f"micro_policy_fwd={t.micro_policy_forward_s:.4f}s "
            f"micro_post_fwd={t.micro_post_forward_s:.4f}s "
            f"micro_raycast={t.micro_raycast_s:.4f}s"
            f"{target_detail} "
            f"micro_target={t.micro_target_s:.4f}s "
            f"micro_book={t.micro_book_s:.4f}s "
            f"micro_sum={micro_sum:.4f}s "
            f"slack={slack:+.4f}s]"
        )

    def _swap_owner(owner: Any) -> Any:
        if owner == 0:
            return 1
        if owner == 1:
            return 0
        return owner

    def _swapped_view_agent(
        base_agent: Callable[[dict[str, Any], Any], list[list[float]]]
    ) -> Callable[[dict[str, Any], Any], list[list[float]]]:
        def wrapped(obs: dict[str, Any], config: Any = None) -> list[list[float]]:
            obs2 = copy.deepcopy(obs)
            if "player" in obs2:
                obs2["player"] = _swap_owner(obs2["player"])
            for planet in obs2.get("planets", []) or []:
                planet[1] = _swap_owner(planet[1])
            for planet in obs2.get("initial_planets", []) or []:
                planet[1] = _swap_owner(planet[1])
            for fleet in obs2.get("fleets", []) or []:
                fleet[1] = _swap_owner(fleet[1])
            return base_agent(obs2, config)

        return wrapped

    def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
        if config is None:
            return default
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    def _timed_agent(
        base_agent: Callable[[dict[str, Any], Any], list[list[float]]],
        *,
        show_internal: bool,
        timing_getter: Callable[[], Any],
    ):
        def wrapped(obs: dict[str, Any], config: Any = None) -> list[list[float]]:
            step = obs.get("step", obs.get("step_count", "?"))
            player = obs.get("player", "?")
            before_overage = obs.get("remainingOverageTime", None)
            act_timeout = _cfg_get(config, "actTimeout", None)
            t0 = time.perf_counter()
            try:
                return base_agent(obs, config)
            finally:
                dt = time.perf_counter() - t0
                overage_spent = 0.0
                if act_timeout is not None:
                    overage_spent = max(0.0, dt - float(act_timeout))
                after_overage = None if before_overage is None else float(before_overage) - overage_spent
                timeout_suffix = ""
                if act_timeout is not None:
                    timeout_suffix = f" actTimeout={float(act_timeout):.3f}s overage_spent={overage_spent:.3f}s"
                overage_suffix = ""
                if before_overage is not None:
                    overage_suffix = f" remainingOverage {float(before_overage):.3f}->{after_overage:.3f}s"
                internal_suffix = _agent_internal_timing_suffix(dt, timing_getter()) if show_internal else ""
                print(
                    f"[timing] step={step} player={player} duration={dt:.6f}s{internal_suffix}"
                    f"{timeout_suffix}{overage_suffix}",
                    flush=True,
                )

        return wrapped

    configuration: dict[str, Any] = {}
    configuration["agentCount"] = int(args.num_agents)
    if args.seed is not None:
        configuration["seed"] = int(args.seed)

    env = make("orbit_wars", configuration=configuration, debug=bool(args.debug))
    greedy_by_seat = [
        greedy_p0,
        greedy_p1,
        greedy_p2,
        greedy_p3,
    ]
    base_checkpoint = Path(args.checkpoint).expanduser()
    checkpoint_by_seat = [
        base_checkpoint if args.checkpoint_p0 is None else Path(args.checkpoint_p0).expanduser(),
        base_checkpoint if args.checkpoint_p1 is None else Path(args.checkpoint_p1).expanduser(),
        base_checkpoint if args.checkpoint_p2 is None else Path(args.checkpoint_p2).expanduser(),
        base_checkpoint if args.checkpoint_p3 is None else Path(args.checkpoint_p3).expanduser(),
    ]
    member_by_seat = [
        member_p0,
        member_p1,
        member_p2,
        member_p3,
    ]
    if args.main_vs_exploiter:
        if args.main_seat is None:
            seat_rng_seed = int(args.seed if args.seed is not None else args.agent_seed)
            main_seat = int(np.random.default_rng(seat_rng_seed).integers(int(args.num_agents)))
        else:
            main_seat = int(args.main_seat)
            if not (0 <= main_seat < int(args.num_agents)):
                raise SystemExit(f"--main-seat must be in [0, {int(args.num_agents) - 1}]")
        print(
            f"[orbit_wars_pt] selfplay main-vs-exploiter mode: main_seat={main_seat} "
            f"base_checkpoint={base_checkpoint}",
            flush=True,
        )
    else:
        main_seat = -1

    unique_checkpoints = {str(checkpoint_by_seat[seat]) for seat in range(int(args.num_agents))}
    if len(unique_checkpoints) > 1:
        checkpoint_summary = ", ".join(
            f"p{seat}={checkpoint_by_seat[seat]}" for seat in range(int(args.num_agents))
        )
        print(f"[orbit_wars_pt] per-seat checkpoints: {checkpoint_summary}", flush=True)

    run_agents: list[Callable[[dict[str, Any], Any], list[list[float]]]] = []
    for seat in range(int(args.num_agents)):
        policy_key = "policy"
        if args.main_vs_exploiter and seat != main_seat:
            policy_key = "exploiter_policy"
        seat_agent = KaggleOrbitWarsAgent(
            checkpoint_by_seat[seat],
            device=str(args.device),
            policy_key=policy_key,
            greedy=bool(greedy_by_seat[seat]),
            population_member=member_by_seat[seat],
            max_micro_steps=(None if args.max_micro_steps is None else int(args.max_micro_steps)),
            seed=int(args.agent_seed) + seat + (1000 if policy_key != "policy" else 0),
            raycast_rays=(None if args.raycast_rays is None else int(args.raycast_rays)),
        )
        run_agent = seat_agent
        if args.swap_player_view:
            run_agent = _swapped_view_agent(run_agent)
        if args.timings:
            run_agent = _timed_agent(
                run_agent,
                show_internal=True,
                timing_getter=lambda inst=seat_agent: getattr(inst, "_last_call_timing", None),
            )
        run_agents.append(run_agent)
    reset_interval_aim_stats()
    env.run(run_agents)

    record = {
        "name": "orbit_wars",
        "configuration": json.loads(json.dumps(env.configuration, default=_json_default)),
        "steps": json.loads(json.dumps(env.steps, default=_json_default)),
    }

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, separators=(",", ":")), encoding="utf-8")
    aim_report = format_interval_aim_stats()
    if aim_report:
        print(aim_report)
    print(f"saved {out_path} ({len(env.steps)} steps)")


if __name__ == "__main__":
    main()

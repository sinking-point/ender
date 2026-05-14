"""Run one official Kaggle Orbit Wars self-play episode and save the record."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

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
        help="Policy checkpoint path used by orbit_wars_pt.kaggle_adapter.agent.",
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
        "--greedy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use argmax action selection instead of rollout-style stochastic sampling.",
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
    args = parser.parse_args()

    if args.cpu_threads > 0:
        for name in CPU_THREAD_ENV_VARS:
            os.environ[name] = str(int(args.cpu_threads))
        os.environ["ORBIT_WARS_CPU_THREADS"] = str(int(args.cpu_threads))
    else:
        os.environ["ORBIT_WARS_CPU_THREADS"] = "0"

    os.environ["ORBIT_WARS_CHECKPOINT"] = str(Path(args.checkpoint).expanduser())
    os.environ["ORBIT_WARS_DEVICE"] = str(args.device)
    os.environ["ORBIT_WARS_GREEDY"] = "1" if args.greedy else "0"
    os.environ["ORBIT_WARS_AGENT_SEED"] = str(int(args.agent_seed))
    if args.raycast_rays is not None:
        os.environ["ORBIT_WARS_RAYCAST_RAYS"] = str(int(args.raycast_rays))
    else:
        os.environ.pop("ORBIT_WARS_RAYCAST_RAYS", None)
    if args.max_micro_steps is not None:
        os.environ["ORBIT_WARS_MAX_MICRO_STEPS"] = str(int(args.max_micro_steps))
    else:
        os.environ.pop("ORBIT_WARS_MAX_MICRO_STEPS", None)

    from kaggle_environments import make

    from agent import agent
    from orbit_wars_pt.kaggle_adapter import get_last_agent_call_timing

    def _agent_internal_timing_suffix(dt_wall: float) -> str:
        t = get_last_agent_call_timing()
        if t is None:
            return " internal=unavailable"
        micro_sum = t.micro_sum_s()
        slack = dt_wall - t.obs_to_state_s - micro_sum
        return (
            " internal["
            f"obs_to_state={t.obs_to_state_s:.4f}s "
            f"micro_iters={t.micro_iters} "
            f"micro_obs_tensors={t.micro_obs_tensors_s:.4f}s "
            f"micro_policy_fwd={t.micro_policy_forward_s:.4f}s "
            f"micro_post_fwd={t.micro_post_forward_s:.4f}s "
            f"micro_raycast={t.micro_raycast_s:.4f}s "
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

    def _swapped_view_agent(obs: dict[str, Any], config: Any = None) -> list[list[float]]:
        obs2 = copy.deepcopy(obs)
        if "player" in obs2:
            obs2["player"] = _swap_owner(obs2["player"])
        for planet in obs2.get("planets", []) or []:
            planet[1] = _swap_owner(planet[1])
        for planet in obs2.get("initial_planets", []) or []:
            planet[1] = _swap_owner(planet[1])
        for fleet in obs2.get("fleets", []) or []:
            fleet[1] = _swap_owner(fleet[1])
        return agent(obs2, config)

    def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
        if config is None:
            return default
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    def _timed_agent(base_agent: Callable[[dict[str, Any], Any], list[list[float]]], *, show_internal: bool):
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
                internal_suffix = _agent_internal_timing_suffix(dt) if show_internal else ""
                print(
                    f"[timing] step={step} player={player} duration={dt:.6f}s{internal_suffix}"
                    f"{timeout_suffix}{overage_suffix}",
                    flush=True,
                )

        return wrapped

    configuration: dict[str, Any] = {}
    if args.seed is not None:
        configuration["seed"] = int(args.seed)

    env = make("orbit_wars", configuration=configuration, debug=bool(args.debug))
    run_agent = _swapped_view_agent if args.swap_player_view else agent
    if args.timings:
        run_agent = _timed_agent(run_agent, show_internal=True)
    env.run([run_agent, run_agent])

    record = {
        "name": "orbit_wars",
        "configuration": json.loads(json.dumps(env.configuration, default=_json_default)),
        "steps": json.loads(json.dumps(env.steps, default=_json_default)),
    }

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, separators=(",", ":")), encoding="utf-8")
    print(f"saved {out_path} ({len(env.steps)} steps)")


if __name__ == "__main__":
    main()

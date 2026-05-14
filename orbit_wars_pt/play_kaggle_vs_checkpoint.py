"""Play one Orbit Wars episode in the official Kaggle env: you vs a policy checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping

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


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _parse_launch_line(line: str) -> tuple[float, float, int] | None:
    """Return (planet_id, angle, ships) or None if line should be ignored."""
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    if s.lower() in {"h", "help", "?"}:
        return None
    parts = re.split(r"[,\s]+", s.strip())
    if len(parts) != 3:
        raise ValueError(f"expected 3 values (planet_id angle ships), got {len(parts)}: {line!r}")
    pid = float(parts[0])
    ang = float(parts[1])
    ships = int(float(parts[2]))
    return pid, ang, ships


def _make_human_agent(
    *,
    human_player: int,
    max_launches: int,
    angle_unit: str,
) -> Callable[[dict[str, Any], Any], list[list[float]]]:
    """Build ``agent(obs, config)`` that reads launches from stdin."""

    def human_agent(obs: Mapping[str, Any], config: Any = None) -> list[list[float]]:
        step = obs.get("step", obs.get("step_count", "?"))
        print(f"\n=== Human turn  player={human_player}  step={step} ===", flush=True)
        planets_raw = obs.get("planets") or []
        print("Your planets:  id    x       y     floor(ships)", flush=True)
        n_mine = 0
        for row in planets_raw:
            if len(row) < 6:
                continue
            pid = int(row[0])
            owner = int(row[1])
            if owner != int(human_player):
                continue
            n_mine += 1
            x, y = float(row[2]), float(row[3])
            ships = math.floor(float(row[5]))
            print(f"  {pid:4d}     {x:7.2f} {y:7.2f}   {ships:5d}", flush=True)
        if n_mine == 0:
            print("  (none listed — you may still pass with an empty line.)", flush=True)

        n_enemy = sum(1 for row in planets_raw if len(row) >= 2 and int(row[1]) != int(human_player))
        print(f"Enemy / neutral planet rows in obs: {n_enemy}", flush=True)

        fleets = obs.get("fleets") or []
        if fleets:
            print(f"Fleets in play: {len(fleets)}", flush=True)

        unit = "radians" if angle_unit == "rad" else "degrees"
        print(
            f"Enter up to {max_launches} launches, one per line:  <planet_id> <angle> <ships>  (angle in {unit})\n"
            "  blank line ends your turn (policy-style micro-turn).\n"
            "  type help for this reminder.",
            flush=True,
        )

        actions: list[list[float]] = []
        avail: dict[int, float] = {}
        for row in planets_raw:
            if len(row) < 6:
                continue
            pid = int(row[0])
            owner = int(row[1])
            if owner != int(human_player):
                continue
            avail[pid] = float(row[5])

        while len(actions) < int(max_launches):
            try:
                raw = input("launch> ").strip()
            except EOFError:
                print("(EOF — ending turn.)", flush=True)
                break
            if not raw:
                break
            if raw.lower() in {"h", "help", "?"}:
                print(
                    f"Format: <planet_id> <angle> <ships>  (angle in {unit}); blank line to finish.",
                    flush=True,
                )
                continue
            try:
                parsed = _parse_launch_line(raw)
            except ValueError as e:
                print(f"  {e}", flush=True)
                continue
            if parsed is None:
                continue
            pid_f, angle_in, ships = parsed
            pid = int(pid_f)
            angle = float(angle_in)
            if angle_unit == "deg":
                angle = math.radians(angle)
            angle = angle % (2.0 * math.pi)

            if pid not in avail:
                print(f"  planet_id {pid} is not yours (or missing from obs).", flush=True)
                continue
            if ships < 1:
                print("  ships must be >= 1.", flush=True)
                continue
            if float(ships) > avail[pid]:
                print(f"  not enough ships on {pid} (have {math.floor(avail[pid])}, need <=).", flush=True)
                continue
            avail[pid] -= float(ships)
            actions.append([float(pid), angle, int(ships)])
            print(f"  queued: id={pid} angle={angle:.4f} rad ships={ships}  (remaining on planet ~{math.floor(avail[pid])})", flush=True)

        return actions

    return human_agent


def main() -> None:
    from orbit_wars_pt.kaggle_local_play import (
        DEFAULT_LOCAL_ACT_TIMEOUT_S,
        DEFAULT_LOCAL_RUN_TIMEOUT_S,
        orbit_wars_local_configuration,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint.pt",
        help="Training checkpoint (.pt) with policy weights for the bot.",
    )
    parser.add_argument(
        "--human-player",
        type=int,
        choices=(0, 1),
        default=0,
        help="Which Kaggle player id you control (the other player runs the checkpoint).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="If set, write episode record JSON (same shape as play_kaggle_selfplay).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional official env seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for the bot policy.",
    )
    parser.add_argument(
        "--greedy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Bot uses argmax actions. Default is off (stochastic, like rollout). Use --greedy for deterministic bot.",
    )
    parser.add_argument(
        "--agent-seed",
        type=int,
        default=0,
        help="Torch sampling seed for the stochastic bot (default).",
    )
    parser.add_argument(
        "--raycast-rays",
        type=int,
        default=None,
        help="Override discrete ray count. Default: from checkpoint training_args.",
    )
    parser.add_argument(
        "--max-micro-steps",
        type=int,
        default=None,
        help="Override bot max micro-actions per turn. Default: checkpoint.",
    )
    parser.add_argument(
        "--max-human-launches",
        type=int,
        default=None,
        help="Max launch lines you may enter per turn (default: same as bot max_micro_steps).",
    )
    parser.add_argument(
        "--angle-unit",
        type=str,
        choices=("rad", "deg"),
        default="rad",
        help="How your typed angle is interpreted.",
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help="CPU worker threads for Torch/BLAS. Use 0 to leave existing settings alone.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        default=False,
        help="Open a Tkinter map instead of typing launches in the terminal.",
    )
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=720,
        help="With --gui: size in pixels of the square board view.",
    )
    parser.add_argument(
        "--act-timeout",
        type=float,
        default=DEFAULT_LOCAL_ACT_TIMEOUT_S,
        help="Kaggle actTimeout (seconds per agent call). Extra think time consumes remainingOverageTime.",
    )
    parser.add_argument(
        "--run-timeout",
        type=float,
        default=DEFAULT_LOCAL_RUN_TIMEOUT_S,
        help="Kaggle runTimeout (wall seconds for the full env.run loop).",
    )
    args = parser.parse_args()

    if getattr(args, "gui", False):
        from orbit_wars_pt.kaggle_vs_checkpoint_gui import run_gui

        run_gui(args)
        return

    if args.cpu_threads > 0:
        for name in CPU_THREAD_ENV_VARS:
            os.environ[name] = str(int(args.cpu_threads))
        os.environ["ORBIT_WARS_CPU_THREADS"] = str(int(args.cpu_threads))
    else:
        os.environ["ORBIT_WARS_CPU_THREADS"] = "0"

    from kaggle_environments import make

    from orbit_wars_pt.kaggle_adapter import KaggleOrbitWarsAgent

    configuration = orbit_wars_local_configuration(
        seed=args.seed,
        act_timeout_s=float(args.act_timeout),
        run_timeout_s=float(args.run_timeout),
    )

    ckpt_path = Path(args.checkpoint).expanduser()
    bot = KaggleOrbitWarsAgent(
        ckpt_path,
        device=args.device,
        greedy=bool(args.greedy),
        max_micro_steps=args.max_micro_steps,
        seed=int(args.agent_seed),
        raycast_rays=args.raycast_rays,
    )
    max_human = int(args.max_human_launches) if args.max_human_launches is not None else int(bot.max_micro_steps)
    human_player = int(args.human_player)
    bot_player = 1 - human_player
    human_fn = _make_human_agent(
        human_player=human_player,
        max_launches=max_human,
        angle_unit=str(args.angle_unit),
    )

    agents: list[Callable[[dict[str, Any], Any], list[list[float]]]]
    if human_player == 0:
        agents = [human_fn, bot]
    else:
        agents = [bot, human_fn]

    print(
        f"You are player {human_player}; checkpoint bot is player {bot_player}.\n"
        f"Checkpoint: {ckpt_path}\n"
        f"device={args.device} greedy_bot={bool(args.greedy)} max_bot_micro_steps={int(bot.max_micro_steps)}",
        flush=True,
    )

    env = make("orbit_wars", configuration=configuration, debug=bool(args.debug))
    env.run(agents)

    if args.out:
        record = {
            "name": "orbit_wars",
            "configuration": json.loads(json.dumps(env.configuration, default=_json_default)),
            "steps": json.loads(json.dumps(env.steps, default=_json_default)),
        }
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, separators=(",", ":")), encoding="utf-8")
        print(f"saved {out_path} ({len(env.steps)} steps)", flush=True)

    print("Episode finished.", flush=True)


if __name__ == "__main__":
    main()

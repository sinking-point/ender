#!/usr/bin/env python3
"""Run an incremental tournament over packaged Orbit Wars submissions."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


STATE_VERSION = 1
LOG_OFFSET = 2  # kaggle_environments keeps two leading empty log slots before step-0 agent calls.


@dataclass(frozen=True)
class AgentSpec:
    name: str
    source_path: str
    source_signature: str
    bundle_dir: str


@dataclass(frozen=True)
class MatchTask:
    match_key: str
    match_dir: str
    num_players: int
    repeat_index: int
    seed: int
    combo_agents: tuple[str, ...]
    seat_agents: tuple[str, ...]
    seat_main_paths: tuple[str, ...]
    act_timeout_s: float
    run_timeout_s: float


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "toJSON"):
        return obj.toJSON()
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return str(obj)


def _safe_name(raw: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw.strip())
    safe = safe.strip(".-")
    if not safe:
        raise ValueError(f"invalid empty agent name derived from {raw!r}")
    return safe


def _strip_archive_suffix(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".tar.gz"):
        return name[:-7]
    if lower.endswith(".tgz"):
        return name[:-4]
    return Path(name).stem


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, row: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, separators=(",", ":")))
        f.write("\n")


def _load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.is_file():
        return {
            "state_version": STATE_VERSION,
            "num_players": None,
            "agents": {},
            "matches": {},
        }
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if int(payload.get("state_version", 0)) != STATE_VERSION:
        raise SystemExit(
            f"unsupported tournament state version in {state_path}: "
            f"{payload.get('state_version')!r} != {STATE_VERSION}"
        )
    payload.setdefault("agents", {})
    payload.setdefault("matches", {})
    payload.setdefault("num_players", None)
    return payload


def _signature_for_source(path: Path) -> str:
    stat = path.stat()
    raw = json.dumps(
        {
            "path": str(path.resolve()),
            "is_dir": path.is_dir(),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _discover_agents(agent_args: Sequence[str], agents_dir: Path | None) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    seen_names: set[str] = set()

    def add(name: str, path: Path) -> None:
        safe = _safe_name(name)
        if safe in seen_names:
            raise SystemExit(f"duplicate agent name: {safe}")
        seen_names.add(safe)
        found.append((safe, path.expanduser().resolve()))

    for raw in agent_args:
        if "=" in raw:
            name, path_str = raw.split("=", 1)
            add(name, Path(path_str))
        else:
            path = Path(raw)
            add(_strip_archive_suffix(path.name), path)

    if agents_dir is not None:
        base = agents_dir.expanduser().resolve()
        if not base.is_dir():
            raise SystemExit(f"--agents-dir is not a directory: {base}")
        children = sorted(base.iterdir())
        for child in children:
            if child.name.startswith("."):
                continue
            if child.is_dir() and (child / "main.py").is_file():
                add(child.name, child)
                continue
            lower = child.name.lower()
            if child.is_file() and (lower.endswith(".tar.gz") or lower.endswith(".tgz")):
                add(_strip_archive_suffix(child.name), child)

    if not found:
        raise SystemExit("no agents found; pass --agent NAME=PATH and/or --agents-dir")
    return found


def _prepare_bundle_cache(out_dir: Path, name: str, source: Path) -> Path:
    cache_root = out_dir / "bundle-cache"
    signature = _signature_for_source(source)
    cache_dir = cache_root / f"{name}-{signature}"
    meta_path = cache_dir / ".source.json"
    if cache_dir.is_dir() and (cache_dir / "main.py").is_file() and meta_path.is_file():
        return cache_dir.resolve()

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if not (source / "main.py").is_file():
            raise SystemExit(f"bundle directory is missing main.py: {source}")
        for item in source.iterdir():
            dest = cache_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
    else:
        lower = source.name.lower()
        if not (lower.endswith(".tar.gz") or lower.endswith(".tgz")):
            raise SystemExit(f"unsupported bundle file type for {source}; expected .tar.gz or .tgz")
        with tarfile.open(source, "r:gz") as tf:
            tf.extractall(cache_dir)
        if not (cache_dir / "main.py").is_file():
            raise SystemExit(f"archive bundle did not extract a root main.py: {source}")

    _atomic_write_json(
        meta_path,
        {
            "name": name,
            "source_path": str(source),
            "source_signature": signature,
        },
    )
    return cache_dir.resolve()


def _register_agents(
    *,
    state: dict[str, Any],
    out_dir: Path,
    discovered: Sequence[tuple[str, Path]],
) -> dict[str, AgentSpec]:
    existing = dict(state.get("agents", {}))
    specs: dict[str, AgentSpec] = {}
    for name, source in discovered:
        if not source.exists():
            raise SystemExit(f"agent source does not exist: {source}")
        signature = _signature_for_source(source)
        cache_dir = _prepare_bundle_cache(out_dir, name, source)
        prior = existing.get(name)
        if prior is not None and prior.get("source_signature") != signature:
            raise SystemExit(
                f"agent {name!r} already exists in this tournament with a different source signature; "
                "use a new agent name or a new output directory"
            )
        spec = AgentSpec(
            name=name,
            source_path=str(source),
            source_signature=signature,
            bundle_dir=str(cache_dir),
        )
        specs[name] = spec
        existing[name] = {
            "source_path": spec.source_path,
            "source_signature": spec.source_signature,
            "bundle_dir": spec.bundle_dir,
        }
    state["agents"] = existing
    return specs


def _stable_seed(seed_base: int, num_players: int, combo_agents: Sequence[str], repeat_index: int) -> int:
    raw = json.dumps(
        {
            "seed_base": int(seed_base),
            "num_players": int(num_players),
            "combo_agents": list(combo_agents),
            "repeat_index": int(repeat_index),
        },
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _seat_assignment(combo_agents: Sequence[str], seed: int) -> tuple[str, ...]:
    items = list(combo_agents)
    keyed = []
    for idx, name in enumerate(items):
        raw = f"{seed}:{idx}:{name}".encode("utf-8")
        keyed.append((hashlib.sha256(raw).hexdigest(), name))
    keyed.sort()
    return tuple(name for _, name in keyed)


def _match_key(num_players: int, combo_agents: Sequence[str], repeat_index: int) -> str:
    combo_part = "__".join(combo_agents)
    return f"{num_players}p__{combo_part}__g{int(repeat_index):04d}"


def _build_tasks(
    *,
    state: dict[str, Any],
    num_players: int,
    matches_per_combo: int,
    seed_base: int,
    out_dir: Path,
    act_timeout_s: float,
    run_timeout_s: float,
) -> list[MatchTask]:
    agent_names = sorted(state["agents"].keys())
    tasks: list[MatchTask] = []
    matches = state.get("matches", {})
    for combo in itertools.combinations(agent_names, int(num_players)):
        combo_agents = tuple(combo)
        for repeat_index in range(int(matches_per_combo)):
            key = _match_key(int(num_players), combo_agents, int(repeat_index))
            prior = matches.get(key)
            if prior is not None and prior.get("status") == "completed":
                continue
            seed = _stable_seed(int(seed_base), int(num_players), combo_agents, int(repeat_index))
            seat_agents = _seat_assignment(combo_agents, seed)
            seat_paths = tuple(
                str(Path(state["agents"][agent]["bundle_dir"]).resolve() / "main.py")
                for agent in seat_agents
            )
            task = MatchTask(
                match_key=key,
                match_dir=str((out_dir / "matches" / key).resolve()),
                num_players=int(num_players),
                repeat_index=int(repeat_index),
                seed=int(seed),
                combo_agents=combo_agents,
                seat_agents=seat_agents,
                seat_main_paths=seat_paths,
                act_timeout_s=float(act_timeout_s),
                run_timeout_s=float(run_timeout_s),
            )
            tasks.append(task)
    return tasks


def _score_from_observation(obs: dict[str, Any], player: int) -> int:
    total = 0
    for planet in obs.get("planets", []) or []:
        if len(planet) >= 6 and int(planet[1]) == int(player):
            total += int(round(float(planet[5])))
    for fleet in obs.get("fleets", []) or []:
        if len(fleet) >= 7 and int(fleet[1]) == int(player):
            total += int(round(float(fleet[6])))
    return total


def _rank_from_scores(scores: Sequence[int]) -> list[float]:
    unique = sorted(set(int(s) for s in scores), reverse=True)
    rank_by_score = {score: float(idx + 1) for idx, score in enumerate(unique)}
    return [rank_by_score[int(score)] for score in scores]


def _summarize_match(record: dict[str, Any], seat_agents: Sequence[str]) -> dict[str, Any]:
    steps = record.get("steps") or []
    if not steps:
        raise RuntimeError("record has no steps")
    final_step = steps[-1]
    rewards: list[float | None] = []
    statuses: list[str | None] = []
    scores: list[int] = []
    for seat, entry in enumerate(final_step[: len(seat_agents)]):
        obs = dict(entry.get("observation") or {})
        player = int(obs.get("player", seat))
        rewards.append(None if entry.get("reward") is None else float(entry.get("reward")))
        statuses.append(entry.get("status"))
        scores.append(_score_from_observation(obs, player))
    max_score = max(scores)
    first_place_seats = [idx for idx, score in enumerate(scores) if score == max_score]
    ranks = _rank_from_scores(scores)
    return {
        "seat_agents": list(seat_agents),
        "scores": scores,
        "ranks": ranks,
        "rewards": rewards,
        "statuses": statuses,
        "first_place_seats": first_place_seats,
        "first_place_agents": [seat_agents[idx] for idx in first_place_seats],
    }


def _save_agent_logs(match_dir: Path, logs: list[Any], num_players: int) -> None:
    logs_dir = match_dir / "agent-logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    seat_jsonl = [logs_dir / f"seat-{seat}.jsonl" for seat in range(int(num_players))]
    seat_stdout = [logs_dir / f"seat-{seat}.stdout.txt" for seat in range(int(num_players))]
    seat_stderr = [logs_dir / f"seat-{seat}.stderr.txt" for seat in range(int(num_players))]
    for path in [*seat_jsonl, *seat_stdout, *seat_stderr]:
        path.write_text("", encoding="utf-8")

    for log_index, row in enumerate(logs):
        if not isinstance(row, list):
            continue
        step = log_index - LOG_OFFSET
        for seat, entry in enumerate(row[: int(num_players)]):
            if not isinstance(entry, dict):
                continue
            out_row = {
                "log_index": int(log_index),
                "step": (int(step) if step >= 0 else None),
                "duration": entry.get("duration"),
                "stdout": entry.get("stdout", ""),
                "stderr": entry.get("stderr", ""),
            }
            _append_jsonl(seat_jsonl[seat], out_row)
            if out_row["stdout"]:
                with seat_stdout[seat].open("a", encoding="utf-8") as f:
                    f.write(str(out_row["stdout"]))
            if out_row["stderr"]:
                with seat_stderr[seat].open("a", encoding="utf-8") as f:
                    f.write(str(out_row["stderr"]))


def _run_match_worker(task: MatchTask) -> dict[str, Any]:
    match_dir = Path(task.match_dir)
    match_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = match_dir / "metadata.json"
    record_path = match_dir / "record.json"
    error_path = match_dir / "error.txt"
    t0 = time.perf_counter()
    try:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                from kaggle_environments import make

                env = make(
                    "orbit_wars",
                    configuration={
                        "agentCount": int(task.num_players),
                        "seed": int(task.seed),
                        "actTimeout": float(task.act_timeout_s),
                        "runTimeout": float(task.run_timeout_s),
                    },
                    debug=False,
                )
                env.run(list(task.seat_main_paths))
        record = {
            "name": "orbit_wars",
            "configuration": json.loads(json.dumps(env.configuration, default=_json_default)),
            "steps": json.loads(json.dumps(env.steps, default=_json_default)),
        }
        logs = json.loads(json.dumps(env.logs, default=_json_default))
        record_path.write_text(json.dumps(record, separators=(",", ":")), encoding="utf-8")
        _save_agent_logs(match_dir, logs, int(task.num_players))
        summary = _summarize_match(record, task.seat_agents)
        wall_s = time.perf_counter() - t0
        result = {
            "match_key": task.match_key,
            "status": "completed",
            "num_players": int(task.num_players),
            "repeat_index": int(task.repeat_index),
            "seed": int(task.seed),
            "combo_agents": list(task.combo_agents),
            "seat_agents": list(task.seat_agents),
            "record_path": str(record_path),
            "match_dir": str(match_dir),
            "wall_time_s": round(float(wall_s), 6),
            "summary": summary,
        }
        _atomic_write_json(metadata_path, result)
        return result
    except Exception:
        wall_s = time.perf_counter() - t0
        tb = traceback.format_exc()
        error_path.write_text(tb, encoding="utf-8")
        result = {
            "match_key": task.match_key,
            "status": "failed",
            "num_players": int(task.num_players),
            "repeat_index": int(task.repeat_index),
            "seed": int(task.seed),
            "combo_agents": list(task.combo_agents),
            "seat_agents": list(task.seat_agents),
            "match_dir": str(match_dir),
            "wall_time_s": round(float(wall_s), 6),
            "error_path": str(error_path),
        }
        _atomic_write_json(metadata_path, result)
        return result


def _fit_bradley_terry(players: Sequence[str], comparisons: Sequence[tuple[str, str, float]]) -> dict[str, float]:
    ids = list(players)
    if not ids:
        return {}
    index = {name: idx for idx, name in enumerate(ids)}
    n = len(ids)
    wins = [0.0] * n
    n_ij = [[0.0] * n for _ in range(n)]
    for left, right, score_left in comparisons:
        i = index[left]
        j = index[right]
        wins[i] += float(score_left)
        wins[j] += 1.0 - float(score_left)
        n_ij[i][j] += 1.0
        n_ij[j][i] += 1.0

    strength = [1.0] * n
    for _ in range(500):
        prev = list(strength)
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if i == j or n_ij[i][j] <= 0.0:
                    continue
                denom += n_ij[i][j] / (strength[i] + strength[j])
            if denom > 0.0 and wins[i] > 0.0:
                strength[i] = wins[i] / denom
            elif wins[i] <= 0.0:
                strength[i] = 1e-12
            strength[i] = max(strength[i], 1e-12)
        mean_strength = sum(strength) / float(n)
        strength = [s / mean_strength for s in strength]
        delta = max(abs(math.log(s) - math.log(p)) for s, p in zip(strength, prev))
        if delta < 1e-8:
            break

    log_strength = [math.log(s) for s in strength]
    mean_log = sum(log_strength) / float(n)
    return {ids[i]: log_strength[i] - mean_log for i in range(n)}


def _build_rankings(state: dict[str, Any]) -> dict[str, Any]:
    agent_names = sorted(state["agents"].keys())
    by_agent: dict[str, dict[str, Any]] = {
        name: {
            "agent": name,
            "matches": 0,
            "pairwise_games": 0,
            "pairwise_wins": 0.0,
            "pairwise_draws": 0.0,
            "pairwise_losses": 0.0,
            "first_places": 0.0,
            "avg_rank_sum": 0.0,
            "avg_score_sum": 0.0,
        }
        for name in agent_names
    }
    comparisons: list[tuple[str, str, float]] = []

    completed = [m for m in state.get("matches", {}).values() if m.get("status") == "completed"]
    for match in completed:
        summary = match.get("summary") or {}
        seat_agents = list(summary.get("seat_agents") or [])
        scores = list(summary.get("scores") or [])
        ranks = list(summary.get("ranks") or [])
        first_place_agents = set(summary.get("first_place_agents") or [])
        for seat, agent in enumerate(seat_agents):
            rec = by_agent[agent]
            rec["matches"] += 1
            rec["avg_rank_sum"] += float(ranks[seat])
            rec["avg_score_sum"] += float(scores[seat])
            if agent in first_place_agents:
                rec["first_places"] += 1.0 / float(len(first_place_agents))
        for i in range(len(seat_agents)):
            for j in range(i + 1, len(seat_agents)):
                ai = seat_agents[i]
                aj = seat_agents[j]
                si = int(scores[i])
                sj = int(scores[j])
                score_i = 0.5
                if si > sj:
                    score_i = 1.0
                elif si < sj:
                    score_i = 0.0
                comparisons.append((ai, aj, score_i))
                comparisons.append((aj, ai, 1.0 - score_i))
                by_agent[ai]["pairwise_games"] += 1
                by_agent[aj]["pairwise_games"] += 1
                if score_i == 1.0:
                    by_agent[ai]["pairwise_wins"] += 1.0
                    by_agent[aj]["pairwise_losses"] += 1.0
                elif score_i == 0.0:
                    by_agent[aj]["pairwise_wins"] += 1.0
                    by_agent[ai]["pairwise_losses"] += 1.0
                else:
                    by_agent[ai]["pairwise_draws"] += 1.0
                    by_agent[aj]["pairwise_draws"] += 1.0

    # Use only one directed comparison per pairwise game in the BT fit.
    fit_rows = comparisons[::2]
    strengths = _fit_bradley_terry(agent_names, fit_rows)
    rankings = []
    for name in agent_names:
        rec = by_agent[name]
        matches = max(1, int(rec["matches"])) if rec["matches"] else 0
        strength = strengths.get(name, 0.0)
        elo = 1500.0 + (400.0 / math.log(10.0)) * float(strength)
        avg_rank = rec["avg_rank_sum"] / float(matches) if matches else None
        avg_score = rec["avg_score_sum"] / float(matches) if matches else None
        first_place_rate = rec["first_places"] / float(matches) if matches else None
        pairwise_games = rec["pairwise_games"]
        rankings.append(
            {
                "agent": name,
                "elo": round(float(elo), 3),
                "matches": int(rec["matches"]),
                "pairwise_games": int(pairwise_games),
                "pairwise_wins": round(float(rec["pairwise_wins"]), 3),
                "pairwise_draws": round(float(rec["pairwise_draws"]), 3),
                "pairwise_losses": round(float(rec["pairwise_losses"]), 3),
                "first_place_rate": (round(float(first_place_rate), 6) if first_place_rate is not None else None),
                "avg_rank": (round(float(avg_rank), 6) if avg_rank is not None else None),
                "avg_score": (round(float(avg_score), 6) if avg_score is not None else None),
            }
        )
    rankings.sort(key=lambda row: (-float(row["elo"]), row["agent"]))
    return {
        "num_players": state.get("num_players"),
        "completed_matches": len(completed),
        "agents": rankings,
    }


def _write_rankings(out_dir: Path, rankings_payload: dict[str, Any]) -> None:
    _atomic_write_json(out_dir / "rankings.json", rankings_payload)
    csv_path = out_dir / "rankings.csv"
    rows = list(rankings_payload.get("agents") or [])
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "agent",
                "elo",
                "matches",
                "pairwise_games",
                "pairwise_wins",
                "pairwise_draws",
                "pairwise_losses",
                "first_place_rate",
                "avg_rank",
                "avg_score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"Tournament rankings ({rankings_payload.get('num_players')}p, completed_matches={rankings_payload.get('completed_matches')})",
        "",
        "rank  elo      agent                           matches  1st_rate  avg_rank  pairwise W-D-L",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            f"{idx:>4}  {row['elo']:>7}  {row['agent']:<30}  "
            f"{row['matches']:>7}  "
            f"{('' if row['first_place_rate'] is None else f'{row['first_place_rate']:.3f}'):>8}  "
            f"{('' if row['avg_rank'] is None else f'{row['avg_rank']:.3f}'):>8}  "
            f"{row['pairwise_wins']:.1f}-{row['pairwise_draws']:.1f}-{row['pairwise_losses']:.1f}"
        )
    (out_dir / "rankings.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent",
        action="append",
        default=[],
        help="Agent bundle as NAME=PATH or PATH. May be passed multiple times.",
    )
    parser.add_argument(
        "--agents-dir",
        type=Path,
        default=None,
        help="Optional directory to scan for bundle dirs (with main.py) and .tar.gz/.tgz files.",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Tournament output directory.")
    parser.add_argument("--num-players", type=int, choices=(2, 4), required=True, help="Tournament type.")
    parser.add_argument(
        "--matches-per-combo",
        type=int,
        default=3,
        help="Random-seed repeats for each unordered agent combination.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help="Base integer folded into deterministic per-match seed generation.",
    )
    parser.add_argument(
        "--max-active-seats",
        type=int,
        default=8,
        help="Concurrency budget in active seats. Example: 8 means up to 4 concurrent 2p games or 2 concurrent 4p games.",
    )
    parser.add_argument(
        "--act-timeout",
        type=float,
        default=1.0,
        help="Per-agent actTimeout passed to the local Kaggle environment.",
    )
    parser.add_argument(
        "--run-timeout",
        type=float,
        default=1200.0,
        help="Per-match runTimeout passed to the local Kaggle environment.",
    )
    args = parser.parse_args()

    if int(args.matches_per_combo) <= 0:
        raise SystemExit("--matches-per-combo must be positive")
    if int(args.max_active_seats) < int(args.num_players):
        raise SystemExit("--max-active-seats must be at least --num-players")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "tournament-state.json"
    results_path = out_dir / "match-results.jsonl"
    state = _load_state(state_path)

    existing_num_players = state.get("num_players")
    if existing_num_players is None:
        state["num_players"] = int(args.num_players)
    elif int(existing_num_players) != int(args.num_players):
        raise SystemExit(
            f"existing tournament at {out_dir} is for {existing_num_players} players, "
            f"not {args.num_players}; use a separate output directory"
        )

    discovered = _discover_agents(args.agent, args.agents_dir)
    _register_agents(state=state, out_dir=out_dir, discovered=discovered)
    _atomic_write_json(state_path, state)

    tasks = _build_tasks(
        state=state,
        num_players=int(args.num_players),
        matches_per_combo=int(args.matches_per_combo),
        seed_base=int(args.seed_base),
        out_dir=out_dir,
        act_timeout_s=float(args.act_timeout),
        run_timeout_s=float(args.run_timeout),
    )

    if not tasks:
        rankings = _build_rankings(state)
        _write_rankings(out_dir, rankings)
        print(f"no new matches to run; rankings refreshed at {out_dir / 'rankings.json'}")
        return

    max_workers = max(1, int(args.max_active_seats) // int(args.num_players))
    print(
        f"running {len(tasks)} pending matches with num_players={args.num_players}, "
        f"max_active_seats={args.max_active_seats}, worker_games={max_workers}",
        flush=True,
    )

    pending_tasks = list(tasks)
    running: dict[Any, MatchTask] = {}
    completed_count = 0
    failed_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        while pending_tasks or running:
            while pending_tasks and len(running) < max_workers:
                task = pending_tasks.pop(0)
                future = pool.submit(_run_match_worker, task)
                running[future] = task
            done, _ = wait(running.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                task = running.pop(future)
                result = future.result()
                state["matches"][task.match_key] = result
                _atomic_write_json(state_path, state)
                _append_jsonl(results_path, result)
                if result.get("status") == "completed":
                    completed_count += 1
                    summary = result.get("summary") or {}
                    winners = ",".join(summary.get("first_place_agents") or [])
                    print(
                        f"[{completed_count + failed_count}/{len(tasks)}] completed {task.match_key} "
                        f"seed={task.seed} winners={winners} wall={result.get('wall_time_s')}s",
                        flush=True,
                    )
                else:
                    failed_count += 1
                    print(
                        f"[{completed_count + failed_count}/{len(tasks)}] failed {task.match_key} "
                        f"error={result.get('error_path')}",
                        flush=True,
                    )

    rankings = _build_rankings(state)
    _write_rankings(out_dir, rankings)
    print(
        f"finished tournament update: completed={completed_count} failed={failed_count} "
        f"rankings={out_dir / 'rankings.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()

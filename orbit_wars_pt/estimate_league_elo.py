"""Estimate checkpoint Elo from league win-rate data stored in training checkpoints.

During league training the current (main) policy periodically plays historical
checkpoints.  Each saved checkpoint carries a cumulative ``league_state`` with
per-opponent ``games``, ``main_wins``, and ``main_winrate_ema``.

This script never runs games.  It reconstructs pairwise match batches by
differencing ``league_state`` between consecutive checkpoint saves, then fits
Bradley-Terry strengths (reported as Elo with anchor 1500).

Assumes every saved main-training checkpoint ``iter_XXXXXXXX.pt`` in the
experiment checkpoint directory is present and evenly spaced in training
iteration (``checkpoint_every``). Imported league opponents may have arbitrary
``.pt`` filenames; those are read from ``league_state`` metadata rather than
from directory scans. Game counts per interval still vary because
league-opponent selection is stochastic.

Usage::

    ./.venv/bin/python -m orbit_wars_pt.estimate_league_elo --experiment league-2p-003
    ./.venv/bin/python -m orbit_wars_pt.estimate_league_elo --experiment league-2p-003 --csv ratings.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@dataclass(frozen=True)
class LeagueOpponentRecord:
    checkpoint_name: str
    checkpoint_path: str
    checkpoint_iteration: int
    games: int = 0
    main_wins: int = 0
    main_winrate_ema: float = 0.5


def _checkpoint_iteration_from_path(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _sanitize_experiment_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("experiment name must be non-empty")
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
    if ".." in name or "/" in name or "\\" in name:
        raise ValueError("experiment name must not contain path separators or '..'")
    return safe


def _deserialize_league_state(obj: Any) -> dict[str, LeagueOpponentRecord]:
    out: dict[str, LeagueOpponentRecord] = {}
    if not isinstance(obj, Mapping):
        return out
    opponents_obj = obj.get("opponents", {})
    if not isinstance(opponents_obj, Mapping):
        return out
    for key, raw in opponents_obj.items():
        if not isinstance(raw, Mapping):
            continue
        try:
            record = LeagueOpponentRecord(
                checkpoint_name=str(raw.get("checkpoint_name", key)),
                checkpoint_path=str(raw.get("checkpoint_path", key)),
                checkpoint_iteration=int(
                    raw.get("checkpoint_iteration", _checkpoint_iteration_from_path(Path(str(key))))
                ),
                games=int(raw.get("games", 0)),
                main_wins=int(raw.get("main_wins", 0)),
                main_winrate_ema=float(raw.get("main_winrate_ema", 0.5)),
            )
        except (TypeError, ValueError):
            continue
        out[str(key)] = record
    return out


@dataclass(frozen=True)
class MatchBatch:
    """Games where main at one checkpoint played one league opponent."""

    main_id: str
    opponent_id: str
    main_wins: int
    games: int


@dataclass(frozen=True)
class PlayerRating:
    player_id: str
    iteration: int
    checkpoint_name: str
    elo: float
    bt_strength: float
    games_as_main: int
    games_as_opponent: int
    ema_main_winrate: Optional[float]
    cumulative_main_winrate: Optional[float]


def _experiment_checkpoints_dir(experiment: str, experiment_root: str) -> Path:
    root = Path(experiment_root)
    if not root.is_absolute():
        root = Path(ROOT) / root
    name = _sanitize_experiment_name(experiment)
    ckpt_dir = root / name / "checkpoints"
    if not ckpt_dir.is_dir():
        raise SystemExit(f"checkpoint directory not found: {ckpt_dir}")
    return ckpt_dir


def _load_checkpoint(path: Path) -> tuple[int, Optional[dict[str, LeagueOpponentRecord]], Mapping[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    iteration = int(payload.get("iteration", _checkpoint_iteration_from_path(path)))
    opponents = _deserialize_league_state(payload.get("league_state"))
    opponents = opponents if opponents else None
    training_args = payload.get("training_args") or {}
    return iteration, opponents, training_args


def _list_checkpoints(ckpt_dir: Path) -> list[Path]:
    paths = sorted(ckpt_dir.glob("iter_*.pt"), key=_checkpoint_iteration_from_path)
    if not paths:
        raise SystemExit(f"no iter_*.pt checkpoints under {ckpt_dir}")
    return paths


def _verify_even_spacing(paths: list[Path]) -> int:
    iters = [_checkpoint_iteration_from_path(p) for p in paths]
    if len(iters) < 2:
        return iters[0] if iters else 0
    gaps = [iters[i + 1] - iters[i] for i in range(len(iters) - 1)]
    expected = gaps[0]
    bad = [(iters[i], iters[i + 1], gaps[i]) for i in range(len(gaps)) if gaps[i] != expected]
    if bad:
        sample = ", ".join(f"{a}->{b} (gap {g})" for a, b, g in bad[:5])
        raise SystemExit(
            "checkpoints are not evenly spaced in iteration; "
            f"expected constant gap {expected}, mismatches include: {sample}"
        )
    return int(expected)


def _extract_match_batches(paths: list[Path]) -> tuple[list[MatchBatch], dict[str, tuple[int, str]]]:
    """Diff consecutive league_state snapshots to recover main-vs-opponent games."""
    batches: list[MatchBatch] = []
    player_meta: dict[str, tuple[int, str]] = {}

    prev_opponents: Optional[dict[str, LeagueOpponentRecord]] = None

    for path in paths:
        iteration, opponents, _ = _load_checkpoint(path)
        main_id = path.name
        player_meta[main_id] = (int(iteration), path.name)
        if prev_opponents is None or opponents is None:
            prev_opponents = opponents
            continue

        for key, rec in opponents.items():
            opponent_id = str(key)
            player_meta.setdefault(opponent_id, (int(rec.checkpoint_iteration), str(rec.checkpoint_name)))
            prev = prev_opponents.get(key)
            if prev is None:
                continue
            delta_games = int(rec.games) - int(prev.games)
            if delta_games <= 0:
                continue
            delta_wins = int(rec.main_wins) - int(prev.main_wins)
            delta_wins = max(0, min(delta_wins, delta_games))
            batches.append(
                MatchBatch(
                    main_id=main_id,
                    opponent_id=opponent_id,
                    main_wins=int(delta_wins),
                    games=int(delta_games),
                )
            )

        prev_opponents = opponents

    return batches, player_meta


def _final_snapshot_stats(
    paths: list[Path],
) -> tuple[dict[str, LeagueOpponentRecord], str, int, Mapping[str, Any]]:
    iteration, opponents, training_args = _load_checkpoint(paths[-1])
    if opponents is None:
        raise SystemExit("final checkpoint has no league_state opponents; is league enabled?")
    by_id = {str(key): rec for key, rec in opponents.items()}
    return by_id, paths[-1].name, iteration, training_args


def _winrate_to_elo_delta(winrate: float) -> float:
    """Map P(player beats anchor) to Elo delta vs anchor at 1500."""
    p = float(np.clip(winrate, 1e-6, 1.0 - 1e-6))
    return 400.0 * math.log10((1.0 - p) / p)


def _fit_bradley_terry(
    batches: list[MatchBatch],
    *,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> dict[str, float]:
    """Return log-strength for each checkpoint/player id."""
    players = sorted({b.main_id for b in batches} | {b.opponent_id for b in batches})
    if not players:
        return {}

    index = {player: i for i, player in enumerate(players)}
    n = len(players)
    wins = np.zeros(n, dtype=np.float64)
    n_ij = np.zeros((n, n), dtype=np.float64)

    for batch in batches:
        i = index[batch.main_id]
        j = index[batch.opponent_id]
        wins[i] += float(batch.main_wins)
        wins[j] += float(batch.games - batch.main_wins)
        n_ij[i, j] += float(batch.games)
        n_ij[j, i] += float(batch.games)

    # Bradley-Terry MM on strengths pi > 0; scale is arbitrary until we anchor Elo.
    pi = np.ones(n, dtype=np.float64)
    for _ in range(max_iter):
        pi_old = pi.copy()
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if i == j or n_ij[i, j] <= 0.0:
                    continue
                denom += n_ij[i, j] / (pi[i] + pi[j])
            if denom > 0.0 and wins[i] > 0.0:
                pi[i] = wins[i] / denom
            elif wins[i] <= 0.0:
                pi[i] = 1e-12
        pi = np.maximum(pi, 1e-12)
        rel = np.max(np.abs(pi - pi_old) / np.maximum(pi_old, 1e-12))
        if rel < tol:
            break

    return {players[i]: float(math.log(pi[i])) for i in range(n)}


def _bt_to_elo(strengths: dict[str, float], *, anchor_id: Optional[str]) -> dict[str, float]:
    if not strengths:
        return {}
    if anchor_id is not None and anchor_id in strengths:
        anchor_strength = strengths[anchor_id]
    else:
        anchor_strength = strengths[sorted(strengths)[-1]]
    out: dict[str, float] = {}
    for player_id, strength in strengths.items():
        # Elo difference ~= (s_i - s_j) * 400 / ln(10) when using 10-base logistic.
        out[player_id] = 1500.0 + (strength - anchor_strength) * (400.0 / math.log(10.0))
    return out


def _build_ratings(
    batches: list[MatchBatch],
    player_meta: dict[str, tuple[int, str]],
    final_by_id: dict[str, LeagueOpponentRecord],
    *,
    anchor_id: Optional[str],
) -> list[PlayerRating]:
    strengths = _fit_bradley_terry(batches)
    elos = _bt_to_elo(strengths, anchor_id=anchor_id)

    games_as_main = {player_id: 0 for player_id in player_meta}
    games_as_opponent = {player_id: 0 for player_id in player_meta}
    for batch in batches:
        games_as_main[batch.main_id] = games_as_main.get(batch.main_id, 0) + batch.games
        games_as_opponent[batch.opponent_id] = games_as_opponent.get(batch.opponent_id, 0) + batch.games

    all_ids = sorted(set(player_meta) | set(final_by_id) | set(strengths))
    ratings: list[PlayerRating] = []
    for player_id in all_ids:
        iteration, checkpoint_name = player_meta.get(player_id, (-1, str(player_id)))
        rec = final_by_id.get(player_id)
        cumulative_wr: Optional[float] = None
        ema_wr: Optional[float] = None
        if rec is not None and int(rec.games) > 0:
            cumulative_wr = float(rec.main_wins) / float(rec.games)
            ema_wr = float(rec.main_winrate_ema)
        ratings.append(
            PlayerRating(
                player_id=str(player_id),
                iteration=int(iteration),
                checkpoint_name=str(checkpoint_name),
                elo=float(elos.get(player_id, float("nan"))),
                bt_strength=float(strengths.get(player_id, float("nan"))),
                games_as_main=int(games_as_main.get(player_id, 0)),
                games_as_opponent=int(games_as_opponent.get(player_id, 0)),
                ema_main_winrate=ema_wr,
                cumulative_main_winrate=cumulative_wr,
            )
        )
    ratings.sort(key=lambda row: (row.iteration, row.checkpoint_name, row.player_id))
    return ratings


def _ema_snapshot_elo(
    final_main_id: str,
    final_by_id: dict[str, LeagueOpponentRecord],
) -> dict[str, float]:
    """Quick relative ratings from the final checkpoint's EMA vs latest main."""
    out: dict[str, float] = {final_main_id: 1500.0}
    for player_id, rec in final_by_id.items():
        if int(rec.games) <= 0:
            continue
        # main_winrate_ema = P(latest main wins vs this opponent).
        out[str(player_id)] = 1500.0 + _winrate_to_elo_delta(float(rec.main_winrate_ema))
    return out


def _print_table(ratings: list[PlayerRating], *, ema_elo: dict[str, float]) -> None:
    print(
        f"{'iter':>8}  {'elo':>8}  {'ema_elo':>8}  "
        f"{'g_main':>8}  {'g_opp':>8}  {'cum_wr':>7}  {'ema_wr':>7}  checkpoint"
    )
    for row in ratings:
        cum = "" if row.cumulative_main_winrate is None else f"{row.cumulative_main_winrate:.3f}"
        ema = "" if row.ema_main_winrate is None else f"{row.ema_main_winrate:.3f}"
        elo = "" if row.elo != row.elo else f"{row.elo:.1f}"
        ema_elo_text = ""
        if row.player_id in ema_elo:
            ema_elo_text = f"{ema_elo[row.player_id]:.1f}"
        print(
            f"{row.iteration:8d}  {elo:>8}  {ema_elo_text:>8}  "
            f"{row.games_as_main:8d}  {row.games_as_opponent:8d}  {cum:>7}  {ema:>7}  "
            f"{row.checkpoint_name}"
        )


def _write_csv(path: Path, ratings: list[PlayerRating], *, ema_elo: dict[str, float]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "player_id",
                "iteration",
                "checkpoint_name",
                "elo_bt",
                "elo_ema_snapshot",
                "games_as_main",
                "games_as_opponent",
                "cumulative_main_winrate",
                "ema_main_winrate",
            ]
        )
        for row in ratings:
            writer.writerow(
                [
                    row.player_id,
                    row.iteration,
                    row.checkpoint_name,
                    "" if row.elo != row.elo else f"{row.elo:.4f}",
                    "" if row.player_id not in ema_elo else f"{ema_elo[row.player_id]:.4f}",
                    row.games_as_main,
                    row.games_as_opponent,
                    "" if row.cumulative_main_winrate is None else f"{row.cumulative_main_winrate:.6f}",
                    "" if row.ema_main_winrate is None else f"{row.ema_main_winrate:.6f}",
                ]
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=str, required=True)
    p.add_argument(
        "--experiment-root",
        type=str,
        default="experiments",
        help="Experiment root (relative to repo root if not absolute).",
    )
    p.add_argument(
        "--anchor-iter",
        type=int,
        default=None,
        help="Anchor this main-training iteration at Elo 1500 (default: latest main checkpoint).",
    )
    p.add_argument("--csv", type=str, default=None, help="Write ratings table to this CSV path.")
    p.add_argument("--json", type=str, default=None, help="Write ratings as JSON to this path.")
    p.add_argument(
        "--skip-spacing-check",
        action="store_true",
        help="Do not require evenly spaced checkpoint iterations.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_dir = _experiment_checkpoints_dir(args.experiment, args.experiment_root)
    paths = _list_checkpoints(ckpt_dir)

    checkpoint_every = None
    if not args.skip_spacing_check:
        checkpoint_every = _verify_even_spacing(paths)

    batches, player_meta = _extract_match_batches(paths)
    if not batches:
        raise SystemExit(
            "no league match batches found; ensure league is enabled and checkpoints contain league_state"
        )

    final_by_id, final_main_id, final_main_iter, training_args = _final_snapshot_stats(paths)
    league_fraction = training_args.get("league_fraction")
    num_agents = training_args.get("num_agents")

    anchor_id = None
    if args.anchor_iter is not None:
        anchor_name = f"iter_{int(args.anchor_iter):08d}.pt"
        if anchor_name not in player_meta:
            raise SystemExit(
                f"--anchor-iter={args.anchor_iter} does not correspond to a saved main checkpoint filename {anchor_name}"
            )
        anchor_id = anchor_name
    else:
        anchor_id = final_main_id
    ratings = _build_ratings(
        batches,
        player_meta,
        final_by_id,
        anchor_id=anchor_id,
    )
    ema_elo = _ema_snapshot_elo(final_main_id, final_by_id)

    total_games = sum(b.games for b in batches)
    print(f"experiment={args.experiment}")
    print(f"checkpoints={len(paths)}  match_batches={len(batches)}  games={total_games}")
    if checkpoint_every is not None:
        print(f"checkpoint_every={checkpoint_every}")
    if league_fraction is not None:
        print(f"league_fraction={league_fraction}  num_agents={num_agents}")
    print(f"anchor_iter={args.anchor_iter if args.anchor_iter is not None else final_main_iter} (Elo 1500)")
    print()
    print("elo_bt: Bradley-Terry fit from checkpoint-to-checkpoint league deltas")
    print("ema_elo: final-checkpoint EMA vs latest main, anchored at 1500 for latest")
    print()
    _print_table(ratings, ema_elo=ema_elo)

    if args.csv:
        out = Path(args.csv)
        _write_csv(out, ratings, ema_elo=ema_elo)
        print(f"\nwrote {out}")

    if args.json:
        out = Path(args.json)
        payload = {
            "experiment": args.experiment,
            "anchor_iter": anchor_iter,
            "checkpoint_every": checkpoint_every,
            "total_games": total_games,
            "match_batches": len(batches),
            "ratings": [
                {
                    "player_id": r.player_id,
                    "iteration": r.iteration,
                    "checkpoint_name": r.checkpoint_name,
                    "elo_bt": None if r.elo != r.elo else r.elo,
                    "elo_ema_snapshot": ema_elo.get(r.player_id),
                    "games_as_main": r.games_as_main,
                    "games_as_opponent": r.games_as_opponent,
                    "cumulative_main_winrate": r.cumulative_main_winrate,
                    "ema_main_winrate": r.ema_main_winrate,
                }
                for r in ratings
            ],
        }
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""Adapter for running a trained policy in the official Kaggle Orbit Wars env.

The Kaggle agent API calls ``agent(obs, config)`` and expects a Python list of
``[from_planet_id, angle, num_ships]`` launches.  Training uses a fixed-table
``OrbitWarsState`` plus a PyTorch policy, so this module bridges between the two.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import sys
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, NamedTuple, Optional, Sequence

import numpy as np
import torch

from orbit_wars_pt.constants import (
    BOARD_SIZE,
    BLOCKED_FRAC_FEATURES,
    CENTER,
    ENTITY_CLS,
    ENTITY_COMET,
    ENTITY_PLANET,
    FEATURE_DIM_MULTI,
    FEATURE_DIM_MULTI_ABORT,
    FRACTIONS,
    INCOMING_SURVIVOR_FLAT,
    INCOMING_TA_BINS,
    MAX_PLANETS,
    NUM_OWNER_SLOTS,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
    obs_feature_dim_for_num_agents,
)
from orbit_wars_pt.geometry import estimate_time_to_hit, planet_pred_velocity
from orbit_wars_pt.model import (
    OrbitWarsPolicy,
    adapt_checkpoint_state_for_model,
    infer_value_head_count_from_state_dict,
)
from orbit_wars_pt.reward_config import resolve_reward_mix


DEFAULT_CHECKPOINT = "checkpoint.pt"
DEFAULT_CHECKPOINT_4P = "checkpoint_4p.pt"
DEFAULT_CHECKPOINT_2P = "checkpoint_2p.pt"
DEFAULT_RAYCAST_RAYS = 256
DEFAULT_INTERVAL_SAMPLES_PER_SPAN = 9
DEFAULT_TARGET_METHOD = "rays"
DEFAULT_MAX_ACTIONS = 64
DEFAULT_CPU_THREADS = 1
SAMPLING_MODE_STOCHASTIC = "stochastic"
SAMPLING_MODE_GREEDY = "greedy"
SAMPLING_MODE_MIXED = "mixed"
_VALID_SAMPLING_MODES = {
    SAMPLING_MODE_STOCHASTIC,
    SAMPLING_MODE_GREEDY,
    SAMPLING_MODE_MIXED,
}
MODEL_SEARCH_MODE_BINARY = "binary"
MODEL_SEARCH_MODE_EGO_BFS = "ego_bfs"
MODEL_SEARCH_MODE_TURN_SAMPLING = "turn_sampling"
_VALID_MODEL_SEARCH_MODES = {
    MODEL_SEARCH_MODE_BINARY,
    MODEL_SEARCH_MODE_EGO_BFS,
    MODEL_SEARCH_MODE_TURN_SAMPLING,
}
MAX_COMET_GROUPS = 5
MAX_COMET_PATH = 40
_SEAT_QTURNS_TO_P0_2P = np.asarray([0, 2], dtype=np.int32)
_SEAT_QTURNS_TO_P0_4P = np.asarray([0, 3, 1, 2], dtype=np.int32)
_NORMALIZED_OWNER_SLOT_4P = np.asarray(
    [
        [1, 3, 4, 2],
        [4, 1, 2, 3],
        [3, 2, 1, 4],
        [2, 4, 3, 1],
    ],
    dtype=np.int32,
)
CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

PLANET_X = 2
PLANET_Y = 3
PLANET_RADIUS = 4
FLEET_ID = 0
FLEET_OWNER = 1
FLEET_X = 2
FLEET_Y = 3
FLEET_ANGLE = 4
FLEET_FROM_PLANET = 5
FLEET_SHIPS = 6
FLEET_ROW_WIDTH = 9

_FLEET_ARRIVAL_POS_ATOL = 1e-4
_FLEET_ARRIVAL_ANGLE_SCALE = 1_000_000.0


@dataclass
class FleetArrivalCacheEntry:
    owner: int
    from_planet_id: int
    ships: int
    angle_key: int
    step_count: int
    position: tuple[float, float]
    hit_slot: int
    hit_tick: int
    no_hit_ticks: int
    comet_layout_key: tuple[int, ...]


@dataclass
class FleetArrivalCache:
    entries: dict[tuple[int, int, int, int, int], FleetArrivalCacheEntry] = field(default_factory=dict)

    def clear(self) -> None:
        self.entries.clear()


def _norm_angle(angle: float) -> float:
    return float(angle) % (2.0 * math.pi)


def _angle_diff(a: float, b: float) -> float:
    return abs(((_norm_angle(a) - _norm_angle(b)) + math.pi) % (2.0 * math.pi) - math.pi)


def _planet_id_match(recorded: float, observed: float) -> bool:
    return abs(float(recorded) - float(observed)) < 0.5


def _ships_match(recorded: int, observed: float) -> bool:
    return int(recorded) == int(round(float(observed)))


def _launch_debug_enabled() -> bool:
    return os.environ.get("ORBIT_WARS_DEBUG_LAUNCH", "0").lower() in {"1", "true", "yes", "on"}


def _launch_debug(msg: str) -> None:
    if _launch_debug_enabled():
        print(f"[orbit_wars:launch] {msg}", file=sys.stderr, flush=True)


def _model_search_debug_enabled() -> bool:
    return os.environ.get("ORBIT_WARS_DEBUG_MODEL_SEARCH", "0").lower() in {"1", "true", "yes", "on"}


def _model_search_debug(msg: str) -> None:
    if _model_search_debug_enabled():
        print(f"[orbit_wars:search] {msg}", file=sys.stderr, flush=True)


def _trace_fleet_ids() -> set[int]:
    raw = os.environ.get("ORBIT_WARS_TRACE_FLEETS", "").strip()
    if not raw:
        return set()
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            continue
    return out


def _trace_launches_enabled() -> bool:
    return os.environ.get("ORBIT_WARS_TRACE_LAUNCHES", "0").lower() in {"1", "true", "yes", "on"}


def _should_trace_fleet(fid: int) -> bool:
    ids = _trace_fleet_ids()
    return fid >= 0 and fid in ids


def _launch_trace(msg: str, *, fleet_id: int | None = None) -> None:
    if fleet_id is not None and _should_trace_fleet(int(fleet_id)):
        print(f"[orbit_wars:trace] {msg}", file=sys.stderr, flush=True)
        return
    if _trace_launches_enabled():
        print(f"[orbit_wars:trace] {msg}", file=sys.stderr, flush=True)


def _active_comet_planet_ids(
    comet_group_active: np.ndarray,
    comet_planet_ids: np.ndarray,
) -> frozenset[int]:
    ids: list[int] = []
    for g in range(int(comet_group_active.shape[0])):
        if not bool(comet_group_active[g]):
            continue
        for k in range(int(comet_planet_ids.shape[1])):
            pid = int(comet_planet_ids[g, k])
            if pid >= 0:
                ids.append(pid)
    return frozenset(ids)


def _comet_layout_cache_key(
    comet_group_active: np.ndarray,
    comet_planet_ids: np.ndarray,
) -> tuple[int, ...]:
    ids: list[int] = []
    for g in range(int(comet_group_active.shape[0])):
        if not bool(comet_group_active[g]):
            continue
        for pid_raw in comet_planet_ids[g]:
            pid = int(pid_raw)
            if pid >= 0:
                ids.append(pid)
    ids.sort()
    return tuple(ids)


def _fleet_arrival_cache_key(row: np.ndarray) -> tuple[int, int, int, int, int]:
    return (
        int(row[0]),
        int(row[1]),
        int(row[5]),
        int(math.floor(float(row[6]))),
        int(round(_norm_angle(float(row[4])) * _FLEET_ARRIVAL_ANGLE_SCALE)),
    )


def _is_comet_planet_id(planet_id: int, comet_planet_ids: np.ndarray) -> bool:
    return planet_id >= 0 and bool(np.any(comet_planet_ids == planet_id))


def _forecast_hits_comet_spawned_after_launch(
    forecast_slot: int,
    planets: np.ndarray,
    comet_planet_ids: np.ndarray,
    comet_planet_ids_at_launch: frozenset[int],
) -> bool:
    """True if forecast first-hit is a comet that was not active at fleet launch."""

    if forecast_slot < 0 or forecast_slot >= MAX_PLANETS:
        return False
    pid = int(planets[forecast_slot, 0])
    if pid in comet_planet_ids_at_launch:
        return False
    return _is_comet_planet_id(pid, comet_planet_ids)


def _fleet_exits_board_next_tick(
    x: float,
    y: float,
    angle: float,
    ships: float,
    *,
    ship_speed: float = 6.0,
) -> bool:
    sp = _fleet_speed(ships, ship_speed)
    nx = x + math.cos(angle) * sp
    ny = y + math.sin(angle) * sp
    return nx < 0.0 or nx > BOARD_SIZE or ny < 0.0 or ny > BOARD_SIZE


@dataclass
class LaunchRaycastRecord:
    """Bookkeeping for one micro-launch decision (raycast inputs and targets)."""

    game_step: int
    ego_player: int
    micro_idx: int
    origin_slot: int
    origin_planet_id: float
    origin_xy: tuple[float, float]
    origin_radius: float
    frac_idx: int
    fraction: float
    ships_avail: float
    planned_send: int
    n_rays: int
    ship_speed: float
    launch_angle: float
    policy_target_slot: int
    policy_target_planet_id: float
    true_target_slot: int
    true_target_planet_id: float
    policy_hit_tick: float
    true_hit_tick: float
    comet_planet_ids_at_launch: frozenset[int] = field(default_factory=frozenset)
    fleet_id: Optional[int] = None
    debug_serial: int = 0

    def debug_summary(self) -> str:
        return (
            f"serial={self.debug_serial} step={self.game_step} ego={self.ego_player} micro={self.micro_idx} "
            f"from_id={self.origin_planet_id:.0f} send={self.planned_send} "
            f"angle={self.launch_angle:.8f} tgt={self.policy_target_planet_id:.0f} "
            f"hit_tick={self.true_hit_tick:.1f} fleet_id={self.fleet_id}"
        )

    def format_warning(self, *, last_x: float, last_y: float, last_ships: float) -> str:
        lines = [
            f"[orbit_wars] friendly fleet left the board"
            + (f" (fleet_id={self.fleet_id})" if self.fleet_id is not None else ""),
            f"  last_pos=({last_x:.4f}, {last_y:.4f}) ships={last_ships:.0f}",
            f"  game_step={self.game_step} ego_player={self.ego_player} micro_idx={self.micro_idx}",
            f"  raycast: origin_slot={self.origin_slot} planet_id={self.origin_planet_id:.0f}"
            f" pos=({self.origin_xy[0]:.4f},{self.origin_xy[1]:.4f}) r={self.origin_radius:.4f}",
            f"    frac_idx={self.frac_idx} fraction={self.fraction} ships_avail={self.ships_avail:.0f}"
            f" planned_send={self.planned_send} n_rays={self.n_rays} ship_speed={self.ship_speed}",
            f"  launch_angle={self.launch_angle:.6f} rad ({math.degrees(self.launch_angle):.2f} deg)",
            f"  policy_target: slot={self.policy_target_slot} planet_id={self.policy_target_planet_id:.0f}"
            f" projected_hit_tick={self.policy_hit_tick:.1f}",
            f"  true_target: slot={self.true_target_slot} planet_id={self.true_target_planet_id:.0f}"
            f" projected_hit_tick={self.true_hit_tick:.1f}",
        ]
        return "\n".join(lines)


@dataclass
class FleetLaunchDebugTracker:
    """Maps fleet ids to launch/raycast records; warns when a friendly fleet exits the map."""

    warn_oob: bool = True
    warn_forecast_mismatch: bool = True
    warn_unmatched_fleet: bool = True
    _pending: list[LaunchRaycastRecord] = field(default_factory=list)
    _by_fleet_id: dict[int, LaunchRaycastRecord] = field(default_factory=dict)
    _last_fleets_all: dict[int, tuple[int, float, float, float, float, float]] = field(default_factory=dict)
    _last_fleets_by_player: dict[int, dict[int, tuple[int, float, float, float, float, float]]] = field(
        default_factory=dict
    )
    _last_step_by_player: dict[int, int] = field(default_factory=dict)
    _game_key: Optional[str] = None
    _call_seq: int = 0
    _next_launch_serial: int = 0
    _warned_forecast_mismatch: set[int] = field(default_factory=set)
    _warned_forecast_tick_mismatch: set[int] = field(default_factory=set)
    _warned_unmatched_fleet: set[int] = field(default_factory=set)

    def reset_game(self) -> None:
        if _launch_debug_enabled() and (self._pending or self._by_fleet_id):
            _launch_debug(
                f"reset_game pending={len(self._pending)} by_fleet_id={len(self._by_fleet_id)} "
                f"keys={sorted(self._by_fleet_id.keys())[:20]}"
            )
        self._pending.clear()
        self._by_fleet_id.clear()
        self._last_fleets_all.clear()
        self._last_fleets_by_player.clear()
        self._last_step_by_player.clear()
        self._warned_forecast_mismatch.clear()
        self._warned_forecast_tick_mismatch.clear()
        self._warned_unmatched_fleet.clear()

    def sync_game(self, game_key: str, *, game_step: int = -1) -> None:
        if game_key != self._game_key:
            if (
                self._game_key is not None
                and game_step > 0
                and (self._pending or self._by_fleet_id)
            ):
                print(
                    f"[orbit_wars] launch tracker reset mid-game at step={game_step} "
                    f"(key {str(self._game_key)[:8]} -> {game_key[:8]}); "
                    f"dropping {len(self._pending)} pending and {len(self._by_fleet_id)} fleet records",
                    file=sys.stderr,
                    flush=True,
                )
            if _launch_debug_enabled():
                _launch_debug(f"sync_game new_key={game_key[:8]} old={str(self._game_key)[:8]} step={game_step}")
            self._game_key = game_key
            self.reset_game()

    def _debug_dump_state(self, label: str, *, ego_player: int, game_step: int, obs_step: Any) -> None:
        if not _launch_debug_enabled():
            return
        pending_by_ego: dict[int, int] = {}
        for rec in self._pending:
            pending_by_ego[rec.ego_player] = pending_by_ego.get(rec.ego_player, 0) + 1
        _launch_debug(
            f"{label} ego={ego_player} game_step={game_step} obs.step={obs_step!r} "
            f"call_seq={self._call_seq} pending={len(self._pending)}{pending_by_ego} "
            f"by_fleet_id={len(self._by_fleet_id)} last_step_by_player={dict(self._last_step_by_player)}"
        )
        for rec in self._pending[:12]:
            _launch_debug(f"  pending: {rec.debug_summary()}")
        if len(self._pending) > 12:
            _launch_debug(f"  ... {len(self._pending) - 12} more pending")

    def _debug_explain_fleet_match(
        self,
        owner: int,
        from_id: float,
        angle: float,
        ships: float,
        *,
        n_rays: int,
        fleet_id: int,
    ) -> None:
        if not _launch_debug_enabled():
            return
        angle_tol = max(0.06, (2.0 * math.pi / float(max(n_rays, 8))) + 1e-6)
        _launch_debug(
            f"no match for fleet_id={fleet_id} owner={owner} from={from_id:.0f} "
            f"ships={ships:.0f} angle={angle:.8f} (tol={angle_tol:.5f})"
        )
        if not self._pending:
            _launch_debug("  pending queue is EMPTY")
            return
        for i, rec in enumerate(self._pending):
            reasons: list[str] = []
            if rec.ego_player != owner:
                reasons.append(f"ego {rec.ego_player}!={owner}")
            if not _planet_id_match(rec.origin_planet_id, from_id):
                reasons.append(f"from {rec.origin_planet_id:.0f}!={from_id:.0f}")
            if not _ships_match(rec.planned_send, ships):
                reasons.append(f"ships {rec.planned_send}!={int(round(ships))}")
            ang_diff = _angle_diff(rec.launch_angle, angle)
            if ang_diff > angle_tol:
                reasons.append(f"angle_diff={ang_diff:.5f}>{angle_tol:.5f}")
            _launch_debug(f"  pending[{i}] {rec.debug_summary()} -> {reasons or 'SHOULD MATCH'}")
        if fleet_id in self._by_fleet_id:
            _launch_debug(f"  NOTE: fleet_id {fleet_id} IS in by_fleet_id: {self._by_fleet_id[fleet_id].debug_summary()}")

    @staticmethod
    def _fleet_tuple(f: Any) -> tuple[int, float, float, float, float, float]:
        return (
            int(f[1]),
            float(f[2]),
            float(f[3]),
            float(f[4]),
            float(f[5]),
            float(f[6]),
        )

    def _match_pending(
        self,
        owner: int,
        from_id: float,
        angle: float,
        ships: float,
        *,
        n_rays: int = DEFAULT_RAYCAST_RAYS,
    ) -> Optional[LaunchRaycastRecord]:
        """Match a Kaggle fleet row to a pending launch (tolerant angle, not exact floats)."""

        angle_tol = max(0.06, (2.0 * math.pi / float(max(n_rays, 8))) + 1e-6)
        best: Optional[LaunchRaycastRecord] = None
        best_i = -1
        best_score = 1e9
        for i, rec in enumerate(self._pending):
            if rec.ego_player != owner:
                continue
            if not _planet_id_match(rec.origin_planet_id, from_id):
                continue
            if not _ships_match(rec.planned_send, ships):
                continue
            ang_diff = _angle_diff(rec.launch_angle, angle)
            if ang_diff < best_score:
                best_score = ang_diff
                best = rec
                best_i = i
        if best is not None and best_score <= angle_tol:
            rec = self._pending.pop(best_i)
            if _launch_debug_enabled():
                _launch_debug(
                    f"matched pending[{rec.debug_serial}] -> fleet owner={owner} from={from_id:.0f} "
                    f"angle_diff={best_score:.6f}"
                )
            return rec

        # Fallback: unique pending launch from this planet (ships can disagree if
        # micro bookkeeping diverged, but angle + owner + from_id should still match).
        candidates = [
            (i, rec)
            for i, rec in enumerate(self._pending)
            if rec.ego_player == owner and _planet_id_match(rec.origin_planet_id, from_id)
        ]
        if len(candidates) == 1:
            i, rec = candidates[0]
            ang_diff = _angle_diff(rec.launch_angle, angle)
            if ang_diff <= angle_tol:
                rec = self._pending.pop(i)
                if _launch_debug_enabled():
                    _launch_debug(
                        f"matched pending[{rec.debug_serial}] (fallback unique from) "
                        f"angle_diff={ang_diff:.6f}"
                    )
                return rec
        return None

    def _attach_pending_to_observed_fleets(
        self,
        current: dict[int, tuple[int, float, float, float, float, float]],
        *,
        n_rays: int,
        ego_player: int,
        game_step: int,
    ) -> None:
        unmatched = 0
        for fid, tup in sorted(current.items()):
            if fid in self._by_fleet_id:
                continue
            owner, x, y, ang, from_id, ships = tup
            rec = self._match_pending(owner, from_id, ang, ships, n_rays=n_rays)
            if rec is not None:
                rec.fleet_id = fid
                self._by_fleet_id[fid] = rec
                _launch_trace(
                    "attach"
                    f" fid={fid} game_step={game_step} ego={ego_player}"
                    f" obs_pos=({x:.6f},{y:.6f}) angle={ang:.12f} from={from_id:.0f} ships={ships:.0f}"
                    f" <- serial={rec.debug_serial} launch_step={rec.game_step} micro={rec.micro_idx}"
                    f" origin={rec.origin_planet_id:.0f} launch_angle={rec.launch_angle:.12f}"
                    f" policy_slot={rec.policy_target_slot} true_slot={rec.true_target_slot}"
                    f" policy_tick={rec.policy_hit_tick:.0f} true_tick={rec.true_hit_tick:.0f}",
                    fleet_id=fid,
                )
                if _launch_debug_enabled():
                    _launch_debug(f"attached fleet_id={fid} <- {rec.debug_summary()}")
            else:
                unmatched += 1
                if owner == ego_player:
                    if _launch_debug_enabled():
                        self._debug_explain_fleet_match(
                            owner, from_id, ang, ships, n_rays=n_rays, fleet_id=fid
                        )
                    if (
                        self.warn_unmatched_fleet
                        and fid not in self._warned_unmatched_fleet
                    ):
                        self._warned_unmatched_fleet.add(fid)
                        print(
                            "[orbit_wars] friendly fleet has no LaunchRaycastRecord"
                            f" (fleet_id={fid})",
                            f"\n  game_step={game_step} ego_player={ego_player}",
                            f"\n  pos=({x:.4f}, {y:.4f}) ships={ships:.0f} angle={ang:.6f} "
                            f"from_planet={from_id:.0f}",
                            f"\n  pending_launches={len(self._pending)}",
                            sep="",
                            file=sys.stderr,
                            flush=True,
                        )
        if _launch_debug_enabled() and unmatched:
            _launch_debug(f"attach pass: {unmatched} fleet(s) still without launch record")

    def observe_fleets(
        self,
        obs: Mapping[str, Any],
        ego_player: int,
        *,
        game_step: int,
        ship_speed: float = 6.0,
        n_rays: int = DEFAULT_RAYCAST_RAYS,
    ) -> None:
        self._call_seq += 1
        obs_step = obs.get("step", obs.get("step_count", None))
        self._debug_dump_state("observe IN", ego_player=ego_player, game_step=game_step, obs_step=obs_step)

        fleets_in = obs.get("fleets") or []
        current: dict[int, tuple[int, float, float, float, float, float]] = {}
        for f in fleets_in:
            fid = int(f[0])
            current[fid] = self._fleet_tuple(f)

        new_ids = sorted(set(current) - set(self._last_fleets_all))
        if _launch_debug_enabled() and new_ids:
            _launch_debug(f"new fleet ids this obs ({len(new_ids)}): {new_ids[:30]}")

        self._attach_pending_to_observed_fleets(
            current, n_rays=n_rays, ego_player=ego_player, game_step=game_step
        )

        prev_step = self._last_step_by_player.get(ego_player)
        prev_snap = self._last_fleets_by_player.get(ego_player, {})
        if prev_step is not None and game_step > prev_step:
            if _launch_debug_enabled():
                vanished = [fid for fid in prev_snap if fid not in current]
                _launch_debug(
                    f"step advanced {prev_step}->{game_step} ego={ego_player}: "
                    f"{len(vanished)} friendly fleet(s) vanished from snap"
                )
            for fid, last in prev_snap.items():
                if fid in current:
                    continue
                owner, x, y, ang, _from_id, ships = last
                if owner != ego_player:
                    continue
                oob = _fleet_exits_board_next_tick(x, y, ang, ships, ship_speed=ship_speed)
                if _launch_debug_enabled():
                    _launch_debug(
                        f"vanished fid={fid} oob={oob} pos=({x:.3f},{y:.3f}) ang={ang:.6f} "
                        f"from={_from_id:.0f} ships={ships:.0f} in_by_fleet_id={fid in self._by_fleet_id}"
                    )
                if not oob:
                    continue
                if not self.warn_oob:
                    continue
                rec = self._by_fleet_id.get(fid)
                if rec is not None:
                    print(rec.format_warning(last_x=x, last_y=y, last_ships=ships), file=sys.stderr, flush=True)
                else:
                    print(
                        f"[orbit_wars] friendly fleet left the board (fleet_id={fid}, no launch record)\n"
                        f"  last_pos=({x:.4f}, {y:.4f}) ships={ships:.0f} angle={ang:.6f} from_planet={_from_id:.0f}\n"
                        f"  game_step={game_step} ego_player={ego_player}",
                        file=sys.stderr,
                        flush=True,
                    )
                    self._debug_explain_fleet_match(
                        owner, _from_id, ang, ships, n_rays=n_rays, fleet_id=fid
                    )
                    self._debug_dump_state("OOB no record", ego_player=ego_player, game_step=game_step, obs_step=obs_step)

        self._last_fleets_all = current
        self._last_fleets_by_player[ego_player] = {
            fid: tup for fid, tup in current.items() if int(tup[0]) == ego_player
        }
        self._last_step_by_player[ego_player] = game_step
        self._debug_dump_state("observe OUT", ego_player=ego_player, game_step=game_step, obs_step=obs_step)

    def record_launch(self, rec: LaunchRaycastRecord) -> None:
        self._next_launch_serial += 1
        rec.debug_serial = self._next_launch_serial
        self._pending.append(rec)
        _launch_trace(
            "record"
            f" serial={rec.debug_serial} launch_step={rec.game_step} ego={rec.ego_player} micro={rec.micro_idx}"
            f" origin={rec.origin_planet_id:.0f} frac_idx={rec.frac_idx} send={rec.planned_send}"
            f" angle={rec.launch_angle:.12f}"
            f" policy_slot={rec.policy_target_slot} true_slot={rec.true_target_slot}"
            f" policy_tick={rec.policy_hit_tick:.0f} true_tick={rec.true_hit_tick:.0f}"
        )
        if _launch_debug_enabled():
            _launch_debug(f"record_launch {rec.debug_summary()}")

    @staticmethod
    def _fleet_forecast_ticks_aligned(
        ray_hit_tick: float,
        forecast_hit_tick: int,
        *,
        launch_step: int,
        observe_step: int,
    ) -> bool:
        """True if launch raycast tick (from origin) matches forecast tick (from current fleet pos).

        Raycast counts ticks forward from launch; ``_forecast_incoming_fleets`` counts from the
        fleet's position in the current obs, which is ``observe_step - launch_step`` ticks later.
        """

        if ray_hit_tick >= 500.0 or forecast_hit_tick < 0:
            return False
        elapsed = max(0, int(observe_step) - int(launch_step))
        return int(forecast_hit_tick) + elapsed == int(ray_hit_tick)

    def check_forecast_vs_raycast(
        self,
        obs: Mapping[str, Any],
        fleet_arrivals: np.ndarray,
        planets: np.ndarray,
        comet_planet_ids: np.ndarray,
        *,
        game_step: int,
        ego_player: int,
    ) -> None:
        """Warn when fleet forecast first-hit slot or ETA disagrees with launch raycast."""

        if not self.warn_forecast_mismatch:
            return
        fleets_in = obs.get("fleets") or []
        n = min(len(fleets_in), int(fleet_arrivals.shape[0]))
        for i in range(n):
            row = fleets_in[i]
            fid = int(row[0])
            owner = int(row[1])
            if owner != ego_player:
                continue
            rec = self._by_fleet_id.get(fid)
            if rec is None:
                continue
            fc_slot = int(fleet_arrivals[i, 0])
            fc_tick = int(fleet_arrivals[i, 1])
            ray_slot = int(rec.true_target_slot)
            ray_tick = int(rec.true_hit_tick) if rec.true_hit_tick < 500.0 else -1
            elapsed = max(0, int(game_step) - int(rec.game_step))
            ticks_aligned = self._fleet_forecast_ticks_aligned(
                float(rec.true_hit_tick),
                fc_tick,
                launch_step=int(rec.game_step),
                observe_step=int(game_step),
            )
            _launch_trace(
                "compare"
                f" fid={fid} game_step={game_step} ego={ego_player} launch_step={rec.game_step}"
                f" obs_angle={float(row[4]):.12f} obs_from={float(row[5]):.0f} obs_ships={float(row[6]):.0f}"
                f" policy_slot={rec.policy_target_slot} true_slot={ray_slot} forecast_slot={fc_slot}"
                f" policy_tick={rec.policy_hit_tick:.0f} true_tick={ray_tick} forecast_tick={fc_tick}"
                f" elapsed={elapsed} ticks_aligned={int(ticks_aligned)}",
                fleet_id=fid,
            )
            if (
                ray_slot == fc_slot
                and ray_slot >= 0
                and fc_slot >= 0
                and ray_tick >= 0
                and fc_tick >= 0
                and not ticks_aligned
                and fid not in self._warned_forecast_tick_mismatch
            ):
                ray_remaining = ray_tick - elapsed
                ray_ta = int(math.floor(max(float(ray_remaining) - 1.0, 0.0)))
                fc_ta = max(0, fc_tick - 1)
                self._warned_forecast_tick_mismatch.add(fid)
                ray_pid = float(planets[ray_slot, 0]) if 0 <= ray_slot < MAX_PLANETS else -1.0
                print(
                    "[orbit_wars] forecast hit_tick/ETA differs from geometry at launch"
                    + (f" (fleet_id={fid})" if fid >= 0 else ""),
                    f"\n  game_step={game_step} ego_player={ego_player} launch_step={rec.game_step}"
                    f" elapsed_ticks={elapsed}",
                    f"\n  geometry at launch: slot={ray_slot} planet_id={ray_pid:.0f} "
                    f"hit_tick_from_launch={ray_tick} remaining={ray_remaining} incoming_TA={ray_ta}",
                    f"\n  forecast: slot={fc_slot} hit_tick_from_obs={fc_tick} incoming_TA={fc_ta}"
                    f" (expect hit_tick_from_obs + elapsed == hit_tick_from_launch)",
                    sep="",
                    file=sys.stderr,
                    flush=True,
                )
            if fid in self._warned_forecast_mismatch:
                continue
            if ray_slot == fc_slot:
                continue
            if _forecast_hits_comet_spawned_after_launch(
                fc_slot, planets, comet_planet_ids, rec.comet_planet_ids_at_launch
            ):
                if _launch_debug_enabled():
                    fc_pid_dbg = float(planets[fc_slot, 0]) if 0 <= fc_slot < MAX_PLANETS else -1.0
                    _launch_debug(
                        f"forecast mismatch suppressed fleet_id={fid}: "
                        f"comet {fc_pid_dbg:.0f} spawned after launch_step={rec.game_step}"
                    )
                continue
            self._warned_forecast_mismatch.add(fid)
            ray_pid = (
                float(planets[ray_slot, 0])
                if 0 <= ray_slot < MAX_PLANETS
                else float(rec.true_target_planet_id)
            )
            fc_pid = float(planets[fc_slot, 0]) if 0 <= fc_slot < MAX_PLANETS else -1.0
            print(
                "[orbit_wars] forecast incoming target differs from geometry at launch"
                + (f" (fleet_id={fid})" if fid >= 0 else ""),
                f"\n  game_step={game_step} ego_player={ego_player} launch_step={rec.game_step}",
                f"\n  geometry at launch: slot={ray_slot} planet_id={ray_pid:.0f} hit_tick={ray_tick}",
                f"\n  forecast_incoming_fleets: slot={fc_slot} planet_id={fc_pid:.0f} hit_tick={fc_tick}",
                f"\n  launch: from_id={rec.origin_planet_id:.0f} angle={rec.launch_angle:.6f} "
                f"send={rec.planned_send} micro={rec.micro_idx}",
                sep="",
                file=sys.stderr,
                flush=True,
            )
            if _launch_debug_enabled():
                _launch_debug(f"forecast mismatch fleet_id={fid} {rec.debug_summary()}")


def _warn_forecast_mismatch_enabled() -> bool:
    raw = os.environ.get("ORBIT_WARS_WARN_FORECAST_MISMATCH")
    if raw is not None:
        return raw.lower() not in {"0", "false", "no", "off"}
    return os.environ.get("ORBIT_WARS_WARN_OOB_LAUNCHES", "1").lower() not in {"0", "false", "no", "off"}


def _interval_geometry_mode() -> str:
    """``sampled``, ``tangent`` (external/internal tangency), or ``orthogonal`` (shelved)."""

    raw = os.environ.get("ORBIT_WARS_INTERVAL_GEOMETRY", "tangent").strip().lower()
    if raw in {"orthogonal", "cone"}:
        return "orthogonal"
    if raw in {"tangent", "external", "internal"}:
        return "tangent"
    return "sampled"


def _check_interval_raycast_enabled() -> bool:
    raw = os.environ.get("ORBIT_WARS_CHECK_INTERVAL_RAYCAST")
    if raw is None:
        return False
    return raw.lower() not in {"0", "false", "no", "off"}


def _warn_unmatched_fleet_enabled() -> bool:
    raw = os.environ.get("ORBIT_WARS_WARN_UNMATCHED_FLEET")
    if raw is not None:
        return raw.lower() not in {"0", "false", "no", "off"}
    return os.environ.get("ORBIT_WARS_WARN_OOB_LAUNCHES", "1").lower() not in {"0", "false", "no", "off"}


def _adapter_sanity_warnings_enabled() -> bool:
    return os.environ.get("ORBIT_WARS_WARN_ADAPTER_SANITY", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _adapter_warn_once(seen: set[str], key: str, msg: str) -> None:
    if not _adapter_sanity_warnings_enabled() or key in seen:
        return
    seen.add(key)
    print(f"[orbit_wars:adapter-warning] {msg}", file=sys.stderr, flush=True)


def _policy_feature_dim(policy: OrbitWarsPolicy) -> int:
    return int(policy.feat_proj.in_features)


def _raw_rows(obs: Mapping[str, Any], name: str, width: int) -> np.ndarray:
    raw = obs.get(name, [])
    arr = np.asarray(raw, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, width), dtype=np.float64)
    try:
        return arr.reshape((-1, width))
    except ValueError:
        return np.zeros((0, width), dtype=np.float64)


@dataclass
class MicroTargetTiming:
    """Breakdown of ``micro_raycast`` (first-hit target selection: rays vs interval)."""

    calls: int = 0
    rays_calls: int = 0
    interval_calls: int = 0
    rays_planet_paths_s: float = 0.0
    rays_sim_s: float = 0.0
    rays_aggregate_s: float = 0.0
    interval_planet_paths_s: float = 0.0
    interval_precompute_s: float = 0.0
    interval_sweep_s: float = 0.0
    interval_check_calls: int = 0
    interval_check_s: float = 0.0
    interval_check_ray_mismatches: int = 0
    interval_check_superset_failures: int = 0

    def rays_total_s(self) -> float:
        return self.rays_planet_paths_s + self.rays_sim_s + self.rays_aggregate_s

    def interval_total_s(self) -> float:
        return (
            self.interval_planet_paths_s
            + self.interval_precompute_s
            + self.interval_sweep_s
            + self.interval_check_s
        )

    def format_suffix(self) -> str:
        if self.calls <= 0:
            return ""
        parts: list[str] = []
        if self.rays_calls:
            parts.append(
                "rays×"
                f"{self.rays_calls}={self.rays_total_s():.4f}s"
                f"(paths={self.rays_planet_paths_s:.4f}"
                f" sim={self.rays_sim_s:.4f}"
                f" agg={self.rays_aggregate_s:.4f})"
            )
        if self.interval_calls:
            chk = (
                f" chk={self.interval_check_s:.4f}"
                if self.interval_check_s > 0.0
                else ""
            )
            parts.append(
                "interval×"
                f"{self.interval_calls}={self.interval_total_s():.4f}s"
                f"(paths={self.interval_planet_paths_s:.4f}"
                f" pre={self.interval_precompute_s:.4f}"
                f" sweep={self.interval_sweep_s:.4f}{chk})"
            )
        return " micro_target{" + " ".join(parts) + "}"


@dataclass
class KaggleAgentCallTiming:
    """Host-side ``perf_counter`` slices for the last ``KaggleOrbitWarsAgent.__call__``."""

    obs_to_state_s: float = 0.0
    micro_iters: int = 0
    micro_obs_tensors_s: float = 0.0
    micro_policy_forward_s: float = 0.0
    micro_post_forward_s: float = 0.0
    micro_raycast_s: float = 0.0
    micro_target: MicroTargetTiming = field(default_factory=MicroTargetTiming)
    micro_target_s: float = 0.0
    micro_book_s: float = 0.0
    model_search: "ModelSearchTiming" = field(default_factory=lambda: ModelSearchTiming())

    def micro_sum_s(self) -> float:
        return (
            self.micro_obs_tensors_s
            + self.micro_policy_forward_s
            + self.micro_post_forward_s
            + self.micro_raycast_s
            + self.micro_target_s
            + self.micro_book_s
        )


@dataclass
class BatchObsPlanetTiming:
    """Breakdown of the 60-planet observation encoding loop."""

    meta_s: float = 0.0
    velocity_s: float = 0.0
    comet_s: float = 0.0
    feat_base_s: float = 0.0
    feat_survivor_s: float = 0.0
    feat_abort_s: float = 0.0
    rope_s: float = 0.0
    mask_s: float = 0.0

    def total_s(self) -> float:
        return (
            self.meta_s
            + self.velocity_s
            + self.comet_s
            + self.feat_base_s
            + self.feat_survivor_s
            + self.feat_abort_s
            + self.rope_s
            + self.mask_s
        )

    def format_suffix(self) -> str:
        if self.total_s() <= 0.0:
            return ""
        return (
            f"[meta={self.meta_s:.4f}"
            f" vel={self.velocity_s:.4f}"
            f" comet={self.comet_s:.4f}"
            f" base={self.feat_base_s:.4f}"
            f" surv={self.feat_survivor_s:.4f}"
            f" abort={self.feat_abort_s:.4f}"
            f" rope={self.rope_s:.4f}"
            f" mask={self.mask_s:.4f}]"
        )


@dataclass
class BatchObsTiming:
    """Breakdown of ``batch_plan`` observation tensor construction."""

    encode_calls: int = 0
    virt_states_s: float = 0.0
    setup_s: float = 0.0
    incoming_s: float = 0.0
    planet_loop_s: float = 0.0
    planets: BatchObsPlanetTiming = field(default_factory=BatchObsPlanetTiming)
    to_device_s: float = 0.0
    cat_s: float = 0.0

    def encode_total_s(self) -> float:
        return self.setup_s + self.incoming_s + self.planet_loop_s + self.to_device_s

    def total_s(self) -> float:
        return self.virt_states_s + self.encode_total_s() + self.cat_s

    def format_suffix(self) -> str:
        if self.encode_calls <= 0 and self.virt_states_s <= 0.0:
            return ""
        return (
            f"(virt={self.virt_states_s:.4f}"
            f" setup={self.setup_s:.4f}"
            f" incoming={self.incoming_s:.4f}"
            f" planets={self.planet_loop_s:.4f}{self.planets.format_suffix()}"
            f" h2d={self.to_device_s:.4f}"
            f" cat={self.cat_s:.4f})"
        )


@dataclass
class ModelSearchTiming:
    """Detailed timing for the optional halt-vs-launch rollout search."""

    choose_calls: int = 0
    choose_s: float = 0.0
    mode: str = ""
    stop_at_turn_end: bool = False
    branch_after_first_env_step: bool = True
    horizon_steps: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_lookup_s: float = 0.0
    cache_identify_s: float = 0.0
    cache_store_s: float = 0.0
    branch_rollouts: int = 0
    branch_rollout_s: float = 0.0
    rollout_steps: int = 0
    branch_setup_launch_geometry_s: float = 0.0
    branch_setup_seat_plans_s: float = 0.0
    branch_setup_root_actions_s: float = 0.0
    simulate_joint_calls: int = 0
    simulate_joint_s: float = 0.0
    opponent_greedy_calls: int = 0
    opponent_greedy_s: float = 0.0
    opponent_greedy_obs_to_state_calls: int = 0
    opponent_greedy_obs_to_state_s: float = 0.0
    opponent_greedy_policy_select_calls: int = 0
    opponent_greedy_policy_select_s: float = 0.0
    opponent_greedy_action_build_calls: int = 0
    opponent_greedy_action_build_s: float = 0.0
    ego_tail_build_calls: int = 0
    ego_tail_build_s: float = 0.0
    state_rebuild_calls: int = 0
    state_rebuild_s: float = 0.0
    kaggle_step_calls: int = 0
    kaggle_step_s: float = 0.0
    value_calls: int = 0
    value_s: float = 0.0
    value_obs_to_state_calls: int = 0
    value_obs_to_state_s: float = 0.0
    value_policy_select_calls: int = 0
    value_policy_select_s: float = 0.0
    value_eval_calls: int = 0
    value_eval_s: float = 0.0
    batch_plan_calls: int = 0
    batch_plan_s: float = 0.0
    batch_plan_rounds: int = 0
    batch_obs_tensors_s: float = 0.0
    batch_obs: BatchObsTiming = field(default_factory=BatchObsTiming)
    batch_policy_forward_s: float = 0.0
    batch_post_forward_s: float = 0.0
    batch_raycast_s: float = 0.0
    batch_target_head_s: float = 0.0
    batch_apply_s: float = 0.0
    public_obs_s: float = 0.0
    reward_s: float = 0.0
    trace_copy_s: float = 0.0
    bfs_expand_batches: int = 0
    bfs_expand_nodes: int = 0
    bfs_expand_candidates: int = 0
    bfs_expand_children: int = 0
    bfs_advance_batches: int = 0
    bfs_advance_nodes: int = 0
    bfs_frontier_peak: int = 0
    bfs_single_path_early_exit: int = 0
    turn_end_sample_batches: int = 0
    turn_end_sample_joint_actions: int = 0
    turn_end_sample_leaves: int = 0
    turn_end_sample_s: float = 0.0
    turn_sampling_completed_sequences: int = 0

    def branch_setup_accounted_s(self) -> float:
        return (
            self.branch_setup_launch_geometry_s
            + self.branch_setup_seat_plans_s
            + self.branch_setup_root_actions_s
        )

    def batch_plan_accounted_s(self) -> float:
        return (
            self.batch_obs_tensors_s
            + self.batch_policy_forward_s
            + self.batch_post_forward_s
            + self.batch_raycast_s
            + self.batch_target_head_s
            + self.batch_apply_s
        )

    def branch_rollout_accounted_s(self) -> float:
        return (
            self.branch_setup_accounted_s()
            + self.batch_plan_s
            + self.kaggle_step_s
            + self.state_rebuild_s
            + self.public_obs_s
            + self.reward_s
            + self.value_s
            + self.trace_copy_s
            + self.cache_store_s
        )

    def choose_accounted_s(self) -> float:
        return self.cache_lookup_s + self.cache_identify_s + self.branch_rollout_s + self.cache_store_s

    def batch_plan_unaccounted_s(self) -> float:
        return max(0.0, float(self.batch_plan_s) - float(self.batch_plan_accounted_s()))

    def branch_rollout_unaccounted_s(self) -> float:
        return max(0.0, float(self.branch_rollout_s) - float(self.branch_rollout_accounted_s()))

    def choose_unaccounted_s(self) -> float:
        return max(0.0, float(self.choose_s) - float(self.choose_accounted_s()))

    def format_suffix(self) -> str:
        if self.choose_calls <= 0:
            return ""
        return (
            " model_search{"
            f"mode={self.mode or 'unknown'}"
            f" stop_turn={int(self.stop_at_turn_end)}"
            f" branch_after_env1={int(self.branch_after_first_env_step)}"
            f" horizon={int(self.horizon_steps)} "
            f"choose×{self.choose_calls}={self.choose_s:.4f}s "
            f"cache(h={self.cache_hits} m={self.cache_misses}"
            f" lookup={self.cache_lookup_s:.4f}"
            f" id={self.cache_identify_s:.4f}"
            f" store={self.cache_store_s:.4f}) "
            f"rollouts×{self.branch_rollouts}={self.branch_rollout_s:.4f}s "
            f"steps={self.rollout_steps} "
            f"bfs(expand×{self.bfs_expand_batches}"
            f" nodes={self.bfs_expand_nodes}"
            f" cand={self.bfs_expand_candidates}"
            f" child={self.bfs_expand_children}"
            f" sampled={self.turn_sampling_completed_sequences}"
            f" adv×{self.bfs_advance_batches}"
            f" adv_nodes={self.bfs_advance_nodes}"
            f" frontier_peak={self.bfs_frontier_peak}"
            f" early_exit={self.bfs_single_path_early_exit}"
            f" turn_end×{self.turn_end_sample_batches}"
            f" joint={self.turn_end_sample_joint_actions}"
            f" leaves={self.turn_end_sample_leaves}"
            f" eval={self.turn_end_sample_s:.4f}s) "
            f"setup={self.branch_setup_accounted_s():.4f}s"
            f"(geom={self.branch_setup_launch_geometry_s:.4f}"
            f" plans={self.branch_setup_seat_plans_s:.4f}"
            f" root={self.branch_setup_root_actions_s:.4f}) "
            f"tail×{self.ego_tail_build_calls}={self.ego_tail_build_s:.4f}s "
            f"joint×{self.simulate_joint_calls}={self.simulate_joint_s:.4f}s "
            f"opp_greedy×{self.opponent_greedy_calls}={self.opponent_greedy_s:.4f}s "
            f"(obs2state={self.opponent_greedy_obs_to_state_s:.4f}"
            f" select={self.opponent_greedy_policy_select_s:.4f}"
            f" build={self.opponent_greedy_action_build_s:.4f}) "
            f"batch_plan×{self.batch_plan_calls}={self.batch_plan_s:.4f}s"
            f"(rounds={self.batch_plan_rounds}"
            f" obs={self.batch_obs_tensors_s:.4f}{self.batch_obs.format_suffix()}"
            f" fwd={self.batch_policy_forward_s:.4f}"
            f" post={self.batch_post_forward_s:.4f}"
            f" ray={self.batch_raycast_s:.4f}"
            f" target={self.batch_target_head_s:.4f}"
            f" apply={self.batch_apply_s:.4f}"
            f" other={self.batch_plan_unaccounted_s():.4f}) "
            f"obs2state×{self.state_rebuild_calls}={self.state_rebuild_s:.4f}s "
            f"kaggle_step×{self.kaggle_step_calls}={self.kaggle_step_s:.4f}s "
            f"public_obs={self.public_obs_s:.4f}s "
            f"reward={self.reward_s:.4f}s "
            f"trace_copy={self.trace_copy_s:.4f}s "
            f"value×{self.value_calls}={self.value_s:.4f}s"
            f"(obs2state={self.value_obs_to_state_s:.4f}"
            f" select={self.value_policy_select_s:.4f}"
            f" eval={self.value_eval_s:.4f}) "
            f"other_rollout={self.branch_rollout_unaccounted_s():.4f}s "
            f"other_choose={self.choose_unaccounted_s():.4f}s"
            "}"
        )


class OrbitWarsState(NamedTuple):
    planets: np.ndarray
    planet_active: np.ndarray
    initial_planets: np.ndarray
    initial_active: np.ndarray
    origin_frac_blocked: np.ndarray
    fleets: np.ndarray
    fleet_active: np.ndarray
    incoming_fleets: np.ndarray
    comet_paths: np.ndarray
    comet_path_lengths: np.ndarray
    comet_ships: np.ndarray
    comet_group_active: np.ndarray
    comet_path_index: np.ndarray
    comet_planet_ids: np.ndarray
    comet_slots: np.ndarray
    planet_collision_rank: np.ndarray
    next_fleet_id: np.ndarray
    angular_velocity: np.ndarray
    step_count: np.ndarray
    num_agents: np.ndarray
    rewards: np.ndarray
    done: np.ndarray
    overflow: np.ndarray


@dataclass(frozen=True)
class RewardSettings:
    ship_mass_share_coef: float = 1.0
    production_share_coef: float = 0.0
    terminal_win_loss_coef: float = 0.0
    terminal_loss: float = -1.0
    terminal_draw: float = 0.0
    terminal_win: float = 1.0
    time_bonus_coef: float = 0.0
    gamma: float = 1.0
    episode_steps: int = 500


@dataclass(frozen=True)
class ModelSearchSettings:
    horizon_steps: int
    reward: RewardSettings
    mode: str = MODEL_SEARCH_MODE_BINARY
    adaptive_horizon: bool = False
    adaptive_horizon_offset: int = 2
    min_overage_s: float = 15.0
    launch_probability_threshold: float | None = None
    greedy_launch_threshold: float | None = None
    branch_probability_threshold: float = 0.2
    max_branching_factor: int = 4
    branch_after_first_env_step: bool = True
    branch_micro_depth: int | None = None
    stop_at_turn_end: bool = False
    turn_end_opponent_samples: int = 0
    turn_sampling_max_samples: int | None = None


def _model_search_enabled(settings: ModelSearchSettings) -> bool:
    return bool(settings.adaptive_horizon) or int(settings.horizon_steps) > 0


def _remaining_overage_s(obs: Mapping[str, Any]) -> float | None:
    if "remainingOverageTime" not in obs:
        return None
    return float(obs.get("remainingOverageTime", 0.0))


def _search_time_scale_from_overage(
    obs: Mapping[str, Any],
    *,
    full_overage_s: float = 60.0,
) -> float:
    """Scale search budget by remaining overage fraction, clamped to ``[0, 1]``."""

    overage = _remaining_overage_s(obs)
    if overage is None:
        return 1.0
    denom = max(float(full_overage_s), 1e-6)
    return max(0.0, min(1.0, float(overage) / denom))


def _search_branching_enabled_for_env_step(
    settings: ModelSearchSettings,
    *,
    search_env_step_from_root: int,
    current_micro_idx: int | None = None,
) -> bool:
    if not bool(settings.branch_after_first_env_step or int(search_env_step_from_root) <= 0):
        return False
    max_micro_branch = getattr(settings, "branch_micro_depth", None)
    if max_micro_branch is None or current_micro_idx is None:
        return True
    return int(current_micro_idx) < max(0, int(max_micro_branch))


def _search_should_advance_closed_turn(
    settings: ModelSearchSettings,
    *,
    search_env_step_from_root: int,
) -> bool:
    if bool(settings.stop_at_turn_end) and int(search_env_step_from_root) == 0:
        return False
    return True


def _search_has_deadline(settings: ModelSearchSettings) -> bool:
    return True


def _search_uses_turn_end_opponent_samples(
    settings: ModelSearchSettings,
    *,
    search_env_step_from_root: int,
) -> bool:
    return bool(
        int(search_env_step_from_root) == 0
        and bool(settings.stop_at_turn_end)
        and int(settings.turn_end_opponent_samples) > 0
    )


def _model_search_mode_uses_turn_planner(mode: str) -> bool:
    return str(mode) in {MODEL_SEARCH_MODE_EGO_BFS, MODEL_SEARCH_MODE_TURN_SAMPLING}


def _model_search_allowed_for_obs(obs: Mapping[str, Any], settings: ModelSearchSettings) -> bool:
    overage = _remaining_overage_s(obs)
    if overage is None:
        return True
    return float(overage) >= float(settings.min_overage_s)


def _model_search_rollout_horizon(
    settings: ModelSearchSettings,
    *,
    launch_true_hit_tick: float,
) -> int:
    """Rollout depth (env steps) for halt-vs-launch search.

    With ``adaptive_horizon``, rollout depth is
    ``floor(launch_true_hit_tick) + adaptive_horizon_offset`` env steps.
    ``launch_true_hit_tick`` uses the same tick convention as
    ``apply_micro_launch_in_place``: tick ``k`` collides on env step ``k``
    after launch (tick 0 is the launch step itself).
    """

    if settings.adaptive_horizon:
        if float(launch_true_hit_tick) >= 500.0:
            cap = int(settings.horizon_steps)
            return max(1, cap) if cap > 0 else INCOMING_TA_BINS
        steps = int(math.floor(float(launch_true_hit_tick))) + int(settings.adaptive_horizon_offset)
        cap = int(settings.horizon_steps)
        if cap > 0:
            steps = min(steps, cap)
        return max(1, steps)
    return int(settings.horizon_steps)


def _log_model_search_horizon(
    settings: ModelSearchSettings,
    *,
    rollout_horizon: int,
    launch_true_hit_tick: float,
    step_count: int,
    ego_player: int,
    origin_slot: int = -1,
    target_slot: int = -1,
    micro_idx: int = -1,
    send: int = -1,
) -> None:
    if not _model_search_debug_enabled():
        return
    hit_tick = float(launch_true_hit_tick)
    if settings.adaptive_horizon:
        cap = int(settings.horizon_steps)
        cap_s = "none" if cap <= 0 else str(cap)
        mode = (
            f"adaptive floor({hit_tick:.3f})+{int(settings.adaptive_horizon_offset)} cap={cap_s}"
        )
    else:
        mode = f"fixed steps={int(settings.horizon_steps)}"
    origin_s = "" if int(origin_slot) < 0 else f" origin={int(origin_slot)}"
    target_s = "" if int(target_slot) < 0 else f" target={int(target_slot)}"
    micro_s = "" if int(micro_idx) < 0 else f" micro={int(micro_idx)}"
    send_s = "" if int(send) < 0 else f" send={int(send)}"
    _model_search_debug(
        f"horizon={int(rollout_horizon)} step={int(step_count)} ego={int(ego_player)}"
        f" true_hit_tick={hit_tick:.3f}{origin_s}{target_s}{micro_s}{send_s} ({mode})"
    )


def _reward_settings_from_training_args(training_args: Mapping[str, Any]) -> RewardSettings:
    def _get_float(name: str, default: float) -> float:
        value = training_args.get(name, default)
        return float(default if value is None else value)

    def _get_int(name: str, default: int) -> int:
        value = training_args.get(name, default)
        return int(default if value is None else value)

    ship_coef, prod_coef, terminal_coef, time_coef = resolve_reward_mix(training_args)

    return RewardSettings(
        ship_mass_share_coef=float(ship_coef),
        production_share_coef=float(prod_coef),
        terminal_win_loss_coef=float(terminal_coef),
        terminal_loss=_get_float("reward_terminal_loss", -1.0),
        terminal_draw=_get_float("reward_terminal_draw", 0.0),
        terminal_win=_get_float("reward_terminal_win", 1.0),
        time_bonus_coef=float(time_coef),
        gamma=_get_float("gamma", 1.0),
        episode_steps=_get_int("episode_steps", 500),
    )


def _reward_mix_ratios_np(state: OrbitWarsState, reward: RewardSettings) -> np.ndarray:
    owners = np.asarray(state.planets[:, 1], dtype=np.int32)
    active = np.asarray(state.planet_active, dtype=bool)
    safe_owners = np.clip(owners, 0, 3)

    planet_mask = active & (owners >= 0)
    planet_scores = np.zeros((4,), dtype=np.float32)
    if bool(np.any(planet_mask)):
        np.add.at(
            planet_scores,
            safe_owners[planet_mask],
            np.asarray(state.planets[planet_mask, 5], dtype=np.float32),
        )

    incoming = np.asarray(state.incoming_fleets, dtype=np.float32)
    incoming_active = incoming * active.astype(np.float32)[None, :, None]
    fleet_scores_a = np.sum(incoming_active, axis=(1, 2), dtype=np.float32)
    fleet_scores = np.pad(fleet_scores_a, (0, max(0, 4 - fleet_scores_a.shape[0]))).astype(np.float32, copy=False)
    ship_scores = planet_scores + fleet_scores[:4]
    ship_total = float(np.sum(ship_scores, dtype=np.float32)) + 1e-6
    ship_ratios = ship_scores / ship_total

    prod_scores = np.zeros((4,), dtype=np.float32)
    if bool(np.any(planet_mask)):
        np.add.at(
            prod_scores,
            safe_owners[planet_mask],
            np.asarray(state.planets[planet_mask, 6], dtype=np.float32),
        )
    prod_total = float(np.sum(prod_scores, dtype=np.float32)) + 1e-6
    prod_ratios = prod_scores / prod_total

    return (
        np.float32(reward.ship_mass_share_coef) * ship_ratios
        + np.float32(reward.production_share_coef) * prod_ratios
    ).astype(np.float32, copy=False)


def _terminal_rewards_np(state: OrbitWarsState, reward: RewardSettings) -> np.ndarray:
    owners = np.asarray(state.planets[:, 1], dtype=np.int32)
    active = np.asarray(state.planet_active, dtype=bool)
    safe_owners = np.clip(owners, 0, 3)
    scores = np.zeros((4,), dtype=np.float32)
    planet_mask = active & (owners >= 0)
    if bool(np.any(planet_mask)):
        np.add.at(
            scores,
            safe_owners[planet_mask],
            np.asarray(state.planets[planet_mask, 5], dtype=np.float32),
        )
    incoming = np.asarray(state.incoming_fleets, dtype=np.float32)
    incoming_active = incoming * active.astype(np.float32)[None, :, None]
    fleet_scores_a = np.sum(incoming_active, axis=(1, 2), dtype=np.float32)
    scores[: min(4, fleet_scores_a.shape[0])] += fleet_scores_a[:4]

    num_agents = int(np.asarray(state.num_agents))
    player_mask = np.arange(4, dtype=np.int32) < num_agents
    masked_scores = np.where(player_mask, scores, -np.inf)
    max_score = float(np.max(masked_scores))
    winner_mask = player_mask & (scores == max_score) & (max_score > 0.0)
    winner_count = int(np.sum(winner_mask.astype(np.int32)))
    draw_mask = winner_mask & (winner_count > 1)
    rewards = np.full((4,), np.float32(reward.terminal_loss), dtype=np.float32)
    rewards = np.where(draw_mask, np.float32(reward.terminal_draw), rewards)
    rewards = np.where(winner_mask & ~draw_mask, np.float32(reward.terminal_win), rewards)
    return rewards


def _reward_delta_np(
    state: OrbitWarsState,
    next_state: OrbitWarsState,
    ratios_pre: np.ndarray,
    reward: RewardSettings,
) -> np.ndarray:
    ratios_post = _reward_mix_ratios_np(next_state, reward)
    out = ratios_post - np.asarray(ratios_pre, dtype=np.float32)
    if bool(np.asarray(next_state.done)):
        out = out + np.float32(reward.terminal_win_loss_coef) * _terminal_rewards_np(next_state, reward)
        timeout_turn = float(max(int(reward.episode_steps) - 2, 1))
        pre_turn = float(np.asarray(state.step_count, dtype=np.float32))
        time_bonus_scale = np.float32(np.clip(1.0 - pre_turn / timeout_turn, 0.0, 1.0))
        timeout_done = pre_turn >= timeout_turn
        term = _terminal_rewards_np(next_state, reward)
        win_mask = (~timeout_done) & np.isclose(term, np.float32(reward.terminal_win))
        out = out + np.float32(reward.time_bonus_coef) * win_mask.astype(np.float32) * time_bonus_scale
    return out.astype(np.float32, copy=False)


def _cfg_get(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _check_4p_adapter_sanity(
    *,
    obs: Mapping[str, Any],
    config: Any,
    state: OrbitWarsState,
    ego_player: int,
    step_count: int,
    policy: OrbitWarsPolicy,
    policy_player_count: int,
    seen: set[str],
    context: str,
    use_4p_policy: Optional[bool] = None,
    live_opponents: Optional[int] = None,
) -> None:
    """Warn about 4p-only adapter mismatches without changing inference behavior."""

    num_agents = int(np.asarray(state.num_agents))
    cfg_agents = int(_cfg_get(config, "agentCount", num_agents))
    if max(num_agents, cfg_agents, policy_player_count) <= 2:
        return

    policy_fd = _policy_feature_dim(policy)
    expected_fd = obs_feature_dim_for_num_agents(
        policy_player_count,
        target_abort_enabled=bool(getattr(policy, "target_abort_enabled", False)),
    )
    if policy_fd != expected_fd:
        _adapter_warn_once(
            seen,
            f"{context}:feature-dim:{policy_fd}:{expected_fd}:{policy_player_count}",
            "policy feature width does not match adapter observation width "
            f"context={context} ego={ego_player} step={step_count} "
            f"policy_feature_dim={policy_fd} expected_feature_dim={expected_fd} "
            f"policy_player_count={policy_player_count} state_num_agents={num_agents}",
        )

    if cfg_agents != num_agents:
        _adapter_warn_once(
            seen,
            f"{context}:agent-count:{cfg_agents}:{num_agents}",
            "config agentCount differs from reconstructed state.num_agents "
            f"context={context} ego={ego_player} step={step_count} "
            f"config_agentCount={cfg_agents} state_num_agents={num_agents}",
        )

    if policy_player_count != num_agents and use_4p_policy is not False:
        _adapter_warn_once(
            seen,
            f"{context}:policy-player-count:{policy_player_count}:{num_agents}",
            "4p adapter is using a policy_player_count different from reconstructed state.num_agents "
            f"context={context} ego={ego_player} step={step_count} "
            f"policy_player_count={policy_player_count} state_num_agents={num_agents}",
        )

    incoming = np.asarray(state.incoming_fleets)
    if incoming.shape[0] != num_agents:
        _adapter_warn_once(
            seen,
            f"{context}:incoming-shape:{incoming.shape[0]}:{num_agents}",
            "incoming_fleets owner plane count differs from reconstructed state.num_agents "
            f"context={context} ego={ego_player} step={step_count} "
            f"incoming_planes={incoming.shape[0]} state_num_agents={num_agents}",
        )

    planets_in = _raw_rows(obs, "planets", 7)
    if planets_in.size:
        owners = planets_in[:, 1].astype(np.int32)
        bad = owners[(owners >= num_agents) | (owners < -1)]
        if bad.size:
            _adapter_warn_once(
                seen,
                f"{context}:bad-planet-owner:{int(bad[0])}:{num_agents}",
                "Kaggle planet owner id is outside expected player range "
                f"context={context} ego={ego_player} step={step_count} "
                f"owner={int(bad[0])} state_num_agents={num_agents}",
            )

    fleets_in = _raw_rows(obs, "fleets", 7)
    if fleets_in.size:
        fleet_owners = fleets_in[:, FLEET_OWNER].astype(np.int32)
        bad_fleet = fleet_owners[(fleet_owners < 0) | (fleet_owners >= num_agents)]
        if bad_fleet.size:
            _adapter_warn_once(
                seen,
                f"{context}:bad-fleet-owner:{int(bad_fleet[0])}:{num_agents}",
                "Kaggle fleet owner id is outside expected player range "
                f"context={context} ego={ego_player} step={step_count} "
                f"owner={int(bad_fleet[0])} state_num_agents={num_agents}",
            )
        visible_valid_owners = set(int(x) for x in fleet_owners if 0 <= int(x) < num_agents)
        missing_planes = [
            owner
            for owner in visible_valid_owners
            if owner < incoming.shape[0] and float(np.asarray(incoming[owner]).sum()) <= 0.0
        ]
        if len(visible_valid_owners) >= 2 and len(missing_planes) == len(visible_valid_owners):
            _adapter_warn_once(
                seen,
                f"{context}:all-visible-fleet-owners-no-incoming:{num_agents}",
                "visible 4p fleets produced no forecast incoming bins for any visible fleet owner "
                f"context={context} ego={ego_player} step={step_count} "
                f"visible_fleet_owners={sorted(visible_valid_owners)} "
                f"state_num_agents={num_agents}",
            )

    if use_4p_policy is not None and live_opponents is not None:
        if live_opponents >= 2 and not use_4p_policy:
            _adapter_warn_once(
                seen,
                f"{context}:unexpected-2p-switch:{live_opponents}",
                "dual-policy adapter selected 2p policy while two or more opponents appear live "
                f"context={context} ego={ego_player} step={step_count} live_opponents={live_opponents}",
            )
        if live_opponents < 2 and use_4p_policy:
            _adapter_warn_once(
                seen,
                f"{context}:unexpected-4p-switch:{live_opponents}",
                "dual-policy adapter selected 4p policy with fewer than two live opponents "
                f"context={context} ego={ego_player} step={step_count} live_opponents={live_opponents}",
            )


def _configure_cpu_threads() -> None:
    raw = os.environ.get("ORBIT_WARS_CPU_THREADS", str(DEFAULT_CPU_THREADS))
    try:
        n_threads = int(raw)
    except ValueError:
        return
    if n_threads < 1:
        return
    for name in CPU_THREAD_ENV_VARS:
        os.environ[name] = str(n_threads)
    torch.set_num_threads(n_threads)
    try:
        torch.set_num_interop_threads(n_threads)
    except RuntimeError:
        pass


def _as_array(rows: Any, width: int) -> np.ndarray:
    if rows is None:
        rows = []
    arr = np.asarray(rows, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, width), dtype=np.float32)
    arr = arr.reshape((-1, width))
    return arr.astype(np.float32, copy=False)


def _place_rows_by_id(
    rows: np.ndarray,
    width: int,
    *,
    dtype: np.dtype | type[np.floating] = np.float32,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    table = np.zeros((MAX_PLANETS, width), dtype=dtype)
    active = np.zeros((MAX_PLANETS,), dtype=np.bool_)
    id_to_slot: dict[int, int] = {}
    next_free = 0

    for row in rows[:MAX_PLANETS]:
        pid = int(row[0])
        if 0 <= pid < MAX_PLANETS and not active[pid]:
            slot = pid
        else:
            while next_free < MAX_PLANETS and active[next_free]:
                next_free += 1
            if next_free >= MAX_PLANETS:
                break
            slot = next_free
        table[slot, : min(width, row.shape[0])] = row[:width]
        active[slot] = True
        id_to_slot[pid] = slot

    return table, active, id_to_slot


def _fleet_speed(ships: float, max_speed: float = 6.0) -> float:
    if ships <= 1.0:
        return 1.0
    return float(min(max_speed, 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5))


def _planned_send(ships_avail: float, frac_idx: int) -> int:
    """Ships dispatched for a fraction choice (matches ``_build_turn_actions_torch_only``)."""

    avail = int(ships_avail)
    if avail <= 0:
        return 0
    send = int(math.floor(float(FRACTIONS[frac_idx]) * ships_avail))
    return min(max(1, send), avail)


def _planet_collision_rank_from_obs(
    planets_in: np.ndarray,
    id_to_slot: dict[int, int],
) -> np.ndarray:
    """Per-slot collision priority mirroring Kaggle ``obs0.planets`` iteration order."""

    rank = np.full((MAX_PLANETS,), MAX_PLANETS, dtype=np.int32)
    for pri, row in enumerate(planets_in[:MAX_PLANETS]):
        slot = id_to_slot.get(int(row[0]), -1)
        if 0 <= slot < MAX_PLANETS:
            rank[slot] = pri
    return rank


def _first_hit_planet_index(
    hit_mask: np.ndarray,
    collision_rank: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """First planet per ray in Kaggle ``obs0.planets`` scan order (``collision_rank``)."""

    n_planets = int(hit_mask.shape[-1])
    rank = np.asarray(collision_rank, dtype=np.int32)
    order = np.argsort(rank)
    priority = np.full((n_planets,), n_planets, dtype=np.int32)
    priority[order] = np.arange(n_planets, dtype=np.int32)
    big = np.int32(n_planets)
    score = np.where(hit_mask, priority[None, :], big)
    idx = np.argmin(score, axis=1).astype(np.int32)
    any_hit = np.any(hit_mask, axis=1)
    return idx, any_hit


def _expire_comets_for_forecast(
    planet_active: np.ndarray,
    initial_active: np.ndarray,
    comet_group_active: np.ndarray,
    comet_path_index: np.ndarray,
    comet_path_lengths: np.ndarray,
    comet_slots: np.ndarray,
    comet_planet_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Drop comets the official interpreter removes before fleet launch."""

    planet_active = planet_active.copy()
    initial_active = initial_active.copy()
    comet_group_active = comet_group_active.copy()
    comet_planet_ids = comet_planet_ids.copy()
    comet_slots = comet_slots.copy()

    for g in range(MAX_COMET_GROUPS):
        if not comet_group_active[g]:
            continue
        idx = int(comet_path_index[g])
        group_dead = True
        for k in range(4):
            slot = int(comet_slots[g, k])
            length = int(comet_path_lengths[g, k])
            if slot < 0 or slot >= MAX_PLANETS:
                continue
            if idx < length:
                group_dead = False
            else:
                planet_active[slot] = False
                initial_active[slot] = False
                comet_slots[g, k] = -1
                comet_planet_ids[g, k] = -1
        if group_dead:
            comet_group_active[g] = False

    return planet_active, initial_active, comet_group_active, comet_planet_ids, comet_slots


def _incoming_interfleet_np(
    incoming: np.ndarray,
    ego: int,
    num_agents: int,
    normalize_to_p0: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Interfleet largest-vs-2nd reduction; returns (signed_net[P,T], survivor_slot[P,T] int)."""

    a = incoming.shape[0]
    padded = np.pad(incoming.astype(np.float32), ((0, 4 - a), (0, 0), (0, 0)))
    ships = np.transpose(padded, (1, 2, 0))
    order = np.argsort(-ships, axis=-1)
    top_s = np.take_along_axis(ships, order[..., :1], axis=-1)[..., 0]
    second_s = np.take_along_axis(ships, order[..., 1:2], axis=-1)[..., 0]
    top_p = order[..., 0].astype(np.int32)
    survivor = np.where(top_s == second_s, 0.0, top_s - second_s)
    ego_j = int(ego)
    signed = np.where(survivor <= 0.0, 0.0, np.where(top_p == ego_j, survivor, -survivor))
    is_self = top_p == ego_j
    slot_le2 = np.where(survivor <= 0.0, 0, np.where(is_self, 1, 2))
    slot_gt2 = np.where(
        survivor <= 0.0,
        0,
        np.where(is_self, 1, _opponent_slot_4p_np(top_p, ego_j, normalize_to_p0)),
    )
    survivor_slot = np.where(num_agents <= 2, slot_le2, slot_gt2)
    survivor_slot = np.clip(survivor_slot, 0, NUM_OWNER_SLOTS - 1).astype(np.int32)
    return signed / 1000.0, survivor_slot


def _hide_enemy_far_right_after_resolution(
    incoming_net: np.ndarray,
    survivor_slot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Hide OOD enemy survivors in the final incoming bucket for model input."""

    if incoming_net.shape[1] == 0:
        return incoming_net, survivor_slot
    enemy_final = incoming_net[:, -1] < 0.0
    if not np.any(enemy_final):
        return incoming_net, survivor_slot
    incoming_net = incoming_net.copy()
    survivor_slot = survivor_slot.copy()
    incoming_net[enemy_final, -1] = 0.0
    survivor_slot[enemy_final, -1] = 0
    return incoming_net, survivor_slot


def _obs_qturns_to_p0_np(ego: int, num_agents: int, normalize_to_p0: bool) -> int:
    if not normalize_to_p0:
        return 0
    if num_agents <= 2:
        return int(_SEAT_QTURNS_TO_P0_2P[min(max(int(ego), 0), 1)])
    return int(_SEAT_QTURNS_TO_P0_4P[min(max(int(ego), 0), 3)])


def _rotate_vec_np(x: float, y: float, qturns: int) -> tuple[float, float]:
    q = int(qturns) & 3
    if q == 0:
        return float(x), float(y)
    if q == 1:
        return float(-y), float(x)
    if q == 2:
        return float(-x), float(-y)
    return float(y), float(-x)


def _rotate_xy_about_center_np(x: float, y: float, qturns: int) -> tuple[float, float]:
    rx, ry = _rotate_vec_np(float(x) - CENTER, float(y) - CENTER, qturns)
    return float(rx + CENTER), float(ry + CENTER)


def _opponent_slot_4p_np(owner: np.ndarray | int, ego: int, normalize_to_p0: bool) -> np.ndarray | int:
    o = np.asarray(owner, dtype=np.int32)
    if normalize_to_p0:
        row = _NORMALIZED_OWNER_SLOT_4P[min(max(int(ego), 0), 3)]
        out = row[np.clip(o, 0, 3)]
        return out if isinstance(owner, np.ndarray) else int(np.asarray(out))
    canonical = np.where(o < ego, 2 + o, 2 + (o - 1))
    return canonical if isinstance(owner, np.ndarray) else int(np.asarray(canonical))


def _remap_owner_np(
    owner: np.ndarray,
    ego: int,
    num_agents: int,
    normalize_to_p0: bool = False,
) -> np.ndarray:
    """Egocentric owner bucket per planet: 0 neutral, 1 self, 2–4 opponents (4p) or 2 (2p)."""

    o = np.asarray(owner, dtype=np.int32)
    ego_j = int(ego)
    is_neutral = o < 0
    is_self = o == ego_j
    if num_agents <= 2:
        opponent_slot = np.full_like(o, 2)
    else:
        opponent_slot = _opponent_slot_4p_np(o, ego_j, normalize_to_p0)
    out = np.where(is_neutral, 0, np.where(is_self, 1, opponent_slot))
    return np.minimum(out, NUM_OWNER_SLOTS - 1).astype(np.int64)


def _remap_owner(owner: float, ego: int, num_agents: int, normalize_to_p0: bool = False) -> int:
    o = int(owner)
    if o < 0:
        return 0
    if o == ego:
        return 1
    if num_agents <= 2:
        return 2
    return int(_opponent_slot_4p_np(o, ego, normalize_to_p0))


def _count_live_opponents(state: OrbitWarsState, ego: int) -> int:
    """Players other than ``ego`` still in the game (planets, fleets, or forecast incoming)."""

    num_agents = int(np.asarray(state.num_agents))
    alive = np.zeros(num_agents, dtype=bool)

    planets = np.asarray(state.planets)
    planet_active = np.asarray(state.planet_active)
    owners = planets[:, 1].astype(np.int32)
    valid_planet = planet_active & (owners >= 0) & (owners < num_agents)
    for p in np.unique(owners[valid_planet]):
        if int(p) != ego:
            alive[int(p)] = True

    fleets = np.asarray(state.fleets)
    fleet_active = np.asarray(state.fleet_active)
    if fleet_active.any():
        fo = fleets[:, FLEET_OWNER].astype(np.int32)
        fs = fleets[:, FLEET_SHIPS]
        in_play = fleet_active & (fo >= 0) & (fo < num_agents) & (fs > 0)
        for p in np.unique(fo[in_play]):
            if int(p) != ego:
                alive[int(p)] = True

    incoming = np.asarray(state.incoming_fleets)
    for p in range(min(num_agents, incoming.shape[0])):
        if p != ego and incoming[p].sum() > 0:
            alive[p] = True

    return int(alive.sum())


def _point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    delta = end - start
    l2 = float(np.dot(delta, delta))
    if l2 <= 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, delta) / l2, 0.0, 1.0))
    projection = start + t * delta
    return float(np.linalg.norm(point - projection))


def _swept_pair_hit(a0: np.ndarray, a1: np.ndarray, p0: np.ndarray, p1: np.ndarray, radius: float) -> bool:
    d0 = a0 - p0
    dv = (a1 - a0) - (p1 - p0)
    qa = float(np.dot(dv, dv))
    qb = float(2.0 * np.dot(d0, dv))
    qc = float(np.dot(d0, d0) - radius * radius)
    if qa < 1e-12:
        return qc <= 0.0
    disc = qb * qb - 4.0 * qa * qc
    if disc < 0.0:
        return False
    sqrt_disc = math.sqrt(max(disc, 0.0))
    t1 = (-qb - sqrt_disc) / (2.0 * qa)
    t2 = (-qb + sqrt_disc) / (2.0 * qa)
    return t2 >= 0.0 and t1 <= 1.0


def _next_planet_positions(
    planets: np.ndarray,
    planet_active: np.ndarray,
    initial_planets: np.ndarray,
    initial_active: np.ndarray,
    comet_paths: np.ndarray,
    comet_path_lengths: np.ndarray,
    comet_group_active: np.ndarray,
    comet_path_index: np.ndarray,
    comet_slots: np.ndarray,
    angular_velocity: float,
    step_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Host mirror of the path-only part used by the JAX raycast forecast."""

    old_pos = planets[:, PLANET_X : PLANET_Y + 1].copy()
    new_pos = old_pos.copy()

    init_pos = initial_planets[:, PLANET_X : PLANET_Y + 1]
    delta = init_pos - np.asarray([CENTER, CENTER], dtype=np.float64)
    orbital_r = np.linalg.norm(delta, axis=1)
    initial_angle = np.arctan2(delta[:, 1], delta[:, 0])
    rotating = planet_active & initial_active & (orbital_r + planets[:, PLANET_RADIUS] < ROTATION_RADIUS_LIMIT)
    current_angle = initial_angle + float(angular_velocity) * float(step_count)
    new_pos[rotating, 0] = CENTER + orbital_r[rotating] * np.cos(current_angle[rotating])
    new_pos[rotating, 1] = CENTER + orbital_r[rotating] * np.sin(current_angle[rotating])

    collision_enabled = planet_active.copy()
    next_path_index = comet_path_index + comet_group_active.astype(np.int32)
    expired_after_move = np.zeros_like(planet_active)

    for g in range(MAX_COMET_GROUPS):
        if not comet_group_active[g]:
            continue
        idx = int(next_path_index[g])
        for k in range(4):
            slot = int(comet_slots[g, k])
            if slot < 0 or slot >= MAX_PLANETS:
                continue
            length = int(comet_path_lengths[g, k])
            expired = idx >= length
            in_path = idx < length
            if not planet_active[slot]:
                # Kaggle removes expired comets after the expiry tick; no ghost collisions.
                collision_enabled[slot] = False
                continue
            if in_path:
                new_pos[slot] = comet_paths[g, k, max(idx, 0)]
                first_placement = planets[slot, PLANET_X] < 0.0
                collision_enabled[slot] = not first_placement
            elif expired:
                # Expiry tick only: stationary segment, then planet_active clears.
                collision_enabled[slot] = True
            else:
                collision_enabled[slot] = False
            expired_after_move[slot] = expired_after_move[slot] or expired

    return old_pos, new_pos, collision_enabled, next_path_index, planet_active & ~expired_after_move, initial_active & ~expired_after_move


def _forecast_incoming_fleets(
    planets: np.ndarray,
    planet_active: np.ndarray,
    initial_planets: np.ndarray,
    initial_active: np.ndarray,
    fleets_in: np.ndarray,
    comet_paths: np.ndarray,
    comet_path_lengths: np.ndarray,
    comet_group_active: np.ndarray,
    comet_path_index: np.ndarray,
    comet_planet_ids: np.ndarray,
    comet_slots: np.ndarray,
    num_agents: int,
    step_count: int,
    angular_velocity: float,
    collision_rank: np.ndarray,
    *,
    ship_speed: float = 6.0,
    horizon: int = INCOMING_TA_BINS,
    per_fleet_arrival: Optional[np.ndarray] = None,
    fleet_arrival_cache: Optional[FleetArrivalCache] = None,
) -> np.ndarray:
    """Fill policy incoming bins with only public fleets that hit a planet in the forecast.

    If ``per_fleet_arrival`` is provided, shape ``(len(fleets_in), 2)`` int32, row ``f`` is set to
    ``[hit_slot, hit_tick]`` on planet hit, else ``[-1, -1]`` (sun / OOB / no hit within horizon).
    """

    incoming = np.zeros((num_agents, MAX_PLANETS, INCOMING_TA_BINS), dtype=np.uint16)
    if fleets_in.size == 0:
        if per_fleet_arrival is not None and per_fleet_arrival.size:
            per_fleet_arrival[...] = -1
        return incoming
    if per_fleet_arrival is not None:
        per_fleet_arrival[...] = -1

    # float64 matches Kaggle orbit_wars.py fleet[2/3] += cos/sin * speed (Python float).
    positions = fleets_in[:, 2:4].astype(np.float64, copy=True)
    alive = np.ones((len(fleets_in),), dtype=np.bool_)
    angles = fleets_in[:, 4].astype(np.float64, copy=False)
    owners = fleets_in[:, 1].astype(np.int32, copy=False)
    ships = np.floor(fleets_in[:, 6].astype(np.float64, copy=False))
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float64)
    speeds = np.asarray([_fleet_speed(float(s), ship_speed) for s in ships], dtype=np.float64)
    comet_layout_key = _comet_layout_cache_key(comet_group_active, comet_planet_ids)
    sim_horizon = min(horizon, INCOMING_TA_BINS)

    p = planets.copy()
    pa = planet_active.copy()
    ia = initial_active.copy()
    cpi = comet_path_index.copy()
    skip_until_tick = np.zeros((len(fleets_in),), dtype=np.int32)
    resume_position = positions.copy()

    if fleet_arrival_cache is not None:
        for f in range(len(fleets_in)):
            row = fleets_in[f]
            key = _fleet_arrival_cache_key(row)
            entry = fleet_arrival_cache.entries.get(key)
            if entry is None:
                continue
            delta = int(step_count) - int(entry.step_count)
            if delta < 0:
                continue
            if entry.comet_layout_key != comet_layout_key:
                continue
            expected = np.asarray(entry.position, dtype=np.float64) + float(delta) * speeds[f] * dirs[f]
            if not np.allclose(positions[f], expected, atol=_FLEET_ARRIVAL_POS_ATOL, rtol=0.0):
                continue
            if int(entry.hit_slot) >= 0:
                remaining_tick = int(entry.hit_tick) - delta
                if remaining_tick < 0 or remaining_tick >= sim_horizon:
                    continue
                hit_slot = int(entry.hit_slot)
                owner = int(owners[f])
                if not (0 <= owner < num_agents and 0 <= hit_slot < MAX_PLANETS):
                    continue
                add = int(min(max(int(ships[f]), 0), 65535))
                cur = int(incoming[owner, hit_slot, remaining_tick])
                incoming[owner, hit_slot, remaining_tick] = min(cur + add, 65535)
                if per_fleet_arrival is not None and f < per_fleet_arrival.shape[0]:
                    per_fleet_arrival[f, 0] = hit_slot
                    per_fleet_arrival[f, 1] = remaining_tick
                alive[f] = False
                continue
            guaranteed_miss_prefix = int(entry.no_hit_ticks) - delta
            if guaranteed_miss_prefix <= 0:
                continue
            if guaranteed_miss_prefix >= sim_horizon:
                alive[f] = False
                continue
            skip_until_tick[f] = guaranteed_miss_prefix
            resume_position[f] = positions[f] + float(guaranteed_miss_prefix) * speeds[f] * dirs[f]

    for t in range(sim_horizon):
        old_pos, new_pos, collision_enabled, cpi_next, pa_next, ia_next = _next_planet_positions(
            p,
            pa,
            initial_planets,
            ia,
            comet_paths,
            comet_path_lengths,
            comet_group_active,
            cpi,
            comet_slots,
            angular_velocity,
            step_count + t,
        )

        for f in range(len(fleets_in)):
            if not alive[f]:
                continue
            owner = int(owners[f])
            if owner < 0 or owner >= num_agents or ships[f] <= 0:
                alive[f] = False
                continue
            if t < int(skip_until_tick[f]):
                continue
            if t == int(skip_until_tick[f]) and skip_until_tick[f] > 0:
                positions[f] = resume_position[f]
            a0 = positions[f]
            a1 = a0 + speeds[f] * dirs[f]

            hit_slot = -1
            for i in np.argsort(collision_rank):
                if collision_rank[i] >= MAX_PLANETS or not collision_enabled[i]:
                    continue
                if _swept_pair_hit(a0, a1, old_pos[i], new_pos[i], float(p[i, PLANET_RADIUS])):
                    hit_slot = int(i)
                    break
            if hit_slot >= 0:
                add = int(min(max(int(ships[f]), 0), 65535))
                cur = int(incoming[owner, hit_slot, t])
                incoming[owner, hit_slot, t] = min(cur + add, 65535)
                if per_fleet_arrival is not None and f < per_fleet_arrival.shape[0]:
                    per_fleet_arrival[f, 0] = hit_slot
                    per_fleet_arrival[f, 1] = t
                if fleet_arrival_cache is not None:
                    row_key = _fleet_arrival_cache_key(fleets_in[f])
                    fleet_arrival_cache.entries[row_key] = FleetArrivalCacheEntry(
                        owner=owner,
                        from_planet_id=int(fleets_in[f, 5]),
                        ships=int(ships[f]),
                        angle_key=int(row_key[4]),
                        step_count=int(step_count),
                        position=(float(fleets_in[f, 2]), float(fleets_in[f, 3])),
                        hit_slot=hit_slot,
                        hit_tick=int(t),
                        no_hit_ticks=0,
                        comet_layout_key=comet_layout_key,
                    )
                alive[f] = False
            else:
                if _point_to_segment_distance(np.asarray([CENTER, CENTER], dtype=np.float64), a0, a1) < SUN_RADIUS:
                    alive[f] = False
                    continue
                if a1[0] < 0.0 or a1[0] > BOARD_SIZE or a1[1] < 0.0 or a1[1] > BOARD_SIZE:
                    alive[f] = False
                    continue
                positions[f] = a1

        p[:, PLANET_X : PLANET_Y + 1] = new_pos
        pa = pa_next
        ia = ia_next
        cpi = cpi_next

    if fleet_arrival_cache is not None:
        for f in range(len(fleets_in)):
            if not alive[f]:
                continue
            row_key = _fleet_arrival_cache_key(fleets_in[f])
            fleet_arrival_cache.entries[row_key] = FleetArrivalCacheEntry(
                owner=int(owners[f]),
                from_planet_id=int(fleets_in[f, 5]),
                ships=int(ships[f]),
                angle_key=int(row_key[4]),
                step_count=int(step_count),
                position=(float(fleets_in[f, 2]), float(fleets_in[f, 3])),
                hit_slot=-1,
                hit_tick=-1,
                no_hit_ticks=int(sim_horizon),
                comet_layout_key=comet_layout_key,
            )

    return incoming


def _forecast_planet_paths_np(state: OrbitWarsState, horizon: int = INCOMING_TA_BINS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    planets = np.asarray(state.planets).copy()
    planet_active = np.asarray(state.planet_active).astype(bool).copy()
    initial_planets = np.asarray(state.initial_planets)
    initial_active = np.asarray(state.initial_active).astype(bool).copy()
    comet_paths = np.asarray(state.comet_paths)
    comet_path_lengths = np.asarray(state.comet_path_lengths)
    comet_group_active = np.asarray(state.comet_group_active).astype(bool)
    comet_path_index = np.asarray(state.comet_path_index).astype(np.int32).copy()
    comet_slots = np.asarray(state.comet_slots).astype(np.int32)
    comet_planet_ids = np.asarray(state.comet_planet_ids)
    planet_active, initial_active, comet_group_active, comet_planet_ids, comet_slots = _expire_comets_for_forecast(
        planet_active,
        initial_active,
        comet_group_active,
        comet_path_index,
        comet_path_lengths,
        comet_slots,
        comet_planet_ids,
    )
    angular_velocity = float(np.asarray(state.angular_velocity))
    step_count = int(np.asarray(state.step_count))

    p0_rows: list[np.ndarray] = []
    p1_rows: list[np.ndarray] = []
    active_rows: list[np.ndarray] = []
    for t in range(horizon):
        old_pos, new_pos, collision_enabled, cpi_next, pa_next, ia_next = _next_planet_positions(
            planets,
            planet_active,
            initial_planets,
            initial_active,
            comet_paths,
            comet_path_lengths,
            comet_group_active,
            comet_path_index,
            comet_slots,
            angular_velocity,
            step_count + t,
        )
        p0_rows.append(old_pos)
        p1_rows.append(new_pos)
        active_rows.append(collision_enabled)
        planets[:, PLANET_X : PLANET_Y + 1] = new_pos
        planet_active = pa_next
        initial_active = ia_next
        comet_path_index = cpi_next
    return (
        np.stack(p0_rows, axis=0).astype(np.float32),
        np.stack(p1_rows, axis=0).astype(np.float32),
        np.stack(active_rows, axis=0).astype(np.bool_),
    )


def _launch_geometry_from_obs(
    obs: Mapping[str, Any],
    config: Any = None,
) -> LaunchGeometryInputs:
    """Raw float64 geometry aligned to slots, for launch-time targeting only."""

    planets_in = np.asarray(obs.get("planets", []), dtype=np.float64)
    if planets_in.size == 0:
        planets_in = np.zeros((0, 7), dtype=np.float64)
    else:
        planets_in = planets_in.reshape((-1, 7))
    planets, planet_active, id_to_slot = _place_rows_by_id(planets_in, 7, dtype=np.float64)

    initial_in = np.asarray(obs.get("initial_planets", []), dtype=np.float64)
    if initial_in.size == 0:
        initial_in = np.zeros((0, 7), dtype=np.float64)
    else:
        initial_in = initial_in.reshape((-1, 7))
    initial_planets = np.zeros((MAX_PLANETS, 7), dtype=np.float64)
    initial_active = np.zeros((MAX_PLANETS,), dtype=np.bool_)
    for row in initial_in[:MAX_PLANETS]:
        pid = int(row[0])
        slot = id_to_slot.get(pid, pid if 0 <= pid < MAX_PLANETS else -1)
        if 0 <= slot < MAX_PLANETS:
            initial_planets[slot, :7] = row[:7]
            initial_active[slot] = True

    missing_initial = planet_active & ~initial_active
    initial_planets[missing_initial] = planets[missing_initial]
    initial_active[missing_initial] = True

    comet_paths = np.zeros((MAX_COMET_GROUPS, 4, MAX_COMET_PATH, 2), dtype=np.float64)
    comet_path_lengths = np.zeros((MAX_COMET_GROUPS, 4), dtype=np.int32)
    comet_group_active = np.zeros((MAX_COMET_GROUPS,), dtype=np.bool_)
    comet_path_index = np.full((MAX_COMET_GROUPS,), -1, dtype=np.int32)
    comet_slots = np.full((MAX_COMET_GROUPS, 4), -1, dtype=np.int32)
    for g, comet in enumerate((obs.get("comets") or [])[:MAX_COMET_GROUPS]):
        ids = list(comet.get("planet_ids", []))[:4]
        paths = list(comet.get("paths", []))[:4]
        comet_group_active[g] = True
        comet_path_index[g] = int(comet.get("path_index", -1))
        for k, pid_raw in enumerate(ids):
            pid = int(pid_raw)
            comet_slots[g, k] = id_to_slot.get(pid, -1)
        for k, path in enumerate(paths):
            p = np.asarray(path, dtype=np.float64).reshape((-1, 2))
            n = min(len(p), MAX_COMET_PATH)
            comet_paths[g, k, :n] = p[:n]
            comet_path_lengths[g, k] = n

    return LaunchGeometryInputs(
        planets=planets,
        planet_active=planet_active,
        initial_planets=initial_planets,
        initial_active=initial_active,
        comet_paths=comet_paths,
        comet_path_lengths=comet_path_lengths,
        comet_group_active=comet_group_active,
        comet_path_index=comet_path_index,
        comet_slots=comet_slots,
        angular_velocity=float(obs.get("angular_velocity", 0.0)),
    )


def _forecast_planet_paths_with_geometry_np(
    state: OrbitWarsState,
    geometry: LaunchGeometryInputs | None,
    *,
    horizon: int = INCOMING_TA_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if geometry is None:
        return _forecast_planet_paths_np(state, horizon=horizon)

    planets = np.asarray(geometry.planets, dtype=np.float64).copy()
    planet_active = np.asarray(geometry.planet_active, dtype=bool).copy()
    initial_planets = np.asarray(geometry.initial_planets, dtype=np.float64)
    initial_active = np.asarray(geometry.initial_active, dtype=bool).copy()
    comet_paths = np.asarray(geometry.comet_paths, dtype=np.float64)
    comet_path_lengths = np.asarray(geometry.comet_path_lengths, dtype=np.int32)
    comet_group_active = np.asarray(geometry.comet_group_active, dtype=bool)
    comet_path_index = np.asarray(geometry.comet_path_index, dtype=np.int32).copy()
    comet_slots = np.asarray(geometry.comet_slots, dtype=np.int32)
    comet_planet_ids = np.asarray(state.comet_planet_ids, dtype=np.int32)
    planet_active, initial_active, comet_group_active, comet_planet_ids, comet_slots = _expire_comets_for_forecast(
        planet_active,
        initial_active,
        comet_group_active,
        comet_path_index,
        comet_path_lengths,
        comet_slots,
        comet_planet_ids,
    )
    angular_velocity = float(geometry.angular_velocity)
    step_count = int(np.asarray(state.step_count))

    p0_rows: list[np.ndarray] = []
    p1_rows: list[np.ndarray] = []
    active_rows: list[np.ndarray] = []
    for t in range(horizon):
        old_pos, new_pos, collision_enabled, cpi_next, pa_next, ia_next = _next_planet_positions(
            planets,
            planet_active,
            initial_planets,
            initial_active,
            comet_paths,
            comet_path_lengths,
            comet_group_active,
            comet_path_index,
            comet_slots,
            angular_velocity,
            step_count + t,
        )
        p0_rows.append(old_pos)
        p1_rows.append(new_pos)
        active_rows.append(collision_enabled)
        planets[:, PLANET_X : PLANET_Y + 1] = new_pos
        planet_active = pa_next
        initial_active = ia_next
        comet_path_index = cpi_next
    return (
        np.stack(p0_rows, axis=0).astype(np.float64),
        np.stack(p1_rows, axis=0).astype(np.float64),
        np.stack(active_rows, axis=0).astype(np.bool_),
    )


def _simulate_discrete_ray_policy_hits_np(
    state: OrbitWarsState,
    origin_idx: int,
    frac_idx: int,
    *,
    ship_speed: float = 6.0,
    horizon: int = INCOMING_TA_BINS,
    n_rays: int = DEFAULT_RAYCAST_RAYS,
    target_timing: Optional[MicroTargetTiming] = None,
    planet_paths: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    launch_geometry: LaunchGeometryInputs | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Discrete per-tick fleet forward model; returns ray angles and first-hit bookkeeping."""

    if launch_geometry is None:
        planets = np.asarray(state.planets)
        current_active = np.asarray(state.planet_active).astype(bool)
    else:
        planets = np.asarray(launch_geometry.planets, dtype=np.float64)
        current_active = np.asarray(launch_geometry.planet_active, dtype=bool)
    if planet_paths is None:
        t_paths = perf_counter()
        p0, p1, active_by_tick = _forecast_planet_paths_with_geometry_np(
            state,
            launch_geometry,
            horizon=horizon,
        )
        if target_timing is not None:
            target_timing.rays_planet_paths_s += perf_counter() - t_paths
    else:
        p0, p1, active_by_tick = planet_paths
    t_sim = perf_counter()
    # Terminal events use per-tick collision flags (same as Kaggle fleet movement),
    # not the static visibility mask used only when marking valid policy targets.
    radii = planets[:, PLANET_RADIUS].astype(np.float64)
    origin_xy = planets[origin_idx, PLANET_X : PLANET_Y + 1].astype(np.float64)
    origin_radius = float(planets[origin_idx, PLANET_RADIUS])
    ships_avail = float(np.asarray(state.planets)[origin_idx, 5])
    send = _planned_send(ships_avail, frac_idx)
    speed = _fleet_speed(float(max(send, 1)), ship_speed)
    collision_rank = np.asarray(state.planet_collision_rank, dtype=np.int32)

    angles = np.arange(n_rays, dtype=np.float64) * (2.0 * math.pi / float(n_rays))
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float64)
    pos = origin_xy[None, :] + (origin_radius + 0.1) * dirs
    done_policy = np.zeros((n_rays,), dtype=np.bool_)
    done_true = np.zeros((n_rays,), dtype=np.bool_)
    policy_code = np.full((n_rays,), -1, dtype=np.int32)
    policy_tick = np.full((n_rays,), 10_000, dtype=np.int32)
    policy_kind = np.zeros((n_rays,), dtype=np.int8)
    true_code = np.full((n_rays,), -1, dtype=np.int32)
    true_tick = np.full((n_rays,), 10_000, dtype=np.int32)

    sun = np.asarray([CENTER, CENTER], dtype=np.float64)
    for t in range(horizon):
        if bool(np.all(done_policy & done_true)):
            break
        a0 = pos
        a1 = pos + speed * dirs

        d0 = a0[:, None, :] - p0[t][None, :, :]
        dv = (a1[:, None, :] - a0[:, None, :]) - (p1[t][None, :, :] - p0[t][None, :, :])
        qa = np.sum(dv * dv, axis=-1)
        qb = 2.0 * np.sum(d0 * dv, axis=-1)
        qc = np.sum(d0 * d0, axis=-1) - radii[None, :] ** 2
        disc = qb * qb - 4.0 * qa * qc
        static_hit = qc <= 0.0
        qa_safe = np.where(qa < 1e-12, 1.0, qa)
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
        t1 = (-qb - sqrt_disc) / (2.0 * qa_safe)
        t2 = (-qb + sqrt_disc) / (2.0 * qa_safe)
        moving_hit = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
        hit_raw = np.where(qa < 1e-12, static_hit, moving_hit)
        hit_true = hit_raw & active_by_tick[t][None, :]
        hit_policy = hit_true

        idx_true, any_true = _first_hit_planet_index(hit_true, collision_rank)
        idx_policy, any_policy = _first_hit_planet_index(hit_policy, collision_rank)

        delta = a1 - a0
        l2 = np.sum(delta * delta, axis=1)
        proj = np.zeros((n_rays,), dtype=np.float64)
        nonzero = l2 > 1e-12
        proj[nonzero] = np.sum((sun[None, :] - a0[nonzero]) * delta[nonzero], axis=1) / l2[nonzero]
        proj = np.clip(proj, 0.0, 1.0)
        closest = a0 + proj[:, None] * delta
        sun_hit = np.linalg.norm(closest - sun[None, :], axis=1) < SUN_RADIUS
        in_bounds = (a1[:, 0] >= 0.0) & (a1[:, 0] <= BOARD_SIZE) & (a1[:, 1] >= 0.0) & (a1[:, 1] <= BOARD_SIZE)
        oob = ~in_bounds

        had_policy = any_policy | sun_hit | oob
        new_policy = (~done_policy) & had_policy
        policy_code[new_policy] = np.where(
            any_policy[new_policy],
            idx_policy[new_policy],
            -1,
        )
        policy_tick[new_policy] = t
        policy_kind[new_policy] = np.where(
            any_policy[new_policy],
            np.int8(1),
            np.where(sun_hit[new_policy], np.int8(2), np.int8(3)),
        )
        done_policy |= had_policy

        had_true = any_true | sun_hit | oob
        new_true = (~done_true) & had_true
        true_code[new_true] = np.where(
            any_true[new_true],
            idx_true[new_true],
            -1,
        )
        true_tick[new_true] = t
        done_true |= had_true

        pos = a1

    if target_timing is not None:
        target_timing.rays_sim_s += perf_counter() - t_sim

    return angles, policy_code, policy_tick, policy_kind, true_code, true_tick, done_policy


def discrete_policy_rays_hit_planet_mask(
    state: OrbitWarsState,
    origin_idx: int,
    frac_idx: int,
    *,
    ship_speed: float = 6.0,
    horizon: int = INCOMING_TA_BINS,
    n_rays: int = DEFAULT_RAYCAST_RAYS,
) -> tuple[np.ndarray, np.ndarray]:
    """Ray angles whose policy-side first stop is an active planet (same discrete sim as training).

    Returns ``(angles, hits)`` with ``hits[i]`` true iff discrete ray ``i`` first resolved to a planet index
    (``policy_code[i] >= 0``), as opposed to sun or out-of-bounds alone.
    """

    angles, policy_code, _, _, _, _, done_policy = _simulate_discrete_ray_policy_hits_np(
        state, origin_idx, frac_idx, ship_speed=ship_speed, horizon=horizon, n_rays=n_rays
    )
    hits = (policy_code >= 0) & done_policy
    return angles, hits


def _raycast_targets_np(
    state: OrbitWarsState,
    origin_idx: int,
    frac_idx: int,
    *,
    ship_speed: float = 6.0,
    horizon: int = INCOMING_TA_BINS,
    n_rays: int = DEFAULT_RAYCAST_RAYS,
    target_timing: Optional[MicroTargetTiming] = None,
    launch_geometry: LaunchGeometryInputs | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """NumPy version of the rollout discrete first-hit ray target sampler."""

    if target_timing is not None:
        target_timing.rays_calls += 1

    angles, policy_code, policy_tick, _, true_code, true_tick, done_policy = (
        _simulate_discrete_ray_policy_hits_np(
            state,
            origin_idx,
            frac_idx,
            ship_speed=ship_speed,
            horizon=horizon,
            n_rays=n_rays,
            target_timing=target_timing,
            launch_geometry=launch_geometry,
        )
    )
    planets = np.asarray(state.planets)
    current_active = np.asarray(state.planet_active).astype(bool)

    out_angle = np.zeros((MAX_PLANETS,), dtype=np.float64)
    valid = np.zeros((MAX_PLANETS,), dtype=np.bool_)
    hit_tick = np.zeros((MAX_PLANETS,), dtype=np.float32)
    true_planet = np.full((MAX_PLANETS,), -1, dtype=np.int32)
    true_hit_tick = np.full((MAX_PLANETS,), 500.0, dtype=np.float32)
    t_agg = perf_counter()
    for target in range(MAX_PLANETS):
        ray_idx = np.flatnonzero((policy_code == target) & done_policy)
        if ray_idx.size == 0:
            continue
        best_pos = int(np.lexsort((ray_idx, policy_tick[ray_idx]))[0])
        ray = int(ray_idx[best_pos])
        out_angle[target] = float(angles[ray] % (2.0 * math.pi))
        valid[target] = True
        hit_tick[target] = float(policy_tick[ray])
        if 0 <= int(true_code[ray]) < MAX_PLANETS:
            true_planet[target] = int(true_code[ray])
            true_hit_tick[target] = float(true_tick[ray])
    if target_timing is not None:
        target_timing.rays_aggregate_s += perf_counter() - t_agg
    valid &= current_active
    true_planet = np.where(valid, true_planet, -1).astype(np.int32)
    true_hit_tick = np.where(valid, true_hit_tick, 500.0).astype(np.float32)
    return out_angle, valid, hit_tick, true_planet, true_hit_tick


def _collision_object_order(collision_rank: np.ndarray) -> list[int]:
    """Planet slot order for same-tick occlusion (matches Kaggle list iteration)."""

    rank = np.asarray(collision_rank, dtype=np.int32)
    return [int(i) for i in np.argsort(rank) if int(rank[i]) < MAX_PLANETS]


@dataclass
class IntervalMicroGeometry:
    """Shared interval occlusion inputs for sweep + per-ray checks."""

    geometry: str
    origin_idx: int
    origin_xy: np.ndarray
    origin_radius: float
    speed: float
    horizon: int
    samples_per_span: int
    object_order: list[int]
    p0_by_tick: np.ndarray
    p1_by_tick: np.ndarray
    active_by_tick: np.ndarray
    radii: np.ndarray
    events: list[Any] | None = None
    precomputed_hits: list[Any] | None = None
    occlusion_cache: Any | None = None


@dataclass
class DiscreteRaycastSim:
    """One ``_simulate_discrete_ray_policy_hits_np`` run (stored for interval checks)."""

    angles: np.ndarray
    policy_code: np.ndarray
    policy_tick: np.ndarray
    policy_kind: np.ndarray
    true_code: np.ndarray
    true_tick: np.ndarray
    done_policy: np.ndarray


@dataclass
class LaunchGeometryInputs:
    """Float64 geometry view from the raw observation for launch-time targeting."""

    planets: np.ndarray
    planet_active: np.ndarray
    initial_planets: np.ndarray
    initial_active: np.ndarray
    comet_paths: np.ndarray
    comet_path_lengths: np.ndarray
    comet_group_active: np.ndarray
    comet_path_index: np.ndarray
    comet_slots: np.ndarray
    angular_velocity: float


@dataclass
class PlannedLaunchAction:
    action_index: int
    micro_idx: int
    origin_slot: int
    frac_idx: int
    target_slot: int
    planned_send: int
    policy_hit_tick: float
    true_hit_tick: float
    coarse_angle: float
    planets_snapshot: np.ndarray
    refine_job: Any | None = None


@dataclass
class SearchFirstContactTargets:
    angles: np.ndarray
    valid: np.ndarray
    eta: np.ndarray
    origin_xy: np.ndarray
    origin_radius: float
    speed: float
    p0_by_tick: np.ndarray
    p1_by_tick: np.ndarray
    active_by_tick: np.ndarray
    radii: np.ndarray
    collision_rank: np.ndarray


def _first_hit_signature(kind: str, code: int, tick: int) -> tuple[str, int]:
    """Compare what is hit, not when (tick ignored)."""

    del tick
    return (str(kind), int(code))


def _discrete_first_hit_at_angle_np(
    angle: float,
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    p0_by_tick: np.ndarray,
    p1_by_tick: np.ndarray,
    radii: np.ndarray,
    active_by_tick: np.ndarray,
    collision_rank: np.ndarray,
    *,
    horizon: int = INCOMING_TA_BINS,
    include_sun: bool = True,
    include_board: bool = True,
) -> tuple[str, int, int]:
    """Discrete per-tick first hit at ``angle`` (Kaggle fleet rules)."""

    from orbit_wars_pt.geometry import TAU

    origin = np.asarray(origin_xy, dtype=np.float64)
    theta = float(angle) % TAU
    direction = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
    launch_off = float(origin_radius) + 0.1
    pos = origin + launch_off * direction
    sun = np.asarray([CENTER, CENTER], dtype=np.float64)
    ticks = int(active_by_tick.shape[0])
    planets = int(radii.shape[0])
    order = _collision_object_order(collision_rank)

    for tick in range(min(ticks, int(horizon))):
        a0 = pos
        a1 = pos + float(speed) * direction

        for slot in order:
            if slot < 0 or slot >= planets or not active_by_tick[tick, slot]:
                continue
            p0 = p0_by_tick[tick, slot]
            p1 = p1_by_tick[tick, slot]
            d0 = a0 - p0
            dv = (a1 - a0) - (p1 - p0)
            qa = float(np.dot(dv, dv))
            qb = float(2.0 * np.dot(d0, dv))
            qc = float(np.dot(d0, d0) - float(radii[slot]) ** 2)
            if qa < 1e-12:
                if qc <= 0.0:
                    return ("planet", int(slot), int(tick))
                continue
            disc = qb * qb - 4.0 * qa * qc
            if disc < 0.0:
                continue
            sd = math.sqrt(max(disc, 0.0))
            t1 = (-qb - sd) / (2.0 * qa)
            t2 = (-qb + sd) / (2.0 * qa)
            if t2 >= 0.0 and t1 <= 1.0:
                return ("planet", int(slot), int(tick))

        if include_sun:
            delta = a1 - a0
            l2 = float(np.dot(delta, delta))
            if l2 > 1e-12:
                proj = float(np.clip(np.dot(sun - a0, delta) / l2, 0.0, 1.0))
                closest = a0 + proj * delta
            else:
                closest = a0
            if float(np.linalg.norm(closest - sun)) < float(SUN_RADIUS):
                return ("sun", -1, int(tick))

        if include_board:
            if not (
                0.0 <= a1[0] <= BOARD_SIZE
                and 0.0 <= a1[1] <= BOARD_SIZE
            ):
                return ("board", -1, int(tick))

        pos = a1

    return ("none", -1, -1)


def _build_interval_micro_geometry(
    state: OrbitWarsState,
    origin_idx: int,
    frac_idx: int,
    *,
    ship_speed: float,
    horizon: int,
    samples_per_span: int,
    target_timing: Optional[MicroTargetTiming],
    launch_geometry: LaunchGeometryInputs | None = None,
    geometry_override: str | None = None,
) -> IntervalMicroGeometry:
    from orbit_wars_pt.interval_geometry_np import (
        collect_hit_events,
        precompute_tick_planet_hits,
    )

    planets = (
        np.asarray(launch_geometry.planets, dtype=np.float64)
        if launch_geometry is not None
        else np.asarray(state.planets)
    )
    origin_xy = np.asarray(planets[origin_idx, PLANET_X : PLANET_Y + 1], dtype=np.float64)
    origin_radius = float(planets[origin_idx, PLANET_RADIUS])
    ships_avail = float(np.asarray(state.planets)[origin_idx, 5])
    send = _planned_send(ships_avail, frac_idx)
    speed = _fleet_speed(float(max(send, 1)), ship_speed)
    collision_rank = np.asarray(state.planet_collision_rank, dtype=np.int32)
    object_order = _collision_object_order(collision_rank)
    geometry = str(geometry_override) if geometry_override is not None else _interval_geometry_mode()

    t_paths = perf_counter()
    p0, p1, active_by_tick = _forecast_planet_paths_with_geometry_np(
        state,
        launch_geometry,
        horizon=horizon,
    )
    if target_timing is not None:
        target_timing.interval_planet_paths_s += perf_counter() - t_paths

    events = None
    pre = None
    t_collect = perf_counter()
    if geometry in ("orthogonal", "tangent"):
        events = collect_hit_events(
            origin_xy,
            origin_radius,
            speed,
            planets.astype(np.float64),
            (
                np.asarray(launch_geometry.planet_active, dtype=bool)
                if launch_geometry is not None
                else np.asarray(state.planet_active, dtype=bool)
            ),
            (
                np.asarray(launch_geometry.initial_planets, dtype=np.float64)
                if launch_geometry is not None
                else np.asarray(state.initial_planets, dtype=np.float64)
            ),
            (
                np.asarray(launch_geometry.initial_active, dtype=bool)
                if launch_geometry is not None
                else np.asarray(state.initial_active, dtype=bool)
            ),
            (
                np.asarray(launch_geometry.comet_paths, dtype=np.float64)
                if launch_geometry is not None
                else np.asarray(state.comet_paths, dtype=np.float64)
            ),
            (
                np.asarray(launch_geometry.comet_path_lengths, dtype=np.int32)
                if launch_geometry is not None
                else np.asarray(state.comet_path_lengths, dtype=np.int32)
            ),
            (
                np.asarray(launch_geometry.comet_group_active, dtype=bool)
                if launch_geometry is not None
                else np.asarray(state.comet_group_active, dtype=bool)
            ),
            (
                np.asarray(launch_geometry.comet_path_index, dtype=np.int32)
                if launch_geometry is not None
                else np.asarray(state.comet_path_index, dtype=np.int32)
            ),
            (
                np.asarray(launch_geometry.comet_slots, dtype=np.int32)
                if launch_geometry is not None
                else np.asarray(state.comet_slots, dtype=np.int32)
            ),
            np.asarray(state.comet_planet_ids, dtype=np.int32),
            (
                float(launch_geometry.angular_velocity)
                if launch_geometry is not None
                else float(np.asarray(state.angular_velocity))
            ),
            int(np.asarray(state.step_count)),
            horizon=float(horizon),
        )
    else:
        pre = precompute_tick_planet_hits(
            origin_xy,
            origin_radius,
            speed,
            p0.astype(np.float64),
            p1.astype(np.float64),
            planets[:, PLANET_RADIUS].astype(np.float64),
            active_by_tick,
            samples_per_span=int(samples_per_span),
        )
    if target_timing is not None:
        target_timing.interval_precompute_s += perf_counter() - t_collect

    occlusion_cache = None
    if geometry in ("orthogonal", "tangent") and events is not None:
        from orbit_wars_pt.interval_geometry_np import build_occlusion_walk_cache

        occlusion_cache = build_occlusion_walk_cache(
            events,
            object_order,
            origin_xy=origin_xy,
            origin_radius=origin_radius,
            speed=float(speed),
            horizon=int(horizon),
        )

    return IntervalMicroGeometry(
        geometry=geometry,
        origin_idx=int(origin_idx),
        origin_xy=origin_xy,
        origin_radius=origin_radius,
        speed=float(speed),
        horizon=int(horizon),
        samples_per_span=int(samples_per_span),
        object_order=object_order,
        p0_by_tick=p0.astype(np.float64),
        p1_by_tick=p1.astype(np.float64),
        active_by_tick=active_by_tick,
        radii=planets[:, PLANET_RADIUS].astype(np.float64),
        events=events,
        precomputed_hits=pre,
        occlusion_cache=occlusion_cache,
    )


def _sweep_interval_from_geometry(
    geom: IntervalMicroGeometry,
    num_planets: int,
    *,
    target_timing: Optional[MicroTargetTiming],
    allow_edge_aim: bool,
    refine_boundaries: bool,
    game_step: int = -1,
    micro_idx: int = -1,
    ego_player: int = -1,
    frac_idx: int = -1,
    phase: str | None = None,
    selected_target_slot: int = -1,
    return_jobs: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Any | None]]:
    from orbit_wars_pt.interval_geometry_np import (
        sweep_interval_best_targets,
        sweep_interval_best_targets_from_events,
    )

    t_sweep = perf_counter()
    if geom.geometry in ("orthogonal", "tangent"):
        sweep_result = (
            sweep_interval_best_targets_from_events(
                geom.events or [],
                num_planets,
                object_order=geom.object_order,
                origin_xy=geom.origin_xy,
                origin_radius=geom.origin_radius,
                speed=geom.speed,
                horizon=geom.horizon,
                p0_by_tick=geom.p0_by_tick,
                p1_by_tick=geom.p1_by_tick,
                radii=geom.radii,
                active_by_tick=geom.active_by_tick,
                occlusion_cache=geom.occlusion_cache,
                allow_edge_aim=allow_edge_aim,
                refine_boundaries=refine_boundaries,
                debug_context={
                    "phase": phase,
                    "game_step": int(game_step),
                    "ego_player": int(ego_player),
                    "origin_slot": int(geom.origin_idx),
                    "frac_idx": int(frac_idx),
                    "micro_idx": int(micro_idx),
                    "selected_target_slot": int(selected_target_slot),
                },
                selected_slots=(
                    {int(selected_target_slot)}
                    if phase == "submit_refine" and 0 <= int(selected_target_slot) < int(num_planets)
                    else None
                ),
                return_jobs=return_jobs,
            )
        )
        if return_jobs:
            out_angle, width, valid, _overflow, hit_tick_i, refine_jobs = sweep_result
        else:
            out_angle, width, valid, _overflow, hit_tick_i = sweep_result
    else:
        out_angle, width, valid, _overflow, hit_tick_i = sweep_interval_best_targets(
            geom.precomputed_hits or [],
            object_order=geom.object_order,
            origin_xy=geom.origin_xy,
            origin_radius=geom.origin_radius,
            speed=geom.speed,
            samples_per_span=geom.samples_per_span,
        )
    if target_timing is not None:
        target_timing.interval_sweep_s += perf_counter() - t_sweep
    del width
    result = (
        np.asarray(out_angle, dtype=np.float64),
        np.asarray(valid, dtype=np.bool_),
        hit_tick_i,
    )
    if return_jobs:
        return (*result, refine_jobs)
    return result


def _run_discrete_raycast_sim(
    state: OrbitWarsState,
    origin_idx: int,
    frac_idx: int,
    *,
    ship_speed: float,
    horizon: int,
    n_rays: int,
    target_timing: Optional[MicroTargetTiming],
    planet_paths: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    launch_geometry: LaunchGeometryInputs | None = None,
) -> DiscreteRaycastSim:
    if target_timing is not None:
        target_timing.rays_calls += 1
    angles, policy_code, policy_tick, policy_kind, true_code, true_tick, done_policy = (
        _simulate_discrete_ray_policy_hits_np(
            state,
            origin_idx,
            frac_idx,
            ship_speed=ship_speed,
            horizon=horizon,
            n_rays=n_rays,
            target_timing=target_timing,
            planet_paths=planet_paths,
            launch_geometry=launch_geometry,
        )
    )
    return DiscreteRaycastSim(
        angles=np.asarray(angles),
        policy_code=np.asarray(policy_code),
        policy_tick=np.asarray(policy_tick),
        policy_kind=np.asarray(policy_kind),
        true_code=np.asarray(true_code),
        true_tick=np.asarray(true_tick),
        done_policy=np.asarray(done_policy, dtype=bool),
    )


def _raycast_signature_from_sim(sim: DiscreteRaycastSim, ray_i: int) -> tuple[str, int]:
    """First-hit kind/code from the stored vectorized sim (no per-ray re-simulation)."""

    if not bool(sim.done_policy[ray_i]):
        return ("none", -1)
    kind_code = int(sim.policy_kind[ray_i])
    if kind_code == 1:
        return ("planet", int(sim.policy_code[ray_i]))
    if kind_code == 2:
        return ("sun", -1)
    if kind_code == 3:
        return ("board", -1)
    return ("none", -1)


def _interval_first_hit_signature(
    geom: IntervalMicroGeometry,
    angle: float,
) -> tuple[str, int]:
    if geom.geometry in ("orthogonal", "tangent"):
        if geom.occlusion_cache is None:
            raise ValueError("occlusion_cache required for orthogonal/tangent check")
        from orbit_wars_pt.interval_geometry_np import first_hit_signature_occlusion_walk

        return first_hit_signature_occlusion_walk(float(angle), geom.occlusion_cache)
    if geom.precomputed_hits is None:
        raise ValueError("precomputed_hits required for sampled interval geometry")
    from orbit_wars_pt.interval_geometry_np import first_hit_at_angle_interval

    kind, code, _tick = first_hit_at_angle_interval(
        float(angle),
        geom.precomputed_hits,
        object_order=geom.object_order,
        origin_xy=geom.origin_xy,
        origin_radius=geom.origin_radius,
        speed=geom.speed,
        samples_per_span=geom.samples_per_span,
    )
    return (str(kind), int(code))


def _check_interval_raycast_micro_consistency(
    geom: IntervalMicroGeometry,
    interval_valid: np.ndarray,
    sim: DiscreteRaycastSim,
    *,
    target_timing: Optional[MicroTargetTiming] = None,
    game_step: int = -1,
    micro_idx: int = -1,
) -> None:
    """Warn if per-ray post-occlusion first hits or hittable sets disagree."""

    n_rays = int(sim.angles.shape[0])
    ray_mismatches: list[str] = []
    n_mismatch = 0
    ray_planet_slots: set[int] = set()
    done = np.flatnonzero(sim.done_policy)

    for i in done:
        i = int(i)
        rv = _raycast_signature_from_sim(sim, i)
        iv = _interval_first_hit_signature(geom, float(sim.angles[i]))
        if rv[0] == "planet":
            ray_planet_slots.add(int(rv[1]))
        if rv != iv:
            n_mismatch += 1
            if len(ray_mismatches) < 8:
                ray_mismatches.append(
                    f"ray={i} angle={float(sim.angles[i]):.6f} "
                    f"interval=({iv[0]}, {iv[1]}) raycast=({rv[0]}, {rv[1]})"
                )

    interval_hittable = {
        int(s)
        for s in np.flatnonzero(np.asarray(interval_valid, dtype=bool))
        if 0 <= int(s) < MAX_PLANETS and int(s) != int(geom.origin_idx)
    }
    missing = sorted((ray_planet_slots - {int(geom.origin_idx)}) - interval_hittable)
    superset_ok = not missing

    if target_timing is not None:
        target_timing.interval_check_calls += 1
        target_timing.interval_check_ray_mismatches += n_mismatch
        if not superset_ok:
            target_timing.interval_check_superset_failures += 1

    if not n_mismatch and superset_ok:
        return

    parts = [
        "[orbit_wars] interval vs discrete raycast micro-step check failed"
        + (f" game_step={game_step}" if game_step >= 0 else "")
        + (f" micro={micro_idx}" if micro_idx >= 0 else "")
        + f" origin_slot={geom.origin_idx} geometry={geom.geometry}",
        f"\n  per-ray first-hit mismatches: {n_mismatch}/{n_rays}",
    ]
    for line in ray_mismatches:
        parts.append(f"\n    {line}")
    if not superset_ok:
        parts.append(
            f"\n  interval hittable slots not a superset of raycast planet hits;"
            f" missing={missing}"
        )
        parts.append(
            f"\n  interval_hittable={sorted(interval_hittable)}"
            f" raycast_planet_slots={sorted(ray_planet_slots)}"
        )
    print("".join(parts), file=sys.stderr, flush=True)


def _interval_targets_np(
    state: OrbitWarsState,
    origin_idx: int,
    frac_idx: int,
    *,
    ship_speed: float = 6.0,
    horizon: int = INCOMING_TA_BINS,
    samples_per_span: int = DEFAULT_INTERVAL_SAMPLES_PER_SPAN,
    n_rays: int = DEFAULT_RAYCAST_RAYS,
    target_timing: Optional[MicroTargetTiming] = None,
    run_raycast_check: bool = False,
    game_step: int = -1,
    micro_idx: int = -1,
    ego_player: int = -1,
    launch_geometry: LaunchGeometryInputs | None = None,
    geometry_override: str | None = None,
    refine_boundaries: bool = True,
    phase: str | None = None,
    selected_target_slot: int = -1,
    return_jobs: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Any | None]]:
    """Interval first-hit targets (``first_hit_interval_best_targets_apply_jax`` semantics)."""

    if target_timing is not None:
        target_timing.interval_calls += 1

    planets = np.asarray(state.planets)
    current_active = np.asarray(state.planet_active).astype(bool)
    geom = _build_interval_micro_geometry(
        state,
        origin_idx,
        frac_idx,
        ship_speed=ship_speed,
        horizon=horizon,
        samples_per_span=samples_per_span,
        target_timing=target_timing,
        launch_geometry=launch_geometry,
        geometry_override=geometry_override,
    )
    sweep_result = _sweep_interval_from_geometry(
        geom,
        int(planets.shape[0]),
        target_timing=target_timing,
        allow_edge_aim=int(np.asarray(state.incoming_fleets).shape[0]) <= 2,
        refine_boundaries=refine_boundaries,
        game_step=game_step,
        micro_idx=micro_idx,
        ego_player=ego_player,
        frac_idx=frac_idx,
        phase=phase,
        selected_target_slot=selected_target_slot,
        return_jobs=return_jobs,
    )
    if return_jobs:
        out_angle, valid, hit_tick_i, refine_jobs = sweep_result
    else:
        out_angle, valid, hit_tick_i = sweep_result

    hit_tick = np.where(valid, hit_tick_i.astype(np.float32), 0.0).astype(np.float32)
    true_planet = np.where(valid, np.arange(MAX_PLANETS, dtype=np.int32), -1)
    true_hit_tick = np.where(valid, hit_tick, 500.0).astype(np.float32)
    valid &= current_active
    valid[origin_idx] = False
    true_planet = np.where(valid, true_planet, -1).astype(np.int32)
    true_hit_tick = np.where(valid, true_hit_tick, 500.0).astype(np.float32)

    if run_raycast_check:
        t_check = perf_counter()
        sim = _run_discrete_raycast_sim(
            state,
            origin_idx,
            frac_idx,
            ship_speed=ship_speed,
            horizon=horizon,
            n_rays=n_rays,
            target_timing=target_timing,
            planet_paths=(geom.p0_by_tick, geom.p1_by_tick, geom.active_by_tick),
            launch_geometry=launch_geometry,
        )
        _check_interval_raycast_micro_consistency(
            geom,
            valid,
            sim,
            target_timing=target_timing,
            game_step=game_step,
            micro_idx=micro_idx,
        )
        if target_timing is not None:
            target_timing.interval_check_s += perf_counter() - t_check

    result = (out_angle, valid, hit_tick, true_planet, true_hit_tick)
    if return_jobs:
        return (*result, refine_jobs)
    return result


def _first_hit_targets_np(
    state: OrbitWarsState,
    origin_idx: int,
    frac_idx: int,
    *,
    ship_speed: float = 6.0,
    horizon: int = INCOMING_TA_BINS,
    n_rays: int = DEFAULT_RAYCAST_RAYS,
    samples_per_span: int = DEFAULT_INTERVAL_SAMPLES_PER_SPAN,
    target_method: str = DEFAULT_TARGET_METHOD,
    target_timing: Optional[MicroTargetTiming] = None,
    game_step: int = -1,
    micro_idx: int = -1,
    ego_player: int = -1,
    launch_geometry: LaunchGeometryInputs | None = None,
    geometry_override: str | None = None,
    refine_boundaries: bool = True,
    phase: str | None = None,
    selected_target_slot: int = -1,
    return_jobs: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Any | None]]:
    if target_timing is not None:
        target_timing.calls += 1
    if target_method == "interval":
        return _interval_targets_np(
            state,
            origin_idx,
            frac_idx,
            ship_speed=ship_speed,
            horizon=horizon,
            samples_per_span=samples_per_span,
            n_rays=n_rays,
            target_timing=target_timing,
            run_raycast_check=_check_interval_raycast_enabled(),
            game_step=game_step,
            micro_idx=micro_idx,
            ego_player=ego_player,
            launch_geometry=launch_geometry,
            geometry_override=geometry_override,
            refine_boundaries=refine_boundaries,
            phase=phase,
            selected_target_slot=selected_target_slot,
            return_jobs=return_jobs,
        )
    result = _raycast_targets_np(
        state,
        origin_idx,
        frac_idx,
        ship_speed=ship_speed,
        horizon=horizon,
        n_rays=n_rays,
        target_timing=target_timing,
        launch_geometry=launch_geometry,
    )
    if return_jobs:
        # Interval refinement jobs are unused for discrete raycast targets.
        return (*result, [None] * MAX_PLANETS)
    return result


def _search_first_contact_targets_np(
    state: OrbitWarsState,
    origin_idx: int,
    frac_idx: int,
    *,
    ship_speed: float = 6.0,
    horizon: int = INCOMING_TA_BINS,
    launch_geometry: LaunchGeometryInputs | None = None,
    geometry_override: str | None = None,
) -> SearchFirstContactTargets:
    if launch_geometry is None:
        planets = np.asarray(state.planets, dtype=np.float64)
        current_active = np.asarray(state.planet_active, dtype=bool)
    else:
        planets = np.asarray(launch_geometry.planets, dtype=np.float64)
        current_active = np.asarray(launch_geometry.planet_active, dtype=bool)

    if geometry_override is not None and str(geometry_override) == "sampled":
        p0_by_tick, p1_by_tick, active_by_tick = _forecast_planet_paths_with_geometry_np(
            state,
            None,
            horizon=horizon,
        )
    else:
        p0_by_tick, p1_by_tick, active_by_tick = _forecast_planet_paths_with_geometry_np(
            state,
            launch_geometry,
            horizon=horizon,
        )
    origin_xy = planets[origin_idx, PLANET_X : PLANET_Y + 1].astype(np.float64)
    origin_radius = float(planets[origin_idx, PLANET_RADIUS])
    ships_avail = float(np.asarray(state.planets)[origin_idx, 5])
    send = _planned_send(ships_avail, frac_idx)
    speed = _fleet_speed(float(max(send, 1)), ship_speed)
    radii = planets[:, PLANET_RADIUS].astype(np.float64)
    collision_rank = np.asarray(state.planet_collision_rank, dtype=np.int32)
    launch_off = float(origin_radius) + 0.1

    out_angle = np.zeros((MAX_PLANETS,), dtype=np.float64)
    valid = np.zeros((MAX_PLANETS,), dtype=np.bool_)
    hit_eta = np.full((MAX_PLANETS,), 500.0, dtype=np.float32)
    if p0_by_tick.shape[0] > 0:
        for target in range(MAX_PLANETS):
            if target == int(origin_idx) or not bool(current_active[target]):
                continue
            if not bool(np.any(active_by_tick[:, target])):
                continue
            target_radius = float(radii[target])
            best_tick = -1
            best_residual = math.inf
            best_pos: np.ndarray | None = None

            for tick in range(int(p0_by_tick.shape[0])):
                if not bool(active_by_tick[tick, target]):
                    continue
                pos = np.asarray(p0_by_tick[tick, target, :], dtype=np.float64)
                center_dist = float(np.linalg.norm(pos - origin_xy))
                expected_center_dist = launch_off + target_radius + float(speed) * float(tick)
                residual = abs(center_dist - expected_center_dist)

                # The fleet can move one full segment during the tick, so accept
                # targets whose radial timing falls within roughly one move.
                if residual > float(speed):
                    continue
                if best_tick < 0 or tick < best_tick or (tick == best_tick and residual < best_residual):
                    best_tick = int(tick)
                    best_residual = float(residual)
                    best_pos = pos

            if best_tick < 0 or best_pos is None:
                continue

            delta = best_pos - origin_xy
            if float(np.dot(delta, delta)) <= 1e-12:
                continue
            angle = math.atan2(float(delta[1]), float(delta[0]))
            out_angle[target] = float(angle % (2.0 * math.pi))
            hit_eta[target] = float(best_tick)
            valid[target] = True

    return SearchFirstContactTargets(
        angles=out_angle,
        valid=valid,
        eta=hit_eta,
        origin_xy=origin_xy,
        origin_radius=float(origin_radius),
        speed=float(speed),
        p0_by_tick=np.asarray(p0_by_tick, dtype=np.float64),
        p1_by_tick=np.asarray(p1_by_tick, dtype=np.float64),
        active_by_tick=np.asarray(active_by_tick, dtype=np.bool_),
        radii=np.asarray(radii, dtype=np.float64),
        collision_rank=collision_rank,
    )


def _refine_interval_launches_in_place(
    actions: list[list[float]],
    planned_launches: list[PlannedLaunchAction],
    base_state: OrbitWarsState,
    launch_geometry: LaunchGeometryInputs | None,
    *,
    ship_speed: float,
    horizon: int,
    n_rays: int,
    samples_per_span: int,
    launch_tracker: Optional[FleetLaunchDebugTracker],
    game_step: int,
    ego_player: int,
    deadline_s: float | None,
) -> None:
    """Refine only submitted interval launches, stopping when the turn budget is tight."""

    from orbit_wars_pt.interval_geometry_np import refine_interval_refine_job

    for planned in planned_launches:
        angle = float(planned.coarse_angle)
        policy_tick = float(planned.policy_hit_tick)
        true_slot = int(planned.target_slot)
        env_hit_tick = float(planned.true_hit_tick)
        virt_state = base_state._replace(planets=np.array(planned.planets_snapshot, copy=True))

        if (deadline_s is None or perf_counter() < deadline_s) and planned.refine_job is not None:
            geom = _build_interval_micro_geometry(
                virt_state,
                planned.origin_slot,
                planned.frac_idx,
                ship_speed=ship_speed,
                horizon=horizon,
                samples_per_span=samples_per_span,
                target_timing=None,
                launch_geometry=launch_geometry,
            )
            refined_angle, refined_tick, refined_reason = refine_interval_refine_job(
                planned.refine_job,
                cache=geom.occlusion_cache,
                origin_xy=geom.origin_xy,
                origin_radius=geom.origin_radius,
                speed=geom.speed,
                p0_by_tick=geom.p0_by_tick,
                p1_by_tick=geom.p1_by_tick,
                radii=geom.radii,
                active_by_tick=geom.active_by_tick,
                object_order=geom.object_order,
            )
            if refined_reason == "ok":
                angle = float(refined_angle)
                policy_tick = float(refined_tick)
                true_slot = int(planned.target_slot)
                env_hit_tick = float(refined_tick)

        actions[planned.action_index][1] = float(angle)

        if launch_tracker is not None:
            launch_tracker.record_launch(
                LaunchRaycastRecord(
                    game_step=int(game_step),
                    ego_player=int(ego_player),
                    micro_idx=int(planned.micro_idx),
                    origin_slot=int(planned.origin_slot),
                    origin_planet_id=float(planned.planets_snapshot[planned.origin_slot, 0]),
                    origin_xy=(
                        float(planned.planets_snapshot[planned.origin_slot, PLANET_X]),
                        float(planned.planets_snapshot[planned.origin_slot, PLANET_Y]),
                    ),
                    origin_radius=float(planned.planets_snapshot[planned.origin_slot, PLANET_RADIUS]),
                    frac_idx=int(planned.frac_idx),
                    fraction=float(FRACTIONS[planned.frac_idx]),
                    ships_avail=float(planned.planets_snapshot[planned.origin_slot, 5]),
                    planned_send=int(planned.planned_send),
                    n_rays=int(n_rays),
                    ship_speed=float(ship_speed),
                    launch_angle=float(angle),
                    policy_target_slot=int(planned.target_slot),
                    policy_target_planet_id=float(planned.planets_snapshot[planned.target_slot, 0]),
                    true_target_slot=int(true_slot),
                    true_target_planet_id=float(planned.planets_snapshot[true_slot, 0]) if 0 <= true_slot < MAX_PLANETS else -1.0,
                    policy_hit_tick=float(policy_tick),
                    true_hit_tick=float(env_hit_tick),
                    comet_planet_ids_at_launch=_active_comet_planet_ids(
                        np.asarray(virt_state.comet_group_active),
                        np.asarray(virt_state.comet_planet_ids),
                    ),
                )
            )


@dataclass
class SearchRuntime:
    settings: ModelSearchSettings
    public_obs: Mapping[str, Any]
    kaggle_config: Any
    game_key: str
    step_count: int
    num_agents: int
    greedy_actions_for_player: Any
    value_for_player: Any
    public_state: Optional[OrbitWarsState] = None
    fleet_arrival_cache: Optional[FleetArrivalCache] = None
    choose_launch: Any = None
    deadline_s: float | None = None
    planned_ego_actions: list[list[float]] | None = None
    planned_turn_complete: bool = False
    turn_end_opponent_joint_action_samples: list[list[list[list[float]]]] | None = None


@dataclass
class SearchPlannedLaunchAction:
    action: list[float]
    origin_slot: int
    frac_idx: int
    target_slot: int
    planned_send: int
    policy_hit_tick: float
    true_hit_tick: float
    planets_snapshot: np.ndarray
    refine_job: Any | None = None


@dataclass
class SearchTurnPlanResult:
    actions: list[SearchPlannedLaunchAction]
    turn_complete: bool


@dataclass
class _SearchSampleTargetOption:
    choice_key: tuple[Any, ...]
    origin_slot: int
    frac_idx: int
    send: int
    is_abort: bool
    angle: float = 0.0
    target_slot: int = -1
    policy_hit_tick: float = -1.0
    true_hit_tick: float = -1.0


@dataclass
class _SearchSampleOriginOptions:
    options: list[_SearchSampleTargetOption]
    logits: torch.Tensor


@dataclass
class _SearchTreeNode:
    env_public_obs: Mapping[str, Any]
    env_step_start_state: OrbitWarsState
    current_state: OrbitWarsState
    step_count: int
    search_env_step_from_root: int
    current_turn_actions: list[SearchPlannedLaunchAction]
    current_micro_idx: int
    turn_closed: bool
    root_turn_actions: list[SearchPlannedLaunchAction]
    root_turn_complete: bool
    discounted_reward: float
    discount: float
    done: bool


@dataclass
class _SearchSamplePrefixNode:
    tree_node: _SearchTreeNode
    policy_out: dict[str, Any] | None = None
    origin_options_cache: dict[int, _SearchSampleOriginOptions] = field(default_factory=dict)
    children_by_choice: dict[tuple[Any, ...], "_SearchSamplePrefixNode"] = field(default_factory=dict)


@dataclass
class CachedSearchPolicyOutputs:
    players: tuple[int, ...]
    halt_logits: torch.Tensor
    value: torch.Tensor
    pair_mask: torch.Tensor
    origin_frac_logits: torch.Tensor
    origin_frac_mask: torch.Tensor
    planet_hidden: tuple[torch.Tensor, ...] | torch.Tensor
    abort_logits: torch.Tensor | None = None


@dataclass
class CachedSearchTransition:
    public_obs: dict[str, Any]
    state: OrbitWarsState
    step_count: int
    step_reward: float
    done: bool
    ego_actions: list[list[float]] | None = None
    policy_outputs: CachedSearchPolicyOutputs | None = None


@dataclass
class CachedSearchRollout:
    game_key: str
    ego_player: int
    root_ego_actions: list[list[float]] | None
    root_public_obs: dict[str, Any]
    root_state: OrbitWarsState
    root_step_count: int
    root_policy_outputs: CachedSearchPolicyOutputs | None
    transitions: list[CachedSearchTransition]


@dataclass
class _BatchedSearchSeatPlan:
    branch_idx: int
    player: int
    state_template: OrbitWarsState
    planets: np.ndarray
    incoming_fleets: np.ndarray
    origin_frac_blocked: np.ndarray
    actions: list[list[float]]
    micro_idx: int
    max_micro_steps: int


def _simulate_current_step_joint_actions(
    runtime: SearchRuntime,
    *,
    ego_player: int,
    ego_actions: list[list[float]],
    timing: ModelSearchTiming | None = None,
) -> list[list[list[float]]]:
    t_total = perf_counter() if timing is not None else 0.0
    joint_actions: list[list[list[float]]] = [[] for _ in range(int(runtime.num_agents))]
    joint_actions[int(ego_player)] = copy.deepcopy(ego_actions)
    for player in range(int(runtime.num_agents)):
        if player == int(ego_player):
            continue
        player_obs = _public_obs_for_player(
            runtime.public_obs,
            player=player,
            step_count=int(runtime.step_count),
        )
        t0 = perf_counter() if timing is not None else 0.0
        joint_actions[player] = runtime.greedy_actions_for_player(
            player_obs,
            player,
            int(runtime.step_count),
        )
    if timing is not None:
        timing.simulate_joint_calls += 1
        timing.simulate_joint_s += perf_counter() - t_total
    return joint_actions


def _rollout_branch_score(
    runtime: SearchRuntime,
    *,
    ego_player: int,
    ego_actions: list[list[float]] | None,
    rollout_horizon: int | None = None,
    timing: ModelSearchTiming | None = None,
    trace_out: list[CachedSearchTransition] | None = None,
) -> float:
    t_rollout = perf_counter() if timing is not None else 0.0
    reward = runtime.settings.reward
    sim_state = _make_sim_state(
        runtime.public_obs,
        num_agents=int(runtime.num_agents),
        step_count=int(runtime.step_count),
    )
    public_obs = _public_obs_for_player(runtime.public_obs, player=0, step_count=int(runtime.step_count))
    sim_step = int(runtime.step_count)
    total = 0.0
    discount = 1.0

    horizon = int(rollout_horizon if rollout_horizon is not None else runtime.settings.horizon_steps)
    for depth in range(horizon):
        if timing is not None:
            timing.rollout_steps += 1
        t0 = perf_counter()
        state_pre = observation_to_state(
            public_obs,
            runtime.kaggle_config,
            max_fleets=DEFAULT_MAX_ACTIONS + len(public_obs.get("fleets", []) or []),
            step_count_override=sim_step,
            num_agents_override=int(runtime.num_agents),
        )
        if timing is not None:
            timing.state_rebuild_calls += 1
            timing.state_rebuild_s += perf_counter() - t0
        ratios_pre = _reward_mix_ratios_np(state_pre, reward)
        if depth == 0 and ego_actions is not None:
            joint_actions = _simulate_current_step_joint_actions(
                runtime,
                ego_player=int(ego_player),
                ego_actions=ego_actions,
                timing=timing,
            )
        else:
            joint_actions = [[] for _ in range(int(runtime.num_agents))]
            for player in range(int(runtime.num_agents)):
                player_obs = _public_obs_for_player(public_obs, player=player, step_count=sim_step)
                joint_actions[player] = runtime.greedy_actions_for_player(player_obs, player, sim_step)
        t0 = perf_counter() if timing is not None else 0.0
        _simulate_joint_step_with_kaggle_model(
            sim_state,
            joint_actions=joint_actions,
            config=runtime.kaggle_config,
        )
        if timing is not None:
            timing.kaggle_step_calls += 1
            timing.kaggle_step_s += perf_counter() - t0
        sim_step += 1
        public_obs = _public_obs_from_sim_state(sim_state, step_count=sim_step)
        if timing is not None:
            t0 = perf_counter()
        state_post = observation_to_state(
            public_obs,
            runtime.kaggle_config,
            max_fleets=DEFAULT_MAX_ACTIONS + len(public_obs.get("fleets", []) or []),
            step_count_override=sim_step,
            num_agents_override=int(runtime.num_agents),
        )
        if timing is not None:
            timing.state_rebuild_calls += 1
            timing.state_rebuild_s += perf_counter() - t0
        step_reward = _reward_delta_np(state_pre, state_post, ratios_pre, reward)
        if trace_out is not None:
            trace_out.append(
                CachedSearchTransition(
                    public_obs=copy.deepcopy(public_obs),
                    state=state_post,
                    step_count=int(sim_step),
                    step_reward=float(step_reward[int(ego_player)]),
                    done=bool(np.asarray(state_post.done)),
                )
            )
        total += discount * float(step_reward[int(ego_player)])
        if bool(np.asarray(state_post.done)):
            if timing is not None:
                timing.branch_rollouts += 1
                timing.branch_rollout_s += perf_counter() - t_rollout
            return total
        discount *= float(reward.gamma)

    final_obs = _public_obs_for_player(public_obs, player=int(ego_player), step_count=sim_step)
    total += discount * float(runtime.value_for_player(final_obs, int(ego_player), sim_step))
    if timing is not None:
        timing.branch_rollouts += 1
        timing.branch_rollout_s += perf_counter() - t_rollout
    return total


def _choose_launch_via_model_search(
    runtime: SearchRuntime,
    *,
    ego_player: int,
    action_prefix: list[list[float]],
    launch_action: list[float],
    launch_tail_actions: list[list[float]],
    launch_true_hit_tick: float,
    timing: ModelSearchTiming | None = None,
) -> bool:
    t_choose = perf_counter() if timing is not None else 0.0
    rollout_horizon = _model_search_rollout_horizon(
        runtime.settings,
        launch_true_hit_tick=float(launch_true_hit_tick),
    )
    _log_model_search_horizon(
        runtime.settings,
        rollout_horizon=int(rollout_horizon),
        launch_true_hit_tick=float(launch_true_hit_tick),
        step_count=int(runtime.step_count),
        ego_player=int(ego_player),
        send=int(launch_action[2]) if len(launch_action) >= 3 else -1,
    )
    halt_score = _rollout_branch_score(
        runtime,
        ego_player=int(ego_player),
        ego_actions=copy.deepcopy(action_prefix),
        rollout_horizon=rollout_horizon,
        timing=timing,
    )
    launch_score = _rollout_branch_score(
        runtime,
        ego_player=int(ego_player),
        ego_actions=copy.deepcopy(action_prefix) + [copy.deepcopy(launch_action)] + copy.deepcopy(launch_tail_actions),
        rollout_horizon=rollout_horizon,
        timing=timing,
    )
    if timing is not None:
        timing.choose_calls += 1
        timing.choose_s += perf_counter() - t_choose
    chose_launch = bool(launch_score > halt_score)
    if _model_search_debug_enabled():
        _model_search_debug(
            f"choose step={int(runtime.step_count)} ego={int(ego_player)} "
            f"halt={halt_score:.6f} launch={launch_score:.6f} -> {'launch' if chose_launch else 'halt'}"
        )
    return chose_launch


def _obs_tensors_for_state(
    state: OrbitWarsState,
    ego_player: int,
    device: torch.device,
    *,
    policy_player_count: Optional[int] = None,
    target_abort_enabled: bool = False,
    normalize_obs_to_p0: bool = False,
    obs_timing: BatchObsTiming | None = None,
) -> dict[str, torch.Tensor]:
    t0 = perf_counter() if obs_timing is not None else 0.0
    planets = np.asarray(state.planets)
    planet_active = np.asarray(state.planet_active)
    initial_planets = np.asarray(state.initial_planets)
    initial_active = np.asarray(state.initial_active)
    incoming_fleets = np.asarray(state.incoming_fleets)
    origin_frac_blocked = np.asarray(
        getattr(state, "origin_frac_blocked", np.zeros((MAX_PLANETS, BLOCKED_FRAC_FEATURES), dtype=np.bool_))
    )
    angular_velocity = float(np.asarray(state.angular_velocity))
    step_count = int(np.asarray(state.step_count))
    num_agents = int(np.asarray(state.num_agents))
    player_count = int(policy_player_count if policy_player_count is not None else num_agents)
    obs_qturns = _obs_qturns_to_p0_np(int(ego_player), player_count, normalize_obs_to_p0)
    comet_ids = np.asarray(state.comet_planet_ids)

    entity_type = np.zeros((1 + MAX_PLANETS,), dtype=np.int64)
    owner_idx = np.zeros((1 + MAX_PLANETS,), dtype=np.int64)
    fdim = obs_feature_dim_for_num_agents(player_count, target_abort_enabled=target_abort_enabled)
    features = np.zeros((1 + MAX_PLANETS, fdim), dtype=np.float32)
    rope_pos = np.zeros((1 + MAX_PLANETS, 3), dtype=np.float32)
    entity_mask = np.zeros((1 + MAX_PLANETS,), dtype=np.bool_)
    planet_mask = np.zeros((1 + MAX_PLANETS,), dtype=np.bool_)

    entity_type[0] = ENTITY_CLS
    owner_idx[0] = 1
    features[0, 6] = np.float32(np.clip(float(step_count) / 498.0, 0.0, 1.0))
    rope_pos[0] = np.asarray([CENTER / BOARD_SIZE, CENTER / BOARD_SIZE, 0.0], dtype=np.float32)
    entity_mask[0] = True

    if obs_timing is not None:
        obs_timing.setup_s += perf_counter() - t0
        obs_timing.encode_calls += 1
        t0 = perf_counter()

    incoming_net, survivor_slot = _incoming_interfleet_np(
        incoming_fleets.astype(np.float32), int(ego_player), player_count, normalize_obs_to_p0
    )
    incoming_net, survivor_slot = _hide_enemy_far_right_after_resolution(incoming_net, survivor_slot)

    if obs_timing is not None:
        obs_timing.incoming_s += perf_counter() - t0
        t0 = perf_counter()

    multi_survivor = fdim in (FEATURE_DIM_MULTI, FEATURE_DIM_MULTI_ABORT) and player_count > 2
    tp = perf_counter() if obs_timing is not None else 0.0
    valid_comet_ids = comet_ids[comet_ids >= 0]
    if valid_comet_ids.size > 0:
        is_comet_arr = np.isin(planets[:, 0], valid_comet_ids)
    else:
        is_comet_arr = np.zeros((MAX_PLANETS,), dtype=np.bool_)
    owner_idx[1 : 1 + MAX_PLANETS] = _remap_owner_np(
        planets[:, 1],
        int(ego_player),
        player_count,
        normalize_obs_to_p0,
    )
    entity_type[1 : 1 + MAX_PLANETS] = np.where(is_comet_arr, ENTITY_COMET, ENTITY_PLANET)
    if obs_timing is not None:
        obs_timing.planets.meta_s += perf_counter() - tp
        tp = perf_counter()

    for i in range(MAX_PLANETS):
        j = 1 + i
        active = bool(planet_active[i])
        is_comet = bool(is_comet_arr[i])

        vx, vy = planet_pred_velocity(
            initial_planets[i, 2:4].astype(np.float64),
            planets[i, 2:4].astype(np.float64),
            float(planets[i, 4]),
            angular_velocity,
            step_count,
            bool(initial_active[i]),
            active,
        )
        if obs_timing is not None:
            obs_timing.planets.velocity_s += perf_counter() - tp
            tp = perf_counter()

        if is_comet and active:
            pid = int(planets[i, 0])
            group_row = np.where(comet_ids == pid)
            if group_row[0].size > 0:
                g, k = int(group_row[0][0]), int(group_row[1][0])
                paths = np.asarray(state.comet_paths[g, k])
                lens = int(np.asarray(state.comet_path_lengths[g, k]))
                idx = int(np.asarray(state.comet_path_index[g]))
                if lens > 1 and 0 <= idx < lens - 1:
                    vx = float(paths[idx + 1, 0] - paths[idx, 0])
                    vy = float(paths[idx + 1, 1] - paths[idx, 1])
        if obs_timing is not None:
            obs_timing.planets.comet_s += perf_counter() - tp
            tp = perf_counter()

        vx, vy = _rotate_vec_np(vx, vy, obs_qturns)

        features[j, 0] = np.log1p(max(float(planets[i, 6]), 0.0))
        features[j, 1] = float(planets[i, 5]) / 1000.0
        features[j, 2] = float(vx) / 5.0
        features[j, 3] = float(vy) / 5.0
        features[j, 4] = float(active)
        features[j, 5] = float(planets[i, 4]) / 10.0
        features[j, 8 : 8 + INCOMING_TA_BINS] = incoming_net[i].astype(np.float32)
        if obs_timing is not None:
            obs_timing.planets.feat_base_s += perf_counter() - tp
            tp = perf_counter()

        if multi_survivor:
            oh = np.eye(NUM_OWNER_SLOTS, dtype=np.float32)[survivor_slot[i]]
            features[
                j, 8 + INCOMING_TA_BINS : 8 + INCOMING_TA_BINS + INCOMING_SURVIVOR_FLAT
            ] = oh.reshape(INCOMING_SURVIVOR_FLAT)
        if obs_timing is not None:
            obs_timing.planets.feat_survivor_s += perf_counter() - tp
            tp = perf_counter()

        if bool(target_abort_enabled):
            features[j, -BLOCKED_FRAC_FEATURES:] = origin_frac_blocked[i].astype(np.float32)
        if obs_timing is not None:
            obs_timing.planets.feat_abort_s += perf_counter() - tp
            tp = perf_counter()

        if active:
            px, py = _rotate_xy_about_center_np(float(planets[i, 2]), float(planets[i, 3]), obs_qturns)
            rope_pos[j, 0] = px / BOARD_SIZE
            rope_pos[j, 1] = py / BOARD_SIZE
        if obs_timing is not None:
            obs_timing.planets.rope_s += perf_counter() - tp
            tp = perf_counter()

        entity_mask[j] = active
        planet_mask[j] = True
        if obs_timing is not None:
            obs_timing.planets.mask_s += perf_counter() - tp
            tp = perf_counter()

    if obs_timing is not None:
        obs_timing.planet_loop_s += perf_counter() - t0
        t0 = perf_counter()

    def tensor(x: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(x)).to(device=device, dtype=dtype).unsqueeze(0)

    out = {
        "entity_type": tensor(entity_type, torch.long),
        "owner_idx": tensor(owner_idx, torch.long),
        "features": tensor(features, torch.float32),
        "rope_pos": tensor(rope_pos, torch.float32),
        "entity_mask": tensor(entity_mask, torch.bool),
        "planet_mask": tensor(planet_mask, torch.bool),
    }
    if obs_timing is not None:
        obs_timing.to_device_s += perf_counter() - t0
    return out


def _obs_tensors_for_states(
    states: list[OrbitWarsState],
    ego_players: list[int],
    device: torch.device,
    *,
    policy_player_count: Optional[int] = None,
    target_abort_enabled: bool = False,
    normalize_obs_to_p0: bool = False,
    obs_timing: BatchObsTiming | None = None,
) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("states must be non-empty")
    parts = [
        _obs_tensors_for_state(
            state,
            int(player),
            device,
            policy_player_count=policy_player_count,
            target_abort_enabled=target_abort_enabled,
            normalize_obs_to_p0=normalize_obs_to_p0,
            obs_timing=obs_timing,
        )
        for state, player in zip(states, ego_players)
    ]
    if obs_timing is not None:
        t0 = perf_counter()
    keys = tuple(parts[0].keys())
    batch = {key: torch.cat([part[key] for part in parts], dim=0) for key in keys}
    if obs_timing is not None:
        obs_timing.cat_s += perf_counter() - t0
    return batch


def _build_turn_actions_torch_only(
    policy: OrbitWarsPolicy,
    state: OrbitWarsState,
    ego_player: int,
    device: torch.device,
    *,
    ship_speed: float = 6.0,
    max_micro_steps: int = DEFAULT_MAX_ACTIONS,
    sampling_mode: str = SAMPLING_MODE_STOCHASTIC,
    rng: Optional[torch.Generator] = None,
    n_rays: int = DEFAULT_RAYCAST_RAYS,
    samples_per_span: int = DEFAULT_INTERVAL_SAMPLES_PER_SPAN,
    target_method: str = DEFAULT_TARGET_METHOD,
    timing: Optional[KaggleAgentCallTiming] = None,
    launch_tracker: Optional[FleetLaunchDebugTracker] = None,
    game_step: int = 0,
    policy_player_count: Optional[int] = None,
    normalize_obs_to_p0: bool = False,
    launch_geometry: LaunchGeometryInputs | None = None,
    deadline_s: float | None = None,
    population_member: Optional[int] = None,
    search_runtime: SearchRuntime | None = None,
    search_launch_probability_threshold: float | None = None,
    search_greedy_launch_threshold: float | None = None,
    search_root_player: int | None = None,
) -> list[list[float]]:
    planets = np.array(np.asarray(state.planets), copy=True)
    incoming_fleets = np.array(np.asarray(state.incoming_fleets), copy=True)
    origin_frac_blocked = np.array(
        np.asarray(getattr(state, "origin_frac_blocked", np.zeros((MAX_PLANETS, len(FRACTIONS)), dtype=np.bool_))),
        copy=True,
    )
    planet_active = np.asarray(state.planet_active).astype(bool)
    actions: list[list[float]] = []
    planned_launches: list[PlannedLaunchAction] = []
    micro_idx = 0
    search_used = False
    forced_plan_actions: list[SearchPlannedLaunchAction] = []
    forced_plan_turn_complete = False

    if (
        search_runtime is not None
        and _model_search_enabled(search_runtime.settings)
        and _model_search_mode_uses_turn_planner(str(search_runtime.settings.mode))
    ):
        forced_plan = search_runtime.choose_launch(
            search_runtime,
            ego_player=int(ego_player),
            current_state=state,
            timing=(timing.model_search if timing is not None else None),
        )
        forced_plan_actions = copy.deepcopy(forced_plan.actions)
        forced_plan_turn_complete = bool(forced_plan.turn_complete)
        search_used = True

    with torch.inference_mode():
        for _ in range(max_micro_steps):
            if timing is not None:
                timing.micro_iters += 1

            if forced_plan_actions:
                planned = forced_plan_actions.pop(0)
                actions.append(copy.deepcopy(planned.action))
                planned_launches.append(
                    PlannedLaunchAction(
                        action_index=len(actions) - 1,
                        micro_idx=int(micro_idx),
                        origin_slot=int(planned.origin_slot),
                        frac_idx=int(planned.frac_idx),
                        target_slot=int(planned.target_slot),
                        planned_send=int(planned.planned_send),
                        policy_hit_tick=float(planned.policy_hit_tick),
                        true_hit_tick=float(planned.true_hit_tick),
                        coarse_angle=float(planned.action[1]),
                        planets_snapshot=np.array(planned.planets_snapshot, copy=True),
                        refine_job=planned.refine_job,
                    )
                )
                micro_idx += 1
                apply_micro_launch_in_place(
                    planets,
                    incoming_fleets,
                    ego_player=int(ego_player),
                    origin_slot=int(planned.origin_slot),
                    send=int(planned.planned_send),
                    true_target_slot=int(planned.target_slot),
                    true_hit_tick=float(planned.true_hit_tick),
                )
                continue
            if forced_plan_turn_complete:
                break

            t0 = perf_counter()
            virt = state._replace(
                planets=planets,
                incoming_fleets=incoming_fleets,
                origin_frac_blocked=origin_frac_blocked,
            )
            batch = _obs_tensors_for_state(
                virt,
                ego_player,
                device,
                policy_player_count=policy_player_count,
                target_abort_enabled=bool(getattr(policy, "target_abort_enabled", False)),
                normalize_obs_to_p0=normalize_obs_to_p0,
            )
            if timing is not None:
                timing.micro_obs_tensors_s += perf_counter() - t0

            t0 = perf_counter()
            population_idx = None
            if population_member is not None:
                population_idx = torch.tensor([int(population_member)], device=device, dtype=torch.long)
            out = _policy_forward_inference(
                policy,
                batch,
                population_idx=population_idx,
            )
            if timing is not None:
                timing.micro_policy_forward_s += perf_counter() - t0

            t0 = perf_counter()
            search_available = (
                search_runtime is not None
                and _model_search_enabled(search_runtime.settings)
                and not search_used
            )
            halt_logits = out["halt_logits"][0]
            if search_available and not _launch_probability_meets_threshold(
                halt_logits,
                threshold=search_launch_probability_threshold,
            ):
                search_available = False
            halt_greedy = sampling_mode == SAMPLING_MODE_GREEDY
            origin_target_greedy = sampling_mode in {SAMPLING_MODE_GREEDY, SAMPLING_MODE_MIXED}
            if halt_greedy:
                policy_halt_action = _greedy_halt_action_from_logits(
                    halt_logits,
                    launch_threshold=search_greedy_launch_threshold,
                )
            else:
                halt_probs = torch.softmax(halt_logits, dim=-1)
                policy_halt_action = int(torch.multinomial(halt_probs, 1, generator=rng).item())
            force_candidate_eval = search_available
            halt_action = 0 if force_candidate_eval else policy_halt_action
            if halt_action == 1:
                if timing is not None:
                    timing.micro_post_forward_s += perf_counter() - t0
                break

            flat_mask = out["origin_frac_mask"].flatten(start_dim=1)[0]
            if not bool(flat_mask.any().item()):
                if timing is not None:
                    timing.micro_post_forward_s += perf_counter() - t0
                break
            flat_logits = out["origin_frac_logits"].flatten(start_dim=1)[0]
            masked_origin_frac = flat_logits.masked_fill(~flat_mask, -1e4)
            if origin_target_greedy:
                origin_frac_flat = int(torch.argmax(masked_origin_frac).item())
            else:
                origin_frac_probs = torch.softmax(masked_origin_frac, dim=-1)
                origin_frac_flat = int(torch.multinomial(origin_frac_probs, 1, generator=rng).item())
            o_idx = origin_frac_flat // len(FRACTIONS)
            frac_idx = origin_frac_flat % len(FRACTIONS)
            if timing is not None:
                timing.micro_post_forward_s += perf_counter() - t0

            t0 = perf_counter()
            target_timing = timing.micro_target if timing is not None else None
            ray_angle, ray_valid, ray_hit_tick, true_planet, true_hit_tick, refine_jobs = _first_hit_targets_np(
                virt,
                int(o_idx),
                int(frac_idx),
                ship_speed=ship_speed,
                horizon=INCOMING_TA_BINS,
                n_rays=n_rays,
                samples_per_span=samples_per_span,
                target_method=target_method,
                target_timing=target_timing,
                game_step=int(game_step),
                micro_idx=int(micro_idx),
                ego_player=int(ego_player),
                launch_geometry=launch_geometry,
                refine_boundaries=False,
                phase="microstep",
                return_jobs=True,
            )
            if timing is not None:
                timing.micro_raycast_s += perf_counter() - t0

            t0 = perf_counter()
            target_logits = policy.target_logits_for_origin_fraction(
                out["planet_hidden"],
                batch["owner_idx"],
                batch["features"],
                torch.tensor([o_idx], device=device, dtype=torch.long),
                torch.tensor([frac_idx], device=device, dtype=torch.long),
                fleet_size=torch.tensor(
                    [float(_planned_send(float(planets[o_idx, 5]), int(frac_idx)))],
                    device=device,
                    dtype=torch.float32,
                ),
                target_eta=torch.from_numpy(ray_hit_tick[None, :]).to(device=device, dtype=torch.float32),
                target_ships=torch.from_numpy(planets[None, :, 5]).to(device=device, dtype=torch.float32),
                population_idx=population_idx,
            )[0]
            abort_logits_all = out.get("abort_logits")
            abort_logit = None
            if abort_logits_all is not None:
                abort_logit = abort_logits_all[
                    0,
                    int(o_idx),
                    int(frac_idx),
                ].reshape(1)
            target_mask = out["pair_mask"][0, o_idx].clone()
            ray_valid_t = torch.from_numpy(ray_valid).to(device=device, dtype=torch.bool)
            target_mask &= ray_valid_t
            if abort_logit is not None:
                combined_target = torch.cat([target_logits.masked_fill(~target_mask, -1e4), abort_logit.reshape(1)], dim=0)
                if origin_target_greedy:
                    target_choice = int(torch.argmax(combined_target).item())
                else:
                    target_probs = torch.softmax(combined_target, dim=-1)
                    target_choice = int(torch.multinomial(target_probs, 1, generator=rng).item())
                if target_choice == MAX_PLANETS:
                    if force_candidate_eval and policy_halt_action == 1:
                        if timing is not None:
                            timing.micro_target_s += perf_counter() - t0
                        break
                    origin_frac_blocked[int(o_idx), int(frac_idx)] = True
                    micro_idx += 1
                    if timing is not None:
                        timing.micro_target_s += perf_counter() - t0
                    continue
                d_idx = int(target_choice)
            else:
                if not bool(target_mask.any().item()):
                    if timing is not None:
                        timing.micro_target_s += perf_counter() - t0
                    break
                masked_target = target_logits.masked_fill(~target_mask, -1e4)
                if origin_target_greedy:
                    d_idx = int(torch.argmax(masked_target).item())
                else:
                    target_probs = torch.softmax(masked_target, dim=-1)
                    d_idx = int(torch.multinomial(target_probs, 1, generator=rng).item())
            if timing is not None:
                timing.micro_target_s += perf_counter() - t0

            t0 = perf_counter()
            ships_avail = float(planets[o_idx, 5])
            send = _planned_send(ships_avail, int(frac_idx))
            if send <= 0:
                if timing is not None:
                    timing.micro_book_s += perf_counter() - t0
                break
            angle = float(ray_angle[d_idx])
            if _search_opponent_should_halt_on_neutral_underlaunch(
                planets,
                incoming_fleets,
                player=int(ego_player),
                search_root_player=search_root_player,
                send=int(send),
                true_target_slot=int(true_planet[d_idx]),
            ):
                if timing is not None:
                    timing.micro_book_s += perf_counter() - t0
                break
            if search_available:
                search_used = True
                launch_action = [float(planets[o_idx, 0]), float(angle), int(send)]
                if search_runtime.choose_launch is not None:
                    if not bool(
                        search_runtime.choose_launch(
                            search_runtime,
                            ego_player=int(ego_player),
                            current_state=virt,
                            current_micro_idx=int(micro_idx),
                            action_prefix=copy.deepcopy(actions),
                            launch_action=launch_action,
                            launch_origin_slot=int(o_idx),
                            launch_send=int(send),
                            launch_true_target_slot=int(true_planet[d_idx]),
                            launch_true_hit_tick=float(true_hit_tick[d_idx]),
                            timing=(timing.model_search if timing is not None else None),
                        )
                    ):
                        if timing is not None:
                            timing.micro_book_s += perf_counter() - t0
                        break
                else:
                    branch_planets = np.array(planets, copy=True)
                    branch_incoming = np.array(incoming_fleets, copy=True)
                    branch_blocked = np.array(origin_frac_blocked, copy=True)
                    apply_micro_launch_in_place(
                        branch_planets,
                        branch_incoming,
                        ego_player=int(ego_player),
                        origin_slot=int(o_idx),
                        send=int(send),
                        true_target_slot=int(true_planet[d_idx]),
                        true_hit_tick=float(true_hit_tick[d_idx]),
                    )
                    branch_state = state._replace(
                        planets=branch_planets,
                        incoming_fleets=branch_incoming,
                        origin_frac_blocked=branch_blocked,
                    )
                    t_search = perf_counter() if timing is not None else 0.0
                    launch_tail_actions = _build_turn_actions_torch_only(
                        policy,
                        branch_state,
                        ego_player,
                        device,
                        ship_speed=ship_speed,
                        max_micro_steps=max(0, int(max_micro_steps) - int(micro_idx) - 1),
                        sampling_mode=SAMPLING_MODE_GREEDY,
                        rng=rng,
                        n_rays=n_rays,
                        samples_per_span=samples_per_span,
                        target_method=target_method,
                        timing=None,
                        launch_tracker=None,
                        game_step=game_step,
                        policy_player_count=policy_player_count,
                        normalize_obs_to_p0=normalize_obs_to_p0,
                        launch_geometry=launch_geometry,
                        deadline_s=deadline_s,
                        population_member=population_member,
                        search_runtime=None,
                        search_launch_probability_threshold=None,
                        search_greedy_launch_threshold=(
                            search_runtime.settings.greedy_launch_threshold if search_runtime is not None else None
                        ),
                        search_root_player=search_root_player,
                    )
                    if timing is not None:
                        timing.model_search.ego_tail_build_calls += 1
                        timing.model_search.ego_tail_build_s += perf_counter() - t_search
                    if not _choose_launch_via_model_search(
                        search_runtime,
                        ego_player=int(ego_player),
                        action_prefix=copy.deepcopy(actions),
                        launch_action=launch_action,
                        launch_tail_actions=launch_tail_actions,
                        launch_true_hit_tick=float(true_hit_tick[d_idx]),
                        timing=(timing.model_search if timing is not None else None),
                    ):
                        if timing is not None:
                            timing.micro_book_s += perf_counter() - t0
                        break
            actions.append([float(planets[o_idx, 0]), float(angle), int(send)])
            planned_launches.append(
                PlannedLaunchAction(
                    action_index=len(actions) - 1,
                    micro_idx=int(micro_idx),
                    origin_slot=int(o_idx),
                    frac_idx=int(frac_idx),
                    target_slot=int(d_idx),
                    planned_send=int(send),
                    policy_hit_tick=float(ray_hit_tick[d_idx]),
                    true_hit_tick=float(true_hit_tick[d_idx]),
                    coarse_angle=float(angle),
                    planets_snapshot=np.array(planets, copy=True),
                    refine_job=refine_jobs[d_idx],
                )
            )
            micro_idx += 1
            apply_micro_launch_in_place(
                planets,
                incoming_fleets,
                ego_player=int(ego_player),
                origin_slot=int(o_idx),
                send=int(send),
                true_target_slot=int(true_planet[d_idx]),
                true_hit_tick=float(true_hit_tick[d_idx]),
            )
            if not planet_active[o_idx] or planets[o_idx, 5] < 1.0:
                if timing is not None:
                    timing.micro_book_s += perf_counter() - t0
                continue
            if timing is not None:
                timing.micro_book_s += perf_counter() - t0

    if target_method == "interval" and planned_launches:
        _refine_interval_launches_in_place(
            actions,
            planned_launches,
            state,
            launch_geometry,
            ship_speed=ship_speed,
            horizon=INCOMING_TA_BINS,
            n_rays=n_rays,
            samples_per_span=samples_per_span,
            launch_tracker=launch_tracker,
            game_step=int(game_step),
            ego_player=int(ego_player),
            deadline_s=deadline_s,
        )
    elif launch_tracker is not None:
        for planned in planned_launches:
            true_slot = int(planned.target_slot)
            launch_tracker.record_launch(
                LaunchRaycastRecord(
                    game_step=int(game_step),
                    ego_player=int(ego_player),
                    micro_idx=int(planned.micro_idx),
                    origin_slot=int(planned.origin_slot),
                    origin_planet_id=float(planned.planets_snapshot[planned.origin_slot, 0]),
                    origin_xy=(
                        float(planned.planets_snapshot[planned.origin_slot, PLANET_X]),
                        float(planned.planets_snapshot[planned.origin_slot, PLANET_Y]),
                    ),
                    origin_radius=float(planned.planets_snapshot[planned.origin_slot, PLANET_RADIUS]),
                    frac_idx=int(planned.frac_idx),
                    fraction=float(FRACTIONS[planned.frac_idx]),
                    ships_avail=float(planned.planets_snapshot[planned.origin_slot, 5]),
                    planned_send=int(planned.planned_send),
                    n_rays=int(n_rays),
                    ship_speed=float(ship_speed),
                    launch_angle=float(actions[planned.action_index][1]),
                    policy_target_slot=int(planned.target_slot),
                    policy_target_planet_id=float(planned.planets_snapshot[planned.target_slot, 0]),
                    true_target_slot=true_slot,
                    true_target_planet_id=float(planned.planets_snapshot[true_slot, 0]) if 0 <= true_slot < MAX_PLANETS else -1.0,
                    policy_hit_tick=float(planned.policy_hit_tick),
                    true_hit_tick=float(planned.policy_hit_tick),
                    comet_planet_ids_at_launch=_active_comet_planet_ids(
                        np.asarray(state.comet_group_active),
                        np.asarray(state.comet_planet_ids),
                    ),
                )
            )

    return actions


def apply_micro_launch_in_place(
    planets: np.ndarray,
    incoming_fleets: np.ndarray,
    *,
    ego_player: int,
    origin_slot: int,
    send: int,
    true_target_slot: int,
    true_hit_tick: float,
) -> None:
    """Mutate ``planets`` / ``incoming_fleets`` for one launched micro-step.

    Matches the inline mutation inside ``_build_turn_actions_torch_only``: ships
    are deducted from the origin planet's stock and added to the destination's
    incoming-fleets bin keyed by ``(ego_player, true_target_slot, ta)`` where
    ``ta = floor(true_hit_tick)`` and ``tick=k`` means the fleet's ``(k+1)``th
    move collides (so ``ta=0`` is "hits during the same env step that processes
    the launch", matching ``jax_orbit_wars._launch_fleets`` after the bin-0
    consume-and-shift). Out-of-range targets are no-ops on the incoming table
    (origin deduction still applies). This is the single source of truth used
    by both the live adapter and the background consistency worker so they
    stay numerically identical.
    """

    planets[int(origin_slot), 5] -= float(send)
    if not (0 <= int(true_target_slot) < MAX_PLANETS):
        return
    ta = int(math.floor(float(true_hit_tick)))
    ta = max(0, min(ta, incoming_fleets.shape[2] - 1))
    owner = max(0, min(int(ego_player), incoming_fleets.shape[0] - 1))
    cur = int(incoming_fleets[owner, int(true_target_slot), ta])
    incoming_fleets[owner, int(true_target_slot), ta] = min(cur + int(send), 65535)


def observation_to_state(
    obs: Mapping[str, Any],
    config: Any = None,
    *,
    max_fleets: int = 512,
    step_count_override: Optional[int] = None,
    fleet_forecast_arrival: Optional[np.ndarray] = None,
    num_agents_override: Optional[int] = None,
    fleet_arrival_cache: Optional[FleetArrivalCache] = None,
) -> OrbitWarsState:
    """Convert an official Kaggle observation dict into a padded ``OrbitWarsState``.

    The policy only needs the current public state.  Fields absent from the
    official observation, such as rollout-only fleet ETA metadata, are filled
    with neutral defaults.
    """

    planets_in = _as_array(obs.get("planets", []), 7)
    planets_forecast_in = np.asarray(obs.get("planets", []), dtype=np.float64)
    if planets_forecast_in.size == 0:
        planets_forecast_in = np.zeros((0, 7), dtype=np.float64)
    else:
        planets_forecast_in = planets_forecast_in.reshape((-1, 7))
    planets, planet_active, id_to_slot = _place_rows_by_id(planets_in, 7)
    planets_forecast, _planet_active_forecast, _ = _place_rows_by_id(
        planets_forecast_in,
        7,
        dtype=np.float64,
    )

    initial_in = _as_array(obs.get("initial_planets", []), 7)
    initial_forecast_in = np.asarray(obs.get("initial_planets", []), dtype=np.float64)
    if initial_forecast_in.size == 0:
        initial_forecast_in = np.zeros((0, 7), dtype=np.float64)
    else:
        initial_forecast_in = initial_forecast_in.reshape((-1, 7))
    initial_planets = np.zeros((MAX_PLANETS, 7), dtype=np.float32)
    initial_planets_forecast = np.zeros((MAX_PLANETS, 7), dtype=np.float64)
    initial_active = np.zeros((MAX_PLANETS,), dtype=np.bool_)
    for row in initial_in[:MAX_PLANETS]:
        pid = int(row[0])
        slot = id_to_slot.get(pid, pid if 0 <= pid < MAX_PLANETS else -1)
        if 0 <= slot < MAX_PLANETS:
            initial_planets[slot, :7] = row[:7]
            initial_active[slot] = True
    for row in initial_forecast_in[:MAX_PLANETS]:
        pid = int(row[0])
        slot = id_to_slot.get(pid, pid if 0 <= pid < MAX_PLANETS else -1)
        if 0 <= slot < MAX_PLANETS:
            initial_planets_forecast[slot, :7] = row[:7]

    # Comets can be present in planets without being present in initial_planets.
    missing_initial = planet_active & ~initial_active
    initial_planets[missing_initial] = planets[missing_initial]
    initial_planets_forecast[missing_initial] = planets_forecast[missing_initial]
    initial_active[missing_initial] = True

    fleets_in = _as_array(obs.get("fleets", []), 7)
    fleets_forecast_in = np.asarray(obs.get("fleets", []), dtype=np.float64)
    if fleets_forecast_in.size == 0:
        fleets_forecast_in = np.zeros((0, 7), dtype=np.float64)
    else:
        fleets_forecast_in = fleets_forecast_in.reshape((-1, 7))
    max_fleets = max(int(max_fleets), len(fleets_in) + DEFAULT_MAX_ACTIONS)
    fleets = np.zeros((max_fleets, FLEET_ROW_WIDTH), dtype=np.float32)
    fleet_active = np.zeros((max_fleets,), dtype=np.bool_)
    for i, row in enumerate(fleets_in[:max_fleets]):
        fleets[i, FLEET_ID] = row[0]
        fleets[i, FLEET_OWNER] = row[1]
        fleets[i, FLEET_X] = row[2]
        fleets[i, FLEET_Y] = row[3]
        fleets[i, FLEET_ANGLE] = row[4]
        fleets[i, FLEET_FROM_PLANET] = row[5]
        fleets[i, FLEET_SHIPS] = row[6]
        fleet_active[i] = True

    if num_agents_override is not None:
        num_agents = max(2, int(num_agents_override))
    else:
        num_agents = max(2, int(_cfg_get(config, "agentCount", 2)))

    comet_paths = np.zeros((MAX_COMET_GROUPS, 4, MAX_COMET_PATH, 2), dtype=np.float32)
    comet_paths_forecast = np.zeros((MAX_COMET_GROUPS, 4, MAX_COMET_PATH, 2), dtype=np.float64)
    comet_path_lengths = np.zeros((MAX_COMET_GROUPS, 4), dtype=np.int32)
    comet_ships = np.zeros((MAX_COMET_GROUPS,), dtype=np.float32)
    comet_group_active = np.zeros((MAX_COMET_GROUPS,), dtype=np.bool_)
    comet_path_index = np.full((MAX_COMET_GROUPS,), -1, dtype=np.int32)
    comet_planet_ids = np.full((MAX_COMET_GROUPS, 4), -1, dtype=np.int32)
    comet_slots = np.full((MAX_COMET_GROUPS, 4), -1, dtype=np.int32)

    for g, comet in enumerate((obs.get("comets") or [])[:MAX_COMET_GROUPS]):
        ids = list(comet.get("planet_ids", []))[:4]
        paths = list(comet.get("paths", []))[:4]
        comet_group_active[g] = True
        comet_path_index[g] = int(comet.get("path_index", -1))
        for k, pid_raw in enumerate(ids):
            pid = int(pid_raw)
            comet_planet_ids[g, k] = pid
            comet_slots[g, k] = id_to_slot.get(pid, -1)
        for k, path in enumerate(paths):
            p = np.asarray(path, dtype=np.float32).reshape((-1, 2))
            p_forecast = np.asarray(path, dtype=np.float64).reshape((-1, 2))
            n = min(len(p), MAX_COMET_PATH)
            comet_paths[g, k, :n] = p[:n]
            comet_paths_forecast[g, k, :n] = p_forecast[:n]
            comet_path_lengths[g, k] = n
        slot0 = comet_slots[g, 0]
        if 0 <= slot0 < MAX_PLANETS:
            comet_ships[g] = planets[slot0, 5]

    angular_velocity = float(obs.get("angular_velocity", 0.0))
    step_count = int(step_count_override if step_count_override is not None else obs.get("step", obs.get("step_count", 0)))
    planet_collision_rank = _planet_collision_rank_from_obs(planets_in, id_to_slot)
    incoming_fleets = _forecast_incoming_fleets(
        planets_forecast,
        planet_active,
        initial_planets_forecast,
        initial_active,
        fleets_forecast_in,
        comet_paths_forecast,
        comet_path_lengths,
        comet_group_active,
        comet_path_index,
        comet_planet_ids,
        comet_slots,
        num_agents,
        step_count,
        angular_velocity,
        planet_collision_rank,
        ship_speed=float(_cfg_get(config, "shipSpeed", 6.0)),
        horizon=INCOMING_TA_BINS,
        per_fleet_arrival=fleet_forecast_arrival,
        fleet_arrival_cache=fleet_arrival_cache,
    )
    rewards = np.zeros((max(num_agents, 4),), dtype=np.float32)

    return OrbitWarsState(
        planets=planets,
        planet_active=planet_active,
        initial_planets=initial_planets,
        initial_active=initial_active,
        origin_frac_blocked=np.zeros((MAX_PLANETS, len(FRACTIONS)), dtype=np.bool_),
        fleets=fleets,
        fleet_active=fleet_active,
        incoming_fleets=incoming_fleets,
        comet_paths=comet_paths,
        comet_path_lengths=comet_path_lengths,
        comet_ships=comet_ships,
        comet_group_active=comet_group_active,
        comet_path_index=comet_path_index,
        comet_planet_ids=comet_planet_ids,
        comet_slots=comet_slots,
        planet_collision_rank=planet_collision_rank,
        next_fleet_id=np.asarray(len(fleets_in), dtype=np.int32),
        angular_velocity=np.asarray(angular_velocity, dtype=np.float32),
        step_count=np.asarray(step_count, dtype=np.int32),
        num_agents=np.asarray(num_agents, dtype=np.int32),
        rewards=rewards,
        done=np.asarray(False),
        overflow=np.asarray(False),
    )


def _public_obs_for_player(
    public_obs: Mapping[str, Any],
    *,
    player: int,
    step_count: int,
) -> dict[str, Any]:
    out = {
        "player": int(player),
        "angular_velocity": float(public_obs.get("angular_velocity", 0.0)),
        "planets": copy.deepcopy(list(public_obs.get("planets", []) or [])),
        "initial_planets": copy.deepcopy(list(public_obs.get("initial_planets", []) or [])),
        "fleets": copy.deepcopy(list(public_obs.get("fleets", []) or [])),
        "next_fleet_id": int(public_obs.get("next_fleet_id", len(public_obs.get("fleets", []) or []))),
        "comets": copy.deepcopy(list(public_obs.get("comets", []) or [])),
        "comet_planet_ids": copy.deepcopy(list(public_obs.get("comet_planet_ids", []) or [])),
        "remainingOverageTime": float(public_obs.get("remainingOverageTime", 60.0)),
        "step": int(step_count),
    }
    return out


def _public_obs_from_sim_state(state: list[Any], *, step_count: int) -> dict[str, Any]:
    obs0 = state[0].observation
    return _public_obs_for_player(obs0, player=0, step_count=step_count)


_CACHE_FLOAT_ATOL = 1e-5


def _state_float_array_match(
    lhs: Any,
    rhs: Any,
    *,
    float_atol: float = _CACHE_FLOAT_ATOL,
) -> bool:
    lhs = np.asarray(lhs)
    rhs = np.asarray(rhs)
    if lhs.shape != rhs.shape:
        return False
    if lhs.size == 0:
        return True
    return bool(np.allclose(lhs, rhs, atol=float_atol, rtol=0.0))


def _state_int_array_match(lhs: Any, rhs: Any) -> bool:
    lhs = np.asarray(lhs)
    rhs = np.asarray(rhs)
    if lhs.shape != rhs.shape:
        return False
    return bool(np.array_equal(lhs, rhs))


def _action_match(lhs: Sequence[float], rhs: Sequence[float], *, float_atol: float = _CACHE_FLOAT_ATOL) -> bool:
    if len(lhs) != len(rhs):
        return False
    return (
        math.isclose(float(lhs[0]), float(rhs[0]), abs_tol=float_atol, rel_tol=0.0)
        and math.isclose(float(lhs[1]), float(rhs[1]), abs_tol=float_atol, rel_tol=0.0)
        and int(round(float(lhs[2]))) == int(round(float(rhs[2])))
    )


def _action_sequence_match(
    lhs: Sequence[Sequence[float]],
    rhs: Sequence[Sequence[float]],
    *,
    float_atol: float = _CACHE_FLOAT_ATOL,
) -> bool:
    if len(lhs) != len(rhs):
        return False
    return all(_action_match(a, b, float_atol=float_atol) for a, b in zip(lhs, rhs))


def _cached_policy_value_for_player(cached: CachedSearchPolicyOutputs, player: int) -> float:
    row = cached.players.index(int(player))
    value = cached.value[row]
    if value.ndim > 0:
        value = value.reshape(-1)[0]
    return float(value.item())


def _cache_state_match(
    lhs: OrbitWarsState,
    rhs: OrbitWarsState,
    *,
    float_atol: float = _CACHE_FLOAT_ATOL,
) -> bool:
    if int(np.asarray(lhs.step_count)) != int(np.asarray(rhs.step_count)):
        return False
    if int(np.asarray(lhs.next_fleet_id)) != int(np.asarray(rhs.next_fleet_id)):
        return False
    if int(np.asarray(lhs.num_agents)) != int(np.asarray(rhs.num_agents)):
        return False
    if bool(np.asarray(lhs.done)) != bool(np.asarray(rhs.done)):
        return False
    if bool(np.asarray(lhs.overflow)) != bool(np.asarray(rhs.overflow)):
        return False
    if not math.isclose(
        float(np.asarray(lhs.angular_velocity)),
        float(np.asarray(rhs.angular_velocity)),
        abs_tol=float_atol,
        rel_tol=0.0,
    ):
        return False
    if not _state_float_array_match(lhs.planets, rhs.planets, float_atol=float_atol):
        return False
    if not _state_int_array_match(lhs.planet_active, rhs.planet_active):
        return False
    if not _state_float_array_match(lhs.initial_planets, rhs.initial_planets, float_atol=float_atol):
        return False
    if not _state_int_array_match(lhs.initial_active, rhs.initial_active):
        return False
    if not _state_float_array_match(lhs.incoming_fleets, rhs.incoming_fleets, float_atol=float_atol):
        return False
    if not _state_float_array_match(lhs.comet_paths, rhs.comet_paths, float_atol=float_atol):
        return False
    if not _state_int_array_match(lhs.comet_path_lengths, rhs.comet_path_lengths):
        return False
    if not _state_float_array_match(lhs.comet_ships, rhs.comet_ships, float_atol=float_atol):
        return False
    if not _state_int_array_match(lhs.comet_group_active, rhs.comet_group_active):
        return False
    if not _state_int_array_match(lhs.comet_path_index, rhs.comet_path_index):
        return False
    if not _state_int_array_match(lhs.comet_planet_ids, rhs.comet_planet_ids):
        return False
    if not _state_int_array_match(lhs.comet_slots, rhs.comet_slots):
        return False
    if not _state_int_array_match(lhs.planet_collision_rank, rhs.planet_collision_rank):
        return False
    return True


def _kaggle_env_modules() -> tuple[Any, Any]:
    from kaggle_environments.envs.orbit_wars import orbit_wars as ow
    from kaggle_environments.utils import structify

    return ow, structify


def _make_sim_state(
    public_obs: Mapping[str, Any],
    *,
    num_agents: int,
    step_count: int,
) -> list[Any]:
    _ow, structify = _kaggle_env_modules()
    state: list[dict[str, Any]] = []
    for player in range(int(num_agents)):
        state.append(
            {
                "action": [],
                "reward": 0.0,
                "status": "ACTIVE",
                "info": {},
                "observation": _public_obs_for_player(public_obs, player=player, step_count=step_count),
            }
        )
    return list(structify(state))


def _simulate_joint_step_with_kaggle_model(
    sim_state: list[Any],
    *,
    joint_actions: list[list[list[float]]],
    config: Any,
) -> None:
    ow, structify = _kaggle_env_modules()
    for player, action in enumerate(joint_actions):
        sim_state[player].action = copy.deepcopy(action)
    env_stub = structify(
        {
            "configuration": {
                "agentCount": len(sim_state),
                "shipSpeed": float(_cfg_get(config, "shipSpeed", 6.0)),
                "episodeSteps": int(_cfg_get(config, "episodeSteps", 500)),
                "cometSpeed": float(_cfg_get(config, "cometSpeed", 8.0)),
            },
            "done": False,
            "info": {},
        }
    )
    ow.interpreter(sim_state, env_stub)
    # Kaggle's outer environment loop advances ``observation.step`` around the
    # interpreter call. Our search stub calls the interpreter directly, so we
    # must mirror that bookkeeping here or future-step forecasts will see
    # stale planet positions paired with newer step counts.
    step0 = int(getattr(sim_state[0].observation, "step", 0))
    next_step = step0 + 1
    for seat in sim_state:
        seat.observation.step = next_step


def _env_float(name: str) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    return float(raw)


def _infer_num_agents_from_planet_owners(
    obs: Mapping[str, Any],
    *,
    fallback: int = 2,
) -> int:
    """Infer player count from distinct non-neutral planet owners."""

    owners: set[int] = set()
    for row in obs.get("planets", []) or []:
        try:
            owner = int(row[1])
        except (IndexError, TypeError, ValueError):
            continue
        if owner >= 0:
            owners.add(owner)
    if owners:
        return max(2, len(owners))
    return max(2, int(fallback))


def _model_search_steps_from_env() -> int:
    raw = _env_int("ORBIT_WARS_MODEL_SEARCH_STEPS")
    if raw is None:
        raw = _env_int("ORBIT_WARS_MODEL_SEARCH_HORIZON")
    if raw is None:
        return 0
    return max(0, int(raw))


def _dual_model_search_steps_from_env() -> tuple[int | None, int | None]:
    steps_4p = _env_int("ORBIT_WARS_MODEL_SEARCH_STEPS_4P")
    steps_2p = _env_int("ORBIT_WARS_MODEL_SEARCH_STEPS_2P")
    return (
        max(0, int(steps_4p)) if steps_4p is not None else None,
        max(0, int(steps_2p)) if steps_2p is not None else None,
    )


def _validate_model_search_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in _VALID_MODEL_SEARCH_MODES:
        raise ValueError(
            f"invalid model search mode {value!r}; expected one of {sorted(_VALID_MODEL_SEARCH_MODES)}"
        )
    return mode


def _model_search_mode_from_env() -> str:
    raw = os.environ.get("ORBIT_WARS_MODEL_SEARCH_MODE", MODEL_SEARCH_MODE_BINARY).strip()
    if not raw:
        return MODEL_SEARCH_MODE_BINARY
    return _validate_model_search_mode(raw)


def _model_search_adaptive_horizon_from_env() -> bool:
    raw = os.environ.get("ORBIT_WARS_MODEL_SEARCH_ADAPTIVE_HORIZON", "").lower()
    return raw in {"1", "true", "yes", "on"}


def _model_search_adaptive_horizon_offset_from_env() -> int:
    raw = _env_int("ORBIT_WARS_MODEL_SEARCH_ADAPTIVE_HORIZON_OFFSET")
    if raw is None:
        return 2
    return max(0, int(raw))


def _model_search_min_overage_from_env() -> float:
    raw = _env_float("ORBIT_WARS_MODEL_SEARCH_MIN_OVERAGE_S")
    if raw is None:
        return 15.0
    return max(0.0, float(raw))


def _model_search_gamma_from_env(fallback: float) -> float:
    raw = _env_float("ORBIT_WARS_MODEL_SEARCH_GAMMA")
    if raw is None:
        return float(fallback)
    return float(raw)


def _validate_probability_threshold(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _model_search_greedy_launch_threshold_from_env() -> float | None:
    raw = _env_float("ORBIT_WARS_MODEL_SEARCH_GREEDY_LAUNCH_THRESHOLD")
    if raw is None:
        return None
    return _validate_probability_threshold(float(raw), name="ORBIT_WARS_MODEL_SEARCH_GREEDY_LAUNCH_THRESHOLD")


def _model_search_launch_probability_threshold_from_env() -> float | None:
    raw = _env_float("ORBIT_WARS_MODEL_SEARCH_LAUNCH_PROB_THRESHOLD")
    if raw is None:
        return None
    return _validate_probability_threshold(float(raw), name="ORBIT_WARS_MODEL_SEARCH_LAUNCH_PROB_THRESHOLD")


def _model_search_branch_probability_threshold_from_env() -> float:
    raw = _env_float("ORBIT_WARS_MODEL_SEARCH_BRANCH_PROB_THRESHOLD")
    if raw is None:
        return 0.2
    value = _validate_probability_threshold(
        float(raw),
        name="ORBIT_WARS_MODEL_SEARCH_BRANCH_PROB_THRESHOLD",
    )
    assert value is not None
    return float(value)


def _model_search_max_branching_factor_from_env() -> int:
    raw = _env_int("ORBIT_WARS_MODEL_SEARCH_MAX_BRANCHING")
    if raw is None:
        return 4
    return max(1, int(raw))


def _model_search_branch_after_first_env_step_from_env() -> bool:
    raw = os.environ.get("ORBIT_WARS_MODEL_SEARCH_BRANCH_AFTER_FIRST_ENV_STEP", "").lower()
    if not raw:
        return True
    return raw in {"1", "true", "yes", "on"}


def _model_search_branch_micro_depth_from_env() -> int | None:
    raw = _env_int("ORBIT_WARS_MODEL_SEARCH_BRANCH_MICRO_DEPTH")
    if raw is None:
        return None
    return max(0, int(raw))


def _model_search_stop_at_turn_end_from_env() -> bool:
    raw = os.environ.get("ORBIT_WARS_MODEL_SEARCH_STOP_AT_TURN_END", "").lower()
    if not raw:
        return False
    return raw in {"1", "true", "yes", "on"}


def _model_search_turn_end_opponent_samples_from_env() -> int:
    raw = _env_int("ORBIT_WARS_MODEL_SEARCH_TURN_END_OPPONENT_SAMPLES")
    if raw is None:
        return 0
    return max(0, int(raw))


def _model_search_turn_sampling_max_samples_from_env() -> int | None:
    raw = _env_int("ORBIT_WARS_MODEL_SEARCH_TURN_SAMPLING_MAX_SAMPLES")
    if raw is None:
        return None
    return max(0, int(raw))


def _launch_probability_from_halt_logits(halt_logits: torch.Tensor) -> float:
    return float(torch.softmax(halt_logits, dim=-1)[0].item())


def _launch_probability_meets_threshold(
    halt_logits: torch.Tensor,
    *,
    threshold: float | None = None,
) -> bool:
    if threshold is None:
        return True
    return _launch_probability_from_halt_logits(halt_logits) >= float(threshold)


def _search_branch_indices_from_probs(
    probs: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    threshold: float,
    max_branching: int,
) -> list[int]:
    probs = probs.detach().to(dtype=torch.float32)
    if mask is not None:
        valid_mask = mask.detach().to(dtype=torch.bool)
        masked = probs.masked_fill(~valid_mask, -1.0)
    else:
        valid_mask = None
        masked = probs
    order = torch.argsort(masked, descending=True).tolist()
    out: list[int] = []
    fallback: int | None = None
    for idx in order:
        idx = int(idx)
        if valid_mask is not None and not bool(valid_mask[idx].item()):
            continue
        p = float(probs[idx].item())
        if fallback is None:
            fallback = idx
        if p + 1e-8 < float(threshold):
            continue
        out.append(idx)
        if len(out) >= max(1, int(max_branching)):
            break
    if not out and fallback is not None:
        out.append(int(fallback))
    return out


def _search_action_key(action: Sequence[float]) -> tuple[float, int, int]:
    return (
        float(action[0]),
        int(round(float(action[1]) * 1_000_000.0)),
        int(round(float(action[2]))),
    )


def _search_turn_signature(
    actions: Sequence[SearchPlannedLaunchAction],
    origin_frac_blocked: np.ndarray,
) -> tuple[tuple[float, int, int], ...] | tuple[tuple[tuple[float, int, int], ...], bytes]:
    action_keys = tuple(sorted(_search_action_key(action.action) for action in actions))
    blocked = np.asarray(origin_frac_blocked, dtype=np.bool_)
    return (action_keys, blocked.tobytes())


def _search_single_root_turn_path(
    nodes: Sequence[_SearchTreeNode],
) -> _SearchTreeNode | None:
    """Return the sole surviving root-turn path if it is fully resolved.

    We only early-exit once exactly one node remains on the root env step and
    that node has already closed the turn, meaning no further same-turn
    branching can occur before the first simulated env advance.
    """

    root_nodes = [node for node in nodes if int(node.search_env_step_from_root) == 0]
    if len(root_nodes) != 1:
        return None
    node = root_nodes[0]
    if not bool(node.turn_closed):
        return None
    return node


def _greedy_halt_action_from_logits(
    halt_logits: torch.Tensor,
    *,
    launch_threshold: float | None = None,
) -> int:
    if launch_threshold is None:
        return int(torch.argmax(halt_logits, dim=-1).item())
    return 0 if _launch_probability_meets_threshold(halt_logits, threshold=launch_threshold) else 1


def _search_opponent_should_halt_on_neutral_underlaunch(
    planets: np.ndarray,
    incoming_fleets: np.ndarray,
    *,
    player: int,
    search_root_player: int | None,
    send: int,
    true_target_slot: int,
) -> bool:
    if search_root_player is None or int(player) == int(search_root_player):
        return False
    if not (0 <= int(true_target_slot) < MAX_PLANETS):
        return False
    if int(send) <= 0:
        return False
    target_owner = int(planets[int(true_target_slot), 1])
    if target_owner >= 0:
        return False
    garrison = int(planets[int(true_target_slot), 5])
    if int(send) > garrison:
        return False
    incoming = np.asarray(incoming_fleets)
    if incoming.ndim != 3:
        return False
    return not bool(np.any(incoming[:, int(true_target_slot), :] > 0))


def _infer_policy_kwargs(payload: Any, *, policy_key: str = "policy") -> dict[str, Any]:
    training_args = payload.get("training_args", {}) if isinstance(payload, Mapping) else {}
    policy_state = payload.get(policy_key, payload) if isinstance(payload, Mapping) else payload
    prefix = "" if policy_key == "policy" else f"{policy_key.removesuffix('_policy')}_"
    kwargs = {
        "d_model": int(training_args.get(f"{prefix}d_model", training_args.get("d_model", 384))),
        "n_heads": int(training_args.get(f"{prefix}n_heads", training_args.get("n_heads", 8))),
        "n_layers": int(training_args.get(f"{prefix}n_layers", training_args.get("n_layers", 4))),
        "activation_checkpointing": False,
        "population_size": int(training_args.get("population_size", 1)),
        "rope_dims": int(training_args.get(f"{prefix}rope_dims", training_args.get("rope_dims", 3))),
        "value_head_count": int(
            training_args.get(f"{prefix}value_head_count", training_args.get("value_head_count", 1))
        ),
        "disjoint_actor_critic": bool(training_args.get("disjoint_actor_critic", False)),
        "target_abort_enabled": bool(training_args.get("target_abort_enabled", False)),
        "future_feature_enabled": bool(training_args.get("future_feature_enabled", False)),
        "halt_init_prob": training_args.get("halt_init_prob"),
    }
    fraction_init_ratio = training_args.get("fraction_init_ratio")
    if fraction_init_ratio:
        parts = [chunk.strip() for chunk in str(fraction_init_ratio).replace(",", ":").split(":")]
        kwargs["fraction_init_weights"] = tuple(float(chunk) for chunk in parts if chunk)
    if isinstance(policy_state, Mapping):
        w = policy_state.get("feat_proj.weight")
        if hasattr(w, "shape") and len(w.shape) >= 2:
            kwargs["d_model"] = int(w.shape[0])
            kwargs["feature_dim"] = int(w.shape[1])
        kwargs["value_head_count"] = int(infer_value_head_count_from_state_dict(policy_state))
        layer_ids = []
        shared_layer_ids = []
        pop_ids = []
        for key in policy_state:
            key_s = str(key)
            if "abort_head." in key_s:
                kwargs["target_abort_enabled"] = True
            if key_s.startswith("future_feat_proj."):
                kwargs["future_feature_enabled"] = True
            if key_s.startswith("critic_"):
                kwargs["disjoint_actor_critic"] = True
            if key_s.startswith("blocks."):
                try:
                    layer_ids.append(int(key_s.split(".")[1]))
                except (IndexError, ValueError):
                    pass
            elif key_s.startswith("shared_blocks."):
                try:
                    shared_layer_ids.append(int(key_s.split(".")[1]))
                except (IndexError, ValueError):
                    pass
            elif key_s.startswith("population_tails."):
                try:
                    pop_ids.append(int(key_s.split(".")[1]))
                except (IndexError, ValueError):
                    pass
        if layer_ids:
            kwargs["n_layers"] = max(layer_ids) + 1
        elif pop_ids:
            kwargs["population_size"] = max(pop_ids) + 1
            kwargs["n_layers"] = (max(shared_layer_ids) + 1 if shared_layer_ids else 0) + 1
    return kwargs


def _checkpoint_training_args(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        training_args = payload.get("training_args", {})
        if isinstance(training_args, Mapping):
            return training_args
    return {}


def _strip_legacy_pair_head_keys(state: Mapping[str, Any]) -> OrderedDict[str, Any]:
    out = OrderedDict()
    for key, value in state.items():
        key_s = str(key)
        if ".pair_q." in key_s or ".pair_k." in key_s or key_s.startswith("pair_q.") or key_s.startswith("pair_k."):
            continue
        out[key] = value
    return out


def _checkpoint_search_roots() -> list[Path]:
    """Directories to try when resolving a relative checkpoint path."""

    pkg = Path(__file__).resolve().parent
    roots: list[Path] = [Path.cwd(), pkg.parent, pkg]
    for entry in sys.path[:4]:
        if not entry or entry in {".", ""}:
            continue
        try:
            roots.append(Path(entry).resolve())
        except OSError:
            continue
    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        unique.append(root)
    return unique


def resolve_checkpoint_path(path: str | os.PathLike[str]) -> Path:
    """Resolve ``checkpoint.pt`` from cwd, submission bundle root, or this package."""

    raw = Path(path).expanduser()
    if raw.is_file():
        return raw.resolve()

    if raw.is_absolute():
        raise FileNotFoundError(f"Checkpoint not found: {raw}")

    candidates = [root / raw for root in _checkpoint_search_roots()]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Checkpoint not found: {raw!s} (searched: {searched})")


def load_policy(
    checkpoint_path: str | os.PathLike[str] = DEFAULT_CHECKPOINT,
    *,
    device: Optional[str | torch.device] = None,
    policy_key: str = "policy",
) -> tuple[OrbitWarsPolicy, torch.device, Mapping[str, Any]]:
    """Load a training checkpoint or raw policy state dict for inference."""

    resolved = resolve_checkpoint_path(checkpoint_path)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, Mapping):
        if policy_key == "policy":
            policy_state = payload.get("policy", payload)
        else:
            if policy_key not in payload:
                raise KeyError(
                    f"Checkpoint {resolved} does not contain policy key {policy_key!r}"
                )
            policy_state = payload[policy_key]
        payload_for_kwargs: Any = {
            policy_key: policy_state,
            "training_args": _checkpoint_training_args(payload),
        }
    else:
        policy_state = payload
        payload_for_kwargs = payload
    policy = OrbitWarsPolicy(**_infer_policy_kwargs(payload_for_kwargs, policy_key=policy_key)).to(torch_device)
    policy_state_adapted, _ = adapt_checkpoint_state_for_model(policy_state, policy)
    policy_state_adapted = _strip_legacy_pair_head_keys(policy_state_adapted)
    policy.load_state_dict(policy_state_adapted)
    policy.eval()
    training_args = dict(_checkpoint_training_args(payload))
    return policy, torch_device, training_args


def _maybe_compile_policy_batched_forward_for_inference(policy: OrbitWarsPolicy) -> OrbitWarsPolicy:
    raw = os.environ.get("ORBIT_WARS_COMPILE_BATCHED_FORWARD", "0").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return policy
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return policy
    mode = os.environ.get("ORBIT_WARS_COMPILE_BATCHED_FORWARD_MODE", "default").strip() or "default"
    try:
        import torch._dynamo as torch_dynamo

        torch_dynamo.config.capture_scalar_outputs = True
        policy.forward_dense_rollout = compile_fn(policy.forward_dense_rollout, mode=mode, dynamic=True)  # type: ignore[assignment]
    except Exception as exc:
        print(
            f"[orbit_wars_pt] torch.compile skipped for policy.forward_dense_rollout: {exc}",
            file=sys.stderr,
            flush=True,
        )
    return policy


def _compile_batched_forward_enabled() -> bool:
    raw = os.environ.get("ORBIT_WARS_COMPILE_BATCHED_FORWARD", "0").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _compile_batched_forward_warmup_batches() -> list[int]:
    raw = os.environ.get("ORBIT_WARS_COMPILE_BATCHED_FORWARD_WARMUP_BATCHES", "7").strip()
    if not raw:
        return [7]
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            val = int(part)
        except ValueError:
            continue
        if val > 0 and val not in out:
            out.append(val)
    return out or [7]


def _policy_forward_inference(
    policy: OrbitWarsPolicy,
    batch: Mapping[str, torch.Tensor],
    *,
    population_idx: Optional[torch.Tensor],
) -> dict[str, Any]:
    batch_size = int(batch["features"].shape[0])
    if batch_size > 1:
        return policy.forward_dense_rollout(
            entity_type=batch["entity_type"],
            owner_idx=batch["owner_idx"],
            features=batch["features"],
            rope_pos=batch["rope_pos"],
            entity_mask=batch["entity_mask"],
            planet_mask=batch["planet_mask"],
            origin_frac_blocked=batch.get("origin_frac_blocked"),
            population_idx=population_idx,
        )
    return policy(**batch, population_idx=population_idx)


def _warmup_compiled_policy_batched_forward(
    policy: OrbitWarsPolicy,
    state: OrbitWarsState,
    ego_player: int,
    device: torch.device,
    *,
    policy_player_count: Optional[int],
    normalize_obs_to_p0: bool,
    population_member: Optional[int],
) -> None:
    if not _compile_batched_forward_enabled():
        return
    batches = _compile_batched_forward_warmup_batches()
    with torch.inference_mode():
        for batch_size in batches:
            states = [state] * int(batch_size)
            players = [int(ego_player)] * int(batch_size)
            batch = _obs_tensors_for_states(
                states,
                players,
                device,
                policy_player_count=policy_player_count,
                target_abort_enabled=bool(getattr(policy, "target_abort_enabled", False)),
                normalize_obs_to_p0=normalize_obs_to_p0,
            )
            population_idx = None
            if population_member is not None:
                population_idx = torch.full(
                    (int(batch_size),),
                    int(population_member),
                    device=device,
                    dtype=torch.long,
                )
            _policy_forward_inference(
                policy,
                batch,
                population_idx=population_idx,
            )


def _policy_value_for_state(
    policy: OrbitWarsPolicy,
    state: OrbitWarsState,
    ego_player: int,
    device: torch.device,
    *,
    policy_player_count: Optional[int],
    normalize_obs_to_p0: bool,
    population_member: Optional[int],
) -> float:
    batch = _obs_tensors_for_state(
        state,
        ego_player,
        device,
        policy_player_count=policy_player_count,
        target_abort_enabled=bool(getattr(policy, "target_abort_enabled", False)),
        normalize_obs_to_p0=normalize_obs_to_p0,
    )
    population_idx = None
    if population_member is not None:
        population_idx = torch.tensor([int(population_member)], device=device, dtype=torch.long)
    with torch.inference_mode():
        out = _policy_forward_inference(
            policy,
            batch,
            population_idx=population_idx,
        )
    return float(out["value"][0].item())


class KaggleOrbitWarsAgent:
    """Callable adapter object suitable for Kaggle's ``agent(obs, config)`` API."""

    def __init__(
        self,
        checkpoint_path: str | os.PathLike[str] = DEFAULT_CHECKPOINT,
        *,
        use_student_for_search: bool = False,
        search_main_policy_for_ego_steps: int = 1,
        device: Optional[str | torch.device] = None,
        policy_key: str = "policy",
        greedy: bool | Mapping[int, bool] = False,
        sampling_mode: str | Mapping[int, str] | None = None,
        population_member: Optional[int | Mapping[int, int]] = None,
        max_micro_steps: Optional[int] = None,
        max_fleets: int = 512,
        seed: Optional[int] = None,
        raycast_rays: Optional[int] = None,
        target_method: Optional[str] = None,
        interval_samples_per_span: Optional[int] = None,
        model_search_steps: Optional[int] = None,
        model_search_steps_4p: Optional[int] = None,
        model_search_steps_2p: Optional[int] = None,
        model_search_mode: Optional[str] = None,
        model_search_gamma: Optional[float] = None,
        model_search_adaptive_horizon: Optional[bool] = None,
        model_search_adaptive_horizon_offset: Optional[int] = None,
        model_search_min_overage_s: Optional[float] = None,
        model_search_launch_prob_threshold: Optional[float] = None,
        model_search_branch_prob_threshold: Optional[float] = None,
        model_search_max_branching: Optional[int] = None,
        model_search_branch_after_first_env_step: Optional[bool] = None,
        model_search_branch_micro_depth: Optional[int] = None,
        model_search_stop_at_turn_end: Optional[bool] = None,
        model_search_turn_end_opponent_samples: Optional[int] = None,
        model_search_turn_sampling_max_samples: Optional[int] = None,
    ):
        _configure_cpu_threads()
        self.checkpoint_path = resolve_checkpoint_path(checkpoint_path)
        self.use_student_for_search = bool(use_student_for_search)
        self.search_main_policy_for_ego_steps = max(0, int(search_main_policy_for_ego_steps))
        self.policy_key = str(policy_key)
        self.policy, self.device, training_args = load_policy(
            self.checkpoint_path,
            device=device,
            policy_key=self.policy_key,
        )
        self.policy = _maybe_compile_policy_batched_forward_for_inference(self.policy)
        self.search_policy = self.policy
        search_training_args = training_args
        if self.use_student_for_search:
            self.search_policy, _search_device, search_training_args = load_policy(
                self.checkpoint_path,
                device=self.device,
                policy_key="student_policy",
            )
            self.search_policy = _maybe_compile_policy_batched_forward_for_inference(self.search_policy)
        self.reward_settings = _reward_settings_from_training_args(training_args)
        self.reward_settings = RewardSettings(
            ship_mass_share_coef=self.reward_settings.ship_mass_share_coef,
            production_share_coef=self.reward_settings.production_share_coef,
            terminal_win_loss_coef=self.reward_settings.terminal_win_loss_coef,
            terminal_loss=self.reward_settings.terminal_loss,
            terminal_draw=self.reward_settings.terminal_draw,
            terminal_win=self.reward_settings.terminal_win,
            time_bonus_coef=self.reward_settings.time_bonus_coef,
            gamma=(
                float(model_search_gamma)
                if model_search_gamma is not None
                else _model_search_gamma_from_env(self.reward_settings.gamma)
            ),
            episode_steps=self.reward_settings.episode_steps,
        )
        self.model_search = ModelSearchSettings(
            horizon_steps=(
                max(0, int(model_search_steps))
                if model_search_steps is not None
                else _model_search_steps_from_env()
            ),
            reward=self.reward_settings,
            mode=(
                _validate_model_search_mode(model_search_mode)
                if model_search_mode is not None
                else _model_search_mode_from_env()
            ),
            adaptive_horizon=(
                bool(model_search_adaptive_horizon)
                if model_search_adaptive_horizon is not None
                else _model_search_adaptive_horizon_from_env()
            ),
            adaptive_horizon_offset=(
                max(0, int(model_search_adaptive_horizon_offset))
                if model_search_adaptive_horizon_offset is not None
                else _model_search_adaptive_horizon_offset_from_env()
            ),
            min_overage_s=(
                max(0.0, float(model_search_min_overage_s))
                if model_search_min_overage_s is not None
                else _model_search_min_overage_from_env()
            ),
            launch_probability_threshold=(
                _validate_probability_threshold(
                    model_search_launch_prob_threshold,
                    name="model_search_launch_prob_threshold",
                )
                if model_search_launch_prob_threshold is not None
                else _model_search_launch_probability_threshold_from_env()
            ),
            greedy_launch_threshold=_model_search_greedy_launch_threshold_from_env(),
            branch_probability_threshold=(
                _validate_probability_threshold(
                    model_search_branch_prob_threshold,
                    name="model_search_branch_prob_threshold",
                )
                if model_search_branch_prob_threshold is not None
                else _model_search_branch_probability_threshold_from_env()
            )
            or 0.2,
            max_branching_factor=(
                max(1, int(model_search_max_branching))
                if model_search_max_branching is not None
                else _model_search_max_branching_factor_from_env()
            ),
            branch_after_first_env_step=(
                bool(model_search_branch_after_first_env_step)
                if model_search_branch_after_first_env_step is not None
                else _model_search_branch_after_first_env_step_from_env()
            ),
            branch_micro_depth=(
                max(0, int(model_search_branch_micro_depth))
                if model_search_branch_micro_depth is not None
                else _model_search_branch_micro_depth_from_env()
            ),
            stop_at_turn_end=(
                bool(model_search_stop_at_turn_end)
                if model_search_stop_at_turn_end is not None
                else _model_search_stop_at_turn_end_from_env()
            ),
            turn_end_opponent_samples=(
                max(0, int(model_search_turn_end_opponent_samples))
                if model_search_turn_end_opponent_samples is not None
                else _model_search_turn_end_opponent_samples_from_env()
            ),
            turn_sampling_max_samples=(
                max(0, int(model_search_turn_sampling_max_samples))
                if model_search_turn_sampling_max_samples is not None
                else _model_search_turn_sampling_max_samples_from_env()
            ),
        )
        self.population_size = int(training_args.get("population_size", 1))
        self._population_member_by_player = _normalize_population_members(
            population_member,
            population_size=self.population_size,
            context="single",
        )
        self.normalize_obs_to_p0 = bool(training_args.get("normalize_obs_to_p0", False))
        self.policy_player_count = 4 if int(training_args.get("num_agents", 2)) > 2 else 2
        self.search_population_size = int(search_training_args.get("population_size", 1))
        self._search_population_member_by_player = _normalize_population_members(
            population_member,
            population_size=self.search_population_size,
            context="single:search",
        )
        self.search_normalize_obs_to_p0 = bool(search_training_args.get("normalize_obs_to_p0", False))
        self.search_policy_player_count = 4 if int(search_training_args.get("num_agents", 2)) > 2 else 2
        self._greedy_by_player = _normalize_greedy(greedy)
        self._sampling_mode_by_player = _normalize_sampling_mode(
            sampling_mode,
            fallback_greedy=self._greedy_by_player,
        )
        self.max_micro_steps = int(
            max_micro_steps
            if max_micro_steps is not None
            else training_args.get("max_micro_steps", DEFAULT_MAX_ACTIONS)
        )
        self.max_fleets = int(max_fleets)
        self.raycast_rays = int(
            raycast_rays
            if raycast_rays is not None
            else training_args.get("first_hit_n_rays", DEFAULT_RAYCAST_RAYS)
        )
        self.target_method = str(
            target_method
            if target_method is not None
            else os.environ.get("ORBIT_WARS_TARGET_METHOD", DEFAULT_TARGET_METHOD)
        ).lower()
        self.interval_samples_per_span = int(
            interval_samples_per_span
            if interval_samples_per_span is not None
            else os.environ.get(
                "ORBIT_WARS_INTERVAL_SAMPLES",
                str(DEFAULT_INTERVAL_SAMPLES_PER_SPAN),
            )
        )
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(int(seed if seed is not None else os.environ.get("ORBIT_WARS_AGENT_SEED", "0")))
        self._game_key: Optional[str] = None
        # ``initial_planets`` in Kaggle obs grows/shrinks as comets appear; hash only at step 0.
        self._frozen_game_key: Optional[str] = None
        self._frozen_num_agents: Optional[int] = None
        self._next_step_count = 0
        # Kaggle omits ``step`` on player 1's observation; after player 0 runs, mirror that value here
        # so a single shared ``KaggleOrbitWarsAgent`` (``agent()``) still builds the correct state.
        self._last_env_step: Optional[int] = None
        self._last_call_timing: Optional[KaggleAgentCallTiming] = None
        self._sanity_warnings: set[str] = set()
        self._compiled_forward_warmup_done = False
        self._search_compiled_forward_warmup_done = self.search_policy is self.policy
        warn_oob = os.environ.get("ORBIT_WARS_WARN_OOB_LAUNCHES", "1").lower() not in {"0", "false", "no", "off"}
        self.launch_tracker = FleetLaunchDebugTracker(
            warn_oob=warn_oob,
            warn_forecast_mismatch=_warn_forecast_mismatch_enabled(),
            warn_unmatched_fleet=_warn_unmatched_fleet_enabled(),
        )
        self._fleet_arrival_cache = FleetArrivalCache()
        self._search_rollout_cache: Optional[CachedSearchRollout] = None
        self._search_cache_hits: int = 0
        self._search_cache_misses: int = 0

    def _population_member_for_player(self, player: int) -> Optional[int]:
        if self.population_size <= 1:
            return None
        return int(self._population_member_by_player.get(int(player), self._population_member_by_player[-1]))

    def _search_population_member_for_player(self, player: int) -> Optional[int]:
        if not hasattr(self, "search_population_size"):
            return self._population_member_for_player(player)
        if self.search_population_size <= 1:
            return None
        return int(
            self._search_population_member_by_player.get(
                int(player),
                self._search_population_member_by_player[-1],
            )
        )

    def _search_policy_obj(self) -> OrbitWarsPolicy:
        return getattr(self, "search_policy", self.policy)

    def _search_policy_player_count_value(self) -> int:
        return int(getattr(self, "search_policy_player_count", self.policy_player_count))

    def _search_normalize_obs_to_p0_value(self) -> bool:
        return bool(getattr(self, "search_normalize_obs_to_p0", self.normalize_obs_to_p0))

    def _search_uses_main_policy_for_ego_at_step(self, search_env_step_from_root: int) -> bool:
        if self._search_policy_obj() is self.policy:
            return True
        return int(search_env_step_from_root) < int(getattr(self, "search_main_policy_for_ego_steps", 1))

    def _search_cached_policy_outputs_match_new_root_step(self, search_env_step_from_root: int) -> bool:
        return self._search_uses_main_policy_for_ego_at_step(int(search_env_step_from_root)) == (
            self._search_uses_main_policy_for_ego_at_step(int(search_env_step_from_root) + 1)
        )

    def _cached_policy_outputs_match_active_policy_selection(
        self,
        cached: CachedSearchPolicyOutputs | None,
        active: Sequence[_BatchedSearchSeatPlan],
        *,
        search_root_player: int | None,
        search_env_step_from_root: int,
    ) -> bool:
        if cached is None:
            return False
        if len(active) != int(cached.halt_logits.shape[0]):
            return False
        hidden_rows = (
            cached.planet_hidden
            if isinstance(cached.planet_hidden, tuple)
            else tuple(cached.planet_hidden[row] for row in range(int(cached.halt_logits.shape[0])))
        )
        if len(hidden_rows) != len(active):
            return False
        main_d_model = int(getattr(self.policy, "d_model", 0))
        search_d_model = int(getattr(self._search_policy_obj(), "d_model", 0))
        for row, plan in enumerate(active):
            hidden = hidden_rows[row]
            if hidden is None or hidden.ndim != 2:
                return False
            use_main = self._search_uses_main_policy_for_player(
                int(plan.player),
                search_root_player,
                int(search_env_step_from_root),
            )
            expected = main_d_model if use_main else search_d_model
            if int(hidden.shape[-1]) != int(expected):
                return False
        return True

    def _search_uses_main_policy_for_player(
        self,
        player: int,
        search_root_player: int | None,
        search_env_step_from_root: int,
    ) -> bool:
        if self._search_policy_obj() is self.policy:
            return True
        if int(player) != int(search_root_player) if search_root_player is not None else False:
            return False
        return self._search_uses_main_policy_for_ego_at_step(int(search_env_step_from_root))

    def _search_policy_context_for_player(
        self,
        player: int,
        search_root_player: int | None,
        search_env_step_from_root: int,
    ) -> tuple[OrbitWarsPolicy, int, bool, Any]:
        if self._search_uses_main_policy_for_player(
            int(player),
            search_root_player,
            int(search_env_step_from_root),
        ):
            return (
                self.policy,
                self.policy_player_count,
                self.normalize_obs_to_p0,
                self._population_member_for_player,
            )
        return (
            self._search_policy_obj(),
            self._search_policy_player_count_value(),
            self._search_normalize_obs_to_p0_value(),
            self._search_population_member_for_player,
        )

    def _sampling_mode_for_player(self, player: int) -> str:
        return str(self._sampling_mode_by_player.get(int(player), SAMPLING_MODE_STOCHASTIC))

    def _num_agents_for_obs(self, obs: Mapping[str, Any], config: Any = None) -> int:
        if self._frozen_num_agents is None:
            self._frozen_num_agents = _infer_num_agents_from_planet_owners(
                obs,
                fallback=int(_cfg_get(config, "agentCount", self.policy_player_count)),
            )
        return int(self._frozen_num_agents)

    def _policy_values_for_states_batched(
        self,
        states: list[OrbitWarsState],
        players: list[int],
        *,
        policy: OrbitWarsPolicy | None = None,
        policy_player_count: int | None = None,
        normalize_obs_to_p0: bool | None = None,
        population_member_for_player: Any = None,
    ) -> list[float]:
        if not states:
            return []
        policy = self.policy if policy is None else policy
        policy_player_count = self.policy_player_count if policy_player_count is None else int(policy_player_count)
        normalize_obs_to_p0 = self.normalize_obs_to_p0 if normalize_obs_to_p0 is None else bool(normalize_obs_to_p0)
        population_member_for_player = (
            self._population_member_for_player
            if population_member_for_player is None
            else population_member_for_player
        )
        batch = _obs_tensors_for_states(
            states,
            players,
            self.device,
            policy_player_count=policy_player_count,
            target_abort_enabled=bool(getattr(policy, "target_abort_enabled", False)),
            normalize_obs_to_p0=normalize_obs_to_p0,
        )
        population_members = [population_member_for_player(player) for player in players]
        population_idx = None
        if any(member is not None for member in population_members):
            population_idx = torch.tensor(
                [0 if member is None else int(member) for member in population_members],
                device=self.device,
                dtype=torch.long,
            )
        with torch.inference_mode():
            out = _policy_forward_inference(
                policy,
                batch,
                population_idx=population_idx,
            )
        values = out["value"].reshape(len(states), -1)[:, 0]
        return [float(v.item()) for v in values]

    def _policy_outputs_for_states_batched(
        self,
        states: list[OrbitWarsState],
        players: list[int],
        *,
        policy: OrbitWarsPolicy | None = None,
        policy_player_count: int | None = None,
        normalize_obs_to_p0: bool | None = None,
        population_member_for_player: Any = None,
    ) -> dict[str, Any]:
        if not states:
            return {}
        policy = self.policy if policy is None else policy
        policy_player_count = self.policy_player_count if policy_player_count is None else int(policy_player_count)
        normalize_obs_to_p0 = self.normalize_obs_to_p0 if normalize_obs_to_p0 is None else bool(normalize_obs_to_p0)
        population_member_for_player = (
            self._population_member_for_player
            if population_member_for_player is None
            else population_member_for_player
        )
        batch = _obs_tensors_for_states(
            states,
            players,
            self.device,
            policy_player_count=policy_player_count,
            target_abort_enabled=bool(getattr(policy, "target_abort_enabled", False)),
            normalize_obs_to_p0=normalize_obs_to_p0,
        )
        population_members = [population_member_for_player(player) for player in players]
        population_idx = None
        if any(member is not None for member in population_members):
            population_idx = torch.tensor(
                [0 if member is None else int(member) for member in population_members],
                device=self.device,
                dtype=torch.long,
            )
        with torch.inference_mode():
            return _policy_forward_inference(
                policy,
                batch,
                population_idx=population_idx,
            )

    def _search_policy_outputs_for_states_batched_mixed(
        self,
        states: list[OrbitWarsState],
        players: list[int],
        *,
        search_root_player: int | None,
        search_env_step_from_root: int,
    ) -> dict[str, Any]:
        if not states:
            return {}
        grouped: dict[bool, list[int]] = {True: [], False: []}
        for idx, player in enumerate(players):
            grouped[
                self._search_uses_main_policy_for_player(
                    int(player),
                    search_root_player,
                    int(search_env_step_from_root),
                )
            ].append(idx)

        merged: dict[str, Any] | None = None
        planet_hidden_rows: list[torch.Tensor | None] = [None] * len(states)
        abort_logits_enabled: bool | None = None
        for use_main, row_indices in grouped.items():
            if not row_indices:
                continue
            sample_player = int(players[row_indices[0]])
            policy, policy_player_count, normalize_obs_to_p0, population_member_for_player = (
                self._search_policy_context_for_player(
                    sample_player,
                    search_root_player,
                    int(search_env_step_from_root),
                )
            )
            out_group = self._policy_outputs_for_states_batched(
                [states[i] for i in row_indices],
                [players[i] for i in row_indices],
                policy=policy,
                policy_player_count=policy_player_count,
                normalize_obs_to_p0=normalize_obs_to_p0,
                population_member_for_player=population_member_for_player,
            )
            group_abort = out_group.get("abort_logits")
            if abort_logits_enabled is None:
                abort_logits_enabled = group_abort is not None
            elif abort_logits_enabled != (group_abort is not None):
                raise RuntimeError("search policies disagree on target_abort_enabled; mixed search requires matching heads")
            if merged is None:
                merged = {
                    "halt_logits": torch.empty((len(states),) + tuple(out_group["halt_logits"].shape[1:]), device=self.device, dtype=out_group["halt_logits"].dtype),
                    "value": torch.empty((len(states),) + tuple(out_group["value"].shape[1:]), device=self.device, dtype=out_group["value"].dtype),
                    "pair_mask": torch.empty((len(states),) + tuple(out_group["pair_mask"].shape[1:]), device=self.device, dtype=out_group["pair_mask"].dtype),
                    "origin_frac_logits": torch.empty((len(states),) + tuple(out_group["origin_frac_logits"].shape[1:]), device=self.device, dtype=out_group["origin_frac_logits"].dtype),
                    "origin_frac_mask": torch.empty((len(states),) + tuple(out_group["origin_frac_mask"].shape[1:]), device=self.device, dtype=out_group["origin_frac_mask"].dtype),
                    "abort_logits": (
                        torch.empty((len(states),) + tuple(group_abort.shape[1:]), device=self.device, dtype=group_abort.dtype)
                        if group_abort is not None
                        else None
                    ),
                }
            row_idx_t = torch.tensor(row_indices, device=self.device, dtype=torch.long)
            merged["halt_logits"].index_copy_(0, row_idx_t, out_group["halt_logits"])
            merged["value"].index_copy_(0, row_idx_t, out_group["value"])
            merged["pair_mask"].index_copy_(0, row_idx_t, out_group["pair_mask"])
            merged["origin_frac_logits"].index_copy_(0, row_idx_t, out_group["origin_frac_logits"])
            merged["origin_frac_mask"].index_copy_(0, row_idx_t, out_group["origin_frac_mask"])
            if merged["abort_logits"] is not None and group_abort is not None:
                merged["abort_logits"].index_copy_(0, row_idx_t, group_abort)
            for local_idx, row_idx in enumerate(row_indices):
                planet_hidden_rows[row_idx] = out_group["planet_hidden"][local_idx]
        assert merged is not None
        if any(row is None for row in planet_hidden_rows):
            raise RuntimeError("mixed search policy output assembly left missing planet_hidden rows")
        merged["planet_hidden_rows"] = list(planet_hidden_rows)
        if merged.get("abort_logits") is None:
            merged.pop("abort_logits", None)
        return merged

    def _cached_policy_outputs_for_states(
        self,
        states: Sequence[OrbitWarsState],
        *,
        num_agents: int,
        search_root_player: int | None,
        search_env_step_from_root: int,
    ) -> list[CachedSearchPolicyOutputs]:
        if not states:
            return []
        batch_states: list[OrbitWarsState] = []
        batch_players: list[int] = []
        for state in states:
            for player in range(int(num_agents)):
                batch_states.append(state)
                batch_players.append(int(player))
        out = self._search_policy_outputs_for_states_batched_mixed(
            batch_states,
            batch_players,
            search_root_player=search_root_player,
            search_env_step_from_root=int(search_env_step_from_root),
        )
        cached: list[CachedSearchPolicyOutputs] = []
        stride = int(num_agents)
        abort_logits_all = out.get("abort_logits")
        planet_hidden_rows = out["planet_hidden_rows"]
        for state_idx in range(len(states)):
            start = state_idx * stride
            stop = start + stride
            cached.append(
                CachedSearchPolicyOutputs(
                    players=tuple(range(int(num_agents))),
                    halt_logits=out["halt_logits"][start:stop].detach().cpu().clone(),
                    value=out["value"][start:stop].detach().cpu().clone(),
                    pair_mask=out["pair_mask"][start:stop].detach().cpu().clone(),
                    origin_frac_logits=out["origin_frac_logits"][start:stop].detach().cpu().clone(),
                    origin_frac_mask=out["origin_frac_mask"][start:stop].detach().cpu().clone(),
                    planet_hidden=tuple(row.detach().cpu().clone() for row in planet_hidden_rows[start:stop]),
                    abort_logits=(
                        abort_logits_all[start:stop].detach().cpu().clone()
                        if abort_logits_all is not None
                        else None
                    ),
                )
            )
        return cached

    def _search_cache_match(
        self,
        runtime: SearchRuntime,
        *,
        ego_player: int,
    ) -> CachedSearchRollout | None:
        cache = self._search_rollout_cache
        if cache is None:
            return None
        if cache.game_key != str(runtime.game_key) or int(cache.ego_player) != int(ego_player):
            return None
        if runtime.public_state is None:
            return None
        if not _cache_state_match(cache.root_state, runtime.public_state):
            return None
        return cache

    def _evaluate_search_branches(
        self,
        runtime: SearchRuntime,
        *,
        ego_player: int,
        current_state: OrbitWarsState,
        current_micro_idx: int,
        action_prefix: list[list[float]],
        launch_action: list[float],
        launch_origin_slot: int,
        launch_send: int,
        launch_true_target_slot: int,
        launch_true_hit_tick: float,
        rollout_horizon: int,
        branch_mask: tuple[bool, bool] = (True, True),
        timing: ModelSearchTiming | None = None,
    ) -> tuple[list[float], list[list[CachedSearchTransition]], list[list[list[float]]]]:
        reward = runtime.settings.reward
        sim_max_micro_steps = int(self.max_micro_steps)
        base_public_state = runtime.public_state
        if base_public_state is None:
            base_public_state = observation_to_state(
                runtime.public_obs,
                runtime.kaggle_config,
                max_fleets=self.max_fleets,
                step_count_override=int(runtime.step_count),
                num_agents_override=int(runtime.num_agents),
                fleet_arrival_cache=runtime.fleet_arrival_cache,
            )

        launched_planets = np.array(np.asarray(current_state.planets), copy=True)
        launched_incoming = np.array(np.asarray(current_state.incoming_fleets), copy=True)
        launched_blocked = np.array(np.asarray(current_state.origin_frac_blocked), copy=True)
        apply_micro_launch_in_place(
            launched_planets,
            launched_incoming,
            ego_player=int(ego_player),
            origin_slot=int(launch_origin_slot),
            send=int(launch_send),
            true_target_slot=int(launch_true_target_slot),
            true_hit_tick=float(launch_true_hit_tick),
        )
        launched_state = current_state._replace(
            planets=launched_planets,
            incoming_fleets=launched_incoming,
            origin_frac_blocked=launched_blocked,
        )

        branch_scores = [0.0, 0.0]
        branch_traces: list[list[CachedSearchTransition]] = [[], []]
        branch_root_actions: list[list[list[float]]] = [[], []]
        discount = 1.0
        branch_public_obs = [copy.deepcopy(runtime.public_obs), copy.deepcopy(runtime.public_obs)]
        branch_states_pre = [base_public_state, base_public_state]
        branch_sim_states = [
            _make_sim_state(runtime.public_obs, num_agents=int(runtime.num_agents), step_count=int(runtime.step_count)),
            _make_sim_state(runtime.public_obs, num_agents=int(runtime.num_agents), step_count=int(runtime.step_count)),
        ]
        branch_steps = [int(runtime.step_count), int(runtime.step_count)]
        branch_done = [not bool(branch_mask[0]), not bool(branch_mask[1])]

        for depth in range(int(rollout_horizon)):
            active_count = int(sum(0 if done else 1 for done in branch_done))
            if active_count <= 0:
                break
            if timing is not None:
                timing.rollout_steps += active_count
                t_setup = perf_counter()

            joint_actions_by_branch: list[list[list[float]]] = [
                [[] for _ in range(int(runtime.num_agents))],
                [[] for _ in range(int(runtime.num_agents))],
            ]
            seat_plans: list[_BatchedSearchSeatPlan] = []
            t_setup_geom = perf_counter() if timing is not None else 0.0
            branch_launch_geometry = [
                _launch_geometry_from_obs(branch_public_obs[0], runtime.kaggle_config),
                _launch_geometry_from_obs(branch_public_obs[1], runtime.kaggle_config),
            ]
            if timing is not None:
                timing.branch_setup_launch_geometry_s += perf_counter() - t_setup_geom

            t_setup_plans = perf_counter() if timing is not None else 0.0
            if depth == 0:
                if not branch_done[0]:
                    joint_actions_by_branch[0][int(ego_player)] = copy.deepcopy(action_prefix)
                if not branch_done[1]:
                    joint_actions_by_branch[1][int(ego_player)] = copy.deepcopy(action_prefix) + [
                        copy.deepcopy(launch_action)
                    ]
                for branch_idx in range(2):
                    if branch_done[branch_idx]:
                        continue
                    for player in range(int(runtime.num_agents)):
                        if player == int(ego_player):
                            if branch_idx == 1:
                                seat_plans.append(
                                    _BatchedSearchSeatPlan(
                                        branch_idx=1,
                                        player=int(ego_player),
                                        state_template=current_state,
                                        planets=np.array(np.asarray(launched_state.planets), copy=True),
                                        incoming_fleets=np.array(np.asarray(launched_state.incoming_fleets), copy=True),
                                        origin_frac_blocked=np.array(
                                            np.asarray(launched_state.origin_frac_blocked),
                                            copy=True,
                                        ),
                                        actions=copy.deepcopy(joint_actions_by_branch[1][int(ego_player)]),
                                        micro_idx=int(current_micro_idx) + 1,
                                        max_micro_steps=int(sim_max_micro_steps),
                                    )
                                )
                            continue
                        seat_plans.append(
                            _BatchedSearchSeatPlan(
                                branch_idx=int(branch_idx),
                                player=int(player),
                                state_template=base_public_state,
                                planets=np.array(np.asarray(base_public_state.planets), copy=True),
                                incoming_fleets=np.array(np.asarray(base_public_state.incoming_fleets), copy=True),
                                origin_frac_blocked=np.zeros((MAX_PLANETS, len(FRACTIONS)), dtype=np.bool_),
                                actions=[],
                                micro_idx=0,
                                max_micro_steps=int(sim_max_micro_steps),
                            )
                        )
            else:
                for branch_idx in range(2):
                    if branch_done[branch_idx]:
                        continue
                    branch_state = branch_states_pre[branch_idx]
                    for player in range(int(runtime.num_agents)):
                        seat_plans.append(
                            _BatchedSearchSeatPlan(
                                branch_idx=int(branch_idx),
                                player=int(player),
                                state_template=branch_state,
                                planets=np.array(np.asarray(branch_state.planets), copy=True),
                                incoming_fleets=np.array(np.asarray(branch_state.incoming_fleets), copy=True),
                                origin_frac_blocked=np.zeros((MAX_PLANETS, len(FRACTIONS)), dtype=np.bool_),
                                actions=[],
                                micro_idx=0,
                                max_micro_steps=int(sim_max_micro_steps),
                            )
                        )
            if timing is not None:
                timing.branch_setup_seat_plans_s += perf_counter() - t_setup_plans

            if seat_plans:
                joint_actions_by_branch = self._plan_joint_actions_batched_single_policy(
                    seat_plans=seat_plans,
                    branch_joint_actions=joint_actions_by_branch,
                    branch_launch_geometry=branch_launch_geometry,
                    sim_step=int(branch_steps[0]),
                    ship_speed=float(_cfg_get(runtime.kaggle_config, "shipSpeed", 6.0)),
                    search_root_player=int(ego_player),
                    search_env_step_from_root=int(depth),
                    timing=timing,
                    search_greedy_launch_threshold=runtime.settings.greedy_launch_threshold,
                )
            if depth == 0:
                t_copy = perf_counter() if timing is not None else 0.0
                branch_root_actions = [copy.deepcopy(actions[int(ego_player)]) for actions in joint_actions_by_branch]
                if timing is not None:
                    dt = perf_counter() - t_copy
                    timing.branch_setup_root_actions_s += dt
                    timing.trace_copy_s += dt

            for branch_idx in range(2):
                if branch_done[branch_idx]:
                    continue
                t_reward = perf_counter() if timing is not None else 0.0
                ratios_pre = _reward_mix_ratios_np(branch_states_pre[branch_idx], reward)
                if timing is not None:
                    timing.reward_s += perf_counter() - t_reward
                if timing is not None:
                    t0 = perf_counter()
                _simulate_joint_step_with_kaggle_model(
                    branch_sim_states[branch_idx],
                    joint_actions=joint_actions_by_branch[branch_idx],
                    config=runtime.kaggle_config,
                )
                if timing is not None:
                    timing.kaggle_step_calls += 1
                    timing.kaggle_step_s += perf_counter() - t0
                branch_steps[branch_idx] += 1
                if timing is not None:
                    t0 = perf_counter()
                branch_public_obs[branch_idx] = _public_obs_from_sim_state(
                    branch_sim_states[branch_idx],
                    step_count=int(branch_steps[branch_idx]),
                )
                if timing is not None:
                    timing.public_obs_s += perf_counter() - t0
                if timing is not None:
                    t0 = perf_counter()
                state_post = observation_to_state(
                    branch_public_obs[branch_idx],
                    runtime.kaggle_config,
                    max_fleets=self.max_fleets,
                    step_count_override=int(branch_steps[branch_idx]),
                    num_agents_override=int(runtime.num_agents),
                    fleet_arrival_cache=runtime.fleet_arrival_cache,
                )
                if timing is not None:
                    timing.state_rebuild_calls += 1
                    timing.state_rebuild_s += perf_counter() - t0
                    t_reward = perf_counter()
                step_reward = _reward_delta_np(branch_states_pre[branch_idx], state_post, ratios_pre, reward)
                if timing is not None:
                    timing.reward_s += perf_counter() - t_reward
                branch_scores[branch_idx] += discount * float(step_reward[int(ego_player)])
                branch_states_pre[branch_idx] = state_post
                t_copy = perf_counter() if timing is not None else 0.0
                branch_traces[branch_idx].append(
                    CachedSearchTransition(
                        public_obs=copy.deepcopy(branch_public_obs[branch_idx]),
                        state=state_post,
                        step_count=int(branch_steps[branch_idx]),
                        step_reward=float(step_reward[int(ego_player)]),
                        done=bool(np.asarray(state_post.done)),
                        ego_actions=copy.deepcopy(joint_actions_by_branch[branch_idx][int(ego_player)]),
                    )
                )
                if timing is not None:
                    timing.trace_copy_s += perf_counter() - t_copy
                if bool(np.asarray(state_post.done)):
                    branch_done[branch_idx] = True

            discount *= float(reward.gamma)

        remaining_states = [branch_states_pre[idx] for idx in range(2) if not branch_done[idx]]
        remaining_players = [int(ego_player) for idx in range(2) if not branch_done[idx]]
        if remaining_states:
            if timing is not None:
                t0 = perf_counter()
            remaining_values = self._policy_values_for_states_batched(remaining_states, remaining_players)
            if timing is not None:
                timing.value_calls += len(remaining_values)
                timing.value_eval_calls += len(remaining_values)
                timing.value_s += perf_counter() - t0
                timing.value_eval_s += perf_counter() - t0
            value_i = 0
            for branch_idx in range(2):
                if branch_done[branch_idx]:
                    continue
                branch_scores[branch_idx] += discount * float(remaining_values[value_i])
                value_i += 1

        return branch_scores, branch_traces, branch_root_actions

    def _evaluate_greedy_continuation_from_state(
        self,
        runtime: SearchRuntime,
        *,
        ego_player: int,
        current_public_obs: Mapping[str, Any],
        current_state: OrbitWarsState,
        current_step: int,
        rollout_horizon: int,
        search_env_step_from_root_start: int = 0,
        current_policy_outputs: CachedSearchPolicyOutputs | None = None,
        timing: ModelSearchTiming | None = None,
    ) -> tuple[float, list[CachedSearchTransition]]:
        reward = runtime.settings.reward
        sim_max_micro_steps = int(self.max_micro_steps)
        total = 0.0
        discount = 1.0
        traces: list[CachedSearchTransition] = []
        branch_public_obs = copy.deepcopy(current_public_obs)
        branch_state_pre = current_state
        branch_sim_state = _make_sim_state(current_public_obs, num_agents=int(runtime.num_agents), step_count=int(current_step))
        branch_step = int(current_step)
        branch_done = bool(np.asarray(current_state.done))

        for _depth in range(int(rollout_horizon)):
            if branch_done:
                break
            if timing is not None:
                timing.rollout_steps += 1
                t_setup = perf_counter()
                t_setup_geom = perf_counter()
            else:
                t_setup_geom = 0.0
            branch_launch_geometry = [_launch_geometry_from_obs(branch_public_obs, runtime.kaggle_config)]
            if timing is not None:
                timing.branch_setup_launch_geometry_s += perf_counter() - t_setup_geom
                t_setup_plans = perf_counter()
            else:
                t_setup_plans = 0.0
            seat_plans = [
                _BatchedSearchSeatPlan(
                    branch_idx=0,
                    player=int(player),
                    state_template=branch_state_pre,
                    planets=np.array(np.asarray(branch_state_pre.planets), copy=True),
                    incoming_fleets=np.array(np.asarray(branch_state_pre.incoming_fleets), copy=True),
                    origin_frac_blocked=np.zeros((MAX_PLANETS, len(FRACTIONS)), dtype=np.bool_),
                    actions=[],
                    micro_idx=0,
                    max_micro_steps=int(sim_max_micro_steps),
                )
                for player in range(int(runtime.num_agents))
            ]
            if timing is not None:
                timing.branch_setup_seat_plans_s += perf_counter() - t_setup_plans
            joint_actions = self._plan_joint_actions_batched_single_policy(
                seat_plans=seat_plans,
                branch_joint_actions=[[[] for _ in range(int(runtime.num_agents))]],
                branch_launch_geometry=branch_launch_geometry,
                sim_step=int(branch_step),
                ship_speed=float(_cfg_get(runtime.kaggle_config, "shipSpeed", 6.0)),
                search_root_player=int(ego_player),
                search_env_step_from_root=int(search_env_step_from_root_start) + int(_depth),
                initial_policy_outputs=current_policy_outputs,
                timing=timing,
                search_greedy_launch_threshold=runtime.settings.greedy_launch_threshold,
            )[0]
            current_policy_outputs = None
            t_reward = perf_counter() if timing is not None else 0.0
            ratios_pre = _reward_mix_ratios_np(branch_state_pre, reward)
            if timing is not None:
                timing.reward_s += perf_counter() - t_reward
            if timing is not None:
                t0 = perf_counter()
            _simulate_joint_step_with_kaggle_model(
                branch_sim_state,
                joint_actions=joint_actions,
                config=runtime.kaggle_config,
            )
            if timing is not None:
                timing.kaggle_step_calls += 1
                timing.kaggle_step_s += perf_counter() - t0
            branch_step += 1
            if timing is not None:
                t0 = perf_counter()
            branch_public_obs = _public_obs_from_sim_state(branch_sim_state, step_count=int(branch_step))
            if timing is not None:
                timing.public_obs_s += perf_counter() - t0
            if timing is not None:
                t0 = perf_counter()
            state_post = observation_to_state(
                branch_public_obs,
                runtime.kaggle_config,
                max_fleets=self.max_fleets,
                step_count_override=int(branch_step),
                num_agents_override=int(runtime.num_agents),
                fleet_arrival_cache=runtime.fleet_arrival_cache,
            )
            if timing is not None:
                timing.state_rebuild_calls += 1
                timing.state_rebuild_s += perf_counter() - t0
                t_reward = perf_counter()
            step_reward = _reward_delta_np(branch_state_pre, state_post, ratios_pre, reward)
            if timing is not None:
                timing.reward_s += perf_counter() - t_reward
            total += discount * float(step_reward[int(ego_player)])
            t_copy = perf_counter() if timing is not None else 0.0
            traces.append(
                CachedSearchTransition(
                    public_obs=copy.deepcopy(branch_public_obs),
                    state=state_post,
                    step_count=int(branch_step),
                    step_reward=float(step_reward[int(ego_player)]),
                    done=bool(np.asarray(state_post.done)),
                    ego_actions=copy.deepcopy(joint_actions[int(ego_player)]),
                    policy_outputs=None,
                )
            )
            if timing is not None:
                timing.trace_copy_s += perf_counter() - t_copy
            branch_state_pre = state_post
            branch_done = bool(np.asarray(state_post.done))
            discount *= float(reward.gamma)

        if not branch_done:
            if timing is not None:
                t0 = perf_counter()
            total += discount * float(self._policy_values_for_states_batched([branch_state_pre], [int(ego_player)])[0])
            if timing is not None:
                timing.value_calls += 1
                timing.value_eval_calls += 1
                timing.value_s += perf_counter() - t0
                timing.value_eval_s += perf_counter() - t0

        return total, traces

    def _score_branch_from_cache(
        self,
        runtime: SearchRuntime,
        *,
        ego_player: int,
        cache: CachedSearchRollout,
        rollout_horizon: int,
        timing: ModelSearchTiming | None = None,
    ) -> tuple[float, list[CachedSearchTransition]]:
        reward = runtime.settings.reward
        total = 0.0
        discount = 1.0
        steps_used = 0
        end_obs = cache.root_public_obs
        end_state = cache.root_state
        end_step = int(cache.root_step_count)
        end_policy_outputs = (
            cache.root_policy_outputs
            if self._search_cached_policy_outputs_match_new_root_step(0)
            else None
        )
        used_transitions: list[CachedSearchTransition] = []

        for trans in cache.transitions[: max(0, int(rollout_horizon))]:
            if timing is not None:
                timing.rollout_steps += 1
            total += discount * float(trans.step_reward)
            steps_used += 1
            t_copy = perf_counter() if timing is not None else 0.0
            used_transitions.append(
                CachedSearchTransition(
                    public_obs=copy.deepcopy(trans.public_obs),
                    state=trans.state,
                    step_count=int(trans.step_count),
                    step_reward=float(trans.step_reward),
                    done=bool(trans.done),
                    ego_actions=copy.deepcopy(trans.ego_actions) if trans.ego_actions is not None else None,
                    policy_outputs=(
                        trans.policy_outputs
                        if self._search_cached_policy_outputs_match_new_root_step(int(steps_used))
                        else None
                    ),
                )
            )
            if timing is not None:
                timing.trace_copy_s += perf_counter() - t_copy
            end_obs = trans.public_obs
            end_state = trans.state
            end_step = int(trans.step_count)
            end_policy_outputs = (
                trans.policy_outputs
                if self._search_cached_policy_outputs_match_new_root_step(int(steps_used))
                else None
            )
            if trans.done:
                if timing is not None:
                    timing.branch_rollouts += 1
                return total, used_transitions
            discount *= float(reward.gamma)

        remaining = int(rollout_horizon) - int(steps_used)
        if remaining > 0:
            tail_total, tail_transitions = self._evaluate_greedy_continuation_from_state(
                runtime,
                ego_player=int(ego_player),
                current_public_obs=end_obs,
                current_state=end_state,
                current_step=int(end_step),
                rollout_horizon=int(remaining),
                search_env_step_from_root_start=int(steps_used),
                current_policy_outputs=end_policy_outputs,
                timing=timing,
            )
            total += discount * float(tail_total)
            used_transitions.extend(tail_transitions)
            return total, used_transitions

        if not bool(np.asarray(end_state.done)):
            if timing is not None:
                t0 = perf_counter()
            total += discount * float(
                self._policy_values_for_states_batched([end_state], [int(ego_player)])[0]
            )
            if timing is not None:
                timing.value_calls += 1
                timing.value_eval_calls += 1
                timing.value_s += perf_counter() - t0
                timing.value_eval_s += perf_counter() - t0

        return total, used_transitions

    def _identify_cached_branch(
        self,
        *,
        action_prefix: list[list[float]],
        launch_action: list[float],
        cache: CachedSearchRollout,
    ) -> str | None:
        if cache.root_ego_actions is None:
            return None
        if _action_sequence_match(cache.root_ego_actions, action_prefix):
            return "halt"
        launch_actions = copy.deepcopy(action_prefix) + [copy.deepcopy(launch_action)]
        if len(cache.root_ego_actions) >= len(launch_actions) and _action_sequence_match(
            cache.root_ego_actions[: len(launch_actions)],
            launch_actions,
        ):
            return "launch"
        return None

    def _store_search_rollout_cache(
        self,
        runtime: SearchRuntime,
        *,
        ego_player: int,
        chose_launch: bool,
        branch_transitions: list[list[CachedSearchTransition]],
        branch_root_ego_actions: list[list[list[float]]],
    ) -> None:
        chosen_idx = 1 if chose_launch else 0
        self._store_search_rollout_cache_from_transitions(
            runtime,
            ego_player=int(ego_player),
            root_ego_actions=branch_root_ego_actions[chosen_idx],
            transitions=branch_transitions[chosen_idx],
        )

    def _store_search_rollout_cache_from_transitions(
        self,
        runtime: SearchRuntime,
        *,
        ego_player: int,
        root_ego_actions: list[list[float]],
        transitions: list[CachedSearchTransition],
    ) -> None:
        chosen = transitions
        if not chosen:
            self._search_rollout_cache = None
            return
        root = chosen[0]
        next_root_actions = (
            copy.deepcopy(chosen[1].ego_actions)
            if len(chosen) > 1 and chosen[1].ego_actions is not None
            else None
        )
        cached_policy_outputs = self._cached_policy_outputs_for_states(
            [trans.state for trans in chosen],
            num_agents=int(runtime.num_agents),
            search_root_player=int(ego_player),
            search_env_step_from_root=1,
        )
        self._search_rollout_cache = CachedSearchRollout(
            game_key=str(runtime.game_key),
            ego_player=int(ego_player),
            root_ego_actions=next_root_actions,
            root_public_obs=copy.deepcopy(root.public_obs),
            root_state=root.state,
            root_step_count=int(root.step_count),
            root_policy_outputs=(cached_policy_outputs[0] if cached_policy_outputs else None),
            transitions=[
                CachedSearchTransition(
                    public_obs=copy.deepcopy(trans.public_obs),
                    state=trans.state,
                    step_count=int(trans.step_count),
                    step_reward=float(trans.step_reward),
                    done=bool(trans.done),
                    ego_actions=copy.deepcopy(trans.ego_actions) if trans.ego_actions is not None else None,
                    policy_outputs=cached_policy_outputs[idx + 1] if (idx + 1) < len(cached_policy_outputs) else None,
                )
                for idx, trans in enumerate(chosen[1:])
            ],
        )

    def _plan_joint_actions_batched_single_policy(
        self,
        *,
        seat_plans: list[_BatchedSearchSeatPlan],
        branch_joint_actions: list[list[list[float]]],
        branch_launch_geometry: list[LaunchGeometryInputs],
        sim_step: int,
        ship_speed: float,
        search_root_player: int | None = None,
        search_env_step_from_root: int = 0,
        initial_policy_outputs: CachedSearchPolicyOutputs | None = None,
        timing: ModelSearchTiming | None = None,
        search_greedy_launch_threshold: float | None = None,
        sampling_mode: str = SAMPLING_MODE_GREEDY,
    ) -> list[list[list[float]]]:
        t_plan = perf_counter() if timing is not None else 0.0
        active = [plan for plan in seat_plans if int(plan.micro_idx) < int(plan.max_micro_steps)]
        used_initial_policy_outputs = False
        while active:
            population_idx = None
            if timing is not None:
                timing.batch_plan_rounds += 1
            use_initial_policy_outputs = (
                initial_policy_outputs is not None
                and not used_initial_policy_outputs
                and all(int(plan.branch_idx) == 0 and int(plan.micro_idx) == 0 for plan in active)
                and tuple(int(plan.player) for plan in active) == initial_policy_outputs.players
            )
            if use_initial_policy_outputs and not self._cached_policy_outputs_match_active_policy_selection(
                initial_policy_outputs,
                active,
                search_root_player=search_root_player,
                search_env_step_from_root=int(search_env_step_from_root),
            ):
                use_initial_policy_outputs = False
            if use_initial_policy_outputs:
                virt_states = [
                    plan.state_template._replace(
                        planets=plan.planets,
                        incoming_fleets=plan.incoming_fleets,
                        origin_frac_blocked=plan.origin_frac_blocked,
                    )
                    for plan in active
                ]
                out = {
                    "halt_logits": initial_policy_outputs.halt_logits.to(device=self.device),
                    "value": initial_policy_outputs.value.to(device=self.device),
                    "pair_mask": initial_policy_outputs.pair_mask.to(device=self.device),
                    "origin_frac_logits": initial_policy_outputs.origin_frac_logits.to(device=self.device),
                    "origin_frac_mask": initial_policy_outputs.origin_frac_mask.to(device=self.device),
                    "planet_hidden_rows": [
                        row.to(device=self.device)
                        for row in (
                            initial_policy_outputs.planet_hidden
                            if isinstance(initial_policy_outputs.planet_hidden, tuple)
                            else tuple(initial_policy_outputs.planet_hidden[row] for row in range(int(initial_policy_outputs.halt_logits.shape[0])))
                        )
                    ],
                }
                if initial_policy_outputs.abort_logits is not None:
                    out["abort_logits"] = initial_policy_outputs.abort_logits.to(device=self.device)
                used_initial_policy_outputs = True
                t0 = perf_counter() if timing is not None else 0.0
            else:
                if timing is not None:
                    t_obs = perf_counter()
                virt_states = [
                    plan.state_template._replace(
                        planets=plan.planets,
                        incoming_fleets=plan.incoming_fleets,
                        origin_frac_blocked=plan.origin_frac_blocked,
                    )
                    for plan in active
                ]
                if timing is not None:
                    timing.batch_obs.virt_states_s += perf_counter() - t_obs
                players_active = [int(plan.player) for plan in active]
                if timing is not None:
                    t0 = perf_counter()
                out = self._search_policy_outputs_for_states_batched_mixed(
                    virt_states,
                    players_active,
                    search_root_player=search_root_player,
                    search_env_step_from_root=int(search_env_step_from_root),
                )
                if timing is not None:
                    timing.batch_policy_forward_s += perf_counter() - t0
                    t0 = perf_counter()

            continue_rows: list[int] = []
            continue_plans: list[_BatchedSearchSeatPlan] = []
            continue_states: list[OrbitWarsState] = []
            origin_slots: list[int] = []
            frac_slots: list[int] = []
            sends: list[int] = []
            search_targets: list[SearchFirstContactTargets] = []
            ray_angles: list[np.ndarray] = []
            ray_valids: list[np.ndarray] = []
            ray_hit_ticks: list[np.ndarray] = []
            true_planets: list[np.ndarray | None] = []
            true_hit_ticks: list[np.ndarray | None] = []
            t_raycast = 0.0

            for row, plan in enumerate(active):
                halt_logits = out["halt_logits"][row]
                if sampling_mode == SAMPLING_MODE_GREEDY:
                    halt_action = _greedy_halt_action_from_logits(
                        halt_logits,
                        launch_threshold=search_greedy_launch_threshold,
                    )
                else:
                    halt_probs = torch.softmax(halt_logits, dim=-1)
                    halt_action = int(torch.multinomial(halt_probs, 1, generator=self.rng).item())
                if halt_action == 1:
                    continue

                flat_mask = out["origin_frac_mask"].flatten(start_dim=1)[row]
                if not bool(flat_mask.any().item()):
                    continue
                flat_logits = out["origin_frac_logits"].flatten(start_dim=1)[row]
                masked_origin_frac = flat_logits.masked_fill(~flat_mask, -1e4)
                if sampling_mode == SAMPLING_MODE_GREEDY:
                    origin_frac_flat = int(torch.argmax(masked_origin_frac).item())
                else:
                    origin_frac_probs = torch.softmax(masked_origin_frac, dim=-1)
                    origin_frac_flat = int(
                        torch.multinomial(origin_frac_probs, 1, generator=self.rng).item()
                    )
                o_idx = origin_frac_flat // len(FRACTIONS)
                frac_idx = origin_frac_flat % len(FRACTIONS)
                ships_avail = float(plan.planets[o_idx, 5])
                send = _planned_send(ships_avail, int(frac_idx))
                if send <= 0:
                    continue

                virt = virt_states[row]
                t_ray0 = perf_counter() if timing is not None else 0.0
                if self.target_method == "interval":
                    coarse = _search_first_contact_targets_np(
                        virt,
                        int(o_idx),
                        int(frac_idx),
                        ship_speed=float(ship_speed),
                        horizon=INCOMING_TA_BINS,
                        launch_geometry=branch_launch_geometry[int(plan.branch_idx)],
                    )
                    ray_angle = coarse.angles
                    ray_valid = coarse.valid
                    ray_hit_tick = coarse.eta
                    true_planet = None
                    true_hit_tick = None
                else:
                    coarse = None
                    ray_angle, ray_valid, ray_hit_tick, true_planet, true_hit_tick, _ = _first_hit_targets_np(
                        virt,
                        int(o_idx),
                        int(frac_idx),
                        ship_speed=float(ship_speed),
                        horizon=INCOMING_TA_BINS,
                        n_rays=self.raycast_rays,
                        samples_per_span=self.interval_samples_per_span,
                        target_method=self.target_method,
                        target_timing=None,
                        game_step=int(sim_step),
                        micro_idx=int(plan.micro_idx),
                        ego_player=int(plan.player),
                        launch_geometry=branch_launch_geometry[int(plan.branch_idx)],
                        refine_boundaries=False,
                        phase="search",
                        return_jobs=True,
                    )
                if timing is not None:
                    t_raycast += perf_counter() - t_ray0
                continue_rows.append(int(row))
                continue_plans.append(plan)
                continue_states.append(virt)
                origin_slots.append(int(o_idx))
                frac_slots.append(int(frac_idx))
                sends.append(int(send))
                search_targets.append(coarse)
                ray_angles.append(np.asarray(ray_angle, dtype=np.float32))
                ray_valids.append(np.asarray(ray_valid, dtype=np.bool_))
                ray_hit_ticks.append(np.asarray(ray_hit_tick, dtype=np.float32))
                true_planets.append(None if true_planet is None else np.asarray(true_planet, dtype=np.int32))
                true_hit_ticks.append(None if true_hit_tick is None else np.asarray(true_hit_tick, dtype=np.float32))
            if timing is not None:
                loop_dt = perf_counter() - t0
                timing.batch_raycast_s += t_raycast
                timing.batch_post_forward_s += max(0.0, loop_dt - t_raycast)

            if not continue_plans:
                break

            if timing is not None:
                t0 = perf_counter()
            row_idx_t = torch.tensor(continue_rows, device=self.device, dtype=torch.long)
            origin_idx_t = torch.tensor(origin_slots, device=self.device, dtype=torch.long)
            frac_idx_t = torch.tensor(frac_slots, device=self.device, dtype=torch.long)
            fleet_size_t = torch.tensor(sends, device=self.device, dtype=torch.float32)
            target_eta_t = torch.from_numpy(np.stack(ray_hit_ticks, axis=0)).to(device=self.device, dtype=torch.float32)
            target_ships_t = torch.from_numpy(
                np.stack([np.asarray(plan.planets[:, 5], dtype=np.float32) for plan in continue_plans], axis=0)
            ).to(device=self.device, dtype=torch.float32)
            target_logits_b = torch.empty((len(continue_plans), MAX_PLANETS), device=self.device, dtype=torch.float32)
            target_grouped: dict[bool, list[int]] = {True: [], False: []}
            for idx, plan in enumerate(continue_plans):
                target_grouped[
                    self._search_uses_main_policy_for_player(
                        int(plan.player),
                        search_root_player,
                        int(search_env_step_from_root),
                    )
                ].append(idx)
            for use_main, local_indices in target_grouped.items():
                if not local_indices:
                    continue
                sample_player = int(continue_plans[local_indices[0]].player)
                policy, _ppc, _norm, population_member_for_player = self._search_policy_context_for_player(
                    sample_player,
                    search_root_player,
                    int(search_env_step_from_root),
                )
                hidden = torch.stack([out["planet_hidden_rows"][continue_rows[idx]] for idx in local_indices], dim=0)
                pop_members = [population_member_for_player(int(continue_plans[idx].player)) for idx in local_indices]
                target_pop_idx = None
                if any(member is not None for member in pop_members):
                    target_pop_idx = torch.tensor(
                        [0 if member is None else int(member) for member in pop_members],
                        device=self.device,
                        dtype=torch.long,
                    )
                group_idx_t = torch.tensor(local_indices, device=self.device, dtype=torch.long)
                obs_group = _obs_tensors_for_states(
                    [continue_states[idx] for idx in local_indices],
                    [int(continue_plans[idx].player) for idx in local_indices],
                    self.device,
                    policy_player_count=_ppc,
                    target_abort_enabled=bool(getattr(policy, "target_abort_enabled", False)),
                    normalize_obs_to_p0=_norm,
                )
                target_logits_group = policy.target_logits_for_origin_fraction(
                    hidden,
                    obs_group["owner_idx"],
                    obs_group["features"],
                    origin_idx_t.index_select(0, group_idx_t),
                    frac_idx_t.index_select(0, group_idx_t),
                    fleet_size=fleet_size_t.index_select(0, group_idx_t),
                    target_eta=target_eta_t.index_select(0, group_idx_t),
                    target_ships=target_ships_t.index_select(0, group_idx_t),
                    population_idx=target_pop_idx,
                )
                target_logits_b.index_copy_(0, group_idx_t, target_logits_group.to(dtype=target_logits_b.dtype))
            abort_logits_all = out.get("abort_logits")
            if timing is not None:
                timing.batch_target_head_s += perf_counter() - t0

            if timing is not None:
                t0 = perf_counter()
            next_active: list[_BatchedSearchSeatPlan] = []
            for idx, plan in enumerate(continue_plans):
                o_idx = int(origin_slots[idx])
                frac_idx = int(frac_slots[idx])
                target_mask = out["pair_mask"][continue_rows[idx], o_idx].clone()
                ray_valid_t = torch.from_numpy(ray_valids[idx]).to(device=self.device, dtype=torch.bool)
                target_mask &= ray_valid_t
                abort_logit = None
                if abort_logits_all is not None:
                    abort_logit = abort_logits_all[continue_rows[idx], o_idx, frac_idx].reshape(1)
                if abort_logit is not None:
                    combined_target = torch.cat(
                        [target_logits_b[idx].masked_fill(~target_mask, -1e4), abort_logit.reshape(1)],
                        dim=0,
                    )
                    if sampling_mode == SAMPLING_MODE_GREEDY:
                        target_choice = int(torch.argmax(combined_target).item())
                    else:
                        target_probs = torch.softmax(combined_target, dim=-1)
                        target_choice = int(
                            torch.multinomial(target_probs, 1, generator=self.rng).item()
                        )
                    if target_choice == MAX_PLANETS:
                        plan.origin_frac_blocked[o_idx, frac_idx] = True
                        plan.micro_idx += 1
                        if int(plan.micro_idx) < int(plan.max_micro_steps):
                            next_active.append(plan)
                        continue
                    sorted_choices = [int(target_choice)]
                    remaining = torch.argsort(
                        target_logits_b[idx].masked_fill(~target_mask, -1e4),
                        descending=True,
                    ).tolist()
                    sorted_choices.extend(
                        int(choice) for choice in remaining if int(choice) != int(target_choice)
                    )
                else:
                    if not bool(target_mask.any().item()):
                        continue
                    masked_target = target_logits_b[idx].masked_fill(~target_mask, -1e4)
                    if sampling_mode == SAMPLING_MODE_GREEDY:
                        sorted_choices = torch.argsort(masked_target, descending=True).tolist()
                    else:
                        sampled_choice = int(
                            torch.multinomial(torch.softmax(masked_target, dim=-1), 1, generator=self.rng).item()
                        )
                        sorted_choices = [sampled_choice]
                        sorted_choices.extend(
                            int(choice)
                            for choice in torch.argsort(masked_target, descending=True).tolist()
                            if int(choice) != sampled_choice
                        )

                d_idx = -1
                true_target_slot = -1
                true_hit_tick = -1.0
                if search_targets[idx] is not None:
                    coarse = search_targets[idx]
                    t_confirm0 = perf_counter() if timing is not None else 0.0
                    for choice in sorted_choices:
                        choice = int(choice)
                        if choice < 0 or choice >= MAX_PLANETS:
                            continue
                        if not bool(target_mask[choice].item()):
                            continue
                        kind, code, tick = _discrete_first_hit_at_angle_np(
                            float(ray_angles[idx][choice]),
                            coarse.origin_xy,
                            float(coarse.origin_radius),
                            float(coarse.speed),
                            coarse.p0_by_tick,
                            coarse.p1_by_tick,
                            coarse.radii,
                            coarse.active_by_tick,
                            coarse.collision_rank,
                            horizon=INCOMING_TA_BINS,
                        )
                        if kind == "planet" and int(code) == choice:
                            d_idx = int(choice)
                            true_target_slot = int(code)
                            true_hit_tick = float(tick)
                            break
                    if timing is not None:
                        timing.batch_raycast_s += perf_counter() - t_confirm0
                else:
                    d_idx = int(sorted_choices[0]) if sorted_choices else -1
                    true_target_slot = (
                        int(true_planets[idx][d_idx])
                        if d_idx >= 0 and true_planets[idx] is not None
                        else int(d_idx)
                    )
                    true_hit_tick = (
                        float(true_hit_ticks[idx][d_idx])
                        if d_idx >= 0 and true_hit_ticks[idx] is not None
                        else (float(ray_hit_ticks[idx][d_idx]) if d_idx >= 0 else -1.0)
                    )

                if d_idx < 0:
                    continue

                if _search_opponent_should_halt_on_neutral_underlaunch(
                    plan.planets,
                    plan.incoming_fleets,
                    player=int(plan.player),
                    search_root_player=search_root_player,
                    send=int(sends[idx]),
                    true_target_slot=int(true_target_slot),
                ):
                    continue

                action = [float(plan.planets[o_idx, 0]), float(ray_angles[idx][d_idx]), int(sends[idx])]
                plan.actions.append(action)
                branch_joint_actions[int(plan.branch_idx)][int(plan.player)] = copy.deepcopy(plan.actions)
                apply_micro_launch_in_place(
                    plan.planets,
                    plan.incoming_fleets,
                    ego_player=int(plan.player),
                    origin_slot=int(o_idx),
                    send=int(sends[idx]),
                    true_target_slot=int(true_target_slot),
                    true_hit_tick=float(true_hit_tick),
                )
                plan.micro_idx += 1
                if int(plan.micro_idx) < int(plan.max_micro_steps):
                    next_active.append(plan)
            if timing is not None:
                timing.batch_apply_s += perf_counter() - t0

            active = next_active
        if timing is not None:
            timing.batch_plan_calls += 1
            timing.batch_plan_s += perf_counter() - t_plan

        return branch_joint_actions

    def _search_geometry_override_for_tree_node(self, node: _SearchTreeNode) -> str | None:
        if int(node.search_env_step_from_root) <= 0:
            return None
        return "sampled"

    def _search_expand_tree_nodes_batched(
        self,
        runtime: SearchRuntime,
        *,
        ego_player: int,
        nodes: list[_SearchTreeNode],
        timing: ModelSearchTiming | None = None,
        stop_s: float | None = None,
    ) -> list[list[_SearchTreeNode]]:
        if not nodes:
            return []
        if any(bool(node.turn_closed) or bool(node.done) for node in nodes):
            raise ValueError("_search_expand_tree_nodes_batched expects only open nodes")
        step_from_root = int(nodes[0].search_env_step_from_root)
        if any(int(node.search_env_step_from_root) != step_from_root for node in nodes):
            raise ValueError("_search_expand_tree_nodes_batched requires a single search depth")
        micro_idx = int(nodes[0].current_micro_idx)
        if any(int(node.current_micro_idx) != micro_idx for node in nodes):
            raise ValueError("_search_expand_tree_nodes_batched requires a single micro depth")
        if timing is not None:
            timing.bfs_expand_batches += 1
            timing.bfs_expand_nodes += len(nodes)

        children_by_node: list[list[_SearchTreeNode]] = [[] for _ in nodes]
        branch_enabled = _search_branching_enabled_for_env_step(
            runtime.settings,
            search_env_step_from_root=step_from_root,
            current_micro_idx=int(nodes[0].current_micro_idx),
        )
        out = self._search_policy_outputs_for_states_batched_mixed(
            [node.current_state for node in nodes],
            [int(ego_player)] * len(nodes),
            search_root_player=int(ego_player),
            search_env_step_from_root=step_from_root,
        )
        abort_logits_all = out.get("abort_logits")

        candidate_rows: list[int] = []
        candidate_states: list[OrbitWarsState] = []
        candidate_origin_slots: list[int] = []
        candidate_frac_slots: list[int] = []
        candidate_sends: list[int] = []
        candidate_ray_angles: list[np.ndarray] = []
        candidate_ray_valids: list[np.ndarray] = []
        candidate_ray_hit_ticks: list[np.ndarray] = []
        candidate_true_planets: list[np.ndarray | None] = []
        candidate_true_hit_ticks: list[np.ndarray | None] = []
        candidate_coarse: list[SearchFirstContactTargets | None] = []

        for row, node in enumerate(nodes):
            if stop_s is not None and perf_counter() >= stop_s:
                break
            halt_logits = out["halt_logits"][row]
            halt_probs = torch.softmax(halt_logits, dim=-1)
            if branch_enabled:
                halt_choices = _search_branch_indices_from_probs(
                    halt_probs,
                    threshold=float(runtime.settings.branch_probability_threshold),
                    max_branching=2,
                )
            else:
                halt_choices = [int(torch.argmax(halt_probs).item())]
            if 1 in halt_choices:
                root_actions = (
                    copy.deepcopy(node.current_turn_actions)
                    if int(node.search_env_step_from_root) == 0
                    else copy.deepcopy(node.root_turn_actions)
                )
                children_by_node[row].append(
                    _SearchTreeNode(
                        env_public_obs=node.env_public_obs,
                        env_step_start_state=node.env_step_start_state,
                        current_state=node.current_state,
                        step_count=int(node.step_count),
                        search_env_step_from_root=int(node.search_env_step_from_root),
                        current_turn_actions=copy.deepcopy(node.current_turn_actions),
                        current_micro_idx=int(node.current_micro_idx),
                        turn_closed=True,
                        root_turn_actions=root_actions,
                        root_turn_complete=bool(node.root_turn_complete or int(node.search_env_step_from_root) == 0),
                        discounted_reward=float(node.discounted_reward),
                        discount=float(node.discount),
                        done=bool(node.done),
                    )
                )

            flat_mask = out["origin_frac_mask"].flatten(start_dim=1)[row]
            if 0 not in halt_choices or not bool(flat_mask.any().item()):
                continue

            flat_logits = out["origin_frac_logits"].flatten(start_dim=1)[row]
            masked_origin_frac = flat_logits.masked_fill(~flat_mask, -1e4)
            origin_frac_choices = [int(torch.argmax(masked_origin_frac).item())]
            geometry_override = self._search_geometry_override_for_tree_node(node)
            launch_geometry = _launch_geometry_from_obs(node.env_public_obs, runtime.kaggle_config)
            planets_now = np.asarray(node.current_state.planets)

            for origin_frac_flat in origin_frac_choices:
                if stop_s is not None and perf_counter() >= stop_s:
                    break
                o_idx = int(origin_frac_flat) // len(FRACTIONS)
                frac_idx = int(origin_frac_flat) % len(FRACTIONS)
                send = _planned_send(float(planets_now[o_idx, 5]), int(frac_idx))
                if send <= 0:
                    continue
                if self.target_method == "interval":
                    coarse = _search_first_contact_targets_np(
                        node.current_state,
                        int(o_idx),
                        int(frac_idx),
                        ship_speed=float(_cfg_get(runtime.kaggle_config, "shipSpeed", 6.0)),
                        horizon=INCOMING_TA_BINS,
                        launch_geometry=launch_geometry,
                        geometry_override=geometry_override,
                    )
                    ray_angle = coarse.angles
                    ray_valid = coarse.valid
                    ray_hit_tick = coarse.eta
                    true_planet = None
                    true_hit_tick = None
                else:
                    ray_angle, ray_valid, ray_hit_tick, true_planet, true_hit_tick, _ = _first_hit_targets_np(
                        node.current_state,
                        int(o_idx),
                        int(frac_idx),
                        ship_speed=float(_cfg_get(runtime.kaggle_config, "shipSpeed", 6.0)),
                        horizon=INCOMING_TA_BINS,
                        n_rays=self.raycast_rays,
                        samples_per_span=self.interval_samples_per_span,
                        target_method=self.target_method,
                        target_timing=None,
                        game_step=int(node.step_count),
                        micro_idx=int(node.current_micro_idx),
                        ego_player=int(ego_player),
                        launch_geometry=launch_geometry,
                        geometry_override=geometry_override,
                        refine_boundaries=False,
                        phase="search",
                        return_jobs=True,
                    )
                    coarse = None
                candidate_rows.append(int(row))
                candidate_states.append(node.current_state)
                candidate_origin_slots.append(int(o_idx))
                candidate_frac_slots.append(int(frac_idx))
                candidate_sends.append(int(send))
                candidate_ray_angles.append(np.asarray(ray_angle, dtype=np.float32))
                candidate_ray_valids.append(np.asarray(ray_valid, dtype=np.bool_))
                candidate_ray_hit_ticks.append(np.asarray(ray_hit_tick, dtype=np.float32))
                candidate_true_planets.append(None if true_planet is None else np.asarray(true_planet, dtype=np.int32))
                candidate_true_hit_ticks.append(None if true_hit_tick is None else np.asarray(true_hit_tick, dtype=np.float32))
                candidate_coarse.append(coarse)

        if timing is not None:
            timing.bfs_expand_candidates += len(candidate_rows)
        if not candidate_rows:
            return children_by_node

        origin_idx_t = torch.tensor(candidate_origin_slots, device=self.device, dtype=torch.long)
        frac_idx_t = torch.tensor(candidate_frac_slots, device=self.device, dtype=torch.long)
        fleet_size_t = torch.tensor(candidate_sends, device=self.device, dtype=torch.float32)
        target_eta_t = torch.from_numpy(np.stack(candidate_ray_hit_ticks, axis=0)).to(device=self.device, dtype=torch.float32)
        target_ships_t = torch.from_numpy(
            np.stack([np.asarray(state.planets[:, 5], dtype=np.float32) for state in candidate_states], axis=0)
        ).to(device=self.device, dtype=torch.float32)
        hidden_t = torch.stack([out["planet_hidden_rows"][row] for row in candidate_rows], dim=0)
        policy_ctx, pcount, normalize_obs_to_p0, population_member_for_player = self._search_policy_context_for_player(
            int(ego_player),
            int(ego_player),
            step_from_root,
        )
        population_members = [population_member_for_player(int(ego_player)) for _ in candidate_rows]
        population_idx = None
        if any(member is not None for member in population_members):
            population_idx = torch.tensor(
                [0 if member is None else int(member) for member in population_members],
                device=self.device,
                dtype=torch.long,
            )
        obs_group = _obs_tensors_for_states(
            candidate_states,
            [int(ego_player)] * len(candidate_states),
            self.device,
            policy_player_count=pcount,
            target_abort_enabled=bool(getattr(policy_ctx, "target_abort_enabled", False)),
            normalize_obs_to_p0=normalize_obs_to_p0,
        )
        target_logits_b = policy_ctx.target_logits_for_origin_fraction(
            hidden_t,
            obs_group["owner_idx"],
            obs_group["features"],
            origin_idx_t,
            frac_idx_t,
            fleet_size=fleet_size_t,
            target_eta=target_eta_t,
            target_ships=target_ships_t,
            population_idx=population_idx,
        )

        for idx, row in enumerate(candidate_rows):
            if stop_s is not None and perf_counter() >= stop_s:
                break
            node = nodes[row]
            o_idx = int(candidate_origin_slots[idx])
            frac_idx = int(candidate_frac_slots[idx])
            target_mask = out["pair_mask"][row, o_idx].clone()
            target_mask &= torch.from_numpy(candidate_ray_valids[idx]).to(device=self.device, dtype=torch.bool)
            if not bool(target_mask.any().item()) and abort_logits_all is None:
                continue
            abort_logit = None
            if abort_logits_all is not None:
                abort_logit = abort_logits_all[row, o_idx, frac_idx].reshape(1)
                combined_logits = torch.cat(
                    [target_logits_b[idx].masked_fill(~target_mask, -1e4), abort_logit.reshape(1)],
                    dim=0,
                )
                combined_mask = torch.cat(
                    [target_mask, torch.ones((1,), device=self.device, dtype=torch.bool)],
                    dim=0,
                )
                if branch_enabled:
                    target_probs = torch.softmax(combined_logits, dim=-1)
                    target_choices = _search_branch_indices_from_probs(
                        target_probs,
                        mask=combined_mask,
                        threshold=float(runtime.settings.branch_probability_threshold),
                        max_branching=int(runtime.settings.max_branching_factor),
                    )
                else:
                    target_choices = [int(torch.argmax(combined_logits.masked_fill(~combined_mask, -1e4)).item())]
            else:
                masked_target = target_logits_b[idx].masked_fill(~target_mask, -1e4)
                if branch_enabled:
                    target_probs = torch.softmax(masked_target, dim=-1)
                    target_choices = _search_branch_indices_from_probs(
                        target_probs,
                        mask=target_mask,
                        threshold=float(runtime.settings.branch_probability_threshold),
                        max_branching=int(runtime.settings.max_branching_factor),
                    )
                else:
                    target_choices = [int(torch.argmax(masked_target).item())]

            for target_choice in target_choices:
                if stop_s is not None and perf_counter() >= stop_s:
                    break
                if int(target_choice) == MAX_PLANETS:
                    next_blocked = np.array(np.asarray(node.current_state.origin_frac_blocked), copy=True)
                    next_blocked[o_idx, frac_idx] = True
                    next_state = node.current_state._replace(origin_frac_blocked=next_blocked)
                    next_actions = copy.deepcopy(node.current_turn_actions)
                    children_by_node[row].append(
                        _SearchTreeNode(
                            env_public_obs=node.env_public_obs,
                            env_step_start_state=node.env_step_start_state,
                            current_state=next_state,
                            step_count=int(node.step_count),
                            search_env_step_from_root=int(node.search_env_step_from_root),
                            current_turn_actions=next_actions,
                            current_micro_idx=int(node.current_micro_idx) + 1,
                            turn_closed=int(node.current_micro_idx) + 1 >= int(self.max_micro_steps),
                            root_turn_actions=(
                                copy.deepcopy(next_actions)
                                if int(node.search_env_step_from_root) == 0
                                else copy.deepcopy(node.root_turn_actions)
                            ),
                            root_turn_complete=bool(
                                node.root_turn_complete
                                or (
                                    int(node.search_env_step_from_root) == 0
                                    and int(node.current_micro_idx) + 1 >= int(self.max_micro_steps)
                                )
                            ),
                            discounted_reward=float(node.discounted_reward),
                            discount=float(node.discount),
                            done=bool(node.done),
                        )
                    )
                    if timing is not None:
                        timing.bfs_expand_children += 1
                    continue

                d_idx = int(target_choice)
                true_target_slot = -1
                env_hit_tick = -1.0
                coarse = candidate_coarse[idx]
                if coarse is not None:
                    kind, code, tick = _discrete_first_hit_at_angle_np(
                        float(candidate_ray_angles[idx][d_idx]),
                        coarse.origin_xy,
                        float(coarse.origin_radius),
                        float(coarse.speed),
                        coarse.p0_by_tick,
                        coarse.p1_by_tick,
                        coarse.radii,
                        coarse.active_by_tick,
                        coarse.collision_rank,
                        horizon=INCOMING_TA_BINS,
                    )
                    if kind != "planet":
                        continue
                    true_target_slot = int(code)
                    env_hit_tick = float(tick)
                else:
                    if candidate_true_planets[idx] is None or candidate_true_hit_ticks[idx] is None:
                        continue
                    true_target_slot = int(candidate_true_planets[idx][d_idx])
                    env_hit_tick = float(candidate_true_hit_ticks[idx][d_idx])
                if _search_opponent_should_halt_on_neutral_underlaunch(
                    np.asarray(node.current_state.planets),
                    np.asarray(node.current_state.incoming_fleets),
                    player=int(ego_player),
                    search_root_player=int(ego_player),
                    send=int(candidate_sends[idx]),
                    true_target_slot=int(true_target_slot),
                ):
                    continue
                planets_next = np.array(np.asarray(node.current_state.planets), copy=True)
                incoming_next = np.array(np.asarray(node.current_state.incoming_fleets), copy=True)
                blocked_next = np.array(np.asarray(node.current_state.origin_frac_blocked), copy=True)
                apply_micro_launch_in_place(
                    planets_next,
                    incoming_next,
                    ego_player=int(ego_player),
                    origin_slot=int(o_idx),
                    send=int(candidate_sends[idx]),
                    true_target_slot=int(true_target_slot),
                    true_hit_tick=float(env_hit_tick),
                )
                action = [float(np.asarray(node.current_state.planets)[o_idx, 0]), float(candidate_ray_angles[idx][d_idx]), int(candidate_sends[idx])]
                next_actions = copy.deepcopy(node.current_turn_actions)
                next_actions.append(
                    SearchPlannedLaunchAction(
                        action=action,
                        origin_slot=int(o_idx),
                        frac_idx=int(frac_idx),
                        target_slot=int(true_target_slot),
                        planned_send=int(candidate_sends[idx]),
                        policy_hit_tick=float(candidate_ray_hit_ticks[idx][d_idx]),
                        true_hit_tick=float(env_hit_tick),
                        planets_snapshot=np.array(np.asarray(node.current_state.planets), copy=True),
                        refine_job=None,
                    )
                )
                turn_closed = int(node.current_micro_idx) + 1 >= int(self.max_micro_steps)
                children_by_node[row].append(
                    _SearchTreeNode(
                        env_public_obs=node.env_public_obs,
                        env_step_start_state=node.env_step_start_state,
                        current_state=node.current_state._replace(
                            planets=planets_next,
                            incoming_fleets=incoming_next,
                            origin_frac_blocked=blocked_next,
                        ),
                        step_count=int(node.step_count),
                        search_env_step_from_root=int(node.search_env_step_from_root),
                        current_turn_actions=next_actions,
                        current_micro_idx=int(node.current_micro_idx) + 1,
                        turn_closed=bool(turn_closed),
                        root_turn_actions=(
                            copy.deepcopy(next_actions)
                            if int(node.search_env_step_from_root) == 0
                            else copy.deepcopy(node.root_turn_actions)
                        ),
                        root_turn_complete=bool(
                            node.root_turn_complete
                            or (int(node.search_env_step_from_root) == 0 and bool(turn_closed))
                        ),
                        discounted_reward=float(node.discounted_reward),
                        discount=float(node.discount),
                        done=bool(node.done),
                    )
                )
                if timing is not None:
                    timing.bfs_expand_children += 1
        return children_by_node

    def _search_expand_tree_node(
        self,
        runtime: SearchRuntime,
        *,
        ego_player: int,
        node: _SearchTreeNode,
    ) -> list[_SearchTreeNode]:
        if bool(node.turn_closed) or bool(node.done):
            return []
        results = self._search_expand_tree_nodes_batched(
            runtime,
            ego_player=ego_player,
            nodes=[node],
        )
        return results[0] if results else []

    def _search_advance_tree_node(
        self,
        runtime: SearchRuntime,
        *,
        ego_player: int,
        node: _SearchTreeNode,
    ) -> _SearchTreeNode:
        results = self._search_advance_tree_nodes_batched(
            runtime,
            ego_player=ego_player,
            nodes=[node],
        )
        return results[0]

    def _search_advance_tree_nodes_batched(
        self,
        runtime: SearchRuntime,
        *,
        ego_player: int,
        nodes: list[_SearchTreeNode],
        timing: ModelSearchTiming | None = None,
        stop_s: float | None = None,
    ) -> list[_SearchTreeNode]:
        if not nodes:
            return []
        step_from_root = int(nodes[0].search_env_step_from_root)
        if any(int(node.search_env_step_from_root) != step_from_root for node in nodes):
            raise ValueError("_search_advance_tree_nodes_batched requires a single search depth")
        if any(not bool(node.turn_closed) for node in nodes):
            raise ValueError("_search_advance_tree_nodes_batched expects closed-turn nodes")
        if stop_s is not None and perf_counter() >= stop_s:
            return list(nodes)
        if timing is not None:
            timing.bfs_advance_batches += 1
            timing.bfs_advance_nodes += len(nodes)

        branch_joint_actions: list[list[list[float]]] = [
            [[] for _ in range(int(runtime.num_agents))]
            for _ in nodes
        ]
        branch_launch_geometry = [
            _launch_geometry_from_obs(node.env_public_obs, runtime.kaggle_config)
            for node in nodes
        ]
        seat_plans: list[_BatchedSearchSeatPlan] = []
        for branch_idx, node in enumerate(nodes):
            branch_joint_actions[branch_idx][int(ego_player)] = [
                copy.deepcopy(action.action) for action in node.current_turn_actions
            ]
            for player in range(int(runtime.num_agents)):
                if int(player) == int(ego_player):
                    continue
                seat_plans.append(
                    _BatchedSearchSeatPlan(
                        branch_idx=int(branch_idx),
                        player=int(player),
                        state_template=node.env_step_start_state,
                        planets=np.array(np.asarray(node.env_step_start_state.planets), copy=True),
                        incoming_fleets=np.array(np.asarray(node.env_step_start_state.incoming_fleets), copy=True),
                        origin_frac_blocked=np.zeros((MAX_PLANETS, len(FRACTIONS)), dtype=np.bool_),
                        actions=[],
                        micro_idx=0,
                        max_micro_steps=int(self.max_micro_steps),
                    )
                )

        if seat_plans:
            branch_joint_actions = self._plan_joint_actions_batched_single_policy(
                seat_plans=seat_plans,
                branch_joint_actions=branch_joint_actions,
                branch_launch_geometry=branch_launch_geometry,
                sim_step=int(nodes[0].step_count),
                ship_speed=float(_cfg_get(runtime.kaggle_config, "shipSpeed", 6.0)),
                search_root_player=int(ego_player),
                search_env_step_from_root=step_from_root,
                initial_policy_outputs=None,
                timing=None,
                search_greedy_launch_threshold=runtime.settings.greedy_launch_threshold,
            )

        advanced: list[_SearchTreeNode] = []
        for branch_idx, node in enumerate(nodes):
            if stop_s is not None and perf_counter() >= stop_s:
                advanced.extend(nodes[branch_idx:])
                break
            sim_state = _make_sim_state(
                node.env_public_obs,
                num_agents=int(runtime.num_agents),
                step_count=int(node.step_count),
            )
            ratios_pre = _reward_mix_ratios_np(node.env_step_start_state, runtime.settings.reward)
            _simulate_joint_step_with_kaggle_model(
                sim_state,
                joint_actions=branch_joint_actions[branch_idx],
                config=runtime.kaggle_config,
            )
            next_step = int(node.step_count) + 1
            next_public_obs = _public_obs_from_sim_state(sim_state, step_count=int(next_step))
            next_state = observation_to_state(
                next_public_obs,
                runtime.kaggle_config,
                max_fleets=self.max_fleets,
                step_count_override=int(next_step),
                num_agents_override=int(runtime.num_agents),
                fleet_arrival_cache=runtime.fleet_arrival_cache,
            )
            step_reward = _reward_delta_np(node.env_step_start_state, next_state, ratios_pre, runtime.settings.reward)
            advanced.append(
                _SearchTreeNode(
                    env_public_obs=next_public_obs,
                    env_step_start_state=next_state,
                    current_state=next_state,
                    step_count=int(next_step),
                    search_env_step_from_root=int(node.search_env_step_from_root) + 1,
                    current_turn_actions=[],
                    current_micro_idx=0,
                    turn_closed=False,
                    root_turn_actions=copy.deepcopy(node.root_turn_actions),
                    root_turn_complete=bool(node.root_turn_complete or int(node.search_env_step_from_root) == 0),
                    discounted_reward=float(node.discounted_reward) + float(node.discount) * float(step_reward[int(ego_player)]),
                    discount=float(node.discount) * float(runtime.settings.reward.gamma),
                    done=bool(np.asarray(next_state.done)),
                )
            )
        return advanced

    def _search_turn_end_opponent_joint_action_samples(
        self,
        runtime: SearchRuntime,
        *,
        ego_player: int,
        timing: ModelSearchTiming | None = None,
    ) -> list[list[list[list[float]]]]:
        cached = runtime.turn_end_opponent_joint_action_samples
        if cached is not None:
            return cached
        sample_count = max(0, int(runtime.settings.turn_end_opponent_samples))
        if sample_count <= 0:
            runtime.turn_end_opponent_joint_action_samples = []
            return []
        root_state = runtime.public_state
        if root_state is None:
            root_state = observation_to_state(
                runtime.public_obs,
                runtime.kaggle_config,
                max_fleets=self.max_fleets,
                step_count_override=int(runtime.step_count),
                num_agents_override=int(runtime.num_agents),
                fleet_arrival_cache=runtime.fleet_arrival_cache,
            )
            runtime.public_state = root_state
        branch_joint_actions: list[list[list[float]]] = [
            [[] for _ in range(int(runtime.num_agents))]
            for _ in range(sample_count)
        ]
        branch_launch_geometry = [
            _launch_geometry_from_obs(runtime.public_obs, runtime.kaggle_config)
            for _ in range(sample_count)
        ]
        seat_plans: list[_BatchedSearchSeatPlan] = []
        for branch_idx in range(sample_count):
            for player in range(int(runtime.num_agents)):
                if int(player) == int(ego_player):
                    continue
                seat_plans.append(
                    _BatchedSearchSeatPlan(
                        branch_idx=int(branch_idx),
                        player=int(player),
                        state_template=root_state,
                        planets=np.array(np.asarray(root_state.planets), copy=True),
                        incoming_fleets=np.array(np.asarray(root_state.incoming_fleets), copy=True),
                        origin_frac_blocked=np.zeros((MAX_PLANETS, len(FRACTIONS)), dtype=np.bool_),
                        actions=[],
                        micro_idx=0,
                        max_micro_steps=int(self.max_micro_steps),
                    )
                )
        if seat_plans:
            branch_joint_actions = self._plan_joint_actions_batched_single_policy(
                seat_plans=seat_plans,
                branch_joint_actions=branch_joint_actions,
                branch_launch_geometry=branch_launch_geometry,
                sim_step=int(runtime.step_count),
                ship_speed=float(_cfg_get(runtime.kaggle_config, "shipSpeed", 6.0)),
                search_root_player=int(ego_player),
                search_env_step_from_root=0,
                timing=timing,
                search_greedy_launch_threshold=runtime.settings.greedy_launch_threshold,
                sampling_mode=SAMPLING_MODE_STOCHASTIC,
            )
        runtime.turn_end_opponent_joint_action_samples = branch_joint_actions
        return branch_joint_actions

    def _score_turn_end_leaves_with_opponent_samples(
        self,
        runtime: SearchRuntime,
        *,
        ego_player: int,
        leaves: list[_SearchTreeNode],
        timing: ModelSearchTiming | None = None,
        stop_s: float | None = None,
    ) -> list[float] | None:
        if not leaves:
            return []
        t0 = perf_counter() if timing is not None else 0.0
        joint_action_samples = self._search_turn_end_opponent_joint_action_samples(
            runtime,
            ego_player=int(ego_player),
            timing=timing,
        )
        if not joint_action_samples:
            return None
        if runtime.public_state is None:
            runtime.public_state = observation_to_state(
                runtime.public_obs,
                runtime.kaggle_config,
                max_fleets=self.max_fleets,
                step_count_override=int(runtime.step_count),
                num_agents_override=int(runtime.num_agents),
                fleet_arrival_cache=runtime.fleet_arrival_cache,
            )
        pre_state = runtime.public_state
        ratios_pre = _reward_mix_ratios_np(pre_state, runtime.settings.reward)
        gamma = float(runtime.settings.reward.gamma)
        post_states: list[OrbitWarsState] = []
        sample_meta: list[tuple[int, float, bool]] = []
        leaf_totals = [0.0 for _ in leaves]
        for leaf_idx, leaf in enumerate(leaves):
            ego_actions = [copy.deepcopy(action.action) for action in leaf.current_turn_actions]
            for sample_joint_actions in joint_action_samples:
                sim_state = _make_sim_state(
                    runtime.public_obs,
                    num_agents=int(runtime.num_agents),
                    step_count=int(runtime.step_count),
                )
                joint_actions = copy.deepcopy(sample_joint_actions)
                joint_actions[int(ego_player)] = ego_actions
                _simulate_joint_step_with_kaggle_model(
                    sim_state,
                    joint_actions=joint_actions,
                    config=runtime.kaggle_config,
                )
                next_step = int(runtime.step_count) + 1
                next_public_obs = _public_obs_from_sim_state(sim_state, step_count=int(next_step))
                next_state = observation_to_state(
                    next_public_obs,
                    runtime.kaggle_config,
                    max_fleets=self.max_fleets,
                    step_count_override=int(next_step),
                    num_agents_override=int(runtime.num_agents),
                    fleet_arrival_cache=runtime.fleet_arrival_cache,
                )
                step_reward = _reward_delta_np(pre_state, next_state, ratios_pre, runtime.settings.reward)
                reward_value = float(step_reward[int(ego_player)])
                done = bool(np.asarray(next_state.done))
                sample_meta.append((leaf_idx, reward_value, done))
                if done:
                    leaf_totals[leaf_idx] += reward_value
                else:
                    post_states.append(next_state)
        t_value = perf_counter() if timing is not None else 0.0
        post_values = self._policy_values_for_states_batched(
            post_states,
            [int(ego_player)] * len(post_states),
        )
        value_idx = 0
        for leaf_idx, reward_value, done in sample_meta:
            if done:
                continue
            leaf_totals[leaf_idx] += reward_value + gamma * float(post_values[value_idx])
            value_idx += 1
        denom = float(len(joint_action_samples))
        scores = [total / denom for total in leaf_totals]
        if timing is not None:
            timing.turn_end_sample_batches += 1
            timing.turn_end_sample_joint_actions += len(joint_action_samples)
            timing.turn_end_sample_leaves += len(leaves)
            timing.turn_end_sample_s += perf_counter() - t0
            timing.value_calls += len(post_values)
            timing.value_eval_calls += len(post_values)
            timing.value_s += perf_counter() - t_value
            timing.value_eval_s += perf_counter() - t_value
        return scores

    def _search_plan_turn_bfs(
        self,
        runtime: SearchRuntime,
        *,
        ego_player: int,
        current_state: OrbitWarsState,
        timing: ModelSearchTiming | None = None,
    ) -> SearchTurnPlanResult:
        max_env_steps = int(runtime.settings.horizon_steps) if int(runtime.settings.horizon_steps) > 0 else INCOMING_TA_BINS
        deadline_s = runtime.deadline_s
        stop_s = None if deadline_s is None else max(0.0, float(deadline_s) - 0.02)
        t0 = perf_counter() if timing is not None else 0.0
        if timing is not None:
            timing.mode = str(runtime.settings.mode)
            timing.stop_at_turn_end = bool(runtime.settings.stop_at_turn_end)
            timing.branch_after_first_env_step = bool(runtime.settings.branch_after_first_env_step)
            timing.horizon_steps = int(runtime.settings.horizon_steps)
        root = _SearchTreeNode(
            env_public_obs=runtime.public_obs,
            env_step_start_state=current_state,
            current_state=current_state,
            step_count=int(runtime.step_count),
            search_env_step_from_root=0,
            current_turn_actions=[],
            current_micro_idx=0,
            turn_closed=False,
            root_turn_actions=[],
            root_turn_complete=False,
            discounted_reward=0.0,
            discount=1.0,
            done=bool(np.asarray(current_state.done)),
        )
        frontier: list[_SearchTreeNode] = [root]
        leaves: list[_SearchTreeNode] = []
        seen_turn_states: set[tuple[Any, ...]] = set()

        while frontier:
            if timing is not None:
                timing.bfs_frontier_peak = max(int(timing.bfs_frontier_peak), len(frontier))
            if stop_s is not None and perf_counter() >= stop_s:
                break
            turn_nodes = frontier
            frontier = []
            next_turn_roots: list[_SearchTreeNode] = []

            for node in turn_nodes:
                if bool(node.done) or int(node.search_env_step_from_root) >= int(max_env_steps):
                    leaves.append(node)
                    continue
                if bool(node.turn_closed):
                    if not _search_should_advance_closed_turn(
                        runtime.settings,
                        search_env_step_from_root=int(node.search_env_step_from_root),
                    ):
                        leaves.append(node)
                    else:
                        next_turn_roots.append(node)
                    continue
                frontier.append(node)

            while frontier:
                if timing is not None:
                    timing.bfs_frontier_peak = max(int(timing.bfs_frontier_peak), len(frontier))
                if stop_s is not None and perf_counter() >= stop_s:
                    break
                probe_frontier_size = len(frontier)
                probe_env_depth = int(frontier[0].search_env_step_from_root) if frontier else -1
                probe_micro_depth = int(frontier[0].current_micro_idx) if frontier else -1
                children_groups = self._search_expand_tree_nodes_batched(
                    runtime,
                    ego_player=int(ego_player),
                    nodes=frontier,
                    timing=timing,
                    stop_s=stop_s,
                )
                next_frontier: list[_SearchTreeNode] = []
                closed_turn_nodes: list[_SearchTreeNode] = []
                generated_children = 0
                pruned_duplicates = 0
                for batch_node, children in zip(frontier, children_groups):
                    if not children:
                        leaves.append(batch_node)
                        continue
                    for child in children:
                        generated_children += 1
                        key = (
                            int(child.step_count),
                            int(child.search_env_step_from_root),
                            int(child.current_micro_idx),
                            bool(child.turn_closed),
                            _search_turn_signature(
                                child.current_turn_actions,
                                np.asarray(child.current_state.origin_frac_blocked),
                            ),
                        )
                        if key in seen_turn_states:
                            pruned_duplicates += 1
                            continue
                        seen_turn_states.add(key)
                        if bool(child.turn_closed):
                            closed_turn_nodes.append(child)
                        else:
                            next_frontier.append(child)
                combined_frontier = next_frontier + closed_turn_nodes
                unique_root_path = _search_single_root_turn_path(combined_frontier)
                if unique_root_path is not None:
                    if timing is not None:
                        timing.bfs_single_path_early_exit += 1
                        timing.choose_calls += 1
                        timing.choose_s += perf_counter() - t0
                    return SearchTurnPlanResult(
                        actions=copy.deepcopy(unique_root_path.root_turn_actions),
                        turn_complete=bool(unique_root_path.root_turn_complete),
                    )
                for closed_node in closed_turn_nodes:
                    if _search_should_advance_closed_turn(
                        runtime.settings,
                        search_env_step_from_root=int(closed_node.search_env_step_from_root),
                    ):
                        next_turn_roots.append(closed_node)
                    else:
                        leaves.append(closed_node)
                frontier = next_frontier

            if frontier:
                leaves.extend(frontier)
            if not next_turn_roots:
                continue
            if stop_s is not None and perf_counter() >= stop_s:
                leaves.extend(next_turn_roots)
                break
            frontier = self._search_advance_tree_nodes_batched(
                runtime,
                ego_player=int(ego_player),
                nodes=next_turn_roots,
                timing=timing,
                stop_s=stop_s,
            )

        if frontier:
            leaves.extend(list(frontier))
        if not leaves:
            return SearchTurnPlanResult(actions=[], turn_complete=False)

        sampled_leaves: list[_SearchTreeNode] = []
        plain_value_leaves: list[_SearchTreeNode] = []
        for leaf in leaves:
            if bool(leaf.done):
                continue
            if _search_uses_turn_end_opponent_samples(
                runtime.settings,
                search_env_step_from_root=int(leaf.search_env_step_from_root),
            ):
                sampled_leaves.append(leaf)
            else:
                plain_value_leaves.append(leaf)
        sampled_scores: list[float] | None = None
        if sampled_leaves:
            sampled_scores = self._score_turn_end_leaves_with_opponent_samples(
                runtime,
                ego_player=int(ego_player),
                leaves=sampled_leaves,
                timing=timing,
                stop_s=stop_s,
            )
        values = []
        if plain_value_leaves or (sampled_leaves and sampled_scores is None):
            value_leaves = plain_value_leaves + (sampled_leaves if sampled_scores is None else [])
            t_value = perf_counter() if timing is not None else 0.0
            values = self._policy_values_for_states_batched(
                [leaf.current_state for leaf in value_leaves],
                [int(ego_player)] * len(value_leaves),
            )
            if timing is not None:
                dt = perf_counter() - t_value
                timing.value_calls += len(values)
                timing.value_eval_calls += len(values)
                timing.value_s += dt
                timing.value_eval_s += dt
        scored: list[tuple[float, _SearchTreeNode]] = []
        sampled_idx = 0
        value_idx = 0
        for leaf in leaves:
            score = float(leaf.discounted_reward)
            if not bool(leaf.done):
                if sampled_idx < len(sampled_leaves) and leaf is sampled_leaves[sampled_idx] and sampled_scores is not None:
                    score += float(leaf.discount) * float(sampled_scores[sampled_idx])
                    sampled_idx += 1
                else:
                    score += float(leaf.discount) * float(values[value_idx])
                    value_idx += 1
            scored.append((score, leaf))
        best_leaf = max(scored, key=lambda item: item[0])[1]
        if timing is not None:
            timing.choose_calls += 1
            timing.choose_s += perf_counter() - t0
        return SearchTurnPlanResult(
            actions=copy.deepcopy(best_leaf.root_turn_actions),
            turn_complete=bool(best_leaf.root_turn_complete),
        )

    def _search_sample_prefix_policy_out(
        self,
        runtime: SearchRuntime,
        *,
        ego_player: int,
        prefix: _SearchSamplePrefixNode,
    ) -> dict[str, Any]:
        if prefix.policy_out is None:
            prefix.policy_out = self._search_policy_outputs_for_states_batched_mixed(
                [prefix.tree_node.current_state],
                [int(ego_player)],
                search_root_player=int(ego_player),
                search_env_step_from_root=int(prefix.tree_node.search_env_step_from_root),
            )
        return prefix.policy_out

    def _search_sample_turn_child(
        self,
        runtime: SearchRuntime,
        *,
        ego_player: int,
        prefix: _SearchSamplePrefixNode,
        stop_s: float | None = None,
    ) -> _SearchSamplePrefixNode:
        node = prefix.tree_node
        if bool(node.turn_closed) or bool(node.done):
            return prefix
        out = self._search_sample_prefix_policy_out(runtime, ego_player=int(ego_player), prefix=prefix)
        branch_enabled = _search_branching_enabled_for_env_step(
            runtime.settings,
            search_env_step_from_root=int(node.search_env_step_from_root),
            current_micro_idx=int(node.current_micro_idx),
        )

        halt_logits = out["halt_logits"][0]
        if branch_enabled:
            halt_probs = torch.softmax(halt_logits, dim=-1)
            halt_choice = int(torch.multinomial(halt_probs, 1, generator=self.rng).item())
        else:
            halt_choice = int(torch.argmax(halt_logits).item())
        if halt_choice == 1:
            choice_key = ("halt",)
            cached = prefix.children_by_choice.get(choice_key)
            if cached is not None:
                return cached
            root_actions = (
                copy.deepcopy(node.current_turn_actions)
                if int(node.search_env_step_from_root) == 0
                else copy.deepcopy(node.root_turn_actions)
            )
            child = _SearchSamplePrefixNode(
                tree_node=_SearchTreeNode(
                    env_public_obs=node.env_public_obs,
                    env_step_start_state=node.env_step_start_state,
                    current_state=node.current_state,
                    step_count=int(node.step_count),
                    search_env_step_from_root=int(node.search_env_step_from_root),
                    current_turn_actions=copy.deepcopy(node.current_turn_actions),
                    current_micro_idx=int(node.current_micro_idx),
                    turn_closed=True,
                    root_turn_actions=root_actions,
                    root_turn_complete=bool(node.root_turn_complete or int(node.search_env_step_from_root) == 0),
                    discounted_reward=float(node.discounted_reward),
                    discount=float(node.discount),
                    done=bool(node.done),
                )
            )
            prefix.children_by_choice[choice_key] = child
            return child

        flat_mask = out["origin_frac_mask"].flatten(start_dim=1)[0]
        if not bool(flat_mask.any().item()):
            choice_key = ("halt_nomask",)
            cached = prefix.children_by_choice.get(choice_key)
            if cached is not None:
                return cached
            child = _SearchSamplePrefixNode(
                tree_node=_SearchTreeNode(
                    env_public_obs=node.env_public_obs,
                    env_step_start_state=node.env_step_start_state,
                    current_state=node.current_state,
                    step_count=int(node.step_count),
                    search_env_step_from_root=int(node.search_env_step_from_root),
                    current_turn_actions=copy.deepcopy(node.current_turn_actions),
                    current_micro_idx=int(node.current_micro_idx),
                    turn_closed=True,
                    root_turn_actions=(
                        copy.deepcopy(node.current_turn_actions)
                        if int(node.search_env_step_from_root) == 0
                        else copy.deepcopy(node.root_turn_actions)
                    ),
                    root_turn_complete=bool(node.root_turn_complete or int(node.search_env_step_from_root) == 0),
                    discounted_reward=float(node.discounted_reward),
                    discount=float(node.discount),
                    done=bool(node.done),
                )
            )
            prefix.children_by_choice[choice_key] = child
            return child

        flat_logits = out["origin_frac_logits"].flatten(start_dim=1)[0]
        masked_origin_frac = flat_logits.masked_fill(~flat_mask, -1e4)
        if branch_enabled:
            origin_frac_probs = torch.softmax(masked_origin_frac, dim=-1)
            origin_frac_flat = int(torch.multinomial(origin_frac_probs, 1, generator=self.rng).item())
        else:
            origin_frac_flat = int(torch.argmax(masked_origin_frac).item())

        origin_options = prefix.origin_options_cache.get(int(origin_frac_flat))
        if origin_options is None:
            o_idx = int(origin_frac_flat) // len(FRACTIONS)
            frac_idx = int(origin_frac_flat) % len(FRACTIONS)
            ships_avail = float(np.asarray(node.current_state.planets)[o_idx, 5])
            send = _planned_send(ships_avail, int(frac_idx))
            options: list[_SearchSampleTargetOption] = []
            logits: list[float] = []
            if send > 0:
                geometry_override = self._search_geometry_override_for_tree_node(node)
                launch_geometry = _launch_geometry_from_obs(node.env_public_obs, runtime.kaggle_config)
                if self.target_method == "interval":
                    coarse = _search_first_contact_targets_np(
                        node.current_state,
                        int(o_idx),
                        int(frac_idx),
                        ship_speed=float(_cfg_get(runtime.kaggle_config, "shipSpeed", 6.0)),
                        horizon=INCOMING_TA_BINS,
                        launch_geometry=launch_geometry,
                        geometry_override=geometry_override,
                    )
                    ray_angle = coarse.angles
                    ray_valid = coarse.valid
                    ray_hit_tick = coarse.eta
                    true_planet = None
                    true_hit_tick = None
                else:
                    ray_angle, ray_valid, ray_hit_tick, true_planet, true_hit_tick, _ = _first_hit_targets_np(
                        node.current_state,
                        int(o_idx),
                        int(frac_idx),
                        ship_speed=float(_cfg_get(runtime.kaggle_config, "shipSpeed", 6.0)),
                        horizon=INCOMING_TA_BINS,
                        n_rays=self.raycast_rays,
                        samples_per_span=self.interval_samples_per_span,
                        target_method=self.target_method,
                        target_timing=None,
                        game_step=int(node.step_count),
                        micro_idx=int(node.current_micro_idx),
                        ego_player=int(ego_player),
                        launch_geometry=launch_geometry,
                        geometry_override=geometry_override,
                        refine_boundaries=False,
                        phase="search",
                        return_jobs=True,
                    )
                    coarse = None
                origin_idx_t = torch.tensor([int(o_idx)], device=self.device, dtype=torch.long)
                frac_idx_t = torch.tensor([int(frac_idx)], device=self.device, dtype=torch.long)
                fleet_size_t = torch.tensor([int(send)], device=self.device, dtype=torch.float32)
                target_eta_t = torch.from_numpy(np.asarray(ray_hit_tick, dtype=np.float32)[None, :]).to(
                    device=self.device,
                    dtype=torch.float32,
                )
                target_ships_t = torch.from_numpy(
                    np.asarray(node.current_state.planets[:, 5], dtype=np.float32)[None, :]
                ).to(device=self.device, dtype=torch.float32)
                hidden_t = torch.stack([out["planet_hidden_rows"][0]], dim=0)
                policy_ctx, pcount, normalize_obs_to_p0, population_member_for_player = (
                    self._search_policy_context_for_player(
                        int(ego_player),
                        int(ego_player),
                        int(node.search_env_step_from_root),
                    )
                )
                population_member = population_member_for_player(int(ego_player))
                population_idx = None
                if population_member is not None:
                    population_idx = torch.tensor([int(population_member)], device=self.device, dtype=torch.long)
                obs_group = _obs_tensors_for_states(
                    [node.current_state],
                    [int(ego_player)],
                    self.device,
                    policy_player_count=pcount,
                    target_abort_enabled=bool(getattr(policy_ctx, "target_abort_enabled", False)),
                    normalize_obs_to_p0=normalize_obs_to_p0,
                )
                target_logits = policy_ctx.target_logits_for_origin_fraction(
                    hidden_t,
                    obs_group["owner_idx"],
                    obs_group["features"],
                    origin_idx_t,
                    frac_idx_t,
                    fleet_size=fleet_size_t,
                    target_eta=target_eta_t,
                    target_ships=target_ships_t,
                    population_idx=population_idx,
                )[0]
                target_mask = out["pair_mask"][0, o_idx].clone()
                target_mask &= torch.from_numpy(np.asarray(ray_valid, dtype=np.bool_)).to(
                    device=self.device,
                    dtype=torch.bool,
                )
                abort_logits_all = out.get("abort_logits")
                if abort_logits_all is not None:
                    abort_logit = abort_logits_all[0, o_idx, frac_idx].reshape(1)
                    options.append(
                        _SearchSampleTargetOption(
                            choice_key=("abort", int(o_idx), int(frac_idx)),
                            origin_slot=int(o_idx),
                            frac_idx=int(frac_idx),
                            send=int(send),
                            is_abort=True,
                        )
                    )
                    logits.append(float(abort_logit.item()))
                if bool(target_mask.any().item()):
                    for d_idx in torch.nonzero(target_mask, as_tuple=False).flatten().tolist():
                        true_target_slot = -1
                        env_hit_tick = -1.0
                        if coarse is not None:
                            kind, code, tick = _discrete_first_hit_at_angle_np(
                                float(ray_angle[d_idx]),
                                coarse.origin_xy,
                                float(coarse.origin_radius),
                                float(coarse.speed),
                                coarse.p0_by_tick,
                                coarse.p1_by_tick,
                                coarse.radii,
                                coarse.active_by_tick,
                                coarse.collision_rank,
                                horizon=INCOMING_TA_BINS,
                            )
                            if kind != "planet":
                                continue
                            true_target_slot = int(code)
                            env_hit_tick = float(tick)
                        else:
                            if true_planet is None or true_hit_tick is None:
                                continue
                            true_target_slot = int(true_planet[d_idx])
                            env_hit_tick = float(true_hit_tick[d_idx])
                        if _search_opponent_should_halt_on_neutral_underlaunch(
                            np.asarray(node.current_state.planets),
                            np.asarray(node.current_state.incoming_fleets),
                            player=int(ego_player),
                            search_root_player=int(ego_player),
                            send=int(send),
                            true_target_slot=int(true_target_slot),
                        ):
                            continue
                        options.append(
                            _SearchSampleTargetOption(
                                choice_key=("launch", int(o_idx), int(frac_idx), int(d_idx)),
                                origin_slot=int(o_idx),
                                frac_idx=int(frac_idx),
                                send=int(send),
                                is_abort=False,
                                angle=float(ray_angle[d_idx]),
                                target_slot=int(true_target_slot),
                                policy_hit_tick=float(ray_hit_tick[d_idx]),
                                true_hit_tick=float(env_hit_tick),
                            )
                        )
                        logits.append(float(target_logits[d_idx].item()))
            logit_tensor = torch.tensor(logits, device=self.device, dtype=torch.float32) if logits else torch.empty((0,), device=self.device, dtype=torch.float32)
            origin_options = _SearchSampleOriginOptions(options=options, logits=logit_tensor)
            prefix.origin_options_cache[int(origin_frac_flat)] = origin_options

        if not origin_options.options:
            choice_key = ("halt_notarget", int(origin_frac_flat))
            cached = prefix.children_by_choice.get(choice_key)
            if cached is not None:
                return cached
            child = _SearchSamplePrefixNode(
                tree_node=_SearchTreeNode(
                    env_public_obs=node.env_public_obs,
                    env_step_start_state=node.env_step_start_state,
                    current_state=node.current_state,
                    step_count=int(node.step_count),
                    search_env_step_from_root=int(node.search_env_step_from_root),
                    current_turn_actions=copy.deepcopy(node.current_turn_actions),
                    current_micro_idx=int(node.current_micro_idx),
                    turn_closed=True,
                    root_turn_actions=(
                        copy.deepcopy(node.current_turn_actions)
                        if int(node.search_env_step_from_root) == 0
                        else copy.deepcopy(node.root_turn_actions)
                    ),
                    root_turn_complete=bool(node.root_turn_complete or int(node.search_env_step_from_root) == 0),
                    discounted_reward=float(node.discounted_reward),
                    discount=float(node.discount),
                    done=bool(node.done),
                )
            )
            prefix.children_by_choice[choice_key] = child
            return child

        if branch_enabled:
            option_probs = torch.softmax(origin_options.logits, dim=-1)
            option_idx = int(torch.multinomial(option_probs, 1, generator=self.rng).item())
        else:
            option_idx = int(torch.argmax(origin_options.logits).item())
        option = origin_options.options[option_idx]
        cached = prefix.children_by_choice.get(option.choice_key)
        if cached is not None:
            return cached

        if option.is_abort:
            next_blocked = np.array(np.asarray(node.current_state.origin_frac_blocked), copy=True)
            next_blocked[int(option.origin_slot), int(option.frac_idx)] = True
            next_state = node.current_state._replace(origin_frac_blocked=next_blocked)
            next_actions = copy.deepcopy(node.current_turn_actions)
            child = _SearchSamplePrefixNode(
                tree_node=_SearchTreeNode(
                    env_public_obs=node.env_public_obs,
                    env_step_start_state=node.env_step_start_state,
                    current_state=next_state,
                    step_count=int(node.step_count),
                    search_env_step_from_root=int(node.search_env_step_from_root),
                    current_turn_actions=next_actions,
                    current_micro_idx=int(node.current_micro_idx) + 1,
                    turn_closed=int(node.current_micro_idx) + 1 >= int(self.max_micro_steps),
                    root_turn_actions=(
                        copy.deepcopy(next_actions)
                        if int(node.search_env_step_from_root) == 0
                        else copy.deepcopy(node.root_turn_actions)
                    ),
                    root_turn_complete=bool(
                        node.root_turn_complete
                        or (
                            int(node.search_env_step_from_root) == 0
                            and int(node.current_micro_idx) + 1 >= int(self.max_micro_steps)
                        )
                    ),
                    discounted_reward=float(node.discounted_reward),
                    discount=float(node.discount),
                    done=bool(node.done),
                )
            )
        else:
            planets_next = np.array(np.asarray(node.current_state.planets), copy=True)
            incoming_next = np.array(np.asarray(node.current_state.incoming_fleets), copy=True)
            blocked_next = np.array(np.asarray(node.current_state.origin_frac_blocked), copy=True)
            apply_micro_launch_in_place(
                planets_next,
                incoming_next,
                ego_player=int(ego_player),
                origin_slot=int(option.origin_slot),
                send=int(option.send),
                true_target_slot=int(option.target_slot),
                true_hit_tick=float(option.true_hit_tick),
            )
            next_actions = copy.deepcopy(node.current_turn_actions)
            next_actions.append(
                SearchPlannedLaunchAction(
                    action=[
                        float(np.asarray(node.current_state.planets)[int(option.origin_slot), 0]),
                        float(option.angle),
                        int(option.send),
                    ],
                    origin_slot=int(option.origin_slot),
                    frac_idx=int(option.frac_idx),
                    target_slot=int(option.target_slot),
                    planned_send=int(option.send),
                    policy_hit_tick=float(option.policy_hit_tick),
                    true_hit_tick=float(option.true_hit_tick),
                    planets_snapshot=np.array(np.asarray(node.current_state.planets), copy=True),
                    refine_job=None,
                )
            )
            turn_closed = int(node.current_micro_idx) + 1 >= int(self.max_micro_steps)
            child = _SearchSamplePrefixNode(
                tree_node=_SearchTreeNode(
                    env_public_obs=node.env_public_obs,
                    env_step_start_state=node.env_step_start_state,
                    current_state=node.current_state._replace(
                        planets=planets_next,
                        incoming_fleets=incoming_next,
                        origin_frac_blocked=blocked_next,
                    ),
                    step_count=int(node.step_count),
                    search_env_step_from_root=int(node.search_env_step_from_root),
                    current_turn_actions=next_actions,
                    current_micro_idx=int(node.current_micro_idx) + 1,
                    turn_closed=bool(turn_closed),
                    root_turn_actions=(
                        copy.deepcopy(next_actions)
                        if int(node.search_env_step_from_root) == 0
                        else copy.deepcopy(node.root_turn_actions)
                    ),
                    root_turn_complete=bool(
                        node.root_turn_complete
                        or (int(node.search_env_step_from_root) == 0 and bool(turn_closed))
                    ),
                    discounted_reward=float(node.discounted_reward),
                    discount=float(node.discount),
                    done=bool(node.done),
                )
            )
        prefix.children_by_choice[option.choice_key] = child
        return child

    def _search_plan_turn_sampling(
        self,
        runtime: SearchRuntime,
        *,
        ego_player: int,
        current_state: OrbitWarsState,
        timing: ModelSearchTiming | None = None,
    ) -> SearchTurnPlanResult:
        deadline_s = runtime.deadline_s
        stop_s = None if deadline_s is None else max(0.0, float(deadline_s) - 0.02)
        t0 = perf_counter() if timing is not None else 0.0
        if timing is not None:
            timing.mode = str(runtime.settings.mode)
            timing.stop_at_turn_end = bool(runtime.settings.stop_at_turn_end)
            timing.branch_after_first_env_step = bool(runtime.settings.branch_after_first_env_step)
            timing.horizon_steps = int(runtime.settings.horizon_steps)
        root = _SearchTreeNode(
            env_public_obs=runtime.public_obs,
            env_step_start_state=current_state,
            current_state=current_state,
            step_count=int(runtime.step_count),
            search_env_step_from_root=0,
            current_turn_actions=[],
            current_micro_idx=0,
            turn_closed=False,
            root_turn_actions=[],
            root_turn_complete=False,
            discounted_reward=0.0,
            discount=1.0,
            done=bool(np.asarray(current_state.done)),
        )
        root_prefix = _SearchSamplePrefixNode(tree_node=root)
        unique_leaves: list[_SearchTreeNode] = []
        seen_leaf_keys: set[tuple[Any, ...]] = set()
        max_attempts_without_deadline = 128
        sampled_limit = runtime.settings.turn_sampling_max_samples
        sampled_limit = None if sampled_limit is None else max(0, int(sampled_limit))
        sample_attempts = 0
        completed_sequences = 0

        while True:
            if sampled_limit is not None and int(completed_sequences) >= int(sampled_limit):
                break
            if stop_s is not None:
                if perf_counter() >= stop_s:
                    break
            elif sample_attempts >= max_attempts_without_deadline:
                break
            sample_attempts += 1
            prefix = root_prefix
            while True:
                node = prefix.tree_node
                if bool(node.done) or bool(node.turn_closed):
                    completed_sequences += 1
                    if timing is not None:
                        timing.turn_sampling_completed_sequences += 1
                    leaf_key = (
                        int(node.step_count),
                        int(node.current_micro_idx),
                        bool(node.turn_closed),
                        _search_turn_signature(
                            node.current_turn_actions,
                            np.asarray(node.current_state.origin_frac_blocked),
                        ),
                    )
                    if leaf_key not in seen_leaf_keys:
                        seen_leaf_keys.add(leaf_key)
                        unique_leaves.append(node)
                    break
                if stop_s is not None and perf_counter() >= stop_s:
                    break
                prefix = self._search_sample_turn_child(
                    runtime,
                    ego_player=int(ego_player),
                    prefix=prefix,
                    stop_s=stop_s,
                )
            if stop_s is not None and perf_counter() >= stop_s:
                break

        if not unique_leaves:
            return SearchTurnPlanResult(actions=[], turn_complete=False)
        if len(unique_leaves) == 1:
            if timing is not None:
                timing.bfs_single_path_early_exit += 1
                timing.choose_calls += 1
                timing.choose_s += perf_counter() - t0
            leaf = unique_leaves[0]
            return SearchTurnPlanResult(
                actions=copy.deepcopy(leaf.root_turn_actions),
                turn_complete=bool(leaf.root_turn_complete),
            )

        sampled_scores: list[float] | None = None
        if int(runtime.settings.turn_end_opponent_samples) > 0:
            sampled_scores = self._score_turn_end_leaves_with_opponent_samples(
                runtime,
                ego_player=int(ego_player),
                leaves=unique_leaves,
                timing=timing,
                stop_s=stop_s,
            )
        values: list[float] = []
        if sampled_scores is None:
            t_value = perf_counter() if timing is not None else 0.0
            values = self._policy_values_for_states_batched(
                [leaf.current_state for leaf in unique_leaves],
                [int(ego_player)] * len(unique_leaves),
            )
            if timing is not None:
                dt = perf_counter() - t_value
                timing.value_calls += len(values)
                timing.value_eval_calls += len(values)
                timing.value_s += dt
                timing.value_eval_s += dt
        scored: list[tuple[float, _SearchTreeNode]] = []
        for idx, leaf in enumerate(unique_leaves):
            score = float(leaf.discounted_reward)
            if not bool(leaf.done):
                if sampled_scores is not None:
                    score += float(leaf.discount) * float(sampled_scores[idx])
                else:
                    score += float(leaf.discount) * float(values[idx])
            scored.append((score, leaf))
        best_leaf = max(scored, key=lambda item: item[0])[1]
        if timing is not None:
            timing.choose_calls += 1
            timing.choose_s += perf_counter() - t0
        return SearchTurnPlanResult(
            actions=copy.deepcopy(best_leaf.root_turn_actions),
            turn_complete=bool(best_leaf.root_turn_complete),
        )

    def _choose_launch_via_model_search_batched_single_policy(
        self,
        runtime: SearchRuntime,
        *,
        ego_player: int,
        current_state: OrbitWarsState,
        current_micro_idx: int,
        action_prefix: list[list[float]],
        launch_action: list[float],
        launch_origin_slot: int,
        launch_send: int,
        launch_true_target_slot: int,
        launch_true_hit_tick: float,
        timing: ModelSearchTiming | None = None,
    ) -> bool:
        t_choose = perf_counter() if timing is not None else 0.0
        rollout_horizon = _model_search_rollout_horizon(
            runtime.settings,
            launch_true_hit_tick=float(launch_true_hit_tick),
        )
        _log_model_search_horizon(
            runtime.settings,
            rollout_horizon=int(rollout_horizon),
            launch_true_hit_tick=float(launch_true_hit_tick),
            step_count=int(runtime.step_count),
            ego_player=int(ego_player),
            origin_slot=int(launch_origin_slot),
            target_slot=int(launch_true_target_slot),
            micro_idx=int(current_micro_idx),
            send=int(launch_send),
        )
        t_cache = perf_counter() if timing is not None else 0.0
        cache = self._search_cache_match(runtime, ego_player=int(ego_player))
        if timing is not None:
            timing.cache_lookup_s += perf_counter() - t_cache
        cache_branch_miss = False
        if cache is not None:
            self._search_cache_hits += 1
            if timing is not None:
                timing.cache_hits += 1
                t_cache = perf_counter()
            cached_branch = self._identify_cached_branch(
                action_prefix=action_prefix,
                launch_action=launch_action,
                cache=cache,
            )
            if timing is not None:
                timing.cache_identify_s += perf_counter() - t_cache
            if cached_branch == "halt":
                halt_score, halt_transitions = self._score_branch_from_cache(
                    runtime,
                    ego_player=int(ego_player),
                    cache=cache,
                    rollout_horizon=int(rollout_horizon),
                    timing=timing,
                )
                launch_scores, launch_branch_traces, launch_root_actions = self._evaluate_search_branches(
                    runtime,
                    ego_player=int(ego_player),
                    current_state=current_state,
                    current_micro_idx=int(current_micro_idx),
                    action_prefix=copy.deepcopy(action_prefix),
                    launch_action=copy.deepcopy(launch_action),
                    launch_origin_slot=int(launch_origin_slot),
                    launch_send=int(launch_send),
                    launch_true_target_slot=int(launch_true_target_slot),
                    launch_true_hit_tick=float(launch_true_hit_tick),
                    rollout_horizon=int(rollout_horizon),
                    branch_mask=(False, True),
                    timing=timing,
                )
                launch_score = float(launch_scores[1])
                launch_transitions = launch_branch_traces[1]
                launch_root_ego_actions = launch_root_actions[1]
                halt_root_ego_actions = cache.root_ego_actions
            elif cached_branch == "launch":
                launch_score, launch_transitions = self._score_branch_from_cache(
                    runtime,
                    ego_player=int(ego_player),
                    cache=cache,
                    rollout_horizon=int(rollout_horizon),
                    timing=timing,
                )
                halt_scores, halt_branch_traces, halt_root_actions = self._evaluate_search_branches(
                    runtime,
                    ego_player=int(ego_player),
                    current_state=current_state,
                    current_micro_idx=int(current_micro_idx),
                    action_prefix=copy.deepcopy(action_prefix),
                    launch_action=copy.deepcopy(launch_action),
                    launch_origin_slot=int(launch_origin_slot),
                    launch_send=int(launch_send),
                    launch_true_target_slot=int(launch_true_target_slot),
                    launch_true_hit_tick=float(launch_true_hit_tick),
                    rollout_horizon=int(rollout_horizon),
                    branch_mask=(True, False),
                    timing=timing,
                )
                halt_score = float(halt_scores[0])
                halt_transitions = halt_branch_traces[0]
                halt_root_ego_actions = halt_root_actions[0]
                launch_root_ego_actions = cache.root_ego_actions
            else:
                self._search_cache_hits = max(0, self._search_cache_hits - 1)
                if timing is not None:
                    timing.cache_hits = max(0, timing.cache_hits - 1)
                if _model_search_debug_enabled():
                    _model_search_debug(
                        f"choose step={int(runtime.step_count)} ego={int(ego_player)} "
                        f"cache stale-on-branch call={self._search_cache_hits} miss={self._search_cache_misses + 1}"
                    )
                cache = None
                cache_branch_miss = True
        if cache is not None:
            if timing is not None:
                timing.choose_calls += 1
                timing.choose_s += perf_counter() - t_choose
            chose_launch = bool(launch_score > halt_score)
            t_cache = perf_counter() if timing is not None else 0.0
            self._store_search_rollout_cache_from_transitions(
                runtime,
                ego_player=int(ego_player),
                root_ego_actions=(launch_root_ego_actions if chose_launch else halt_root_ego_actions),
                transitions=(launch_transitions if chose_launch else halt_transitions),
            )
            if timing is not None:
                timing.cache_store_s += perf_counter() - t_cache
            if _model_search_debug_enabled():
                _model_search_debug(
                    f"choose step={int(runtime.step_count)} ego={int(ego_player)} "
                    f"halt={halt_score:.6f} launch={launch_score:.6f} -> "
                    f"{'launch' if chose_launch else 'halt'} "
                    f"(cache hit call={self._search_cache_hits} miss={self._search_cache_misses})"
                )
            return chose_launch

        if cache is None:
            self._search_cache_misses += 1
            if timing is not None:
                timing.cache_misses += 1
            if _model_search_debug_enabled():
                _model_search_debug(
                    f"choose step={int(runtime.step_count)} ego={int(ego_player)} "
                    f"cache {'stale' if cache_branch_miss else 'miss'} "
                    f"call={self._search_cache_hits} miss={self._search_cache_misses}"
                )
        branch_scores, branch_traces, branch_root_ego_actions = self._evaluate_search_branches(
            runtime,
            ego_player=int(ego_player),
            current_state=current_state,
            current_micro_idx=int(current_micro_idx),
            action_prefix=copy.deepcopy(action_prefix),
            launch_action=copy.deepcopy(launch_action),
            launch_origin_slot=int(launch_origin_slot),
            launch_send=int(launch_send),
            launch_true_target_slot=int(launch_true_target_slot),
            launch_true_hit_tick=float(launch_true_hit_tick),
            rollout_horizon=int(rollout_horizon),
            branch_mask=(True, True),
            timing=timing,
        )
        if timing is not None:
            timing.branch_rollouts += 2
            timing.branch_rollout_s += perf_counter() - t_choose
            timing.choose_calls += 1
            timing.choose_s += perf_counter() - t_choose
        chose_launch = bool(branch_scores[1] > branch_scores[0])
        t_cache = perf_counter() if timing is not None else 0.0
        self._store_search_rollout_cache(
            runtime,
            ego_player=int(ego_player),
            chose_launch=chose_launch,
            branch_transitions=branch_traces,
            branch_root_ego_actions=branch_root_ego_actions,
        )
        if timing is not None:
            timing.cache_store_s += perf_counter() - t_cache
        if _model_search_debug_enabled():
            _model_search_debug(
                f"choose step={int(runtime.step_count)} ego={int(ego_player)} "
                f"halt={branch_scores[0]:.6f} launch={branch_scores[1]:.6f} -> "
                f"{'launch' if chose_launch else 'halt'}"
            )
        return chose_launch

    def _search_greedy_actions_for_player(
        self,
        obs: Mapping[str, Any],
        player: int,
        step_count: int,
        *,
        search_root_player: int | None = None,
    ) -> list[list[float]]:
        state = observation_to_state(
            obs,
            None,
            max_fleets=self.max_fleets,
            step_count_override=step_count,
            num_agents_override=self._num_agents_for_obs(obs, None),
            fleet_arrival_cache=self._fleet_arrival_cache,
        )
        return _build_turn_actions_torch_only(
            self.policy,
            state,
            player,
            self.device,
            ship_speed=6.0,
            max_micro_steps=self.max_micro_steps,
            sampling_mode=SAMPLING_MODE_GREEDY,
            rng=self.rng,
            n_rays=self.raycast_rays,
            samples_per_span=self.interval_samples_per_span,
            target_method=self.target_method,
            timing=None,
            launch_tracker=None,
            game_step=step_count,
            policy_player_count=self.policy_player_count,
            normalize_obs_to_p0=self.normalize_obs_to_p0,
            launch_geometry=_launch_geometry_from_obs(obs, None),
            population_member=self._population_member_for_player(player),
            search_runtime=None,
            search_launch_probability_threshold=None,
            search_greedy_launch_threshold=self.model_search.greedy_launch_threshold,
            search_root_player=search_root_player,
        )

    def _search_value_for_player(
        self,
        obs: Mapping[str, Any],
        player: int,
        step_count: int,
    ) -> float:
        state = observation_to_state(
            obs,
            None,
            max_fleets=self.max_fleets,
            step_count_override=step_count,
            num_agents_override=self._num_agents_for_obs(obs, None),
            fleet_arrival_cache=self._fleet_arrival_cache,
        )
        return _policy_value_for_state(
            self.policy,
            state,
            player,
            self.device,
            policy_player_count=self.policy_player_count,
            normalize_obs_to_p0=self.normalize_obs_to_p0,
            population_member=self._population_member_for_player(player),
        )

    def _obs_game_key(self, obs: Mapping[str, Any]) -> str:
        """Stable per-episode id.

        Kaggle mutates ``initial_planets`` during play (e.g. comets added/removed), so we only
        refresh the key on step 0; mid-game changes must not reset launch bookkeeping.
        """

        step_raw = obs.get("step", obs.get("step_count", None))
        step = int(step_raw) if step_raw is not None else None

        initial = obs.get("initial_planets")
        if not initial:
            if self._frozen_game_key is not None:
                return self._frozen_game_key
            initial = obs.get("planets", [])

        arr = np.asarray(initial, dtype=np.float32)
        h = hashlib.blake2b(digest_size=16)
        h.update(arr.tobytes())
        h.update(str(obs.get("angular_velocity", 0.0)).encode("ascii", errors="ignore"))
        key = h.hexdigest()

        if step == 0 or self._frozen_game_key is None:
            self._frozen_game_key = key
        return self._frozen_game_key

    def _step_count_for_obs(self, obs: Mapping[str, Any]) -> int:
        step_raw = obs.get("step", obs.get("step_count", None))
        if step_raw is not None:
            s = int(step_raw)
            self._last_env_step = s
            self._next_step_count = s + 1
            key = self._obs_game_key(obs)
            if key != self._game_key:
                self._frozen_num_agents = None
                self._fleet_arrival_cache.clear()
                self._search_rollout_cache = None
                self._search_cache_hits = 0
                self._search_cache_misses = 0
            self._game_key = key
            return s

        key = self._obs_game_key(obs)
        if key != self._game_key:
            self._game_key = key
            self._next_step_count = 0
            self._last_env_step = None
            self._frozen_num_agents = None
            self._fleet_arrival_cache.clear()
            self._search_rollout_cache = None
            self._search_cache_hits = 0
            self._search_cache_misses = 0

        if self._last_env_step is not None:
            ego = int(obs.get("player", 0))
            if ego >= 2:
                _adapter_warn_once(
                    self._sanity_warnings,
                    f"single:missing-step-mirrored:ego{ego}",
                    "Kaggle observation omitted step for player >=2; reusing last observed env step "
                    f"context=single ego={ego} reused_step={int(self._last_env_step)}",
                )
            return int(self._last_env_step)

        step_count = self._next_step_count
        ego = int(obs.get("player", 0))
        if ego >= 2:
            _adapter_warn_once(
                self._sanity_warnings,
                f"single:missing-step-inferred:ego{ego}",
                "Kaggle observation omitted step for player >=2 and no prior step was available; "
                f"context=single ego={ego} inferred_step={step_count}",
            )
        self._next_step_count += 1
        return step_count

    @torch.inference_mode()
    def __call__(self, obs: Mapping[str, Any], config: Any = None) -> list[list[float]]:
        call_t0 = perf_counter()
        self._last_call_timing = None
        timing = KaggleAgentCallTiming()
        ego_player = int(obs.get("player", 0))
        step_count = self._step_count_for_obs(obs)
        game_key = self._obs_game_key(obs)
        self.launch_tracker.sync_game(game_key, game_step=step_count)
        ship_speed = float(_cfg_get(config, "shipSpeed", 6.0))
        self.launch_tracker.observe_fleets(
            obs,
            ego_player,
            game_step=step_count,
            ship_speed=ship_speed,
            n_rays=self.raycast_rays,
        )
        fleets_in = obs.get("fleets") or []
        fleet_arrivals = np.full((len(fleets_in), 2), -1, dtype=np.int32)
        t0 = perf_counter()
        state = observation_to_state(
            obs,
            config,
            max_fleets=self.max_fleets,
            step_count_override=step_count,
            fleet_forecast_arrival=fleet_arrivals,
            num_agents_override=self._num_agents_for_obs(obs, config),
            fleet_arrival_cache=self._fleet_arrival_cache,
        )
        if not self._compiled_forward_warmup_done:
            _warmup_compiled_policy_batched_forward(
                self.policy,
                state,
                ego_player,
                self.device,
                policy_player_count=self.policy_player_count,
                normalize_obs_to_p0=self.normalize_obs_to_p0,
                population_member=self._population_member_for_player(ego_player),
            )
            self._compiled_forward_warmup_done = True
        if not getattr(
            self,
            "_search_compiled_forward_warmup_done",
            self._search_policy_obj() is self.policy,
        ):
            _warmup_compiled_policy_batched_forward(
                self._search_policy_obj(),
                state,
                ego_player,
                self.device,
                policy_player_count=self._search_policy_player_count_value(),
                normalize_obs_to_p0=self._search_normalize_obs_to_p0_value(),
                population_member=self._search_population_member_for_player(ego_player),
            )
            self._search_compiled_forward_warmup_done = True
        launch_geometry = _launch_geometry_from_obs(obs, config)
        timing.obs_to_state_s = perf_counter() - t0
        self.launch_tracker.check_forecast_vs_raycast(
            obs,
            fleet_arrivals,
            np.asarray(state.planets),
            np.asarray(state.comet_planet_ids),
            game_step=step_count,
            ego_player=ego_player,
        )
        _check_4p_adapter_sanity(
            obs=obs,
            config=config,
            state=state,
            ego_player=ego_player,
            step_count=step_count,
            policy=self.policy,
            policy_player_count=self.policy_player_count,
            seen=self._sanity_warnings,
            context="single",
        )
        search_runtime = None
        search_active = False
        act_timeout = _cfg_get(config, "actTimeout", None)
        action_deadline_s = (
            call_t0 + max(0.0, float(act_timeout))
            if act_timeout is not None
            else None
        )
        search_deadline_s = None
        if act_timeout is not None and _search_has_deadline(self.model_search):
            search_scale = _search_time_scale_from_overage(obs)
            search_deadline_s = call_t0 + max(0.0, float(act_timeout)) * float(search_scale)
        if _model_search_enabled(self.model_search) and _model_search_allowed_for_obs(obs, self.model_search):
            search_active = True
            search_timing = timing.model_search
            search_timing.mode = str(self.model_search.mode)
            search_fleet_arrival_cache = FleetArrivalCache()

            def _search_greedy_actions(sim_obs: Mapping[str, Any], player: int, sim_step: int) -> list[list[float]]:
                t_total = perf_counter()
                t0 = perf_counter()
                sim_state = observation_to_state(
                    sim_obs,
                    config,
                    max_fleets=self.max_fleets,
                    step_count_override=sim_step,
                    num_agents_override=int(np.asarray(state.num_agents)),
                    fleet_arrival_cache=search_fleet_arrival_cache,
                )
                search_timing.opponent_greedy_obs_to_state_calls += 1
                search_timing.opponent_greedy_obs_to_state_s += perf_counter() - t0
                t0 = perf_counter()
                actions = _build_turn_actions_torch_only(
                    self._search_policy_obj(),
                    sim_state,
                    player,
                    self.device,
                    ship_speed=ship_speed,
                    max_micro_steps=self.max_micro_steps,
                    sampling_mode=SAMPLING_MODE_GREEDY,
                    rng=self.rng,
                    n_rays=self.raycast_rays,
                    samples_per_span=self.interval_samples_per_span,
                    target_method=self.target_method,
                    timing=None,
                    launch_tracker=None,
                    game_step=sim_step,
                    policy_player_count=self._search_policy_player_count_value(),
                    normalize_obs_to_p0=self._search_normalize_obs_to_p0_value(),
                    launch_geometry=_launch_geometry_from_obs(sim_obs, config),
                    population_member=self._search_population_member_for_player(player),
                    search_runtime=None,
                    search_launch_probability_threshold=None,
                    search_greedy_launch_threshold=self.model_search.greedy_launch_threshold,
                    search_root_player=int(ego_player),
                )
                search_timing.opponent_greedy_action_build_calls += 1
                search_timing.opponent_greedy_action_build_s += perf_counter() - t0
                search_timing.opponent_greedy_calls += 1
                search_timing.opponent_greedy_s += perf_counter() - t_total
                return actions

            def _search_value(sim_obs: Mapping[str, Any], player: int, sim_step: int) -> float:
                t_total = perf_counter()
                t0 = perf_counter()
                sim_state = observation_to_state(
                    sim_obs,
                    config,
                    max_fleets=self.max_fleets,
                    step_count_override=sim_step,
                    num_agents_override=int(np.asarray(state.num_agents)),
                    fleet_arrival_cache=search_fleet_arrival_cache,
                )
                search_timing.value_obs_to_state_calls += 1
                search_timing.value_obs_to_state_s += perf_counter() - t0
                t0 = perf_counter()
                value = _policy_value_for_state(
                    self.policy,
                    sim_state,
                    player,
                    self.device,
                    policy_player_count=self.policy_player_count,
                    normalize_obs_to_p0=self.normalize_obs_to_p0,
                    population_member=self._population_member_for_player(player),
                )
                search_timing.value_eval_calls += 1
                search_timing.value_eval_s += perf_counter() - t0
                search_timing.value_calls += 1
                search_timing.value_s += perf_counter() - t_total
                return value

            search_runtime = SearchRuntime(
                settings=self.model_search,
                public_obs=_public_obs_for_player(obs, player=0, step_count=step_count),
                kaggle_config=config,
                game_key=game_key,
                step_count=step_count,
                num_agents=int(np.asarray(state.num_agents)),
                public_state=state,
                fleet_arrival_cache=search_fleet_arrival_cache,
                greedy_actions_for_player=_search_greedy_actions,
                value_for_player=_search_value,
                choose_launch=(
                    self._search_plan_turn_bfs
                    if str(self.model_search.mode) == MODEL_SEARCH_MODE_EGO_BFS
                    else (
                        self._search_plan_turn_sampling
                        if str(self.model_search.mode) == MODEL_SEARCH_MODE_TURN_SAMPLING
                        else self._choose_launch_via_model_search_batched_single_policy
                    )
                ),
                deadline_s=search_deadline_s,
            )
        elif _model_search_enabled(self.model_search):
            overage = _remaining_overage_s(obs)
            if overage is not None and _model_search_debug_enabled():
                _model_search_debug(
                    f"disabled step={int(step_count)} ego={int(ego_player)} "
                    f"remainingOverageTime={float(overage):.3f}s "
                    f"< min={float(self.model_search.min_overage_s):.3f}s"
                )
        # While halt-vs-launch search is available, force the main per-turn policy
        # path to greedy selection so the searched branch and executed branch align.
        actions = _build_turn_actions_torch_only(
            self.policy,
            state,
            ego_player,
            self.device,
            ship_speed=ship_speed,
            max_micro_steps=self.max_micro_steps,
            sampling_mode=(
                SAMPLING_MODE_GREEDY
                if search_active and not _model_search_mode_uses_turn_planner(str(self.model_search.mode))
                else self._sampling_mode_for_player(ego_player)
            ),
            rng=self.rng,
            n_rays=self.raycast_rays,
            samples_per_span=self.interval_samples_per_span,
            target_method=self.target_method,
            timing=timing,
            launch_tracker=self.launch_tracker,
            game_step=step_count,
            policy_player_count=self.policy_player_count,
            normalize_obs_to_p0=self.normalize_obs_to_p0,
            launch_geometry=launch_geometry,
            population_member=self._population_member_for_player(ego_player),
            search_runtime=search_runtime,
            search_launch_probability_threshold=(
                search_runtime.settings.launch_probability_threshold if search_runtime is not None else None
            ),
            search_greedy_launch_threshold=(
                search_runtime.settings.greedy_launch_threshold if search_runtime is not None else None
            ),
            deadline_s=action_deadline_s,
        )
        self._last_call_timing = timing
        if _launch_debug_enabled():
            action_summaries = [
                f"[{float(a[0]):.0f},{float(a[1]):.6f},{int(a[2])}]" for a in actions[:8]
            ]
            _launch_debug(
                f"call OUT ego={ego_player} game_step={step_count} obs.step={obs.get('step')!r} "
                f"actions={len(actions)} {action_summaries}"
                + (f" (+{len(actions) - 8} more)" if len(actions) > 8 else "")
            )
        return actions


class KaggleOrbitWarsDualPolicyAgent:
    """Submission wrapper that picks 4p or 2p policy once per episode.

    When both checkpoints are provided, the first observation decides the mode
    from ``configuration.agentCount`` when available, otherwise from distinct
    non-neutral planet owners in the public observation. There is no mid-game
    policy switching.
    """

    def __init__(
        self,
        checkpoint_4p: str | os.PathLike[str],
        checkpoint_2p: str | os.PathLike[str],
        *,
        use_student_for_search_4p: bool = False,
        use_student_for_search_2p: bool = False,
        search_main_policy_for_ego_steps_4p: int = 1,
        search_main_policy_for_ego_steps_2p: int = 1,
        device: Optional[str | torch.device] = None,
        greedy: bool | Mapping[int, bool] = False,
        sampling_mode: str | Mapping[int, str] | None = None,
        greedy_4p: bool | Mapping[int, bool] | None = None,
        greedy_2p: bool | Mapping[int, bool] | None = None,
        sampling_mode_4p: str | Mapping[int, str] | None = None,
        sampling_mode_2p: str | Mapping[int, str] | None = None,
        population_member_4p: Optional[int] = None,
        population_member_2p: Optional[int] = None,
        max_micro_steps: Optional[int] = None,
        max_fleets: int = 512,
        seed: Optional[int] = None,
        raycast_rays: Optional[int] = None,
        target_method: Optional[str] = None,
        interval_samples_per_span: Optional[int] = None,
        model_search_steps: Optional[int] = None,
        model_search_steps_4p: Optional[int] = None,
        model_search_steps_2p: Optional[int] = None,
        model_search_mode: Optional[str] = None,
        model_search_gamma: Optional[float] = None,
        model_search_adaptive_horizon: Optional[bool] = None,
        model_search_adaptive_horizon_offset: Optional[int] = None,
        model_search_min_overage_s: Optional[float] = None,
        model_search_launch_prob_threshold: Optional[float] = None,
        model_search_branch_prob_threshold: Optional[float] = None,
        model_search_max_branching: Optional[int] = None,
        model_search_branch_after_first_env_step: Optional[bool] = None,
        model_search_branch_micro_depth: Optional[int] = None,
        model_search_stop_at_turn_end: Optional[bool] = None,
        model_search_turn_end_opponent_samples: Optional[int] = None,
        model_search_turn_sampling_max_samples: Optional[int] = None,
    ):
        self.checkpoint_4p = resolve_checkpoint_path(checkpoint_4p)
        self.checkpoint_2p = resolve_checkpoint_path(checkpoint_2p)
        self.use_student_for_search_4p = bool(use_student_for_search_4p)
        self.use_student_for_search_2p = bool(use_student_for_search_2p)
        self.search_main_policy_for_ego_steps_4p = max(0, int(search_main_policy_for_ego_steps_4p))
        self.search_main_policy_for_ego_steps_2p = max(0, int(search_main_policy_for_ego_steps_2p))
        self.device = device
        self.greedy_default = greedy
        self.sampling_mode_default = sampling_mode
        self.greedy_4p = greedy if greedy_4p is None else greedy_4p
        self.greedy_2p = greedy if greedy_2p is None else greedy_2p
        self.sampling_mode_4p = sampling_mode if sampling_mode_4p is None else sampling_mode_4p
        self.sampling_mode_2p = sampling_mode if sampling_mode_2p is None else sampling_mode_2p
        self.population_member_4p = population_member_4p
        self.population_member_2p = population_member_2p
        self.max_micro_steps = max_micro_steps
        self.max_fleets = int(max_fleets)
        self.seed = seed
        self.raycast_rays = raycast_rays
        self.target_method = target_method
        self.interval_samples_per_span = interval_samples_per_span
        self.model_search_steps = model_search_steps
        self.model_search_steps_4p = model_search_steps if model_search_steps_4p is None else model_search_steps_4p
        self.model_search_steps_2p = model_search_steps if model_search_steps_2p is None else model_search_steps_2p
        self.model_search_mode = model_search_mode
        self.model_search_gamma = model_search_gamma
        self.model_search_adaptive_horizon = model_search_adaptive_horizon
        self.model_search_adaptive_horizon_offset = model_search_adaptive_horizon_offset
        self.model_search_min_overage_s = model_search_min_overage_s
        self.model_search_launch_prob_threshold = model_search_launch_prob_threshold
        self.model_search_branch_prob_threshold = model_search_branch_prob_threshold
        self.model_search_max_branching = model_search_max_branching
        self.model_search_branch_after_first_env_step = model_search_branch_after_first_env_step
        self.model_search_branch_micro_depth = model_search_branch_micro_depth
        self.model_search_stop_at_turn_end = model_search_stop_at_turn_end
        self.model_search_turn_end_opponent_samples = model_search_turn_end_opponent_samples
        self.model_search_turn_sampling_max_samples = model_search_turn_sampling_max_samples
        self._delegate: Optional[KaggleOrbitWarsAgent] = None
        self._delegate_mode: Optional[str] = None
        self._game_key: Optional[str] = None
        self._frozen_game_key: Optional[str] = None
        self._last_call_timing: Optional[KaggleAgentCallTiming] = None

    def _obs_game_key(self, obs: Mapping[str, Any]) -> str:
        step_raw = obs.get("step", obs.get("step_count", None))
        step = int(step_raw) if step_raw is not None else None
        initial = obs.get("initial_planets")
        if not initial:
            if self._frozen_game_key is not None:
                return self._frozen_game_key
            initial = obs.get("planets", [])
        arr = np.asarray(initial, dtype=np.float32)
        h = hashlib.blake2b(digest_size=16)
        h.update(arr.tobytes())
        h.update(str(obs.get("angular_velocity", 0.0)).encode("ascii", errors="ignore"))
        key = h.hexdigest()
        if step == 0 or self._frozen_game_key is None:
            self._frozen_game_key = key
        return self._frozen_game_key

    def _mode_for_obs(self, obs: Mapping[str, Any], config: Any = None) -> str:
        cfg_agents = _cfg_get(config, "agentCount", None)
        if cfg_agents is not None:
            agents = max(2, int(cfg_agents))
        else:
            agents = _infer_num_agents_from_planet_owners(obs, fallback=2)
        return "4p" if agents > 2 else "2p"

    def _build_delegate(self, mode: str) -> KaggleOrbitWarsAgent:
        use_4p = mode == "4p"
        return KaggleOrbitWarsAgent(
            self.checkpoint_4p if use_4p else self.checkpoint_2p,
            use_student_for_search=(
                self.use_student_for_search_4p if use_4p else self.use_student_for_search_2p
            ),
            search_main_policy_for_ego_steps=(
                self.search_main_policy_for_ego_steps_4p
                if use_4p
                else self.search_main_policy_for_ego_steps_2p
            ),
            device=self.device,
            greedy=self.greedy_4p if use_4p else self.greedy_2p,
            sampling_mode=self.sampling_mode_4p if use_4p else self.sampling_mode_2p,
            population_member=self.population_member_4p if use_4p else self.population_member_2p,
            max_micro_steps=self.max_micro_steps,
            max_fleets=self.max_fleets,
            seed=self.seed,
            raycast_rays=self.raycast_rays,
            target_method=self.target_method,
            interval_samples_per_span=self.interval_samples_per_span,
            model_search_steps=self.model_search_steps_4p if use_4p else self.model_search_steps_2p,
            model_search_mode=self.model_search_mode,
            model_search_gamma=self.model_search_gamma,
            model_search_adaptive_horizon=self.model_search_adaptive_horizon,
            model_search_adaptive_horizon_offset=self.model_search_adaptive_horizon_offset,
            model_search_min_overage_s=self.model_search_min_overage_s,
            model_search_launch_prob_threshold=self.model_search_launch_prob_threshold,
            model_search_branch_prob_threshold=self.model_search_branch_prob_threshold,
            model_search_max_branching=self.model_search_max_branching,
            model_search_branch_after_first_env_step=self.model_search_branch_after_first_env_step,
            model_search_branch_micro_depth=self.model_search_branch_micro_depth,
            model_search_stop_at_turn_end=self.model_search_stop_at_turn_end,
            model_search_turn_end_opponent_samples=self.model_search_turn_end_opponent_samples,
            model_search_turn_sampling_max_samples=self.model_search_turn_sampling_max_samples,
        )

    @torch.inference_mode()
    def __call__(self, obs: Mapping[str, Any], config: Any = None) -> list[list[float]]:
        key = self._obs_game_key(obs)
        if key != self._game_key:
            self._game_key = key
            self._delegate = None
            self._delegate_mode = None
        if self._delegate is None:
            self._delegate_mode = self._mode_for_obs(obs, config)
            self._delegate = self._build_delegate(self._delegate_mode)
        actions = self._delegate(obs, config)
        self._last_call_timing = getattr(self._delegate, "_last_call_timing", None)
        return actions


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _timing_logging_enabled() -> bool:
    return _env_bool("ORBIT_WARS_LOG_TIMING", False)


def _agent_internal_timing_suffix(duration_s: float, timing: Any) -> str:
    if timing is None:
        return ""
    target_detail = timing.micro_target.format_suffix()
    search_detail = timing.model_search.format_suffix()
    micro_sum = timing.micro_sum_s()
    slack = float(duration_s) - float(timing.obs_to_state_s + micro_sum)
    return (
        " internal["
        f"obs_to_state={timing.obs_to_state_s:.4f}s"
        f" micro_iters={int(timing.micro_iters)}"
        f" micro_obs_tensors={timing.micro_obs_tensors_s:.4f}s"
        f" micro_policy_fwd={timing.micro_policy_forward_s:.4f}s"
        f" micro_post_fwd={timing.micro_post_forward_s:.4f}s"
        f" micro_raycast={timing.micro_raycast_s:.4f}s"
        f" micro_target{target_detail}"
        f" micro_target={timing.micro_target_s:.4f}s"
        f" micro_book={timing.micro_book_s:.4f}s"
        f"{search_detail}"
        f" micro_sum={micro_sum:.4f}s"
        f" slack={slack:+.4f}s"
        "]"
    )


def _env_int(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    return int(raw)


def _normalize_greedy(greedy: bool | Mapping[int, bool]) -> dict[int, bool]:
    if isinstance(greedy, Mapping):
        out = {i: False for i in range(4)}
        for k, v in greedy.items():
            out[int(k)] = bool(v)
        return out
    g = bool(greedy)
    return {i: g for i in range(4)}


def _normalize_sampling_mode_value(value: str) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    aliases = {
        "sample": SAMPLING_MODE_STOCHASTIC,
        "stochastic": SAMPLING_MODE_STOCHASTIC,
        "random": SAMPLING_MODE_STOCHASTIC,
        "argmax": SAMPLING_MODE_GREEDY,
        "greedy": SAMPLING_MODE_GREEDY,
        "mixed": SAMPLING_MODE_MIXED,
        "hybrid": SAMPLING_MODE_MIXED,
    }
    mode = aliases.get(mode, mode)
    if mode not in _VALID_SAMPLING_MODES:
        raise ValueError(
            f"invalid sampling mode {value!r}; expected one of {sorted(_VALID_SAMPLING_MODES)}"
        )
    return mode


def _normalize_sampling_mode(
    sampling_mode: str | Mapping[int, str] | None,
    *,
    fallback_greedy: bool | Mapping[int, bool] = False,
) -> dict[int, str]:
    fallback_modes = {
        player: (SAMPLING_MODE_GREEDY if greedy else SAMPLING_MODE_STOCHASTIC)
        for player, greedy in _normalize_greedy(fallback_greedy).items()
    }
    if sampling_mode is None:
        return fallback_modes
    if isinstance(sampling_mode, Mapping):
        out = dict(fallback_modes)
        for k, v in sampling_mode.items():
            out[int(k)] = _normalize_sampling_mode_value(str(v))
        return out
    mode = _normalize_sampling_mode_value(str(sampling_mode))
    return {i: mode for i in range(4)}


def _greedy_from_env() -> bool | dict[int, bool]:
    per_player_set = [f"ORBIT_WARS_GREEDY_P{i}" in os.environ for i in range(4)]
    if any(per_player_set):
        fallback = _env_bool("ORBIT_WARS_GREEDY", False)
        return {
            i: _env_bool(f"ORBIT_WARS_GREEDY_P{i}", fallback) if per_player_set[i] else fallback
            for i in range(4)
        }
    return _env_bool("ORBIT_WARS_GREEDY", False)


def _sampling_mode_from_env() -> str | dict[int, str] | None:
    per_player_set = [f"ORBIT_WARS_SAMPLING_MODE_P{i}" in os.environ for i in range(4)]
    global_set = "ORBIT_WARS_SAMPLING_MODE" in os.environ
    if any(per_player_set):
        fallback = (
            _normalize_sampling_mode_value(os.environ["ORBIT_WARS_SAMPLING_MODE"])
            if global_set
            else SAMPLING_MODE_STOCHASTIC
        )
        return {
            i: (
                _normalize_sampling_mode_value(os.environ[f"ORBIT_WARS_SAMPLING_MODE_P{i}"])
                if per_player_set[i]
                else fallback
            )
            for i in range(4)
        }
    if global_set:
        return _normalize_sampling_mode_value(os.environ["ORBIT_WARS_SAMPLING_MODE"])
    return None


def _greedy_from_env_with_fallback(fallback: bool) -> bool | dict[int, bool]:
    per_player_set = [f"ORBIT_WARS_GREEDY_P{i}" in os.environ for i in range(4)]
    if any(per_player_set):
        return {
            i: _env_bool(f"ORBIT_WARS_GREEDY_P{i}", fallback) if per_player_set[i] else fallback
            for i in range(4)
        }
    return fallback


def _sampling_mode_from_env_with_fallback(fallback: str) -> str | dict[int, str]:
    per_player_set = [f"ORBIT_WARS_SAMPLING_MODE_P{i}" in os.environ for i in range(4)]
    if any(per_player_set):
        return {
            i: (
                _normalize_sampling_mode_value(os.environ[f"ORBIT_WARS_SAMPLING_MODE_P{i}"])
                if per_player_set[i]
                else fallback
            )
            for i in range(4)
        }
    return fallback


def _dual_greedy_from_env() -> tuple[bool | dict[int, bool], bool | dict[int, bool]]:
    greedy_default = _greedy_from_env()
    greedy_4p = greedy_default
    greedy_2p = greedy_default
    if "ORBIT_WARS_GREEDY_4P" in os.environ:
        greedy_4p = _greedy_from_env_with_fallback(_env_bool("ORBIT_WARS_GREEDY_4P", False))
    if "ORBIT_WARS_GREEDY_2P" in os.environ:
        greedy_2p = _greedy_from_env_with_fallback(_env_bool("ORBIT_WARS_GREEDY_2P", False))
    return greedy_4p, greedy_2p


def _dual_sampling_mode_from_env() -> tuple[str | dict[int, str] | None, str | dict[int, str] | None]:
    sampling_mode_default = _sampling_mode_from_env()
    sampling_mode_4p = sampling_mode_default
    sampling_mode_2p = sampling_mode_default
    if "ORBIT_WARS_SAMPLING_MODE_4P" in os.environ:
        sampling_mode_4p = _sampling_mode_from_env_with_fallback(
            _normalize_sampling_mode_value(os.environ["ORBIT_WARS_SAMPLING_MODE_4P"])
        )
    if "ORBIT_WARS_SAMPLING_MODE_2P" in os.environ:
        sampling_mode_2p = _sampling_mode_from_env_with_fallback(
            _normalize_sampling_mode_value(os.environ["ORBIT_WARS_SAMPLING_MODE_2P"])
        )
    return sampling_mode_4p, sampling_mode_2p


def _normalize_population_member(
    member: Optional[int],
    *,
    population_size: int,
    context: str,
) -> Optional[int]:
    if population_size <= 1:
        return None
    selected = 0 if member is None else int(member)
    if selected < 0 or selected >= population_size:
        raise ValueError(
            f"{context}: population member {selected} out of range for population_size={population_size}"
        )
    return selected


def _normalize_population_members(
    members: Optional[int | Mapping[int, int]],
    *,
    population_size: int,
    context: str,
) -> dict[int, int]:
    if population_size <= 1:
        return {}
    fallback = 0
    out: dict[int, int] = {}
    if isinstance(members, Mapping):
        if -1 in members:
            fallback = int(members[-1])
        for player, member in members.items():
            if int(player) == -1:
                continue
            out[int(player)] = _normalize_population_member(
                int(member),
                population_size=population_size,
                context=f"{context}:player{int(player)}",
            )
    elif members is not None:
        fallback = int(members)
    fallback = _normalize_population_member(
        fallback,
        population_size=population_size,
        context=context,
    )
    out[-1] = 0 if fallback is None else int(fallback)
    return out


def _population_members_single_from_env() -> Optional[int | dict[int, int]]:
    fallback = _env_int("ORBIT_WARS_MEMBER")
    per_player = [f"ORBIT_WARS_MEMBER_P{i}" in os.environ for i in range(4)]
    if any(per_player):
        out: dict[int, int] = {}
        if fallback is not None:
            out[-1] = int(fallback)
        for i in range(4):
            value = _env_int(f"ORBIT_WARS_MEMBER_P{i}")
            if value is not None:
                out[i] = int(value)
        return out
    return fallback


def _population_members_dual_from_env() -> tuple[Optional[int], Optional[int]]:
    fallback = _env_int("ORBIT_WARS_MEMBER")
    member_4p = _env_int("ORBIT_WARS_MEMBER_4P")
    member_2p = _env_int("ORBIT_WARS_MEMBER_2P")
    return (
        fallback if member_4p is None else int(member_4p),
        fallback if member_2p is None else int(member_2p),
    )


_AGENT: Optional[KaggleOrbitWarsAgent | KaggleOrbitWarsDualPolicyAgent] = None
_ERROR_REPORTED = False


def get_last_agent_call_timing() -> Optional[KaggleAgentCallTiming]:
    """Timing from the last successful ``KaggleOrbitWarsAgent.__call__`` (same global as ``agent``)."""

    inst = _AGENT
    if inst is None:
        return None
    return getattr(inst, "_last_call_timing", None)


def _report_once(exc: BaseException) -> None:
    global _ERROR_REPORTED
    if not _ERROR_REPORTED:
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        _ERROR_REPORTED = True


def agent(obs: Mapping[str, Any], config: Any = None) -> list[list[float]]:
    """Kaggle entry point.

    Single-policy: set ``ORBIT_WARS_CHECKPOINT`` (default ``checkpoint.pt``).
    Optionally set ``ORBIT_WARS_USE_STUDENT_FOR_SEARCH=1`` to use the checkpoint's
    embedded student model for search rollouts only, and optionally set
    ``ORBIT_WARS_SEARCH_MAIN_POLICY_FOR_EGO_STEPS=N`` to keep using the main model
    for the ego seat during the first ``N`` simulated search env steps.

    Two-checkpoint submission mode: set ``ORBIT_WARS_CHECKPOINT_4P`` and
    ``ORBIT_WARS_CHECKPOINT_2P``. Optionally set
    ``ORBIT_WARS_USE_STUDENT_FOR_SEARCH_4P=1`` and/or
    ``ORBIT_WARS_USE_STUDENT_FOR_SEARCH_2P=1``, plus
    ``ORBIT_WARS_SEARCH_MAIN_POLICY_FOR_EGO_STEPS_4P=N`` and/or
    ``ORBIT_WARS_SEARCH_MAIN_POLICY_FOR_EGO_STEPS_2P=N``. The first observation
    picks the 4p or 2p policy for the whole episode; there is no mid-game switching.
    """

    global _AGENT
    if _AGENT is None:
        device = os.environ.get("ORBIT_WARS_DEVICE")
        greedy = _greedy_from_env()
        greedy_4p, greedy_2p = _dual_greedy_from_env()
        sampling_mode = _sampling_mode_from_env()
        sampling_mode_4p, sampling_mode_2p = _dual_sampling_mode_from_env()
        model_search_steps_4p, model_search_steps_2p = _dual_model_search_steps_from_env()
        population_members = _population_members_single_from_env()
        population_member_4p, population_member_2p = _population_members_dual_from_env()
        seed_raw = os.environ.get("ORBIT_WARS_AGENT_SEED")
        seed = int(seed_raw) if seed_raw is not None else None
        rays_raw = os.environ.get("ORBIT_WARS_RAYCAST_RAYS")
        rays = int(rays_raw) if rays_raw is not None else None
        target_method = os.environ.get("ORBIT_WARS_TARGET_METHOD")
        interval_samples_raw = os.environ.get("ORBIT_WARS_INTERVAL_SAMPLES")
        interval_samples = int(interval_samples_raw) if interval_samples_raw is not None else None
        max_micro_raw = os.environ.get("ORBIT_WARS_MAX_MICRO_STEPS")
        max_micro_steps = int(max_micro_raw) if max_micro_raw is not None else None
        ckpt_4p = os.environ.get("ORBIT_WARS_CHECKPOINT_4P")
        ckpt_2p = os.environ.get("ORBIT_WARS_CHECKPOINT_2P")
        use_student_search = _env_bool("ORBIT_WARS_USE_STUDENT_FOR_SEARCH", False)
        use_student_search_4p = _env_bool("ORBIT_WARS_USE_STUDENT_FOR_SEARCH_4P", use_student_search)
        use_student_search_2p = _env_bool("ORBIT_WARS_USE_STUDENT_FOR_SEARCH_2P", use_student_search)
        search_main_policy_for_ego_steps_raw = os.environ.get("ORBIT_WARS_SEARCH_MAIN_POLICY_FOR_EGO_STEPS")
        search_main_policy_for_ego_steps = (
            max(0, int(search_main_policy_for_ego_steps_raw))
            if search_main_policy_for_ego_steps_raw is not None
            else 1
        )
        search_main_policy_for_ego_steps_4p_raw = os.environ.get(
            "ORBIT_WARS_SEARCH_MAIN_POLICY_FOR_EGO_STEPS_4P"
        )
        search_main_policy_for_ego_steps_2p_raw = os.environ.get(
            "ORBIT_WARS_SEARCH_MAIN_POLICY_FOR_EGO_STEPS_2P"
        )
        search_main_policy_for_ego_steps_4p = (
            max(0, int(search_main_policy_for_ego_steps_4p_raw))
            if search_main_policy_for_ego_steps_4p_raw is not None
            else search_main_policy_for_ego_steps
        )
        search_main_policy_for_ego_steps_2p = (
            max(0, int(search_main_policy_for_ego_steps_2p_raw))
            if search_main_policy_for_ego_steps_2p_raw is not None
            else search_main_policy_for_ego_steps
        )
        try:
            if ckpt_4p and ckpt_2p:
                _AGENT = KaggleOrbitWarsDualPolicyAgent(
                    resolve_checkpoint_path(ckpt_4p),
                    resolve_checkpoint_path(ckpt_2p),
                    use_student_for_search_4p=use_student_search_4p,
                    use_student_for_search_2p=use_student_search_2p,
                    search_main_policy_for_ego_steps_4p=search_main_policy_for_ego_steps_4p,
                    search_main_policy_for_ego_steps_2p=search_main_policy_for_ego_steps_2p,
                    device=device,
                    greedy=greedy,
                    sampling_mode=sampling_mode,
                    greedy_4p=greedy_4p,
                    greedy_2p=greedy_2p,
                    sampling_mode_4p=sampling_mode_4p,
                    sampling_mode_2p=sampling_mode_2p,
                    population_member_4p=population_member_4p,
                    population_member_2p=population_member_2p,
                    max_micro_steps=max_micro_steps,
                    seed=seed,
                    raycast_rays=rays,
                    target_method=target_method,
                    interval_samples_per_span=interval_samples,
                    model_search_steps_4p=model_search_steps_4p,
                    model_search_steps_2p=model_search_steps_2p,
                )
            else:
                ckpt = resolve_checkpoint_path(
                    os.environ.get("ORBIT_WARS_CHECKPOINT", DEFAULT_CHECKPOINT)
                )
                _AGENT = KaggleOrbitWarsAgent(
                    ckpt,
                    use_student_for_search=use_student_search,
                    search_main_policy_for_ego_steps=search_main_policy_for_ego_steps,
                    device=device,
                    greedy=greedy,
                    sampling_mode=sampling_mode,
                    population_member=population_members,
                    max_micro_steps=max_micro_steps,
                    seed=seed,
                    raycast_rays=rays,
                    target_method=target_method,
                    interval_samples_per_span=interval_samples,
                )
        except Exception as exc:
            _report_once(exc)
            return []
    before_overage = obs.get("remainingOverageTime", None)
    act_timeout = _cfg_get(config, "actTimeout", None)
    t0 = perf_counter()
    try:
        actions = _AGENT(obs, config)
    except Exception as exc:
        _report_once(exc)
        return []
    dt = perf_counter() - t0
    if _timing_logging_enabled():
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
        internal_suffix = _agent_internal_timing_suffix(dt, get_last_agent_call_timing())
        print(
            f"[timing] step={obs.get('step', obs.get('step_count', '?'))} "
            f"player={obs.get('player', '?')} duration={dt:.6f}s"
            f"{internal_suffix}{timeout_suffix}{overage_suffix}",
            file=sys.stderr,
            flush=True,
        )
    return actions

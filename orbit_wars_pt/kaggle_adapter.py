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
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, NamedTuple, Optional

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
    adapt_legacy_value_heads_for_model,
    infer_value_head_count_from_state_dict,
)


DEFAULT_CHECKPOINT = "checkpoint.pt"
DEFAULT_CHECKPOINT_4P = "checkpoint_4p.pt"
DEFAULT_CHECKPOINT_2P = "checkpoint_2p.pt"
DEFAULT_RAYCAST_RAYS = 256
DEFAULT_INTERVAL_SAMPLES_PER_SPAN = 9
DEFAULT_TARGET_METHOD = "rays"
DEFAULT_MAX_ACTIONS = 64
DEFAULT_CPU_THREADS = 0
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
    cache_hits: int = 0
    cache_misses: int = 0
    branch_rollouts: int = 0
    branch_rollout_s: float = 0.0
    rollout_steps: int = 0
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

    def format_suffix(self) -> str:
        if self.choose_calls <= 0:
            return ""
        return (
            " model_search{"
            f"choose×{self.choose_calls}={self.choose_s:.4f}s "
            f"cache(h={self.cache_hits} m={self.cache_misses}) "
            f"rollouts×{self.branch_rollouts}={self.branch_rollout_s:.4f}s "
            f"steps={self.rollout_steps} "
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
            f" apply={self.batch_apply_s:.4f}) "
            f"obs2state×{self.state_rebuild_calls}={self.state_rebuild_s:.4f}s "
            f"kaggle_step×{self.kaggle_step_calls}={self.kaggle_step_s:.4f}s "
            f"value×{self.value_calls}={self.value_s:.4f}s"
            f"(obs2state={self.value_obs_to_state_s:.4f}"
            f" select={self.value_policy_select_s:.4f}"
            f" eval={self.value_eval_s:.4f})"
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
    adaptive_horizon: bool = False
    adaptive_horizon_offset: int = 2
    min_overage_s: float = 15.0


def _model_search_enabled(settings: ModelSearchSettings) -> bool:
    return bool(settings.adaptive_horizon) or int(settings.horizon_steps) > 0


def _remaining_overage_s(obs: Mapping[str, Any]) -> float | None:
    if "remainingOverageTime" not in obs:
        return None
    return float(obs.get("remainingOverageTime", 0.0))


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

    return RewardSettings(
        ship_mass_share_coef=_get_float("reward_ship_mass_share_coef", 1.0),
        production_share_coef=_get_float("reward_production_share_coef", 0.0),
        terminal_win_loss_coef=_get_float("reward_terminal_win_loss_coef", 0.0),
        terminal_loss=_get_float("reward_terminal_loss", -1.0),
        terminal_draw=_get_float("reward_terminal_draw", 0.0),
        terminal_win=_get_float("reward_terminal_win", 1.0),
        time_bonus_coef=_get_float("reward_time_bonus_coef", 0.0),
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
        os.environ.setdefault(name, str(n_threads))
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
    comet_slots: np.ndarray,
    num_agents: int,
    step_count: int,
    angular_velocity: float,
    collision_rank: np.ndarray,
    *,
    ship_speed: float = 6.0,
    horizon: int = INCOMING_TA_BINS,
    per_fleet_arrival: Optional[np.ndarray] = None,
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

    p = planets.copy()
    pa = planet_active.copy()
    ia = initial_active.copy()
    cpi = comet_path_index.copy()

    for t in range(min(horizon, INCOMING_TA_BINS)):
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
    geometry = _interval_geometry_mode()

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
    )
    sweep_result = _sweep_interval_from_geometry(
        geom,
        int(planets.shape[0]),
        target_timing=target_timing,
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
) -> SearchFirstContactTargets:
    from orbit_wars_pt.tangent_geometry_np import tangent_hit_time_polyline

    if launch_geometry is None:
        planets = np.asarray(state.planets, dtype=np.float64)
        current_active = np.asarray(state.planet_active, dtype=bool)
    else:
        planets = np.asarray(launch_geometry.planets, dtype=np.float64)
        current_active = np.asarray(launch_geometry.planet_active, dtype=bool)

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

    out_angle = np.zeros((MAX_PLANETS,), dtype=np.float64)
    valid = np.zeros((MAX_PLANETS,), dtype=np.bool_)
    hit_eta = np.full((MAX_PLANETS,), 500.0, dtype=np.float32)
    if p0_by_tick.shape[0] > 0:
        for target in range(MAX_PLANETS):
            if target == int(origin_idx) or not bool(current_active[target]):
                continue
            if not bool(np.any(active_by_tick[:, target])):
                continue
            pts = np.concatenate(
                [
                    np.asarray(p0_by_tick[:, target, :], dtype=np.float64),
                    np.asarray(p1_by_tick[-1:, target, :], dtype=np.float64),
                ],
                axis=0,
            )
            hit = tangent_hit_time_polyline(
                pts,
                float(radii[target]),
                origin_xy,
                float(speed),
                float(origin_radius) + 0.1,
                float(max(0, pts.shape[0] - 1)),
                mode="external",
                return_all=False,
            )
            if hit is None:
                continue
            hit_t, _kind, angle = hit
            out_angle[target] = float(angle % (2.0 * math.pi))
            hit_eta[target] = float(hit_t)
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
        env_hit_tick = float(planned.policy_hit_tick)
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


@dataclass(frozen=True)
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
    choose_launch: Any = None


@dataclass
class CachedSearchTransition:
    public_obs: dict[str, Any]
    state: OrbitWarsState
    step_count: int
    step_reward: float
    done: bool


@dataclass
class CachedSearchRollout:
    game_key: str
    ego_player: int
    root_public_obs: dict[str, Any]
    root_state: OrbitWarsState
    root_step_count: int
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
    greedy: bool = False,
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

    with torch.inference_mode():
        for _ in range(max_micro_steps):
            if timing is not None:
                timing.micro_iters += 1

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
            use_search = (
                search_runtime is not None
                and _model_search_enabled(search_runtime.settings)
                and int(micro_idx) == 0
            )
            halt_logits = out["halt_logits"][0]
            if use_search:
                halt_action = 0
            elif greedy:
                halt_action = int(torch.argmax(halt_logits, dim=-1).item())
            else:
                halt_probs = torch.softmax(halt_logits, dim=-1)
                halt_action = int(torch.multinomial(halt_probs, 1, generator=rng).item())
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
            if greedy or use_search:
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
                if greedy or use_search:
                    target_choice = int(torch.argmax(combined_target).item())
                else:
                    target_probs = torch.softmax(combined_target, dim=-1)
                    target_choice = int(torch.multinomial(target_probs, 1, generator=rng).item())
                if target_choice == MAX_PLANETS:
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
                if greedy or use_search:
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
            if use_search:
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
                        greedy=True,
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
        comet_slots,
        num_agents,
        step_count,
        angular_velocity,
        planet_collision_rank,
        ship_speed=float(_cfg_get(config, "shipSpeed", 6.0)),
        horizon=INCOMING_TA_BINS,
        per_fleet_arrival=fleet_forecast_arrival,
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


def _infer_policy_kwargs(payload: Any) -> dict[str, Any]:
    training_args = payload.get("training_args", {}) if isinstance(payload, Mapping) else {}
    policy_state = payload.get("policy", payload) if isinstance(payload, Mapping) else payload
    kwargs = {
        "d_model": int(training_args.get("d_model", 384)),
        "n_heads": int(training_args.get("n_heads", 8)),
        "n_layers": int(training_args.get("n_layers", 4)),
        "activation_checkpointing": False,
        "population_size": int(training_args.get("population_size", 1)),
        "rope_dims": int(training_args.get("rope_dims", 3)),
        "value_head_count": int(training_args.get("value_head_count", 1)),
        "target_abort_enabled": bool(training_args.get("target_abort_enabled", False)),
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
            if "abort_head." in str(key):
                kwargs["target_abort_enabled"] = True
            if key.startswith("blocks."):
                try:
                    layer_ids.append(int(key.split(".")[1]))
                except (IndexError, ValueError):
                    pass
            elif key.startswith("shared_blocks."):
                try:
                    shared_layer_ids.append(int(key.split(".")[1]))
                except (IndexError, ValueError):
                    pass
            elif key.startswith("population_tails."):
                try:
                    pop_ids.append(int(key.split(".")[1]))
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
            "policy": policy_state,
            "training_args": _checkpoint_training_args(payload),
        }
    else:
        policy_state = payload
        payload_for_kwargs = payload
    policy = OrbitWarsPolicy(**_infer_policy_kwargs(payload_for_kwargs)).to(torch_device)
    policy_state_adapted, _ = adapt_legacy_value_heads_for_model(policy_state, policy)
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
        device: Optional[str | torch.device] = None,
        policy_key: str = "policy",
        greedy: bool | Mapping[int, bool] = False,
        population_member: Optional[int | Mapping[int, int]] = None,
        max_micro_steps: Optional[int] = None,
        max_fleets: int = 512,
        seed: Optional[int] = None,
        raycast_rays: Optional[int] = None,
        target_method: Optional[str] = None,
        interval_samples_per_span: Optional[int] = None,
        model_search_steps: Optional[int] = None,
        model_search_gamma: Optional[float] = None,
        model_search_adaptive_horizon: Optional[bool] = None,
        model_search_adaptive_horizon_offset: Optional[int] = None,
        model_search_min_overage_s: Optional[float] = None,
    ):
        _configure_cpu_threads()
        self.checkpoint_path = resolve_checkpoint_path(checkpoint_path)
        self.policy_key = str(policy_key)
        self.policy, self.device, training_args = load_policy(
            self.checkpoint_path,
            device=device,
            policy_key=self.policy_key,
        )
        self.policy = _maybe_compile_policy_batched_forward_for_inference(self.policy)
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
        )
        self.population_size = int(training_args.get("population_size", 1))
        self._population_member_by_player = _normalize_population_members(
            population_member,
            population_size=self.population_size,
            context="single",
        )
        self.normalize_obs_to_p0 = bool(training_args.get("normalize_obs_to_p0", False))
        self.policy_player_count = 4 if int(training_args.get("num_agents", 2)) > 2 else 2
        self._greedy_by_player = _normalize_greedy(greedy)
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
        warn_oob = os.environ.get("ORBIT_WARS_WARN_OOB_LAUNCHES", "1").lower() not in {"0", "false", "no", "off"}
        self.launch_tracker = FleetLaunchDebugTracker(
            warn_oob=warn_oob,
            warn_forecast_mismatch=_warn_forecast_mismatch_enabled(),
            warn_unmatched_fleet=_warn_unmatched_fleet_enabled(),
        )
        self._search_rollout_cache: Optional[CachedSearchRollout] = None
        self._search_cache_hits: int = 0
        self._search_cache_misses: int = 0

    def _population_member_for_player(self, player: int) -> Optional[int]:
        if self.population_size <= 1:
            return None
        return int(self._population_member_by_player.get(int(player), self._population_member_by_player[-1]))

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
    ) -> list[float]:
        if not states:
            return []
        batch = _obs_tensors_for_states(
            states,
            players,
            self.device,
            policy_player_count=self.policy_player_count,
            target_abort_enabled=bool(getattr(self.policy, "target_abort_enabled", False)),
            normalize_obs_to_p0=self.normalize_obs_to_p0,
        )
        population_members = [self._population_member_for_player(player) for player in players]
        population_idx = None
        if any(member is not None for member in population_members):
            population_idx = torch.tensor(
                [0 if member is None else int(member) for member in population_members],
                device=self.device,
                dtype=torch.long,
            )
        with torch.inference_mode():
            out = _policy_forward_inference(
                self.policy,
                batch,
                population_idx=population_idx,
            )
        values = out["value"].reshape(len(states), -1)[:, 0]
        return [float(v.item()) for v in values]

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
    ) -> tuple[list[float], list[list[CachedSearchTransition]]]:
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

            joint_actions_by_branch: list[list[list[float]]] = [
                [[] for _ in range(int(runtime.num_agents))],
                [[] for _ in range(int(runtime.num_agents))],
            ]
            seat_plans: list[_BatchedSearchSeatPlan] = []
            branch_launch_geometry = [
                _launch_geometry_from_obs(branch_public_obs[0], runtime.kaggle_config),
                _launch_geometry_from_obs(branch_public_obs[1], runtime.kaggle_config),
            ]

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

            if seat_plans:
                joint_actions_by_branch = self._plan_joint_actions_batched_single_policy(
                    seat_plans=seat_plans,
                    branch_joint_actions=joint_actions_by_branch,
                    branch_launch_geometry=branch_launch_geometry,
                    sim_step=int(branch_steps[0]),
                    ship_speed=float(_cfg_get(runtime.kaggle_config, "shipSpeed", 6.0)),
                    timing=timing,
                )

            for branch_idx in range(2):
                if branch_done[branch_idx]:
                    continue
                ratios_pre = _reward_mix_ratios_np(branch_states_pre[branch_idx], reward)
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
                branch_public_obs[branch_idx] = _public_obs_from_sim_state(
                    branch_sim_states[branch_idx],
                    step_count=int(branch_steps[branch_idx]),
                )
                if timing is not None:
                    t0 = perf_counter()
                state_post = observation_to_state(
                    branch_public_obs[branch_idx],
                    runtime.kaggle_config,
                    max_fleets=self.max_fleets,
                    step_count_override=int(branch_steps[branch_idx]),
                    num_agents_override=int(runtime.num_agents),
                )
                if timing is not None:
                    timing.state_rebuild_calls += 1
                    timing.state_rebuild_s += perf_counter() - t0
                step_reward = _reward_delta_np(branch_states_pre[branch_idx], state_post, ratios_pre, reward)
                branch_scores[branch_idx] += discount * float(step_reward[int(ego_player)])
                branch_states_pre[branch_idx] = state_post
                branch_traces[branch_idx].append(
                    CachedSearchTransition(
                        public_obs=copy.deepcopy(branch_public_obs[branch_idx]),
                        state=state_post,
                        step_count=int(branch_steps[branch_idx]),
                        step_reward=float(step_reward[int(ego_player)]),
                        done=bool(np.asarray(state_post.done)),
                    )
                )
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

        return branch_scores, branch_traces

    def _evaluate_greedy_continuation_from_state(
        self,
        runtime: SearchRuntime,
        *,
        ego_player: int,
        current_public_obs: Mapping[str, Any],
        current_state: OrbitWarsState,
        current_step: int,
        rollout_horizon: int,
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
            joint_actions = self._plan_joint_actions_batched_single_policy(
                seat_plans=seat_plans,
                branch_joint_actions=[[[] for _ in range(int(runtime.num_agents))]],
                branch_launch_geometry=[_launch_geometry_from_obs(branch_public_obs, runtime.kaggle_config)],
                sim_step=int(branch_step),
                ship_speed=float(_cfg_get(runtime.kaggle_config, "shipSpeed", 6.0)),
                timing=timing,
            )[0]
            ratios_pre = _reward_mix_ratios_np(branch_state_pre, reward)
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
            branch_public_obs = _public_obs_from_sim_state(branch_sim_state, step_count=int(branch_step))
            if timing is not None:
                t0 = perf_counter()
            state_post = observation_to_state(
                branch_public_obs,
                runtime.kaggle_config,
                max_fleets=self.max_fleets,
                step_count_override=int(branch_step),
                num_agents_override=int(runtime.num_agents),
            )
            if timing is not None:
                timing.state_rebuild_calls += 1
                timing.state_rebuild_s += perf_counter() - t0
            step_reward = _reward_delta_np(branch_state_pre, state_post, ratios_pre, reward)
            total += discount * float(step_reward[int(ego_player)])
            traces.append(
                CachedSearchTransition(
                    public_obs=copy.deepcopy(branch_public_obs),
                    state=state_post,
                    step_count=int(branch_step),
                    step_reward=float(step_reward[int(ego_player)]),
                    done=bool(np.asarray(state_post.done)),
                )
            )
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
        used_transitions: list[CachedSearchTransition] = []

        for trans in cache.transitions[: max(0, int(rollout_horizon))]:
            if timing is not None:
                timing.rollout_steps += 1
            total += discount * float(trans.step_reward)
            steps_used += 1
            used_transitions.append(
                CachedSearchTransition(
                    public_obs=copy.deepcopy(trans.public_obs),
                    state=trans.state,
                    step_count=int(trans.step_count),
                    step_reward=float(trans.step_reward),
                    done=bool(trans.done),
                )
            )
            end_obs = trans.public_obs
            end_state = trans.state
            end_step = int(trans.step_count)
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
        cache: CachedSearchRollout,
        timing: ModelSearchTiming | None = None,
    ) -> tuple[str | None, list[CachedSearchTransition], list[CachedSearchTransition]]:
        if not cache.transitions:
            return None, [], []
        cached_next = cache.transitions[0].state

        probe_scores, probe_traces = self._evaluate_search_branches(
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
            rollout_horizon=1,
            branch_mask=(True, True),
            timing=timing,
        )
        halt_probe = probe_traces[0]
        if halt_probe and _cache_state_match(halt_probe[0].state, cached_next):
            return "halt", halt_probe, []

        launch_probe = probe_traces[1]
        if launch_probe and _cache_state_match(launch_probe[0].state, cached_next):
            return "launch", halt_probe, launch_probe
        return None, halt_probe, launch_probe

    def _store_search_rollout_cache(
        self,
        runtime: SearchRuntime,
        *,
        ego_player: int,
        chose_launch: bool,
        branch_transitions: list[list[CachedSearchTransition]],
    ) -> None:
        chosen_idx = 1 if chose_launch else 0
        self._store_search_rollout_cache_from_transitions(
            runtime,
            ego_player=int(ego_player),
            transitions=branch_transitions[chosen_idx],
        )

    def _store_search_rollout_cache_from_transitions(
        self,
        runtime: SearchRuntime,
        *,
        ego_player: int,
        transitions: list[CachedSearchTransition],
    ) -> None:
        chosen = transitions
        if not chosen:
            self._search_rollout_cache = None
            return
        root = chosen[0]
        self._search_rollout_cache = CachedSearchRollout(
            game_key=str(runtime.game_key),
            ego_player=int(ego_player),
            root_public_obs=copy.deepcopy(root.public_obs),
            root_state=root.state,
            root_step_count=int(root.step_count),
            transitions=[
                CachedSearchTransition(
                    public_obs=copy.deepcopy(trans.public_obs),
                    state=trans.state,
                    step_count=int(trans.step_count),
                    step_reward=float(trans.step_reward),
                    done=bool(trans.done),
                )
                for trans in chosen[1:]
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
        timing: ModelSearchTiming | None = None,
    ) -> list[list[list[float]]]:
        t_plan = perf_counter() if timing is not None else 0.0
        active = [plan for plan in seat_plans if int(plan.micro_idx) < int(plan.max_micro_steps)]
        while active:
            if timing is not None:
                timing.batch_plan_rounds += 1
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
            batch = _obs_tensors_for_states(
                virt_states,
                [int(plan.player) for plan in active],
                self.device,
                policy_player_count=self.policy_player_count,
                target_abort_enabled=bool(getattr(self.policy, "target_abort_enabled", False)),
                normalize_obs_to_p0=self.normalize_obs_to_p0,
                obs_timing=timing.batch_obs if timing is not None else None,
            )
            if timing is not None:
                timing.batch_obs_tensors_s += perf_counter() - t_obs
                t0 = perf_counter()
            population_members = [self._population_member_for_player(int(plan.player)) for plan in active]
            population_idx = None
            if any(member is not None for member in population_members):
                population_idx = torch.tensor(
                    [0 if member is None else int(member) for member in population_members],
                    device=self.device,
                    dtype=torch.long,
                )
            with torch.inference_mode():
                out = _policy_forward_inference(
                    self.policy,
                    batch,
                    population_idx=population_idx,
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
                halt_action = int(torch.argmax(halt_logits, dim=-1).item())
                if halt_action == 1:
                    continue

                flat_mask = out["origin_frac_mask"].flatten(start_dim=1)[row]
                if not bool(flat_mask.any().item()):
                    continue
                flat_logits = out["origin_frac_logits"].flatten(start_dim=1)[row]
                masked_origin_frac = flat_logits.masked_fill(~flat_mask, -1e4)
                origin_frac_flat = int(torch.argmax(masked_origin_frac).item())
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
            target_pop_idx = None
            if population_idx is not None:
                target_pop_idx = population_idx.index_select(0, row_idx_t)
            target_logits_b = self.policy.target_logits_for_origin_fraction(
                out["planet_hidden"].index_select(0, row_idx_t),
                origin_idx_t,
                frac_idx_t,
                fleet_size=fleet_size_t,
                target_eta=target_eta_t,
                target_ships=target_ships_t,
                population_idx=target_pop_idx,
            )
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
                    target_choice = int(torch.argmax(combined_target).item())
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
                    sorted_choices = torch.argsort(
                        target_logits_b[idx].masked_fill(~target_mask, -1e4),
                        descending=True,
                    ).tolist()

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
        cache = self._search_cache_match(runtime, ego_player=int(ego_player))
        cache_branch_miss = False
        if cache is not None:
            self._search_cache_hits += 1
            if timing is not None:
                timing.cache_hits += 1
            cached_branch, _halt_probe, _launch_probe = self._identify_cached_branch(
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
                rollout_horizon=int(rollout_horizon),
                cache=cache,
                timing=timing,
            )
            if cached_branch == "halt":
                halt_score, halt_transitions = self._score_branch_from_cache(
                    runtime,
                    ego_player=int(ego_player),
                    cache=cache,
                    rollout_horizon=int(rollout_horizon),
                    timing=timing,
                )
                launch_scores, launch_branch_traces = self._evaluate_search_branches(
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
            elif cached_branch == "launch":
                launch_score, launch_transitions = self._score_branch_from_cache(
                    runtime,
                    ego_player=int(ego_player),
                    cache=cache,
                    rollout_horizon=int(rollout_horizon),
                    timing=timing,
                )
                halt_scores, halt_branch_traces = self._evaluate_search_branches(
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
            self._store_search_rollout_cache_from_transitions(
                runtime,
                ego_player=int(ego_player),
                transitions=(launch_transitions if chose_launch else halt_transitions),
            )
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
        branch_scores, branch_traces = self._evaluate_search_branches(
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
        self._store_search_rollout_cache(
            runtime,
            ego_player=int(ego_player),
            chose_launch=chose_launch,
            branch_transitions=branch_traces,
        )
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
    ) -> list[list[float]]:
        state = observation_to_state(
            obs,
            None,
            max_fleets=self.max_fleets,
            step_count_override=step_count,
            num_agents_override=self._num_agents_for_obs(obs, None),
        )
        return _build_turn_actions_torch_only(
            self.policy,
            state,
            player,
            self.device,
            ship_speed=6.0,
            max_micro_steps=self.max_micro_steps,
            greedy=True,
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
        if _model_search_enabled(self.model_search) and _model_search_allowed_for_obs(obs, self.model_search):
            search_timing = timing.model_search

            def _search_greedy_actions(sim_obs: Mapping[str, Any], player: int, sim_step: int) -> list[list[float]]:
                t_total = perf_counter()
                t0 = perf_counter()
                sim_state = observation_to_state(
                    sim_obs,
                    config,
                    max_fleets=self.max_fleets,
                    step_count_override=sim_step,
                    num_agents_override=int(np.asarray(state.num_agents)),
                )
                search_timing.opponent_greedy_obs_to_state_calls += 1
                search_timing.opponent_greedy_obs_to_state_s += perf_counter() - t0
                t0 = perf_counter()
                actions = _build_turn_actions_torch_only(
                    self.policy,
                    sim_state,
                    player,
                    self.device,
                    ship_speed=ship_speed,
                    max_micro_steps=self.max_micro_steps,
                    greedy=True,
                    rng=self.rng,
                    n_rays=self.raycast_rays,
                    samples_per_span=self.interval_samples_per_span,
                    target_method=self.target_method,
                    timing=None,
                    launch_tracker=None,
                    game_step=sim_step,
                    policy_player_count=self.policy_player_count,
                    normalize_obs_to_p0=self.normalize_obs_to_p0,
                    launch_geometry=_launch_geometry_from_obs(sim_obs, config),
                    population_member=self._population_member_for_player(player),
                    search_runtime=None,
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
                greedy_actions_for_player=_search_greedy_actions,
                value_for_player=_search_value,
                choose_launch=self._choose_launch_via_model_search_batched_single_policy,
            )
        elif _model_search_enabled(self.model_search):
            overage = _remaining_overage_s(obs)
            if overage is not None and _model_search_debug_enabled():
                _model_search_debug(
                    f"disabled step={int(step_count)} ego={int(ego_player)} "
                    f"remainingOverageTime={float(overage):.3f}s "
                    f"< min={float(self.model_search.min_overage_s):.3f}s"
                )
        actions = _build_turn_actions_torch_only(
            self.policy,
            state,
            ego_player,
            self.device,
            ship_speed=ship_speed,
            max_micro_steps=self.max_micro_steps,
            greedy=self._greedy_by_player.get(ego_player, False),
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
            deadline_s=(
                call_t0 + max(0.0, float(_cfg_get(config, "actTimeout", 1.0)) - 0.1)
                if _cfg_get(config, "actTimeout", None) is not None
                else None
            ),
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
        device: Optional[str | torch.device] = None,
        greedy: bool | Mapping[int, bool] = False,
        greedy_4p: bool | Mapping[int, bool] | None = None,
        greedy_2p: bool | Mapping[int, bool] | None = None,
        population_member_4p: Optional[int] = None,
        population_member_2p: Optional[int] = None,
        max_micro_steps: Optional[int] = None,
        max_fleets: int = 512,
        seed: Optional[int] = None,
        raycast_rays: Optional[int] = None,
        target_method: Optional[str] = None,
        interval_samples_per_span: Optional[int] = None,
        model_search_steps: Optional[int] = None,
        model_search_gamma: Optional[float] = None,
        model_search_adaptive_horizon: Optional[bool] = None,
        model_search_adaptive_horizon_offset: Optional[int] = None,
        model_search_min_overage_s: Optional[float] = None,
    ):
        self.checkpoint_4p = resolve_checkpoint_path(checkpoint_4p)
        self.checkpoint_2p = resolve_checkpoint_path(checkpoint_2p)
        self.device = device
        self.greedy_default = greedy
        self.greedy_4p = greedy if greedy_4p is None else greedy_4p
        self.greedy_2p = greedy if greedy_2p is None else greedy_2p
        self.population_member_4p = population_member_4p
        self.population_member_2p = population_member_2p
        self.max_micro_steps = max_micro_steps
        self.max_fleets = int(max_fleets)
        self.seed = seed
        self.raycast_rays = raycast_rays
        self.target_method = target_method
        self.interval_samples_per_span = interval_samples_per_span
        self.model_search_steps = model_search_steps
        self.model_search_gamma = model_search_gamma
        self.model_search_adaptive_horizon = model_search_adaptive_horizon
        self.model_search_adaptive_horizon_offset = model_search_adaptive_horizon_offset
        self.model_search_min_overage_s = model_search_min_overage_s
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
            device=self.device,
            greedy=self.greedy_4p if use_4p else self.greedy_2p,
            population_member=self.population_member_4p if use_4p else self.population_member_2p,
            max_micro_steps=self.max_micro_steps,
            max_fleets=self.max_fleets,
            seed=self.seed,
            raycast_rays=self.raycast_rays,
            target_method=self.target_method,
            interval_samples_per_span=self.interval_samples_per_span,
            model_search_steps=self.model_search_steps,
            model_search_gamma=self.model_search_gamma,
            model_search_adaptive_horizon=self.model_search_adaptive_horizon,
            model_search_adaptive_horizon_offset=self.model_search_adaptive_horizon_offset,
            model_search_min_overage_s=self.model_search_min_overage_s,
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


def _greedy_from_env() -> bool | dict[int, bool]:
    per_player_set = [f"ORBIT_WARS_GREEDY_P{i}" in os.environ for i in range(4)]
    if any(per_player_set):
        fallback = _env_bool("ORBIT_WARS_GREEDY", False)
        return {
            i: _env_bool(f"ORBIT_WARS_GREEDY_P{i}", fallback) if per_player_set[i] else fallback
            for i in range(4)
        }
    return _env_bool("ORBIT_WARS_GREEDY", False)


def _greedy_from_env_with_fallback(fallback: bool) -> bool | dict[int, bool]:
    per_player_set = [f"ORBIT_WARS_GREEDY_P{i}" in os.environ for i in range(4)]
    if any(per_player_set):
        return {
            i: _env_bool(f"ORBIT_WARS_GREEDY_P{i}", fallback) if per_player_set[i] else fallback
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

    Two-checkpoint submission mode: set ``ORBIT_WARS_CHECKPOINT_4P`` and
    ``ORBIT_WARS_CHECKPOINT_2P``. The first observation picks the 4p or 2p
    policy for the whole episode; there is no mid-game switching.
    """

    global _AGENT
    if _AGENT is None:
        device = os.environ.get("ORBIT_WARS_DEVICE")
        greedy = _greedy_from_env()
        greedy_4p, greedy_2p = _dual_greedy_from_env()
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
        try:
            if ckpt_4p and ckpt_2p:
                _AGENT = KaggleOrbitWarsDualPolicyAgent(
                    resolve_checkpoint_path(ckpt_4p),
                    resolve_checkpoint_path(ckpt_2p),
                    device=device,
                    greedy=greedy,
                    greedy_4p=greedy_4p,
                    greedy_2p=greedy_2p,
                    population_member_4p=population_member_4p,
                    population_member_2p=population_member_2p,
                    max_micro_steps=max_micro_steps,
                    seed=seed,
                    raycast_rays=rays,
                    target_method=target_method,
                    interval_samples_per_span=interval_samples,
                )
            else:
                ckpt = resolve_checkpoint_path(
                    os.environ.get("ORBIT_WARS_CHECKPOINT", DEFAULT_CHECKPOINT)
                )
                _AGENT = KaggleOrbitWarsAgent(
                    ckpt,
                    device=device,
                    greedy=greedy,
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

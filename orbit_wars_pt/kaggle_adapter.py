"""Adapter for running a trained policy in the official Kaggle Orbit Wars env.

The Kaggle agent API calls ``agent(obs, config)`` and expects a Python list of
``[from_planet_id, angle, num_ships]`` launches.  Training uses a fixed-table
``OrbitWarsState`` plus a PyTorch policy, so this module bridges between the two.
"""

from __future__ import annotations

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
    CENTER,
    ENTITY_CLS,
    ENTITY_COMET,
    ENTITY_PLANET,
    FEATURE_DIM_MULTI,
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
DEFAULT_CPU_THREADS = 1
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


def _warn_greedy_bool_4p(greedy: bool | Mapping[int, bool], seen: set[str], context: str) -> None:
    if isinstance(greedy, Mapping) or not bool(greedy):
        return
    _adapter_warn_once(
        seen,
        f"{context}:greedy-bool-p23",
        "greedy=True currently normalizes only players 0 and 1; "
        f"context={context} players 2 and 3 will use the adapter fallback unless set explicitly "
        "with a per-player mapping or ORBIT_WARS_GREEDY_P2/P3",
    )


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

    def micro_sum_s(self) -> float:
        return (
            self.micro_obs_tensors_s
            + self.micro_policy_forward_s
            + self.micro_post_forward_s
            + self.micro_raycast_s
            + self.micro_target_s
            + self.micro_book_s
        )


class OrbitWarsState(NamedTuple):
    planets: np.ndarray
    planet_active: np.ndarray
    initial_planets: np.ndarray
    initial_active: np.ndarray
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
    expected_fd = obs_feature_dim_for_num_agents(policy_player_count)
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
    canonical = np.where(o < ego, 2 + o, 2 + (o - 1))
    if not normalize_to_p0:
        return canonical if isinstance(owner, np.ndarray) else int(np.asarray(canonical))
    row = _NORMALIZED_OWNER_SLOT_4P[min(max(int(ego), 0), 3)]
    normalized = row[np.clip(o, 0, 3)]
    return normalized if isinstance(owner, np.ndarray) else int(np.asarray(normalized))


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
    return _raycast_targets_np(
        state,
        origin_idx,
        frac_idx,
        ship_speed=ship_speed,
        horizon=horizon,
        n_rays=n_rays,
        target_timing=target_timing,
        launch_geometry=launch_geometry,
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


def _obs_tensors_for_state(
    state: OrbitWarsState,
    ego_player: int,
    device: torch.device,
    *,
    policy_player_count: Optional[int] = None,
    normalize_obs_to_p0: bool = False,
) -> dict[str, torch.Tensor]:
    planets = np.asarray(state.planets)
    planet_active = np.asarray(state.planet_active)
    initial_planets = np.asarray(state.initial_planets)
    initial_active = np.asarray(state.initial_active)
    incoming_fleets = np.asarray(state.incoming_fleets)
    angular_velocity = float(np.asarray(state.angular_velocity))
    step_count = int(np.asarray(state.step_count))
    num_agents = int(np.asarray(state.num_agents))
    player_count = int(policy_player_count if policy_player_count is not None else num_agents)
    obs_qturns = _obs_qturns_to_p0_np(int(ego_player), player_count, normalize_obs_to_p0)
    comet_ids = np.asarray(state.comet_planet_ids)
    comet_set = set(float(x) for x in comet_ids.flatten() if int(x) >= 0)

    entity_type = np.zeros((1 + MAX_PLANETS,), dtype=np.int64)
    owner_idx = np.zeros((1 + MAX_PLANETS,), dtype=np.int64)
    fdim = obs_feature_dim_for_num_agents(player_count)
    features = np.zeros((1 + MAX_PLANETS, fdim), dtype=np.float32)
    rope_pos = np.zeros((1 + MAX_PLANETS, 3), dtype=np.float32)
    entity_mask = np.zeros((1 + MAX_PLANETS,), dtype=np.bool_)
    planet_mask = np.zeros((1 + MAX_PLANETS,), dtype=np.bool_)

    entity_type[0] = ENTITY_CLS
    owner_idx[0] = 1
    features[0, 6] = np.float32(np.clip(float(step_count) / 498.0, 0.0, 1.0))
    rope_pos[0] = np.asarray([CENTER / BOARD_SIZE, CENTER / BOARD_SIZE, 0.0], dtype=np.float32)
    entity_mask[0] = True

    incoming_net, survivor_slot = _incoming_interfleet_np(
        incoming_fleets.astype(np.float32), int(ego_player), player_count, normalize_obs_to_p0
    )
    incoming_net, survivor_slot = _hide_enemy_far_right_after_resolution(incoming_net, survivor_slot)

    for i in range(MAX_PLANETS):
        j = 1 + i
        active = bool(planet_active[i])
        pid = float(planets[i, 0])
        is_comet = pid in comet_set
        entity_type[j] = ENTITY_COMET if is_comet else ENTITY_PLANET
        owner_idx[j] = min(
            _remap_owner(float(planets[i, 1]), ego_player, player_count, normalize_obs_to_p0),
            NUM_OWNER_SLOTS - 1,
        )
        vx, vy = planet_pred_velocity(
            initial_planets[i, 2:4].astype(np.float64),
            planets[i, 2:4].astype(np.float64),
            float(planets[i, 4]),
            angular_velocity,
            step_count,
            bool(initial_active[i]),
            active,
        )
        if is_comet and active:
            group_row = np.where(comet_ids == int(pid))
            if group_row[0].size > 0:
                g, k = int(group_row[0][0]), int(group_row[1][0])
                paths = np.asarray(state.comet_paths[g, k])
                lens = int(np.asarray(state.comet_path_lengths[g, k]))
                idx = int(np.asarray(state.comet_path_index[g]))
                if lens > 1 and 0 <= idx < lens - 1:
                    vx = float(paths[idx + 1, 0] - paths[idx, 0])
                    vy = float(paths[idx + 1, 1] - paths[idx, 1])
        vx, vy = _rotate_vec_np(vx, vy, obs_qturns)

        features[j, 0] = np.log1p(max(float(planets[i, 6]), 0.0))
        features[j, 1] = float(planets[i, 5]) / 1000.0
        features[j, 2] = float(vx) / 5.0
        features[j, 3] = float(vy) / 5.0
        features[j, 4] = float(active)
        features[j, 5] = float(planets[i, 4]) / 10.0
        features[j, 8 : 8 + INCOMING_TA_BINS] = incoming_net[i].astype(np.float32)
        if fdim == FEATURE_DIM_MULTI and player_count > 2:
            oh = np.eye(NUM_OWNER_SLOTS, dtype=np.float32)[survivor_slot[i]]
            features[
                j, 8 + INCOMING_TA_BINS : 8 + INCOMING_TA_BINS + INCOMING_SURVIVOR_FLAT
            ] = oh.reshape(INCOMING_SURVIVOR_FLAT)
        if active:
            px, py = _rotate_xy_about_center_np(float(planets[i, 2]), float(planets[i, 3]), obs_qturns)
            rope_pos[j, 0] = px / BOARD_SIZE
            rope_pos[j, 1] = py / BOARD_SIZE
        entity_mask[j] = active
        planet_mask[j] = True

    def tensor(x: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(x)).to(device=device, dtype=dtype).unsqueeze(0)

    return {
        "entity_type": tensor(entity_type, torch.long),
        "owner_idx": tensor(owner_idx, torch.long),
        "features": tensor(features, torch.float32),
        "rope_pos": tensor(rope_pos, torch.float32),
        "entity_mask": tensor(entity_mask, torch.bool),
        "planet_mask": tensor(planet_mask, torch.bool),
    }


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
) -> list[list[float]]:
    planets = np.array(np.asarray(state.planets), copy=True)
    incoming_fleets = np.array(np.asarray(state.incoming_fleets), copy=True)
    planet_active = np.asarray(state.planet_active).astype(bool)
    actions: list[list[float]] = []
    planned_launches: list[PlannedLaunchAction] = []
    micro_idx = 0

    for _ in range(max_micro_steps):
        if timing is not None:
            timing.micro_iters += 1

        t0 = perf_counter()
        virt = state._replace(planets=planets, incoming_fleets=incoming_fleets)
        batch = _obs_tensors_for_state(
            virt,
            ego_player,
            device,
            policy_player_count=policy_player_count,
            normalize_obs_to_p0=normalize_obs_to_p0,
        )
        if timing is not None:
            timing.micro_obs_tensors_s += perf_counter() - t0

        t0 = perf_counter()
        population_idx = None
        if population_member is not None:
            population_idx = torch.tensor([int(population_member)], device=device, dtype=torch.long)
        out = policy(**batch, population_idx=population_idx)
        if timing is not None:
            timing.micro_policy_forward_s += perf_counter() - t0

        t0 = perf_counter()
        halt_logits = out["halt_logits"][0]
        if greedy:
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
        if greedy:
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
        target_mask = out["pair_mask"][0, o_idx].clone()
        ray_valid_t = torch.from_numpy(ray_valid).to(device=device, dtype=torch.bool)
        target_mask &= ray_valid_t
        if not bool(target_mask.any().item()):
            if timing is not None:
                timing.micro_target_s += perf_counter() - t0
            break
        masked_target = target_logits.masked_fill(~target_mask, -1e4)
        if greedy:
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

    num_agents = int(_cfg_get(config, "agentCount", 2))
    num_agents = max(num_agents, int(obs.get("player", 0)) + 1, 2)

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
    }
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


def _infer_policy_player_count_from_payload(payload: Any) -> int:
    training_args = _checkpoint_training_args(payload)
    for key in ("policy_player_count", "inference_policy_player_count"):
        if key in training_args:
            return 4 if int(training_args[key]) > 2 else 2
    policy_state = payload.get("policy", payload) if isinstance(payload, Mapping) else payload
    if isinstance(policy_state, Mapping):
        w = policy_state.get("feat_proj.weight")
        if hasattr(w, "shape") and len(w.shape) >= 2:
            feat_dim = int(w.shape[1])
            return 4 if feat_dim > 32 else 2
    if "num_agents" in training_args:
        return 4 if int(training_args["num_agents"]) > 2 else 2
    return 2


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
    training_args.setdefault(
        "policy_player_count",
        _infer_policy_player_count_from_payload(payload),
    )
    return policy, torch_device, training_args


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
    ):
        _configure_cpu_threads()
        self.checkpoint_path = resolve_checkpoint_path(checkpoint_path)
        self.policy_key = str(policy_key)
        self.policy, self.device, training_args = load_policy(
            self.checkpoint_path,
            device=device,
            policy_key=self.policy_key,
        )
        self.population_size = int(training_args.get("population_size", 1))
        self._population_member_by_player = _normalize_population_members(
            population_member,
            population_size=self.population_size,
            context="single",
        )
        self.normalize_obs_to_p0 = bool(training_args.get("normalize_obs_to_p0", False))
        self.policy_player_count = 4 if int(training_args.get("policy_player_count", 2)) > 2 else 2
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
        self._next_step_count = 0
        # Kaggle omits ``step`` on player 1's observation; after player 0 runs, mirror that value here
        # so a single shared ``KaggleOrbitWarsAgent`` (``agent()``) still builds the correct state.
        self._last_env_step: Optional[int] = None
        self._last_call_timing: Optional[KaggleAgentCallTiming] = None
        self._sanity_warnings: set[str] = set()
        self._greedy_bool_true = (not isinstance(greedy, Mapping)) and bool(greedy)
        warn_oob = os.environ.get("ORBIT_WARS_WARN_OOB_LAUNCHES", "1").lower() not in {"0", "false", "no", "off"}
        self.launch_tracker = FleetLaunchDebugTracker(
            warn_oob=warn_oob,
            warn_forecast_mismatch=_warn_forecast_mismatch_enabled(),
            warn_unmatched_fleet=_warn_unmatched_fleet_enabled(),
        )

    def _population_member_for_player(self, player: int) -> Optional[int]:
        if self.population_size <= 1:
            return None
        return int(self._population_member_by_player.get(int(player), self._population_member_by_player[-1]))

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
            self._game_key = self._obs_game_key(obs)
            return s

        key = self._obs_game_key(obs)
        if key != self._game_key:
            self._game_key = key
            self._next_step_count = 0
            self._last_env_step = None

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
        )
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
        if self._greedy_bool_true and int(np.asarray(state.num_agents)) > 2:
            _warn_greedy_bool_4p(True, self._sanity_warnings, "single")
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
    """4p policy while two or more opponents are alive; 2p policy once only one remains.

    Both checkpoints are loaded eagerly at construction so a mid-game switch does not
    pay load/JIT cost under the 1s turn timer.
    """

    def __init__(
        self,
        checkpoint_4p: str | os.PathLike[str],
        checkpoint_2p: str | os.PathLike[str],
        *,
        device: Optional[str | torch.device] = None,
        greedy: bool | Mapping[int, bool] = False,
        population_member_4p: Optional[int] = None,
        population_member_2p: Optional[int] = None,
        max_micro_steps: Optional[int] = None,
        max_fleets: int = 512,
        seed: Optional[int] = None,
        raycast_rays: Optional[int] = None,
        target_method: Optional[str] = None,
        interval_samples_per_span: Optional[int] = None,
    ):
        _configure_cpu_threads()
        self.checkpoint_4p = resolve_checkpoint_path(checkpoint_4p)
        self.checkpoint_2p = resolve_checkpoint_path(checkpoint_2p)
        self.policy_4p, self.device, training_args_4p = load_policy(self.checkpoint_4p, device=device)
        self.policy_2p, _, training_args_2p = load_policy(self.checkpoint_2p, device=self.device)
        self.population_size_4p = int(training_args_4p.get("population_size", 1))
        self.population_size_2p = int(training_args_2p.get("population_size", 1))
        self.population_member_4p = _normalize_population_member(
            population_member_4p,
            population_size=self.population_size_4p,
            context="dual:4p",
        )
        self.population_member_2p = _normalize_population_member(
            population_member_2p,
            population_size=self.population_size_2p,
            context="dual:2p",
        )
        self.normalize_obs_to_p0_4p = bool(training_args_4p.get("normalize_obs_to_p0", False))
        self.normalize_obs_to_p0_2p = bool(training_args_2p.get("normalize_obs_to_p0", False))
        self.policy_player_count_4p = 4 if int(training_args_4p.get("policy_player_count", 4)) > 2 else 2
        self.policy_player_count_2p = 4 if int(training_args_2p.get("policy_player_count", 2)) > 2 else 2
        self._greedy_by_player = _normalize_greedy(greedy)
        micro_4p = int(training_args_4p.get("max_micro_steps", DEFAULT_MAX_ACTIONS))
        micro_2p = int(training_args_2p.get("max_micro_steps", DEFAULT_MAX_ACTIONS))
        self.max_micro_steps = int(
            max_micro_steps if max_micro_steps is not None else max(micro_4p, micro_2p)
        )
        self.max_fleets = int(max_fleets)
        rays_4p = int(training_args_4p.get("first_hit_n_rays", DEFAULT_RAYCAST_RAYS))
        rays_2p = int(training_args_2p.get("first_hit_n_rays", DEFAULT_RAYCAST_RAYS))
        self.raycast_rays = int(
            raycast_rays if raycast_rays is not None else max(rays_4p, rays_2p)
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
        self._frozen_game_key: Optional[str] = None
        self._next_step_count = 0
        self._last_env_step: Optional[int] = None
        self._last_call_timing: Optional[KaggleAgentCallTiming] = None
        self._sanity_warnings: set[str] = set()
        self._greedy_bool_true = (not isinstance(greedy, Mapping)) and bool(greedy)
        warn_oob = os.environ.get("ORBIT_WARS_WARN_OOB_LAUNCHES", "1").lower() not in {"0", "false", "no", "off"}
        self.launch_tracker = FleetLaunchDebugTracker(
            warn_oob=warn_oob,
            warn_forecast_mismatch=_warn_forecast_mismatch_enabled(),
            warn_unmatched_fleet=_warn_unmatched_fleet_enabled(),
        )

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

    def _step_count_for_obs(self, obs: Mapping[str, Any]) -> int:
        step_raw = obs.get("step", obs.get("step_count", None))
        if step_raw is not None:
            s = int(step_raw)
            self._last_env_step = s
            self._next_step_count = s + 1
            self._game_key = self._obs_game_key(obs)
            return s

        key = self._obs_game_key(obs)
        if key != self._game_key:
            self._game_key = key
            self._next_step_count = 0
            self._last_env_step = None

        if self._last_env_step is not None:
            ego = int(obs.get("player", 0))
            if ego >= 2:
                _adapter_warn_once(
                    self._sanity_warnings,
                    f"dual:missing-step-mirrored:ego{ego}",
                    "Kaggle observation omitted step for player >=2; reusing last observed env step "
                    f"context=dual ego={ego} reused_step={int(self._last_env_step)}",
                )
            return int(self._last_env_step)

        step_count = self._next_step_count
        ego = int(obs.get("player", 0))
        if ego >= 2:
            _adapter_warn_once(
                self._sanity_warnings,
                f"dual:missing-step-inferred:ego{ego}",
                "Kaggle observation omitted step for player >=2 and no prior step was available; "
                f"context=dual ego={ego} inferred_step={step_count}",
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
        )
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
        live_opponents = _count_live_opponents(state, ego_player)
        use_4p_policy = live_opponents >= 2
        policy = self.policy_4p if use_4p_policy else self.policy_2p
        policy_player_count = self.policy_player_count_4p if use_4p_policy else self.policy_player_count_2p
        normalize_obs_to_p0 = self.normalize_obs_to_p0_4p if use_4p_policy else self.normalize_obs_to_p0_2p
        _check_4p_adapter_sanity(
            obs=obs,
            config=config,
            state=state,
            ego_player=ego_player,
            step_count=step_count,
            policy=policy,
            policy_player_count=policy_player_count,
            seen=self._sanity_warnings,
            context="dual",
            use_4p_policy=use_4p_policy,
            live_opponents=live_opponents,
        )
        if self._greedy_bool_true and int(np.asarray(state.num_agents)) > 2:
            _warn_greedy_bool_4p(True, self._sanity_warnings, "dual")
        actions = _build_turn_actions_torch_only(
            policy,
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
            policy_player_count=policy_player_count,
            normalize_obs_to_p0=normalize_obs_to_p0,
            launch_geometry=launch_geometry,
            population_member=self.population_member_4p if use_4p_policy else self.population_member_2p,
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
            mode = "4p" if use_4p_policy else "2p"
            _launch_debug(
                f"call OUT ego={ego_player} policy={mode} live_opponents={live_opponents} "
                f"game_step={step_count} obs.step={obs.get('step')!r} "
                f"actions={len(actions)} {action_summaries}"
                + (f" (+{len(actions) - 8} more)" if len(actions) > 8 else "")
            )
        return actions


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


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
        out = {0: False, 1: False}
        for k, v in greedy.items():
            out[int(k)] = bool(v)
        return out
    g = bool(greedy)
    return {0: g, 1: g}


def _greedy_from_env() -> bool | dict[int, bool]:
    per_player_set = [f"ORBIT_WARS_GREEDY_P{i}" in os.environ for i in range(4)]
    if any(per_player_set):
        fallback = _env_bool("ORBIT_WARS_GREEDY", False)
        return {
            i: _env_bool(f"ORBIT_WARS_GREEDY_P{i}", fallback) if per_player_set[i] else fallback
            for i in range(4)
        }
    return _env_bool("ORBIT_WARS_GREEDY", False)


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

    Dual-policy (4p FFA + 2p endgame): set ``ORBIT_WARS_CHECKPOINT_4P`` and
    ``ORBIT_WARS_CHECKPOINT_2P``.  Both are loaded at startup.
    """

    global _AGENT
    if _AGENT is None:
        device = os.environ.get("ORBIT_WARS_DEVICE")
        greedy = _greedy_from_env()
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
    try:
        return _AGENT(obs, config)
    except Exception as exc:
        _report_once(exc)
        return []

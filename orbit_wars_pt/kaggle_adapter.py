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
    FEATURE_DIM,
    FRACTIONS,
    INCOMING_TA_BINS,
    MAX_PLANETS,
    NUM_OWNER_SLOTS,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
)
from orbit_wars_pt.geometry import estimate_time_to_hit, planet_pred_velocity
from orbit_wars_pt.model import OrbitWarsPolicy


DEFAULT_CHECKPOINT = "checkpoint.pt"
DEFAULT_RAYCAST_RAYS = 256
DEFAULT_MAX_ACTIONS = 64
DEFAULT_CPU_THREADS = 1
MAX_COMET_GROUPS = 5
MAX_COMET_PATH = 40
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
    ) -> None:
        unmatched = 0
        for fid, tup in sorted(current.items()):
            if fid in self._by_fleet_id:
                continue
            owner, _x, _y, ang, from_id, ships = tup
            rec = self._match_pending(owner, from_id, ang, ships, n_rays=n_rays)
            if rec is not None:
                rec.fleet_id = fid
                self._by_fleet_id[fid] = rec
                if _launch_debug_enabled():
                    _launch_debug(f"attached fleet_id={fid} <- {rec.debug_summary()}")
            else:
                unmatched += 1
                if _launch_debug_enabled() and owner == ego_player:
                    self._debug_explain_fleet_match(
                        owner, from_id, ang, ships, n_rays=n_rays, fleet_id=fid
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

        self._attach_pending_to_observed_fleets(current, n_rays=n_rays, ego_player=ego_player)

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
        if _launch_debug_enabled():
            _launch_debug(f"record_launch {rec.debug_summary()}")

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
        """Warn when ``_forecast_incoming_fleets`` first-hit slot differs from raycast at launch."""

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
            if rec is None or fid in self._warned_forecast_mismatch:
                continue
            fc_slot = int(fleet_arrivals[i, 0])
            fc_tick = int(fleet_arrivals[i, 1])
            ray_slot = int(rec.true_target_slot)
            ray_tick = int(rec.true_hit_tick) if rec.true_hit_tick < 500.0 else -1
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
                "[orbit_wars] forecast incoming target differs from raycast at launch"
                + (f" (fleet_id={fid})" if fid >= 0 else ""),
                f"\n  game_step={game_step} ego_player={ego_player} launch_step={rec.game_step}",
                f"\n  raycast: slot={ray_slot} planet_id={ray_pid:.0f} hit_tick={ray_tick}",
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


@dataclass
class KaggleAgentCallTiming:
    """Host-side ``perf_counter`` slices for the last ``KaggleOrbitWarsAgent.__call__``."""

    obs_to_state_s: float = 0.0
    micro_iters: int = 0
    micro_obs_tensors_s: float = 0.0
    micro_policy_forward_s: float = 0.0
    micro_post_forward_s: float = 0.0
    micro_raycast_s: float = 0.0
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


def _place_rows_by_id(rows: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    table = np.zeros((MAX_PLANETS, width), dtype=np.float32)
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


def _earliest_hit_planet_index(
    hit_mask: np.ndarray,
    t_enter: np.ndarray,
    collision_rank: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """First planet along each ray's segment (earliest ``t``, then Kaggle list order)."""

    big = np.float32(1e9)
    score = np.where(hit_mask, t_enter, big)
    score = score + collision_rank.astype(np.float32)[None, :] * 1e-6
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


def _collapse_opponents_enabled() -> bool:
    return os.environ.get("ORBIT_WARS_COLLAPSE_OPPONENTS", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _remap_owner(owner: float, ego: int, num_agents: int) -> int:
    o = int(owner)
    if o < 0:
        return 0
    if o == ego:
        return 1
    if num_agents <= 2 or _collapse_opponents_enabled():
        return 2
    return 2 + o if o < ego else 2 + (o - 1)


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
    delta = init_pos - np.asarray([CENTER, CENTER], dtype=np.float32)
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

    positions = fleets_in[:, 2:4].astype(np.float32, copy=True)
    alive = np.ones((len(fleets_in),), dtype=np.bool_)
    angles = fleets_in[:, 4].astype(np.float32, copy=False)
    owners = fleets_in[:, 1].astype(np.int32, copy=False)
    ships = np.floor(fleets_in[:, 6].astype(np.float32, copy=False))
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    speeds = np.asarray([_fleet_speed(float(s), ship_speed) for s in ships], dtype=np.float32)

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
                if _point_to_segment_distance(np.asarray([CENTER, CENTER], dtype=np.float32), a0, a1) < SUN_RADIUS:
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


def _simulate_discrete_ray_policy_hits_np(
    state: OrbitWarsState,
    origin_idx: int,
    frac_idx: int,
    *,
    ship_speed: float = 6.0,
    horizon: int = INCOMING_TA_BINS,
    n_rays: int = DEFAULT_RAYCAST_RAYS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Discrete per-tick fleet forward model; returns ray angles and first-hit bookkeeping."""

    planets = np.asarray(state.planets)
    current_active = np.asarray(state.planet_active).astype(bool)
    p0, p1, active_by_tick = _forecast_planet_paths_np(state, horizon=horizon)
    # Terminal events use per-tick collision flags (same as Kaggle fleet movement),
    # not the static visibility mask used only when marking valid policy targets.
    radii = planets[:, PLANET_RADIUS].astype(np.float32)
    origin_xy = planets[origin_idx, PLANET_X : PLANET_Y + 1].astype(np.float32)
    origin_radius = float(planets[origin_idx, PLANET_RADIUS])
    ships_avail = float(planets[origin_idx, 5])
    send = _planned_send(ships_avail, frac_idx)
    speed = _fleet_speed(float(max(send, 1)), ship_speed)
    collision_rank = np.asarray(state.planet_collision_rank, dtype=np.int32)

    angles = np.arange(n_rays, dtype=np.float32) * (2.0 * math.pi / float(n_rays))
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    pos = origin_xy[None, :] + (origin_radius + 0.1) * dirs
    done_policy = np.zeros((n_rays,), dtype=np.bool_)
    done_true = np.zeros((n_rays,), dtype=np.bool_)
    policy_code = np.full((n_rays,), -1, dtype=np.int32)
    policy_tick = np.full((n_rays,), 10_000, dtype=np.int32)
    true_code = np.full((n_rays,), -1, dtype=np.int32)
    true_tick = np.full((n_rays,), 10_000, dtype=np.int32)

    sun = np.asarray([CENTER, CENTER], dtype=np.float32)
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
        t_enter = np.where(
            qa < 1e-12,
            np.where(static_hit, 0.0, np.float32(1e9)),
            np.where(moving_hit, np.clip(t1, 0.0, 1.0), np.float32(1e9)),
        ).astype(np.float32)

        idx_true, any_true = _earliest_hit_planet_index(hit_true, t_enter, collision_rank)
        idx_policy, any_policy = _earliest_hit_planet_index(hit_policy, t_enter, collision_rank)

        delta = a1 - a0
        l2 = np.sum(delta * delta, axis=1)
        proj = np.zeros((n_rays,), dtype=np.float32)
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

    return angles, policy_code, policy_tick, true_code, true_tick, done_policy


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

    angles, policy_code, _, _, _, done_policy = _simulate_discrete_ray_policy_hits_np(
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """NumPy version of the rollout discrete first-hit ray target sampler."""

    angles, policy_code, policy_tick, true_code, true_tick, done_policy = _simulate_discrete_ray_policy_hits_np(
        state, origin_idx, frac_idx, ship_speed=ship_speed, horizon=horizon, n_rays=n_rays
    )
    planets = np.asarray(state.planets)
    current_active = np.asarray(state.planet_active).astype(bool)

    out_angle = np.zeros((MAX_PLANETS,), dtype=np.float32)
    valid = np.zeros((MAX_PLANETS,), dtype=np.bool_)
    hit_tick = np.zeros((MAX_PLANETS,), dtype=np.float32)
    true_planet = np.full((MAX_PLANETS,), -1, dtype=np.int32)
    true_hit_tick = np.full((MAX_PLANETS,), 500.0, dtype=np.float32)
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
    valid &= current_active
    valid[origin_idx] = False
    true_planet = np.where(valid, true_planet, -1).astype(np.int32)
    true_hit_tick = np.where(valid, true_hit_tick, 500.0).astype(np.float32)
    return out_angle, valid, hit_tick, true_planet, true_hit_tick


def _obs_tensors_for_state(state: OrbitWarsState, ego_player: int, device: torch.device) -> dict[str, torch.Tensor]:
    planets = np.asarray(state.planets)
    planet_active = np.asarray(state.planet_active)
    initial_planets = np.asarray(state.initial_planets)
    initial_active = np.asarray(state.initial_active)
    incoming_fleets = np.asarray(state.incoming_fleets)
    angular_velocity = float(np.asarray(state.angular_velocity))
    step_count = int(np.asarray(state.step_count))
    num_agents = int(np.asarray(state.num_agents))
    comet_ids = np.asarray(state.comet_planet_ids)
    comet_set = set(float(x) for x in comet_ids.flatten() if int(x) >= 0)

    entity_type = np.zeros((1 + MAX_PLANETS,), dtype=np.int64)
    owner_idx = np.zeros((1 + MAX_PLANETS,), dtype=np.int64)
    features = np.zeros((1 + MAX_PLANETS, FEATURE_DIM), dtype=np.float32)
    rope_pos = np.zeros((1 + MAX_PLANETS, 3), dtype=np.float32)
    entity_mask = np.zeros((1 + MAX_PLANETS,), dtype=np.bool_)
    planet_mask = np.zeros((1 + MAX_PLANETS,), dtype=np.bool_)

    entity_type[0] = ENTITY_CLS
    owner_idx[0] = 1
    features[0, 6] = np.float32(np.clip(float(step_count) / 498.0, 0.0, 1.0))
    rope_pos[0] = np.asarray([CENTER / BOARD_SIZE, CENTER / BOARD_SIZE, 0.0], dtype=np.float32)
    entity_mask[0] = True

    incoming = incoming_fleets.astype(np.float32)
    self_incoming = incoming[int(ego_player)]
    enemy_incoming = incoming[np.arange(incoming.shape[0]) != int(ego_player)].sum(axis=0)
    incoming_net = (self_incoming - enemy_incoming) / 1000.0

    for i in range(MAX_PLANETS):
        j = 1 + i
        active = bool(planet_active[i])
        pid = float(planets[i, 0])
        is_comet = pid in comet_set
        entity_type[j] = ENTITY_COMET if is_comet else ENTITY_PLANET
        owner_idx[j] = min(_remap_owner(float(planets[i, 1]), ego_player, num_agents), NUM_OWNER_SLOTS - 1)
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

        features[j, 0] = np.log1p(max(float(planets[i, 6]), 0.0))
        features[j, 1] = float(planets[i, 5]) / 1000.0
        features[j, 2] = float(vx) / 5.0
        features[j, 3] = float(vy) / 5.0
        features[j, 4] = float(active)
        features[j, 5] = float(planets[i, 4]) / 10.0
        features[j, 8:] = incoming_net[i]
        if active:
            rope_pos[j, 0] = float(planets[i, 2]) / BOARD_SIZE
            rope_pos[j, 1] = float(planets[i, 3]) / BOARD_SIZE
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
    timing: Optional[KaggleAgentCallTiming] = None,
    launch_tracker: Optional[FleetLaunchDebugTracker] = None,
    game_step: int = 0,
) -> list[list[float]]:
    planets = np.array(np.asarray(state.planets), copy=True)
    incoming_fleets = np.array(np.asarray(state.incoming_fleets), copy=True)
    planet_active = np.asarray(state.planet_active).astype(bool)
    actions: list[list[float]] = []
    micro_idx = 0

    for _ in range(max_micro_steps):
        if timing is not None:
            timing.micro_iters += 1

        t0 = perf_counter()
        virt = state._replace(planets=planets, incoming_fleets=incoming_fleets)
        batch = _obs_tensors_for_state(virt, ego_player, device)
        if timing is not None:
            timing.micro_obs_tensors_s += perf_counter() - t0

        t0 = perf_counter()
        out = policy(**batch)
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
        ray_angle, ray_valid, ray_hit_tick, true_planet, true_hit_tick = _raycast_targets_np(
            virt,
            int(o_idx),
            int(frac_idx),
            ship_speed=ship_speed,
            horizon=INCOMING_TA_BINS,
            n_rays=n_rays,
        )
        if timing is not None:
            timing.micro_raycast_s += perf_counter() - t0

        t0 = perf_counter()
        target_logits = policy.target_logits_for_origin_fraction(
            out["planet_hidden"],
            torch.tensor([o_idx], device=device, dtype=torch.long),
            torch.tensor([frac_idx], device=device, dtype=torch.long),
            torch.tensor([float(_planned_send(float(planets[o_idx, 5]), int(frac_idx)))], device=device, dtype=torch.float32),
            torch.from_numpy(ray_hit_tick[None, :]).to(device=device, dtype=torch.float32),
            torch.from_numpy(planets[None, :, 5]).to(device=device, dtype=torch.float32),
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
        if launch_tracker is not None:
            true_slot = int(true_planet[d_idx])
            launch_tracker.record_launch(
                LaunchRaycastRecord(
                    game_step=int(game_step),
                    ego_player=int(ego_player),
                    micro_idx=micro_idx,
                    origin_slot=int(o_idx),
                    origin_planet_id=float(planets[o_idx, 0]),
                    origin_xy=(float(planets[o_idx, PLANET_X]), float(planets[o_idx, PLANET_Y])),
                    origin_radius=float(planets[o_idx, PLANET_RADIUS]),
                    frac_idx=int(frac_idx),
                    fraction=float(FRACTIONS[frac_idx]),
                    ships_avail=float(ships_avail),
                    planned_send=int(send),
                    n_rays=int(n_rays),
                    ship_speed=float(ship_speed),
                    launch_angle=angle,
                    policy_target_slot=int(d_idx),
                    policy_target_planet_id=float(planets[d_idx, 0]),
                    true_target_slot=true_slot,
                    true_target_planet_id=float(planets[true_slot, 0]) if 0 <= true_slot < MAX_PLANETS else -1.0,
                    policy_hit_tick=float(ray_hit_tick[d_idx]),
                    true_hit_tick=float(true_hit_tick[d_idx]),
                    comet_planet_ids_at_launch=_active_comet_planet_ids(
                        np.asarray(virt.comet_group_active),
                        np.asarray(virt.comet_planet_ids),
                    ),
                )
            )
            micro_idx += 1
        actions.append([float(planets[o_idx, 0]), float(angle), int(send)])
        planets[o_idx, 5] -= float(send)
        env_target = int(true_planet[d_idx])
        if 0 <= env_target < MAX_PLANETS:
            ta = int(math.floor(max(float(true_hit_tick[d_idx]) - 1.0, 0.0)))
            ta = max(0, min(ta, incoming_fleets.shape[2] - 1))
            owner = max(0, min(int(ego_player), incoming_fleets.shape[0] - 1))
            cur = int(incoming_fleets[owner, env_target, ta])
            incoming_fleets[owner, env_target, ta] = min(cur + int(send), 65535)
        if not planet_active[o_idx] or planets[o_idx, 5] < 1.0:
            if timing is not None:
                timing.micro_book_s += perf_counter() - t0
            continue
        if timing is not None:
            timing.micro_book_s += perf_counter() - t0

    return actions


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
    planets, planet_active, id_to_slot = _place_rows_by_id(planets_in, 7)

    initial_in = _as_array(obs.get("initial_planets", []), 7)
    initial_planets = np.zeros((MAX_PLANETS, 7), dtype=np.float32)
    initial_active = np.zeros((MAX_PLANETS,), dtype=np.bool_)
    for row in initial_in[:MAX_PLANETS]:
        pid = int(row[0])
        slot = id_to_slot.get(pid, pid if 0 <= pid < MAX_PLANETS else -1)
        if 0 <= slot < MAX_PLANETS:
            initial_planets[slot, :7] = row[:7]
            initial_active[slot] = True

    # Comets can be present in planets without being present in initial_planets.
    missing_initial = planet_active & ~initial_active
    initial_planets[missing_initial] = planets[missing_initial]
    initial_active[missing_initial] = True

    fleets_in = _as_array(obs.get("fleets", []), 7)
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
            n = min(len(p), MAX_COMET_PATH)
            comet_paths[g, k, :n] = p[:n]
            comet_path_lengths[g, k] = n
        slot0 = comet_slots[g, 0]
        if 0 <= slot0 < MAX_PLANETS:
            comet_ships[g] = planets[slot0, 5]

    angular_velocity = float(obs.get("angular_velocity", 0.0))
    step_count = int(step_count_override if step_count_override is not None else obs.get("step", obs.get("step_count", 0)))
    planet_collision_rank = _planet_collision_rank_from_obs(planets_in, id_to_slot)
    incoming_fleets = _forecast_incoming_fleets(
        planets,
        planet_active,
        initial_planets,
        initial_active,
        fleets_in,
        comet_paths,
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
    }
    if isinstance(policy_state, Mapping):
        w = policy_state.get("feat_proj.weight")
        if hasattr(w, "shape"):
            kwargs["d_model"] = int(w.shape[0])
        layer_ids = []
        for key in policy_state:
            if key.startswith("blocks."):
                try:
                    layer_ids.append(int(key.split(".")[1]))
                except (IndexError, ValueError):
                    pass
        if layer_ids:
            kwargs["n_layers"] = max(layer_ids) + 1
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
) -> tuple[OrbitWarsPolicy, torch.device, Mapping[str, Any]]:
    """Load a training checkpoint or raw policy state dict for inference."""

    resolved = resolve_checkpoint_path(checkpoint_path)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    policy_state = payload.get("policy", payload) if isinstance(payload, Mapping) else payload
    policy = OrbitWarsPolicy(**_infer_policy_kwargs(payload)).to(torch_device)
    policy.load_state_dict(policy_state)
    policy.eval()
    return policy, torch_device, _checkpoint_training_args(payload)


class KaggleOrbitWarsAgent:
    """Callable adapter object suitable for Kaggle's ``agent(obs, config)`` API."""

    def __init__(
        self,
        checkpoint_path: str | os.PathLike[str] = DEFAULT_CHECKPOINT,
        *,
        device: Optional[str | torch.device] = None,
        greedy: bool = False,
        max_micro_steps: Optional[int] = None,
        max_fleets: int = 512,
        seed: Optional[int] = None,
        raycast_rays: Optional[int] = None,
    ):
        _configure_cpu_threads()
        self.checkpoint_path = resolve_checkpoint_path(checkpoint_path)
        self.policy, self.device, training_args = load_policy(self.checkpoint_path, device=device)
        self.greedy = bool(greedy)
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
        warn_oob = os.environ.get("ORBIT_WARS_WARN_OOB_LAUNCHES", "1").lower() not in {"0", "false", "no", "off"}
        self.launch_tracker = FleetLaunchDebugTracker(
            warn_oob=warn_oob,
            warn_forecast_mismatch=_warn_forecast_mismatch_enabled(),
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
            self._game_key = self._obs_game_key(obs)
            return s

        key = self._obs_game_key(obs)
        if key != self._game_key:
            self._game_key = key
            self._next_step_count = 0
            self._last_env_step = None

        if self._last_env_step is not None:
            return int(self._last_env_step)

        step_count = self._next_step_count
        self._next_step_count += 1
        return step_count

    @torch.inference_mode()
    def __call__(self, obs: Mapping[str, Any], config: Any = None) -> list[list[float]]:
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
        timing.obs_to_state_s = perf_counter() - t0
        self.launch_tracker.check_forecast_vs_raycast(
            obs,
            fleet_arrivals,
            np.asarray(state.planets),
            np.asarray(state.comet_planet_ids),
            game_step=step_count,
            ego_player=ego_player,
        )
        actions = _build_turn_actions_torch_only(
            self.policy,
            state,
            ego_player,
            self.device,
            ship_speed=ship_speed,
            max_micro_steps=self.max_micro_steps,
            greedy=self.greedy,
            rng=self.rng,
            n_rays=self.raycast_rays,
            timing=timing,
            launch_tracker=self.launch_tracker,
            game_step=step_count,
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


_AGENT: Optional[KaggleOrbitWarsAgent] = None
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

    Set ``ORBIT_WARS_CHECKPOINT`` to choose a checkpoint path.  By default the
    adapter looks for ``checkpoint.pt`` in the process cwd, the submission bundle
    root (parent of ``orbit_wars_pt/``), and next to this module.
    """

    global _AGENT
    if _AGENT is None:
        ckpt = resolve_checkpoint_path(os.environ.get("ORBIT_WARS_CHECKPOINT", DEFAULT_CHECKPOINT))
        device = os.environ.get("ORBIT_WARS_DEVICE")
        greedy = os.environ.get("ORBIT_WARS_GREEDY", "0").lower() in {"1", "true", "yes", "on"}
        seed_raw = os.environ.get("ORBIT_WARS_AGENT_SEED")
        seed = int(seed_raw) if seed_raw is not None else None
        rays_raw = os.environ.get("ORBIT_WARS_RAYCAST_RAYS")
        rays = int(rays_raw) if rays_raw is not None else None
        max_micro_raw = os.environ.get("ORBIT_WARS_MAX_MICRO_STEPS")
        max_micro_steps = int(max_micro_raw) if max_micro_raw is not None else None
        try:
            _AGENT = KaggleOrbitWarsAgent(
                ckpt,
                device=device,
                greedy=greedy,
                max_micro_steps=max_micro_steps,
                seed=seed,
                raycast_rays=rays,
            )
        except Exception as exc:
            _report_once(exc)
            return []
    try:
        return _AGENT(obs, config)
    except Exception as exc:
        _report_once(exc)
        return []

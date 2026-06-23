"""CPU NumPy port of ``first_hit_interval_best_targets_apply_jax`` (no JAX).

Uses continuous angular intervals (union / subtract on the circle), sampled
intra-tick hulls matching ``tick_hit_intervals_jax`` (``samples_per_span`` on
radial tau windows — not a fixed ray grid).  Per-planet targets (tangent /
event sweep) aim at earliest-arrival headings from first-contact bearings,
with an edge-hugging alternative when it shares the same hit tick.
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from orbit_wars_pt.constants import BOARD_SIZE, CENTER, ROTATION_RADIUS_LIMIT, SUN_RADIUS
from orbit_wars_pt.geometry import (
    AngleInterval,
    TAU,
    _circle_disk_angle_interval,
    _merge_pieces,
    _norm_angle,
    _radial_active_windows,
    board_exit_angle_intervals,
    subtract_angle_intervals,
    union_angle_intervals,
)


GEOM_EPS = 1e-5
ANGLE_PAD = 1e-4
AIM_INSIDE_EDGE_EPS = 1e-6
LAUNCH_RIM_OFFSET = 0.1

# Shelved: ``orthogonal_hit_circular_orbit`` (exact cos/sin motion). The env linearises
# each turn as a chord; chord polylines match swept collision more closely.
_USE_EXACT_CIRCULAR_ORBIT_ORTHOGONAL = False


@dataclass(frozen=True)
class IntervalRefineJob:
    variant: str
    slot: int
    coarse_angle: float
    owner_lo: float
    owner_hi: float
    bound: float | None
    outward_dir: float



@dataclass(frozen=True, slots=True)
class OrthogonalHitEvent:
    """One orthogonal hit: launch cone; ``t`` is for occlusion ordering only (not ETA)."""

    t: float
    angle_lo: float
    angle_hi: float
    slot: int
    kind: str  # planet | sun
    first_contact_angle: float = float("nan")


@dataclass(slots=True)
class IntervalAimStats:
    planet_evals: int = 0
    selected: Counter[str] = field(default_factory=Counter)
    rejected: Counter[str] = field(default_factory=Counter)
    all_rejected: Counter[str] = field(default_factory=Counter)
    polish: Counter[str] = field(default_factory=Counter)

    def note_rejected(self, label: str, reason: str) -> None:
        self.rejected[f"{label}:{reason}"] += 1

    def note_polish(self, key: str, amount: int = 1) -> None:
        self.polish[key] += int(amount)

    def note_polish_gap(self, keep: float, reject: float) -> None:
        gap = abs(float(reject) - float(keep))
        if gap == 0.0:
            self.polish["gap:zero"] += 1
            return
        toward = float(reject)
        step = abs(float(np.nextafter(float(keep), toward)) - float(keep))
        if step <= 0.0:
            self.polish["gap:unknown"] += 1
            return
        ulps = gap / step
        if ulps <= 1.5:
            self.polish["gap:<=1ulp"] += 1
        elif ulps <= 2.5:
            self.polish["gap:<=2ulp"] += 1
        elif ulps <= 4.5:
            self.polish["gap:<=4ulp"] += 1
        elif ulps <= 8.5:
            self.polish["gap:<=8ulp"] += 1
        else:
            self.polish["gap:>8ulp"] += 1

    def format_report(self) -> str:
        if self.planet_evals <= 0:
            return ""
        lines = [
            "[orbit_wars] interval aim breakdown",
            f"  planet evaluations: {self.planet_evals}",
        ]
        if self.selected:
            lines.append("  selected:")
            for key, value in sorted(self.selected.items()):
                lines.append(f"    {key}: {value}")
        if self.rejected:
            lines.append("  rejections:")
            for key, value in sorted(self.rejected.items()):
                lines.append(f"    {key}: {value}")
        if self.all_rejected:
            total_all = int(sum(self.all_rejected.values()))
            lines.append(f"  all three rejected: {total_all}")
            for key, value in sorted(self.all_rejected.items()):
                lines.append(f"    {key}: {value}")
        if self.polish:
            lines.append("  polish:")
            for key, value in sorted(self.polish.items()):
                lines.append(f"    {key}: {value}")
        return "\n".join(lines)


_INTERVAL_AIM_STATS = IntervalAimStats()


def reset_interval_aim_stats() -> None:
    global _INTERVAL_AIM_STATS
    _INTERVAL_AIM_STATS = IntervalAimStats()


def format_interval_aim_stats() -> str:
    return _INTERVAL_AIM_STATS.format_report()


def angle_in_intervals(
    angle: float,
    intervals: Sequence[AngleInterval],
    *,
    eps: float = 0.0,
) -> bool:
    """Whether ``angle`` lies in the union of (possibly wrapping) intervals."""

    a = _norm_angle(float(angle))
    for iv in union_angle_intervals(intervals):
        lo = _norm_angle(iv.lo)
        hi = _norm_angle(iv.hi)
        raw_width = float(iv.hi - iv.lo)
        width = raw_width % TAU
        if raw_width >= TAU - eps or (width <= eps and raw_width > TAU - eps):
            return True
        if lo <= hi:
            if lo - eps <= a <= hi + eps:
                return True
        elif a >= lo - eps or a <= hi + eps:
            return True
    return False


def _intervals_to_pieces(intervals: Sequence[AngleInterval]) -> list[tuple[float, float]]:
    pieces: list[tuple[float, float]] = []
    for iv in union_angle_intervals(intervals):
        lo = _norm_angle(iv.lo)
        hi = _norm_angle(iv.hi)
        raw_width = float(iv.hi - iv.lo)
        width = raw_width % TAU
        if raw_width >= TAU - GEOM_EPS or (width < GEOM_EPS and raw_width > TAU - GEOM_EPS):
            pieces.append((0.0, TAU))
        elif lo <= hi:
            if hi - lo > GEOM_EPS:
                pieces.append((lo, hi))
        else:
            if hi > GEOM_EPS:
                pieces.append((0.0, hi))
            if TAU - lo > GEOM_EPS:
                pieces.append((lo, TAU))
    return _merge_pieces(pieces, eps=GEOM_EPS)


def set_subtract_cells(
    hit: Sequence[AngleInterval],
    blocked: Sequence[AngleInterval],
    *,
    eps: float = GEOM_EPS,
) -> list[tuple[float, float]]:
    """Elementary non-wrapping cells in ``hit - blocked`` (mirrors ``geometry_jax._set_subtract_cells``)."""

    hit_p = _intervals_to_pieces(hit)
    block_p = _intervals_to_pieces(blocked)
    if not hit_p:
        return []
    endpoints = [0.0, TAU]
    for lo, hi in hit_p:
        endpoints.extend([lo, hi])
    for lo, hi in block_p:
        endpoints.extend([lo, hi])
    endpoints = sorted(set(max(0.0, min(TAU, float(x))) for x in endpoints))
    cells: list[tuple[float, float]] = []
    for lo, hi in zip(endpoints, endpoints[1:]):
        if hi - lo <= eps:
            continue
        mid = 0.5 * (lo + hi)
        in_hit = any(lo_h <= mid <= hi_h for lo_h, hi_h in hit_p)
        in_block = any(lo_b <= mid <= hi_b for lo_b, hi_b in block_p)
        if in_hit and not in_block:
            cells.append((lo, hi))
    return cells


def _signed_angle_delta(a: float, ref: float) -> float:
    return ((a - ref + math.pi) % TAU) - math.pi


def _unwrap_angle_near(a: float, ref: float) -> float:
    return ref + _signed_angle_delta(a, ref)


def _angular_hull_sampled(intervals: Sequence[AngleInterval]) -> list[AngleInterval]:
    """Hull sampled disk intervals on one radial span (matches JAX tick_hit hull step)."""

    present = [iv for iv in intervals if iv is not None]
    if not present:
        return []
    any_full = False
    mids: list[float] = []
    shifted_pts: list[float] = []
    for iv in present:
        raw_width = float(iv.hi - iv.lo)
        width = raw_width % TAU
        if raw_width >= TAU - GEOM_EPS or (width <= GEOM_EPS and raw_width > TAU - GEOM_EPS):
            any_full = True
        mid = _norm_angle(iv.lo + 0.5 * width)
        mids.append(mid)
    if any_full:
        return [AngleInterval(0.0, TAU)]
    ref = mids[0]
    for iv in present:
        width = (iv.hi - iv.lo) % TAU
        mid = _norm_angle(iv.lo + 0.5 * width)
        shifted_pts.extend(
            [
                _unwrap_angle_near(iv.lo, ref),
                _unwrap_angle_near(mid, ref),
                _unwrap_angle_near(iv.hi, ref),
            ]
        )
    hull_lo = min(shifted_pts) - ANGLE_PAD
    hull_hi = max(shifted_pts) + ANGLE_PAD
    return union_angle_intervals(
        [AngleInterval(_norm_angle(hull_lo), _norm_angle(hull_hi))]
    )


def tick_hit_intervals_sampled(
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    tick: int,
    object_p0: np.ndarray,
    object_p1: np.ndarray,
    object_radius: float,
    *,
    object_active: bool = True,
    samples_per_span: int = 9,
) -> list[AngleInterval]:
    """Non-wrapping hit-angle pieces for one object/tick (matches ``tick_hit_intervals_jax``)."""

    if not object_active:
        return []
    origin = np.asarray(origin_xy, dtype=np.float64)
    p0 = np.asarray(object_p0, dtype=np.float64)
    p1 = np.asarray(object_p1, dtype=np.float64)
    launch_offset = float(origin_radius) + LAUNCH_RIM_OFFSET
    radius = float(object_radius)
    windows = _radial_active_windows(
        origin, launch_offset, float(speed), int(tick), p0, p1, radius
    )
    if not windows:
        return []

    n_s = max(2, int(samples_per_span))
    span_hits: list[AngleInterval] = []
    for tau_lo, tau_hi in windows:
        samples: list[AngleInterval] = []
        for k in range(n_s):
            frac = float(k) / float(n_s - 1)
            tau = tau_lo + (tau_hi - tau_lo) * frac
            target_xy = p0 + tau * (p1 - p0)
            iv = _circle_disk_angle_interval(
                origin,
                launch_offset,
                float(speed),
                int(tick),
                target_xy,
                radius,
                tau,
            )
            if iv is not None:
                samples.append(iv)
        span_hits.extend(_angular_hull_sampled(samples))
    return union_angle_intervals(span_hits)


def precompute_tick_planet_hits(
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    object_p0_by_tick: np.ndarray,
    object_p1_by_tick: np.ndarray,
    object_radii: np.ndarray,
    object_active_by_tick: np.ndarray,
    *,
    samples_per_span: int = 9,
) -> list[list[list[AngleInterval]]]:
    """``hits[tick][planet]`` interval lists (no occlusion)."""

    p0 = np.asarray(object_p0_by_tick, dtype=np.float64)
    p1 = np.asarray(object_p1_by_tick, dtype=np.float64)
    radii = np.asarray(object_radii, dtype=np.float64)
    active = np.asarray(object_active_by_tick, dtype=bool)
    ticks, planets = active.shape
    out: list[list[list[AngleInterval]]] = []
    for t in range(ticks):
        row: list[list[AngleInterval]] = []
        for j in range(planets):
            row.append(
                tick_hit_intervals_sampled(
                    origin_xy,
                    origin_radius,
                    speed,
                    t,
                    p0[t, j],
                    p1[t, j],
                    float(radii[j]),
                    object_active=bool(active[t, j]),
                    samples_per_span=samples_per_span,
                )
            )
        out.append(row)
    return out


def _widest_cell_midpoint_and_width(
    cells: Sequence[tuple[float, float]],
) -> tuple[float | None, float]:
    best_w = -1.0
    best_mid: float | None = None
    for lo, hi in cells:
        w = hi - lo
        if w > best_w:
            best_w = w
            best_mid = _norm_angle(0.5 * (lo + hi))
    return best_mid, best_w


def _cells_total_width(cells: Sequence[tuple[float, float]]) -> float:
    return float(sum(max(0.0, hi - lo) for lo, hi in cells))


def _event_first_contact_angle(event: OrthogonalHitEvent) -> float:
    if math.isfinite(event.first_contact_angle):
        return _norm_angle(float(event.first_contact_angle))
    mid = _norm_angle(0.5 * (float(event.angle_lo) + float(event.angle_hi)))
    return mid


def _candidate_tick_single_planet(
    theta: float | None,
    *,
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    p0_by_tick: np.ndarray,
    p1_by_tick: np.ndarray,
    radii: np.ndarray,
    active_by_tick: np.ndarray,
    slot: int,
) -> tuple[float | None, int, str]:
    if theta is None:
        return None, -1, "absent"
    tick = first_hit_tick_single_planet_raycast(
        float(theta),
        origin_xy,
        origin_radius,
        speed,
        p0_by_tick[:, slot, :],
        p1_by_tick[:, slot, :],
        float(radii[slot]),
        active_by_tick[:, slot],
    )
    if tick < 0:
        return float(theta), -1, "tick_reject"
    return float(theta), int(tick), "ok"


def _sun_hit_tick(
    angle: float,
    *,
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    active_by_tick: np.ndarray,
) -> int:
    origin = np.asarray(origin_xy, dtype=np.float64)
    theta = float(angle) % TAU
    direction = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
    launch_off = float(origin_radius) + LAUNCH_RIM_OFFSET
    pos = origin + launch_off * direction
    sun = np.asarray([CENTER, CENTER], dtype=np.float64)
    ticks = int(active_by_tick.shape[0])

    for tick in range(ticks):
        a0 = pos
        a1 = pos + float(speed) * direction
        delta = a1 - a0
        l2 = float(np.dot(delta, delta))
        if l2 > 1e-12:
            proj = float(np.clip(np.dot(sun - a0, delta) / l2, 0.0, 1.0))
            closest = a0 + proj * delta
        else:
            closest = a0
        if float(np.linalg.norm(closest - sun)) < float(SUN_RADIUS):
            return int(tick)
        pos = a1
    return -1


def _launch_segment_at_tick(
    angle: float,
    *,
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    tick: int,
) -> tuple[np.ndarray, np.ndarray]:
    origin = np.asarray(origin_xy, dtype=np.float64)
    theta = float(angle) % TAU
    direction = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
    a0 = origin + (float(origin_radius) + LAUNCH_RIM_OFFSET) * direction
    step = float(speed) * direction
    for _ in range(int(tick)):
        a0 = a0 + step
    a1 = a0 + float(speed) * direction
    return a0, a1


def _swept_hit_on_tick(
    angle: float,
    *,
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    tick: int,
    p0: np.ndarray,
    p1: np.ndarray,
    radius: float,
) -> bool:
    a0, a1 = _launch_segment_at_tick(
        angle,
        origin_xy=origin_xy,
        origin_radius=origin_radius,
        speed=speed,
        tick=int(tick),
    )
    d0 = a0 - np.asarray(p0, dtype=np.float64)
    dv = (a1 - a0) - (np.asarray(p1, dtype=np.float64) - np.asarray(p0, dtype=np.float64))
    qa = float(np.dot(dv, dv))
    qb = float(2.0 * np.dot(d0, dv))
    qc = float(np.dot(d0, d0) - float(radius) * float(radius))
    if qa < 1e-12:
        return qc <= 0.0
    disc = qb * qb - 4.0 * qa * qc
    if disc < 0.0:
        return False
    sd = math.sqrt(max(disc, 0.0))
    t1 = (-qb - sd) / (2.0 * qa)
    t2 = (-qb + sd) / (2.0 * qa)
    return t2 >= 0.0 and t1 <= 1.0


def _planet_hit_on_tick(
    angle: float,
    *,
    slot: int,
    tick: int,
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    p0_by_tick: np.ndarray,
    p1_by_tick: np.ndarray,
    radii: np.ndarray,
    active_by_tick: np.ndarray,
) -> bool:
    if tick < 0 or tick >= int(active_by_tick.shape[0]) or not bool(active_by_tick[tick, slot]):
        return False
    return _swept_hit_on_tick(
        angle,
        origin_xy=origin_xy,
        origin_radius=origin_radius,
        speed=speed,
        tick=int(tick),
        p0=p0_by_tick[tick, slot],
        p1=p1_by_tick[tick, slot],
        radius=float(radii[slot]),
    )


def _planet_hit_near_tick(
    angle: float,
    *,
    slot: int,
    tick: int,
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    p0_by_tick: np.ndarray,
    p1_by_tick: np.ndarray,
    radii: np.ndarray,
    active_by_tick: np.ndarray,
    tick_margin: int = 1,
) -> bool:
    for t in range(max(0, int(tick) - int(tick_margin)), min(int(active_by_tick.shape[0]), int(tick) + int(tick_margin) + 1)):
        if _planet_hit_on_tick(
            angle,
            slot=slot,
            tick=t,
            origin_xy=origin_xy,
            origin_radius=origin_radius,
            speed=speed,
            p0_by_tick=p0_by_tick,
            p1_by_tick=p1_by_tick,
            radii=radii,
            active_by_tick=active_by_tick,
        ):
            return True
    return False


def _sun_hit_on_tick(
    angle: float,
    *,
    tick: int,
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
) -> bool:
    a0, a1 = _launch_segment_at_tick(
        angle,
        origin_xy=origin_xy,
        origin_radius=origin_radius,
        speed=speed,
        tick=int(tick),
    )
    delta = a1 - a0
    l2 = float(np.dot(delta, delta))
    sun = np.asarray([CENTER, CENTER], dtype=np.float64)
    if l2 > 1e-12:
        proj = float(np.clip(np.dot(sun - a0, delta) / l2, 0.0, 1.0))
        closest = a0 + proj * delta
    else:
        closest = a0
    return float(np.linalg.norm(closest - sun)) < float(SUN_RADIUS)


def _sun_hit_near_tick(
    angle: float,
    *,
    tick: int,
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    tick_margin: int = 1,
) -> bool:
    for t in range(max(0, int(tick) - int(tick_margin)), int(tick) + int(tick_margin) + 1):
        if _sun_hit_on_tick(
            angle,
            tick=t,
            origin_xy=origin_xy,
            origin_radius=origin_radius,
            speed=speed,
        ):
            return True
    return False


def _board_hit_on_tick(
    angle: float,
    *,
    tick: int,
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
) -> bool:
    _a0, a1 = _launch_segment_at_tick(
        angle,
        origin_xy=origin_xy,
        origin_radius=origin_radius,
        speed=speed,
        tick=int(tick),
    )
    return not (0.0 <= a1[0] <= BOARD_SIZE and 0.0 <= a1[1] <= BOARD_SIZE)


def _board_hit_near_tick(
    angle: float,
    *,
    tick: int,
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    tick_margin: int = 1,
) -> bool:
    for t in range(max(0, int(tick) - int(tick_margin)), int(tick) + int(tick_margin) + 1):
        if _board_hit_on_tick(
            angle,
            tick=t,
            origin_xy=origin_xy,
            origin_radius=origin_radius,
            speed=speed,
        ):
            return True
    return False


def _body_first_hit_near_tick(
    angle: float,
    *,
    body_sig: tuple[str, int],
    tick: int,
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    p0_by_tick: np.ndarray,
    p1_by_tick: np.ndarray,
    radii: np.ndarray,
    active_by_tick: np.ndarray,
    tick_margin: int = 1,
) -> int:
    kind, code = body_sig
    lo = max(0, int(tick) - int(tick_margin))
    hi = min(int(active_by_tick.shape[0]) - 1, int(tick) + int(tick_margin))
    if hi < lo:
        return -1
    for t in range(lo, hi + 1):
        if kind == "planet" and 0 <= int(code) < int(radii.shape[0]):
            if _planet_hit_on_tick(
                angle,
                slot=int(code),
                tick=t,
                origin_xy=origin_xy,
                origin_radius=origin_radius,
                speed=speed,
                p0_by_tick=p0_by_tick,
                p1_by_tick=p1_by_tick,
                radii=radii,
                active_by_tick=active_by_tick,
            ):
                return int(t)
        elif kind == "sun":
            if _sun_hit_on_tick(
                angle,
                tick=t,
                origin_xy=origin_xy,
                origin_radius=origin_radius,
                speed=speed,
            ):
                return int(t)
        elif kind == "board":
            if _board_hit_on_tick(
                angle,
                tick=t,
                origin_xy=origin_xy,
                origin_radius=origin_radius,
                speed=speed,
            ):
                return int(t)
    return -1


def _hit_signature_priority(
    sig: tuple[str, int],
    *,
    object_order: Sequence[int],
    order_rank: dict[int, int] | None = None,
) -> int:
    kind, code = sig
    if kind == "planet":
        rank = order_rank or {int(s): i for i, s in enumerate(object_order)}
        return int(rank.get(int(code), 10_000))
    if kind == "sun":
        return 10_001
    if kind == "board":
        return 10_002
    return 10_003


def _board_hit_tick(
    angle: float,
    *,
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    active_by_tick: np.ndarray,
) -> int:
    origin = np.asarray(origin_xy, dtype=np.float64)
    theta = float(angle) % TAU
    direction = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
    launch_off = float(origin_radius) + LAUNCH_RIM_OFFSET
    pos = origin + launch_off * direction
    ticks = int(active_by_tick.shape[0])
    for tick in range(ticks):
        a1 = pos + float(speed) * direction
        if not (0.0 <= a1[0] <= BOARD_SIZE and 0.0 <= a1[1] <= BOARD_SIZE):
            return int(tick)
        pos = a1
    return -1


def _body_hit_tick(
    angle: float,
    *,
    body_sig: tuple[str, int],
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    p0_by_tick: np.ndarray,
    p1_by_tick: np.ndarray,
    radii: np.ndarray,
    active_by_tick: np.ndarray,
) -> int:
    kind, code = body_sig
    if kind == "planet" and 0 <= int(code) < int(radii.shape[0]):
        slot = int(code)
        return first_hit_tick_single_planet_raycast(
            angle,
            origin_xy,
            origin_radius,
            speed,
            p0_by_tick[:, slot, :],
            p1_by_tick[:, slot, :],
            float(radii[slot]),
            active_by_tick[:, slot],
        )
    if kind == "sun":
        return _sun_hit_tick(
            angle,
            origin_xy=origin_xy,
            origin_radius=origin_radius,
            speed=speed,
            active_by_tick=active_by_tick,
        )
    if kind == "board":
        return _board_hit_tick(
            angle,
            origin_xy=origin_xy,
            origin_radius=origin_radius,
            speed=speed,
            active_by_tick=active_by_tick,
        )
    return -1


def _find_body_tick_hint_from_boundary(
    bound: float,
    *,
    inward_dir: float,
    body_sig: tuple[str, int],
    search_span: float,
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    p0_by_tick: np.ndarray,
    p1_by_tick: np.ndarray,
    radii: np.ndarray,
    active_by_tick: np.ndarray,
) -> int:
    """Find a hit tick for one body by probing at/near a contested boundary."""

    tick = _body_hit_tick(
        float(bound),
        body_sig=body_sig,
        origin_xy=origin_xy,
        origin_radius=origin_radius,
        speed=speed,
        p0_by_tick=p0_by_tick,
        p1_by_tick=p1_by_tick,
        radii=radii,
        active_by_tick=active_by_tick,
    )
    if tick >= 0:
        return int(tick)

    probe = np.nextafter(float(bound), float("inf") if inward_dir > 0.0 else float("-inf"))
    tick = _body_hit_tick(
        float(probe),
        body_sig=body_sig,
        origin_xy=origin_xy,
        origin_radius=origin_radius,
        speed=speed,
        p0_by_tick=p0_by_tick,
        p1_by_tick=p1_by_tick,
        radii=radii,
        active_by_tick=active_by_tick,
    )
    if tick >= 0:
        return int(tick)

    step = max(np.spacing(max(1.0, abs(bound))), min(max(search_span, 0.0), 1e-3))
    max_span = max(search_span, step)
    for _ in range(64):
        probe = float(bound + inward_dir * step)
        tick = _body_hit_tick(
            float(probe),
            body_sig=body_sig,
            origin_xy=origin_xy,
            origin_radius=origin_radius,
            speed=speed,
            p0_by_tick=p0_by_tick,
            p1_by_tick=p1_by_tick,
            radii=radii,
            active_by_tick=active_by_tick,
        )
        if tick >= 0:
            return int(tick)
        if step >= max_span:
            break
        step = min(max_span, step * 2.0)
    return -1


def _local_first_hit_signature_at_angle(
    angle: float,
    *,
    target_slot: int,
    competitor_sig: tuple[str, int],
    target_tick_hint: int | None = None,
    competitor_tick_hint: int | None = None,
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    p0_by_tick: np.ndarray,
    p1_by_tick: np.ndarray,
    radii: np.ndarray,
    active_by_tick: np.ndarray,
    object_order: Sequence[int],
    order_rank: dict[int, int] | None = None,
) -> tuple[str, int]:
    if target_tick_hint is None or competitor_tick_hint is None:
        target_tick = first_hit_tick_single_planet_raycast(
            angle,
            origin_xy,
            origin_radius,
            speed,
            p0_by_tick[:, target_slot, :],
            p1_by_tick[:, target_slot, :],
            float(radii[target_slot]),
            active_by_tick[:, target_slot],
        )
        comp_kind, comp_code = competitor_sig
        comp_tick = -1
        if comp_kind == "planet" and 0 <= int(comp_code) < int(radii.shape[0]):
            cslot = int(comp_code)
            comp_tick = first_hit_tick_single_planet_raycast(
                angle,
                origin_xy,
                origin_radius,
                speed,
                p0_by_tick[:, cslot, :],
                p1_by_tick[:, cslot, :],
                float(radii[cslot]),
                active_by_tick[:, cslot],
            )
        elif comp_kind == "sun":
            comp_tick = _sun_hit_tick(
                angle,
                origin_xy=origin_xy,
                origin_radius=origin_radius,
                speed=speed,
                active_by_tick=active_by_tick,
            )
        elif comp_kind == "board":
            comp_tick = _board_hit_tick(
                angle,
                origin_xy=origin_xy,
                origin_radius=origin_radius,
                speed=speed,
                active_by_tick=active_by_tick,
            )
    else:
        target_tick = int(target_tick_hint)
        comp_tick = int(competitor_tick_hint)

    target_sig = ("planet", int(target_slot))
    target_near = -1
    if target_tick >= 0:
        target_near = _body_first_hit_near_tick(
            angle,
            body_sig=target_sig,
            tick=int(target_tick),
            origin_xy=origin_xy,
            origin_radius=origin_radius,
            speed=speed,
            p0_by_tick=p0_by_tick,
            p1_by_tick=p1_by_tick,
            radii=radii,
            active_by_tick=active_by_tick,
        )

    comp_near = -1
    if comp_tick >= 0:
        comp_near = _body_first_hit_near_tick(
            angle,
            body_sig=competitor_sig,
            tick=int(comp_tick),
            origin_xy=origin_xy,
            origin_radius=origin_radius,
            speed=speed,
            p0_by_tick=p0_by_tick,
            p1_by_tick=p1_by_tick,
            radii=radii,
            active_by_tick=active_by_tick,
        )

    if target_near < 0 and comp_near < 0:
        return ("none", -1)
    if target_near < 0:
        return (str(competitor_sig[0]), int(competitor_sig[1]))
    if comp_near < 0:
        return target_sig
    if target_near < comp_near:
        return target_sig
    if comp_near < target_near:
        return (str(competitor_sig[0]), int(competitor_sig[1]))
    target_pri = _hit_signature_priority(
        target_sig,
        object_order=object_order,
        order_rank=order_rank,
    )
    comp_pri = _hit_signature_priority(
        competitor_sig,
        object_order=object_order,
        order_rank=order_rank,
    )
    if target_pri <= comp_pri:
        return target_sig
    return (str(competitor_sig[0]), int(competitor_sig[1]))


def _approx_boundary_competitor_signature(
    bound: float,
    *,
    outward_dir: float,
    target_sig: tuple[str, int],
    cache: "OcclusionWalkCache" | None,
) -> tuple[str, int]:
    if cache is None:
        return ("none", -1)
    dest = float("-inf") if outward_dir < 0.0 else float("inf")
    probe = np.nextafter(float(bound), dest)
    for _ in range(64):
        sig = first_hit_signature_occlusion_walk(_norm_angle(probe), cache)
        if sig != target_sig:
            return (str(sig[0]), int(sig[1]))
        nxt = np.nextafter(probe, dest)
        if nxt == probe:
            break
        probe = nxt
    return ("none", -1)


def _find_target_seed_in_cell(
    owner_cell: tuple[float, float],
    *,
    target_sig: tuple[str, int],
    competitor_sig: tuple[str, int],
    outward_dir: float,
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    p0_by_tick: np.ndarray,
    p1_by_tick: np.ndarray,
    radii: np.ndarray,
    active_by_tick: np.ndarray,
    object_order: Sequence[int],
    order_rank: dict[int, int] | None = None,
) -> float | None:
    lo, hi = owner_cell
    width = hi - lo
    inward_dir = -1.0 if outward_dir > 0.0 else 1.0
    near_bound = lo if outward_dir < 0.0 else hi
    near_inside = np.nextafter(float(near_bound), float("inf") if inward_dir > 0.0 else float("-inf"))
    probes = [
        near_inside,
        0.5 * (lo + hi),
        lo + 0.25 * width,
        hi - 0.25 * width,
        lo + 0.125 * width,
        hi - 0.125 * width,
    ]
    for probe in probes:
        if not (lo <= probe <= hi):
            continue
        if (
            _local_first_hit_signature_at_angle(
                _norm_angle(probe),
                target_slot=int(target_sig[1]),
                competitor_sig=competitor_sig,
                target_tick_hint=None,
                competitor_tick_hint=None,
                origin_xy=origin_xy,
                origin_radius=origin_radius,
                speed=speed,
                p0_by_tick=p0_by_tick,
                p1_by_tick=p1_by_tick,
                radii=radii,
                active_by_tick=active_by_tick,
                object_order=object_order,
                order_rank=order_rank,
            )
            == target_sig
        ):
            return float(probe)
    return None


def _target_hit_with_hint_at_angle(
    angle: float,
    *,
    target_slot: int,
    target_tick_hint: int | None,
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    p0_by_tick: np.ndarray,
    p1_by_tick: np.ndarray,
    radii: np.ndarray,
    active_by_tick: np.ndarray,
) -> bool:
    if target_tick_hint is None or int(target_tick_hint) < 0:
        tick = first_hit_tick_single_planet_raycast(
            angle,
            origin_xy,
            origin_radius,
            speed,
            p0_by_tick[:, int(target_slot), :],
            p1_by_tick[:, int(target_slot), :],
            float(radii[int(target_slot)]),
            active_by_tick[:, int(target_slot)],
        )
        return tick >= 0
    return _planet_hit_near_tick(
        angle,
        slot=int(target_slot),
        tick=int(target_tick_hint),
        origin_xy=origin_xy,
        origin_radius=origin_radius,
        speed=speed,
        p0_by_tick=p0_by_tick,
        p1_by_tick=p1_by_tick,
        radii=radii,
        active_by_tick=active_by_tick,
    )


def _polish_boundary_ground_truth(
    bound: float,
    *,
    owner_cell: tuple[float, float],
    target_sig: tuple[str, int],
    outward_dir: float,
    cache: "OcclusionWalkCache" | None,
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    p0_by_tick: np.ndarray,
    p1_by_tick: np.ndarray,
    radii: np.ndarray,
    active_by_tick: np.ndarray,
    object_order: Sequence[int],
    order_rank: dict[int, int] | None = None,
) -> float:
    """Return the extreme target angle near ``bound`` using local raycast ground truth."""

    _INTERVAL_AIM_STATS.note_polish("attempts")
    competitor_sig = _approx_boundary_competitor_signature(
        bound,
        outward_dir=outward_dir,
        target_sig=target_sig,
        cache=cache,
    )
    target_seed = _find_target_seed_in_cell(
        owner_cell,
        target_sig=target_sig,
        competitor_sig=competitor_sig,
        outward_dir=outward_dir,
        origin_xy=origin_xy,
        origin_radius=origin_radius,
        speed=speed,
        p0_by_tick=p0_by_tick,
        p1_by_tick=p1_by_tick,
        radii=radii,
        active_by_tick=active_by_tick,
        object_order=object_order,
        order_rank=order_rank,
    )
    if target_seed is None:
        _INTERVAL_AIM_STATS.note_polish("fallback:no_target_seed")
        lo, hi = owner_cell
        return _boundary_angle_inside(lo, hi, bound)

    lo, hi = owner_cell
    width = max(hi - lo, np.spacing(max(1.0, abs(bound))))
    target_tick_hint = _find_body_tick_hint_from_boundary(
        float(bound),
        inward_dir=(-outward_dir),
        body_sig=target_sig,
        search_span=width,
        origin_xy=origin_xy,
        origin_radius=origin_radius,
        speed=speed,
        p0_by_tick=p0_by_tick,
        p1_by_tick=p1_by_tick,
        radii=radii,
        active_by_tick=active_by_tick,
    )
    if target_tick_hint < 0:
        _INTERVAL_AIM_STATS.note_polish("fallback:missing_tick_hint")
        return _boundary_angle_inside(lo, hi, bound)

    if competitor_sig == ("none", -1):
        keep = float(target_seed)
        probe = float(bound)
        if outward_dir > 0.0:
            probe = max(probe, keep)
        else:
            probe = min(probe, keep)
        if _target_hit_with_hint_at_angle(
            _norm_angle(probe),
            target_slot=int(target_sig[1]),
            target_tick_hint=int(target_tick_hint),
            origin_xy=origin_xy,
            origin_radius=origin_radius,
            speed=speed,
            p0_by_tick=p0_by_tick,
            p1_by_tick=p1_by_tick,
            radii=radii,
            active_by_tick=active_by_tick,
        ):
            keep = float(probe)
        reject: float | None = None
        step = 1e-3
        for _ in range(80):
            cand = float(probe + outward_dir * step)
            if not _target_hit_with_hint_at_angle(
                _norm_angle(cand),
                target_slot=int(target_sig[1]),
                target_tick_hint=int(target_tick_hint),
                origin_xy=origin_xy,
                origin_radius=origin_radius,
                speed=speed,
                p0_by_tick=p0_by_tick,
                p1_by_tick=p1_by_tick,
                radii=radii,
                active_by_tick=active_by_tick,
            ):
                reject = cand
                break
            keep = cand
            probe = cand
            step *= 2.0
        if reject is None:
            _INTERVAL_AIM_STATS.note_polish("success:no_competitor_expand_only")
            return _norm_angle(keep)
        bisect_iters = 0
        for _ in range(80):
            mid = 0.5 * (keep + reject)
            if mid == keep or mid == reject:
                break
            bisect_iters += 1
            if _target_hit_with_hint_at_angle(
                _norm_angle(mid),
                target_slot=int(target_sig[1]),
                target_tick_hint=int(target_tick_hint),
                origin_xy=origin_xy,
                origin_radius=origin_radius,
                speed=speed,
                p0_by_tick=p0_by_tick,
                p1_by_tick=p1_by_tick,
                radii=radii,
                active_by_tick=active_by_tick,
            ):
                keep = float(mid)
            else:
                reject = float(mid)
        _INTERVAL_AIM_STATS.note_polish("success:no_competitor_bisected")
        _INTERVAL_AIM_STATS.note_polish("bisect_iters_total", bisect_iters)
        _INTERVAL_AIM_STATS.note_polish_gap(keep, reject)
        return _norm_angle(keep)

    competitor_tick_hint = _find_body_tick_hint_from_boundary(
        float(bound),
        inward_dir=outward_dir,
        body_sig=competitor_sig,
        search_span=width,
        origin_xy=origin_xy,
        origin_radius=origin_radius,
        speed=speed,
        p0_by_tick=p0_by_tick,
        p1_by_tick=p1_by_tick,
        radii=radii,
        active_by_tick=active_by_tick,
    )
    if competitor_tick_hint < 0:
        _INTERVAL_AIM_STATS.note_polish("fallback:missing_tick_hint")
        return _boundary_angle_inside(lo, hi, bound)

    keep = float(target_seed)
    probe = float(bound)
    if outward_dir > 0.0:
        probe = max(probe, keep)
    else:
        probe = min(probe, keep)
    probe_sig = _local_first_hit_signature_at_angle(
        _norm_angle(probe),
        target_slot=int(target_sig[1]),
        competitor_sig=competitor_sig,
        target_tick_hint=int(target_tick_hint),
        competitor_tick_hint=int(competitor_tick_hint),
        origin_xy=origin_xy,
        origin_radius=origin_radius,
        speed=speed,
        p0_by_tick=p0_by_tick,
        p1_by_tick=p1_by_tick,
        radii=radii,
        active_by_tick=active_by_tick,
        object_order=object_order,
        order_rank=order_rank,
    )
    if probe_sig == target_sig:
        keep = float(probe)
    reject: float | None = None
    if probe_sig != target_sig:
        reject = float(probe)
    else:
        step = 1e-3
        for _ in range(80):
            cand = float(probe + outward_dir * step)
            cand_sig = _local_first_hit_signature_at_angle(
                _norm_angle(cand),
                target_slot=int(target_sig[1]),
                competitor_sig=competitor_sig,
                target_tick_hint=int(target_tick_hint),
                competitor_tick_hint=int(competitor_tick_hint),
                origin_xy=origin_xy,
                origin_radius=origin_radius,
                speed=speed,
                p0_by_tick=p0_by_tick,
                p1_by_tick=p1_by_tick,
                radii=radii,
                active_by_tick=active_by_tick,
                object_order=object_order,
                order_rank=order_rank,
            )
            if cand_sig != target_sig:
                reject = cand
                break
            keep = cand
            probe = cand
            step *= 2.0
    if reject is None:
        _INTERVAL_AIM_STATS.note_polish("success:expand_only")
        return _norm_angle(keep)

    bisect_iters = 0
    for _ in range(80):
        mid = 0.5 * (keep + reject)
        if mid == keep or mid == reject:
            break
        bisect_iters += 1
        if (
            _local_first_hit_signature_at_angle(
                _norm_angle(mid),
                target_slot=int(target_sig[1]),
                competitor_sig=competitor_sig,
                target_tick_hint=int(target_tick_hint),
                competitor_tick_hint=int(competitor_tick_hint),
                origin_xy=origin_xy,
                origin_radius=origin_radius,
                speed=speed,
                p0_by_tick=p0_by_tick,
                p1_by_tick=p1_by_tick,
                radii=radii,
                active_by_tick=active_by_tick,
                object_order=object_order,
                order_rank=order_rank,
            )
            == target_sig
        ):
            keep = float(mid)
        else:
            reject = float(mid)
    _INTERVAL_AIM_STATS.note_polish("success:bisected")
    _INTERVAL_AIM_STATS.note_polish("bisect_iters_total", bisect_iters)
    _INTERVAL_AIM_STATS.note_polish_gap(keep, reject)
    return _norm_angle(keep)


def _closest_angle_in_cells(
    theta: float,
    cells: Sequence[tuple[float, float]],
    *,
    refine_boundaries: bool = True,
    target_sig: tuple[str, int] | None = None,
    cache: "OcclusionWalkCache" | None = None,
    origin_xy: np.ndarray | None = None,
    origin_radius: float = 0.0,
    speed: float = 0.0,
    p0_by_tick: np.ndarray | None = None,
    p1_by_tick: np.ndarray | None = None,
    radii: np.ndarray | None = None,
    active_by_tick: np.ndarray | None = None,
    object_order: Sequence[int] | None = None,
    order_rank: dict[int, int] | None = None,
    return_meta: bool = False,
) -> float | tuple[float | None, dict[str, float | bool | None]]:
    """Nearest heading just inside the union of non-wrapping visible cells."""

    if not cells:
        return None
    target = _norm_angle(theta)
    best: float | None = None
    best_d = float("inf")
    best_meta: dict[str, float | bool | None] = {
        "inside_cell": False,
        "owner_lo": None,
        "owner_hi": None,
        "bound": None,
        "outward_dir": 0.0,
    }
    for lo, hi in cells:
        if hi - lo <= GEOM_EPS:
            continue
        meta = {
            "inside_cell": False,
            "owner_lo": float(lo),
            "owner_hi": float(hi),
            "bound": None,
            "outward_dir": 0.0,
        }
        if lo <= target <= hi:
            clamped = target
            meta["inside_cell"] = True
        else:
            raw_bound = lo if abs(_signed_angle_delta(lo, target)) <= abs(_signed_angle_delta(hi, target)) else hi
            outward_dir = -1.0 if abs(raw_bound - lo) <= abs(raw_bound - hi) else 1.0
            meta["bound"] = float(raw_bound)
            meta["outward_dir"] = float(outward_dir)
            if refine_boundaries and target_sig is not None:
                clamped = _polish_boundary_ground_truth(
                    raw_bound,
                    owner_cell=(lo, hi),
                    target_sig=target_sig,
                    outward_dir=outward_dir,
                    cache=cache,
                    origin_xy=np.asarray(origin_xy, dtype=np.float64),
                    origin_radius=origin_radius,
                    speed=speed,
                    p0_by_tick=np.asarray(p0_by_tick, dtype=np.float64),
                    p1_by_tick=np.asarray(p1_by_tick, dtype=np.float64),
                    radii=np.asarray(radii, dtype=np.float64),
                    active_by_tick=np.asarray(active_by_tick, dtype=bool),
                    object_order=list(object_order or []),
                    order_rank=order_rank,
                )
            else:
                clamped = _boundary_angle_inside(lo, hi, raw_bound)
        d = abs(_signed_angle_delta(clamped, target))
        if d < best_d:
            best_d = d
            best = _norm_angle(clamped)
            best_meta = meta
    if return_meta:
        return best, best_meta
    return best


def _boundary_angle_inside(
    lo: float,
    hi: float,
    bound: float,
    *,
    inside_eps: float = AIM_INSIDE_EDGE_EPS,
) -> float:
    if abs(bound - lo) <= abs(bound - hi):
        inner = lo + inside_eps
        if inner >= hi - inside_eps:
            return _norm_angle(0.5 * (lo + hi))
        return _norm_angle(inner)
    inner = hi - inside_eps
    if inner <= lo + inside_eps:
        return _norm_angle(0.5 * (lo + hi))
    return _norm_angle(inner)


def _edge_angle_inside_furthest_on_side(
    theta_fc: float,
    cells: Sequence[tuple[float, float]],
    *,
    side: int,
    refine_boundaries: bool = True,
    target_sig: tuple[str, int] | None = None,
    cache: "OcclusionWalkCache" | None = None,
    origin_xy: np.ndarray | None = None,
    origin_radius: float = 0.0,
    speed: float = 0.0,
    p0_by_tick: np.ndarray | None = None,
    p1_by_tick: np.ndarray | None = None,
    radii: np.ndarray | None = None,
    active_by_tick: np.ndarray | None = None,
    object_order: Sequence[int] | None = None,
    order_rank: dict[int, int] | None = None,
    return_meta: bool = False,
) -> float | tuple[float | None, dict[str, float | None]]:
    """Heading just inside the visible boundary farthest on one side of ``theta_fc``."""

    if side not in (-1, 1):
        raise ValueError("side must be -1 (right/cw) or +1 (left/ccw)")
    if not cells:
        return None

    ref = _norm_angle(theta_fc)
    best_bound: float | None = None
    best_dist = -1.0
    owner: tuple[float, float] | None = None
    for lo, hi in cells:
        if hi - lo <= GEOM_EPS:
            continue
        for bound in (lo, hi):
            delta = _signed_angle_delta(bound, ref)
            if side * delta <= 0.0:
                continue
            dist = abs(delta)
            if dist > best_dist:
                best_dist = dist
                best_bound = bound
                owner = (lo, hi)
    if best_bound is None or owner is None:
        if return_meta:
            return None, {"owner_lo": None, "owner_hi": None, "bound": None, "outward_dir": 0.0}
        return None
    lo, hi = owner
    meta = {
        "owner_lo": float(lo),
        "owner_hi": float(hi),
        "bound": float(best_bound),
        "outward_dir": float(-1.0 if abs(best_bound - lo) <= abs(best_bound - hi) else 1.0),
    }
    if refine_boundaries and target_sig is not None:
        outward_dir = float(meta["outward_dir"])
        angle = _polish_boundary_ground_truth(
            best_bound,
            owner_cell=(lo, hi),
            target_sig=target_sig,
            outward_dir=outward_dir,
            cache=cache,
            origin_xy=np.asarray(origin_xy, dtype=np.float64),
            origin_radius=origin_radius,
            speed=speed,
            p0_by_tick=np.asarray(p0_by_tick, dtype=np.float64),
            p1_by_tick=np.asarray(p1_by_tick, dtype=np.float64),
            radii=np.asarray(radii, dtype=np.float64),
            active_by_tick=np.asarray(active_by_tick, dtype=bool),
            object_order=list(object_order or []),
            order_rank=order_rank,
        )
        if return_meta:
            return angle, meta
        return angle
    angle = _boundary_angle_inside(lo, hi, best_bound)
    if return_meta:
        return angle, meta
    return angle


def _pick_planet_aim_from_visible_cells(
    cells: Sequence[tuple[float, float]],
    first_contact_angle: float,
    *,
    allow_edge_aim: bool = True,
    refine_boundaries: bool,
    cache: "OcclusionWalkCache" | None,
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    p0_by_tick: np.ndarray,
    p1_by_tick: np.ndarray,
    radii: np.ndarray,
    active_by_tick: np.ndarray,
    slot: int,
    object_order: Sequence[int],
    debug_context: dict[str, Any] | None = None,
    return_job: bool = False,
) -> tuple[float | None, int, float] | tuple[float | None, int, float, IntervalRefineJob | None]:
    """Pick launch angle prioritising earliest planet hit tick, then edge precision."""

    global _INTERVAL_AIM_STATS
    _INTERVAL_AIM_STATS.planet_evals += 1
    width = _cells_total_width(cells)
    if width <= GEOM_EPS:
        for label in ("primary", "edge_same_side", "edge_other_side"):
            _INTERVAL_AIM_STATS.note_rejected(label, "no_visible_cells")
        _INTERVAL_AIM_STATS.selected["none_all_three_rejected"] += 1
        _INTERVAL_AIM_STATS.all_rejected["no_visible_cells"] += 1
        return (None, -1, 0.0, None) if return_job else (None, -1, 0.0)

    theta_fc = _norm_angle(first_contact_angle)
    target_sig = ("planet", int(slot))
    order_rank = {int(s): i for i, s in enumerate(object_order)}
    theta_primary, primary_meta = _closest_angle_in_cells(
        theta_fc,
        cells,
        refine_boundaries=refine_boundaries,
        target_sig=target_sig,
        cache=cache,
        origin_xy=origin_xy,
        origin_radius=origin_radius,
        speed=speed,
        p0_by_tick=p0_by_tick,
        p1_by_tick=p1_by_tick,
        radii=radii,
        active_by_tick=active_by_tick,
        object_order=object_order,
        order_rank=order_rank,
        return_meta=True,
    )
    primary_angle, tick_primary, reason_primary = _candidate_tick_single_planet(
        theta_primary,
        origin_xy=origin_xy,
        origin_radius=origin_radius,
        speed=speed,
        p0_by_tick=p0_by_tick,
        p1_by_tick=p1_by_tick,
        radii=radii,
        active_by_tick=active_by_tick,
        slot=slot,
    )
    if reason_primary != "ok":
        _INTERVAL_AIM_STATS.note_rejected("primary", reason_primary)

    if theta_primary is None:
        primary_side = 1
    else:
        primary_side = 1 if _signed_angle_delta(theta_primary, theta_fc) >= 0.0 else -1
    secondary_side = -primary_side

    edge_angles_by_label: dict[str, float | None] = {}
    edge_meta_by_label: dict[str, dict[str, float | None]] = {}
    if allow_edge_aim:
        edge_angles_by_label["edge_same_side"], edge_meta_by_label["edge_same_side"] = _edge_angle_inside_furthest_on_side(
                theta_fc,
                cells,
                side=primary_side,
                refine_boundaries=refine_boundaries,
                target_sig=target_sig,
                cache=cache,
                origin_xy=origin_xy,
                origin_radius=origin_radius,
                speed=speed,
                p0_by_tick=p0_by_tick,
                p1_by_tick=p1_by_tick,
                radii=radii,
                active_by_tick=active_by_tick,
                object_order=object_order,
                order_rank=order_rank,
                return_meta=True,
            )
        edge_angles_by_label["edge_other_side"], edge_meta_by_label["edge_other_side"] = _edge_angle_inside_furthest_on_side(
                theta_fc,
                cells,
                side=secondary_side,
                refine_boundaries=refine_boundaries,
                target_sig=target_sig,
                cache=cache,
                origin_xy=origin_xy,
                origin_radius=origin_radius,
                speed=speed,
                p0_by_tick=p0_by_tick,
                p1_by_tick=p1_by_tick,
                radii=radii,
                active_by_tick=active_by_tick,
                object_order=object_order,
                order_rank=order_rank,
                return_meta=True,
            )
    else:
        for label in ("edge_same_side", "edge_other_side"):
            edge_angles_by_label[label] = None
            edge_meta_by_label[label] = {}

    seen_angles: list[float] = []
    candidate_results: dict[str, tuple[float | None, int, str]] = {"primary": (primary_angle, tick_primary, reason_primary)}
    if primary_angle is not None:
        seen_angles.append(float(primary_angle))
    for label in ("edge_same_side", "edge_other_side"):
        theta_edge = edge_angles_by_label[label]
        if theta_edge is None:
            candidate_results[label] = (None, -1, "absent")
            _INTERVAL_AIM_STATS.note_rejected(label, "absent")
            continue
        if any(abs(_signed_angle_delta(theta_edge, prev)) <= AIM_INSIDE_EDGE_EPS for prev in seen_angles):
            candidate_results[label] = (float(theta_edge), -1, "duplicate_angle")
            _INTERVAL_AIM_STATS.note_rejected(label, "duplicate_angle")
            continue
        edge_angle, tick_edge, reason_edge = _candidate_tick_single_planet(
            theta_edge,
            origin_xy=origin_xy,
            origin_radius=origin_radius,
            speed=speed,
            p0_by_tick=p0_by_tick,
            p1_by_tick=p1_by_tick,
            radii=radii,
            active_by_tick=active_by_tick,
            slot=slot,
        )
        candidate_results[label] = (edge_angle, tick_edge, reason_edge)
        seen_angles.append(float(theta_edge))
        if reason_edge != "ok":
            _INTERVAL_AIM_STATS.note_rejected(label, reason_edge)

    if reason_primary != "ok":
        edge_debug = []
        for label in ("edge_same_side", "edge_other_side"):
            edge_angle, tick_edge, reason_edge = candidate_results[label]
            edge_debug.append(
                f"{label}=(angle={edge_angle!r}, tick={tick_edge}, reason={reason_edge})"
            )
        ctx = debug_context or {}
        ctx_parts: list[str] = []
        for key in (
            "phase",
            "game_step",
            "ego_player",
            "origin_slot",
            "frac_idx",
            "micro_idx",
            "selected_target_slot",
        ):
            if key not in ctx:
                continue
            value = ctx[key]
            if isinstance(value, int) and value < 0:
                continue
            if value is None:
                continue
            ctx_parts.append(f"{key}={value!r}")
        msg = (
            "[orbit_wars] primary interval aim invalid"
            + (f"\n  context: {' '.join(ctx_parts)}" if ctx_parts else "")
            + f"\n  slot={slot} first_contact_angle={theta_fc!r}"
            + f"\n  cells={list(cells)!r}"
            + f"\n  primary=(angle={primary_angle!r}, tick={tick_primary}, reason={reason_primary})"
            + f"\n  {' '.join(edge_debug)}"
        )
        print(msg, file=sys.stderr, flush=True)

    if reason_primary == "ok":
        for label in ("edge_same_side", "edge_other_side"):
            edge_angle, tick_edge, reason_edge = candidate_results[label]
            if reason_edge == "ok" and tick_edge == tick_primary:
                _INTERVAL_AIM_STATS.selected[label] += 1
                job = IntervalRefineJob(
                    variant=label,
                    slot=int(slot),
                    coarse_angle=float(edge_angle),
                    owner_lo=float(edge_meta_by_label[label]["owner_lo"]),
                    owner_hi=float(edge_meta_by_label[label]["owner_hi"]),
                    bound=float(edge_meta_by_label[label]["bound"]),
                    outward_dir=float(edge_meta_by_label[label]["outward_dir"]),
                )
                return (float(edge_angle), int(tick_edge), width, job) if return_job else (float(edge_angle), int(tick_edge), width)
        _INTERVAL_AIM_STATS.selected["primary"] += 1
        if primary_meta.get("inside_cell"):
            job = IntervalRefineJob(
                variant="primary",
                slot=int(slot),
                coarse_angle=float(primary_angle),
                owner_lo=float(primary_meta["owner_lo"]),
                owner_hi=float(primary_meta["owner_hi"]),
                bound=None,
                outward_dir=0.0,
            )
        else:
            bound = primary_meta.get("bound")
            outward_dir = primary_meta.get("outward_dir", 0.0)
            job = IntervalRefineJob(
                variant="primary",
                slot=int(slot),
                coarse_angle=float(primary_angle),
                owner_lo=float(primary_meta["owner_lo"]),
                owner_hi=float(primary_meta["owner_hi"]),
                bound=None if bound is None else float(bound),
                outward_dir=float(outward_dir),
            )
        return (float(primary_angle), int(tick_primary), width, job) if return_job else (float(primary_angle), int(tick_primary), width)

    edge_valid = [
        label
        for label in ("edge_same_side", "edge_other_side")
        if candidate_results[label][2] == "ok"
    ]
    reason_all = "primary_invalid_with_edge_valid" if edge_valid else "all_three_rejected"
    if not edge_valid:
        reasons = []
        for label in ("primary", "edge_same_side", "edge_other_side"):
            reasons.append(f"{label}={candidate_results[label][2]}")
        _INTERVAL_AIM_STATS.all_rejected[" ".join(reasons)] += 1
    _INTERVAL_AIM_STATS.selected[f"none_{reason_all}"] += 1
    return (None, -1, 0.0, None) if return_job else (None, -1, 0.0)


def refine_interval_refine_job(
    job: IntervalRefineJob,
    *,
    cache: "OcclusionWalkCache" | None,
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    p0_by_tick: np.ndarray,
    p1_by_tick: np.ndarray,
    radii: np.ndarray,
    active_by_tick: np.ndarray,
    object_order: Sequence[int],
) -> tuple[float | None, int, str]:
    angle = float(job.coarse_angle)
    if job.bound is None or abs(float(job.outward_dir)) <= 0.0:
        return _candidate_tick_single_planet(
            angle,
            origin_xy=origin_xy,
            origin_radius=origin_radius,
            speed=speed,
            p0_by_tick=p0_by_tick,
            p1_by_tick=p1_by_tick,
            radii=radii,
            active_by_tick=active_by_tick,
            slot=int(job.slot),
        )

    refined = _polish_boundary_ground_truth(
        float(job.bound),
        owner_cell=(float(job.owner_lo), float(job.owner_hi)),
        target_sig=("planet", int(job.slot)),
        outward_dir=float(job.outward_dir),
        cache=cache,
        origin_xy=np.asarray(origin_xy, dtype=np.float64),
        origin_radius=float(origin_radius),
        speed=float(speed),
        p0_by_tick=np.asarray(p0_by_tick, dtype=np.float64),
        p1_by_tick=np.asarray(p1_by_tick, dtype=np.float64),
        radii=np.asarray(radii, dtype=np.float64),
        active_by_tick=np.asarray(active_by_tick, dtype=bool),
        object_order=list(object_order),
        order_rank={int(s): i for i, s in enumerate(object_order)},
    )
    return _candidate_tick_single_planet(
        refined,
        origin_xy=origin_xy,
        origin_radius=origin_radius,
        speed=speed,
        p0_by_tick=p0_by_tick,
        p1_by_tick=p1_by_tick,
        radii=radii,
        active_by_tick=active_by_tick,
        slot=int(job.slot),
    )


def _first_contact_bearing_at_enter(
    origin: np.ndarray,
    launch_off: float,
    grow_rate: float,
    t_enter: float,
    circle_center: np.ndarray,
    circle_radius: float,
) -> float:
    """External tangent bearing at window entry (tangent first-contact time)."""

    from orbit_wars_pt.tangent_geometry_np import intersection_angles_from_grow_center

    gr = float(launch_off) + float(grow_rate) * float(t_enter)
    c = np.asarray(circle_center, dtype=np.float64)
    q = c - np.asarray(origin, dtype=np.float64)
    angs = intersection_angles_from_grow_center(q, float(circle_radius), gr)
    if angs is not None:
        return _norm_angle(float(angs[0]))
    return _norm_angle(math.atan2(q[1], q[0]))


def sweep_interval_best_targets(
    precomputed_hits: Sequence[Sequence[Sequence[AngleInterval]]],
    *,
    object_order: Sequence[int] | None = None,
    include_board: bool = True,
    include_sun: bool = True,
    origin_xy: np.ndarray | None = None,
    origin_radius: float = 0.0,
    speed: float = 0.0,
    board_size: float = BOARD_SIZE,
    sun_radius: float = SUN_RADIUS,
    samples_per_span: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, np.ndarray]:
    """Occlusion sweep; returns ``(angle[P], width[P], valid[P], overflow, hit_tick[P])``."""

    ticks = len(precomputed_hits)
    planets = len(precomputed_hits[0]) if ticks else 0
    order = list(range(planets)) if object_order is None else list(object_order)
    blocked: list[AngleInterval] = []
    best_angle = np.zeros((planets,), dtype=np.float64)
    best_width = np.zeros((planets,), dtype=np.float64)
    best_tick = np.full((planets,), -1, dtype=np.int32)
    overflow = False

    origin = None if origin_xy is None else np.asarray(origin_xy, dtype=np.float64)

    for tick in range(ticks):
        row = precomputed_hits[tick]
        for obj_idx in order:
            if obj_idx < 0 or obj_idx >= planets:
                continue
            hit = row[obj_idx]
            if not hit:
                continue
            cells = set_subtract_cells(hit, blocked)
            mid, width = _widest_cell_midpoint_and_width(cells)
            if mid is not None and width > best_width[obj_idx]:
                best_width[obj_idx] = width
                best_angle[obj_idx] = mid
                best_tick[obj_idx] = tick
            blocked = union_angle_intervals([*blocked, *hit])

        if include_board and origin is not None:
            b_hit = board_exit_angle_intervals(
                origin, origin_radius, speed, tick, board_size=board_size
            )
            blocked = union_angle_intervals([*blocked, *b_hit])

        if include_sun and origin is not None:
            sun_xy = np.asarray([CENTER, CENTER], dtype=np.float64)
            s_hit = tick_hit_intervals_sampled(
                origin,
                origin_radius,
                speed,
                tick,
                sun_xy,
                sun_xy,
                max(0.0, float(sun_radius) - 1e-9),
                object_active=True,
                samples_per_span=samples_per_span,
            )
            blocked = union_angle_intervals([*blocked, *s_hit])

    valid = best_width > GEOM_EPS
    return (
        best_angle.astype(np.float64),
        best_width.astype(np.float32),
        valid,
        overflow,
        best_tick,
    )


def first_hit_at_angle_interval(
    angle: float,
    precomputed_hits: Sequence[Sequence[Sequence[AngleInterval]]],
    *,
    object_order: Sequence[int] | None = None,
    include_board: bool = True,
    include_sun: bool = True,
    origin_xy: np.ndarray | None = None,
    origin_radius: float = 0.0,
    speed: float = 0.0,
    board_size: float = BOARD_SIZE,
    sun_radius: float = SUN_RADIUS,
    samples_per_span: int = 9,
) -> tuple[str, int, int]:
    """First terminal event at ``angle`` under interval occlusion (matches sweep rules).

    Returns ``(kind, code, tick)`` with ``kind`` in ``planet``, ``sun``, ``board``, ``none``.
    ``code`` is the planet slot for ``planet``, else ``-1``.
    """

    ticks = len(precomputed_hits)
    planets = len(precomputed_hits[0]) if ticks else 0
    order = list(range(planets)) if object_order is None else list(object_order)
    blocked: list[AngleInterval] = []
    origin = None if origin_xy is None else np.asarray(origin_xy, dtype=np.float64)
    theta = float(angle)

    for tick in range(ticks):
        row = precomputed_hits[tick]
        for obj_idx in order:
            if obj_idx < 0 or obj_idx >= planets:
                continue
            hit = row[obj_idx]
            if not hit:
                continue
            available = subtract_angle_intervals(hit, blocked)
            if angle_in_intervals(theta, available):
                return ("planet", int(obj_idx), int(tick))
            blocked = union_angle_intervals([*blocked, *hit])

        if include_board and origin is not None:
            b_hit = board_exit_angle_intervals(
                origin, origin_radius, speed, tick, board_size=board_size
            )
            if angle_in_intervals(theta, b_hit):
                return ("board", -1, int(tick))
            blocked = union_angle_intervals([*blocked, *b_hit])

        if include_sun and origin is not None:
            sun_xy = np.asarray([CENTER, CENTER], dtype=np.float64)
            s_hit = tick_hit_intervals_sampled(
                origin,
                origin_radius,
                speed,
                tick,
                sun_xy,
                sun_xy,
                max(0.0, float(sun_radius) - 1e-9),
                object_active=True,
                samples_per_span=samples_per_span,
            )
            if angle_in_intervals(theta, s_hit):
                return ("sun", -1, int(tick))
            blocked = union_angle_intervals([*blocked, *s_hit])

    return ("none", -1, -1)


def first_hit_tick_single_planet_raycast(
    angle: float,
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    p0_by_tick: np.ndarray,
    p1_by_tick: np.ndarray,
    radius: float,
    active_by_tick: np.ndarray,
    *,
    rim_offset: float = LAUNCH_RIM_OFFSET,
) -> int:
    """Earliest discrete tick where fleet at ``angle`` hits one planet (swept segment test)."""

    origin = np.asarray(origin_xy, dtype=np.float64)
    theta = float(angle) % TAU
    direction = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
    launch_off = float(origin_radius) + float(rim_offset)
    pos = origin + launch_off * direction
    ticks = int(active_by_tick.shape[0])
    r = float(radius)
    v = float(speed)

    for tick in range(ticks):
        a1 = pos + v * direction
        if bool(active_by_tick[tick]):
            p0 = np.asarray(p0_by_tick[tick], dtype=np.float64)
            p1 = np.asarray(p1_by_tick[tick], dtype=np.float64)
            d0 = pos - p0
            dv = (a1 - pos) - (p1 - p0)
            qa = float(np.dot(dv, dv))
            qb = float(2.0 * np.dot(d0, dv))
            qc = float(np.dot(d0, d0) - r * r)
            if qa < 1e-12:
                if qc <= 0.0:
                    return tick
            else:
                disc = qb * qb - 4.0 * qa * qc
                if disc >= 0.0:
                    sd = math.sqrt(max(disc, 0.0))
                    t1 = (-qb - sd) / (2.0 * qa)
                    t2 = (-qb + sd) / (2.0 * qa)
                    if t2 >= 0.0 and t1 <= 1.0:
                        return tick
        pos = a1

    return -1


def _refine_planet_hit_ticks_single_rays(
    best_angle: np.ndarray,
    best_tick: np.ndarray,
    valid: np.ndarray,
    *,
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    p0_by_tick: np.ndarray,
    p1_by_tick: np.ndarray,
    radii: np.ndarray,
    active_by_tick: np.ndarray,
) -> None:
    """Replace ``best_tick`` for valid planets with per-target swept raycast ETAs."""

    for slot in range(int(best_angle.shape[0])):
        if not valid[slot]:
            continue
        tick = first_hit_tick_single_planet_raycast(
            float(best_angle[slot]),
            origin_xy,
            origin_radius,
            speed,
            p0_by_tick[:, slot, :],
            p1_by_tick[:, slot, :],
            float(radii[slot]),
            active_by_tick[:, slot],
        )
        if tick >= 0:
            best_tick[slot] = tick
        else:
            valid[slot] = False
            best_tick[slot] = -1


def first_hit_at_angle_orthogonal(
    angle: float,
    events: Sequence[OrthogonalHitEvent],
    *,
    object_order: Sequence[int] | None = None,
    include_board: bool = True,
    include_sun: bool = True,
    origin_xy: np.ndarray | None = None,
    origin_radius: float = 0.0,
    speed: float = 0.0,
    horizon: int = 24,
    board_size: float = BOARD_SIZE,
    p0_by_tick: np.ndarray | None = None,
    p1_by_tick: np.ndarray | None = None,
    radii: np.ndarray | None = None,
    active_by_tick: np.ndarray | None = None,
) -> tuple[str, int, int]:
    """First terminal event at ``angle`` under occlusion (``floor(t)``, collision rank)."""

    from orbit_wars_pt.orthogonal_geometry_np import cone_to_angle_intervals

    planets = max(
        (int(e.slot) for e in events if e.kind == "planet" and e.slot >= 0),
        default=-1,
    ) + 1
    order = list(range(planets)) if object_order is None else list(object_order)
    order_rank = {int(s): i for i, s in enumerate(order)}
    sorted_events = sorted(events, key=lambda e: _event_sort_key(e, order_rank))
    blocked: list[AngleInterval] = []
    origin = None if origin_xy is None else np.asarray(origin_xy, dtype=np.float64)
    theta = float(angle)
    use_ray_eta = (
        p0_by_tick is not None
        and p1_by_tick is not None
        and radii is not None
        and active_by_tick is not None
    )

    for event in sorted_events:
        hit = cone_to_angle_intervals(event.angle_lo, event.angle_hi)
        if event.kind == "planet" and 0 <= event.slot < planets:
            available = subtract_angle_intervals(hit, blocked)
            if angle_in_intervals(theta, available):
                slot = int(event.slot)
                if use_ray_eta:
                    tick = first_hit_tick_single_planet_raycast(
                        theta,
                        origin,
                        origin_radius,
                        speed,
                        p0_by_tick[:, slot, :],
                        p1_by_tick[:, slot, :],
                        float(radii[slot]),
                        active_by_tick[:, slot],
                    )
                    if tick < 0:
                        blocked = union_angle_intervals([*blocked, *hit])
                        continue
                    return ("planet", slot, tick)
                tick = int(min(max(math.floor(event.t), 0), max(horizon - 1, 0)))
                return ("planet", slot, tick)
        elif event.kind == "sun" and include_sun:
            available = subtract_angle_intervals(hit, blocked)
            if angle_in_intervals(theta, available):
                tick = int(min(max(math.floor(event.t), 0), max(horizon - 1, 0)))
                return ("sun", -1, tick)
        blocked = union_angle_intervals([*blocked, *hit])

    if origin is not None:
        for tick in range(int(horizon)):
            if include_board:
                b_hit = board_exit_angle_intervals(
                    origin, origin_radius, speed, tick, board_size=board_size
                )
                if angle_in_intervals(theta, b_hit):
                    return ("board", -1, tick)
                blocked = union_angle_intervals([*blocked, *b_hit])

    return ("none", -1, -1)


def _rotating_planet_chord_polyline(
    centre_xy: np.ndarray,
    orbital_radius: float,
    angular_velocity: float,
    *,
    horizon: int,
    step_count: int = 0,
    orbit_center: np.ndarray | None = None,
) -> np.ndarray:
    """Planet-centre polyline with one chord per tick (env-style linearised orbit)."""

    o = np.asarray(
        [CENTER, CENTER] if orbit_center is None else orbit_center,
        dtype=np.float64,
    )
    p0 = np.asarray(centre_xy, dtype=np.float64)
    rho = float(orbital_radius)
    w = float(angular_velocity)
    th0 = math.atan2(p0[1] - o[1], p0[0] - o[0])
    rows: list[np.ndarray] = [p0.copy()]
    for dt in range(1, int(horizon) + 1):
        advance = int(dt) - 1 if int(step_count) <= 0 else int(dt)
        th = th0 + w * float(advance)
        rows.append(
            o
            + rho
            * np.array([math.cos(th), math.sin(th)], dtype=np.float64)
        )
    return np.stack(rows, axis=0)


def _comet_chord_polyline_points(
    pos: np.ndarray,
    slot: int,
    *,
    comet_paths: np.ndarray,
    comet_path_lengths: np.ndarray,
    comet_group_active: np.ndarray,
    comet_path_index: np.ndarray,
    comet_slots: np.ndarray,
    horizon: float,
) -> np.ndarray:
    """Chord polyline for a comet; duplicate the terminal knot when the path ends.

    Matches per-tick swept hits after the last path point: one tick with zero comet
    motion at the endpoint (see ``_forecast_incoming_fleets`` when ``c0 == c1``).
    """

    pts_list = [np.asarray(pos, dtype=np.float64).copy()]
    path_exhausted = False
    for g in range(int(comet_group_active.shape[0])):
        if not comet_group_active[g]:
            continue
        for k in range(int(comet_slots.shape[1])):
            if int(comet_slots[g, k]) != slot:
                continue
            idx = int(comet_path_index[g])
            length = int(comet_path_lengths[g, k])
            path = comet_paths[g, k]
            for dt in range(1, int(horizon) + 2):
                pi = idx + dt
                if pi >= length:
                    path_exhausted = True
                    break
                pts_list.append(np.asarray(path[pi], dtype=np.float64))
            break
        else:
            continue
        break

    if len(pts_list) < 2:
        return np.stack([pts_list[0], pts_list[0]], axis=0)

    pts = np.stack(pts_list, axis=0)
    if path_exhausted:
        pts = np.vstack([pts, pts[-1:]])
    return pts


def collect_tangent_hit_events(
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    planets: np.ndarray,
    planet_active: np.ndarray,
    initial_planets: np.ndarray,
    initial_active: np.ndarray,
    comet_paths: np.ndarray,
    comet_path_lengths: np.ndarray,
    comet_group_active: np.ndarray,
    comet_path_index: np.ndarray,
    comet_slots: np.ndarray,
    comet_planet_ids: np.ndarray,
    angular_velocity: float,
    step_count: int,
    *,
    horizon: float,
    include_sun: bool = True,
    sun_radius: float = SUN_RADIUS,
) -> list[OrthogonalHitEvent]:
    """Tangent growing-circle hits: overlap windows with min/max intersection bearings.

    Reuses ``OrthogonalHitEvent`` for the occlusion sweep (``angle_lo``/``angle_hi`` span).
    """

    from orbit_wars_pt.tangent_geometry_np import (
        angular_intersection_extrema_polyline,
        angular_intersection_extrema_stationary,
        intersection_windows,
        make_polyline_motion,
    )

    origin = np.asarray(origin_xy, dtype=np.float64)
    launch_off = float(origin_radius) + LAUNCH_RIM_OFFSET
    v = float(speed)
    t_hor = float(horizon)
    pa = np.asarray(planet_active, dtype=bool)
    ia = np.asarray(initial_active, dtype=bool)
    planets = np.asarray(planets, dtype=np.float64)
    init_pos = np.asarray(initial_planets, dtype=np.float64)[:, 2:4]
    delta = init_pos - np.asarray([CENTER, CENTER], dtype=np.float64)
    orbital_r = np.linalg.norm(delta, axis=1)
    rotating = pa & ia & (orbital_r + planets[:, 4] < ROTATION_RADIUS_LIMIT)

    comet_slot_set: set[int] = set()
    for g in range(int(comet_group_active.shape[0])):
        if not comet_group_active[g]:
            continue
        for k in range(4):
            slot = int(comet_slots[g, k])
            if 0 <= slot < len(pa):
                comet_slot_set.add(slot)

    events: list[OrthogonalHitEvent] = []
    o_center = np.array([CENTER, CENTER], dtype=np.float64)

    def add_planet_windows(
        slot: int,
        center_at,
        velocity_at,
        knot_times: Sequence[float],
        r: float,
        *,
        kind: str = "planet",
        polyline_points: np.ndarray | None = None,
        stationary_center: np.ndarray | None = None,
    ) -> None:
        windows = intersection_windows(
            center_at,
            origin,
            r,
            launch_off,
            v,
            t_hor,
            polyline_points=polyline_points,
            stationary_center=stationary_center,
        )
        for t_enter, t_exit in windows:
            if polyline_points is not None:
                ext = angular_intersection_extrema_polyline(
                    polyline_points,
                    r,
                    origin,
                    v,
                    launch_off,
                    t_enter,
                    t_exit,
                )
            else:
                ext = angular_intersection_extrema_stationary(
                    stationary_center,
                    r,
                    origin,
                    v,
                    launch_off,
                    t_enter,
                    t_exit,
                )
            if ext is None:
                continue
            a_lo = float(ext["min"]["angle"])
            a_hi = float(ext["max"]["angle"])
            if stationary_center is not None:
                c_enter = np.asarray(stationary_center, dtype=np.float64)
            else:
                c_enter = np.asarray(center_at(t_enter), dtype=np.float64)
            fc = _first_contact_bearing_at_enter(
                origin, launch_off, v, t_enter, c_enter, r
            )
            events.append(
                OrthogonalHitEvent(
                    float(t_enter),
                    _norm_angle(a_lo),
                    _norm_angle(a_hi),
                    slot,
                    kind,
                    fc,
                )
            )

    for slot in range(int(planets.shape[0])):
        if not pa[slot]:
            continue
        r = float(planets[slot, 4])
        pos = planets[slot, 2:4]

        if slot in comet_slot_set:
            pts = _comet_chord_polyline_points(
                pos,
                slot,
                comet_paths=comet_paths,
                comet_path_lengths=comet_path_lengths,
                comet_group_active=comet_group_active,
                comet_path_index=comet_path_index,
                comet_slots=comet_slots,
                horizon=t_hor,
            )
            center_at, velocity_at, knot_times = make_polyline_motion(pts)
            add_planet_windows(
                slot,
                center_at,
                velocity_at,
                knot_times,
                r,
                polyline_points=pts,
            )
            continue

        if rotating[slot]:
            pts = _rotating_planet_chord_polyline(
                pos,
                float(orbital_r[slot]),
                float(angular_velocity),
                horizon=int(t_hor),
                step_count=int(step_count),
                orbit_center=o_center,
            )
            center_at, velocity_at, knot_times = make_polyline_motion(pts)
            add_planet_windows(
                slot,
                center_at,
                velocity_at,
                knot_times,
                r,
                polyline_points=pts,
            )
        else:
            center_at = lambda t, p=pos.copy(): p  # noqa: E731
            velocity_at = lambda t: np.zeros(2, dtype=np.float64)  # noqa: E731
            add_planet_windows(
                slot,
                center_at,
                velocity_at,
                [],
                r,
                stationary_center=pos,
            )

    if include_sun:
        center_at = lambda t, c=o_center.copy(): c  # noqa: E731
        velocity_at = lambda t: np.zeros(2, dtype=np.float64)  # noqa: E731
        add_planet_windows(
            -1,
            center_at,
            velocity_at,
            [],
            float(sun_radius),
            kind="sun",
            stationary_center=o_center,
        )

    return events


def _interval_use_tangent_geometry() -> bool:
    import os

    raw = os.environ.get("ORBIT_WARS_INTERVAL_GEOMETRY", "tangent").strip().lower()
    return raw not in {"orthogonal", "cone", "sampled", "0", "false", "off", "no"}


def collect_hit_events(
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    planets: np.ndarray,
    planet_active: np.ndarray,
    initial_planets: np.ndarray,
    initial_active: np.ndarray,
    comet_paths: np.ndarray,
    comet_path_lengths: np.ndarray,
    comet_group_active: np.ndarray,
    comet_path_index: np.ndarray,
    comet_slots: np.ndarray,
    comet_planet_ids: np.ndarray,
    angular_velocity: float,
    step_count: int,
    *,
    horizon: float,
    include_sun: bool = True,
    sun_radius: float = SUN_RADIUS,
) -> list[OrthogonalHitEvent]:
    """Dispatch orthogonal (shelved) vs external/internal tangent geometry."""

    if _interval_use_tangent_geometry():
        return collect_tangent_hit_events(
            origin_xy,
            origin_radius,
            speed,
            planets,
            planet_active,
            initial_planets,
            initial_active,
            comet_paths,
            comet_path_lengths,
            comet_group_active,
            comet_path_index,
            comet_slots,
            comet_planet_ids,
            angular_velocity,
            step_count,
            horizon=horizon,
            include_sun=include_sun,
            sun_radius=sun_radius,
        )
    return collect_orthogonal_hit_events(
        origin_xy,
        origin_radius,
        speed,
        planets,
        planet_active,
        initial_planets,
        initial_active,
        comet_paths,
        comet_path_lengths,
        comet_group_active,
        comet_path_index,
        comet_slots,
        comet_planet_ids,
        angular_velocity,
        step_count,
        horizon=horizon,
        include_sun=include_sun,
        sun_radius=sun_radius,
    )


def collect_orthogonal_hit_events(
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    planets: np.ndarray,
    planet_active: np.ndarray,
    initial_planets: np.ndarray,
    initial_active: np.ndarray,
    comet_paths: np.ndarray,
    comet_path_lengths: np.ndarray,
    comet_group_active: np.ndarray,
    comet_path_index: np.ndarray,
    comet_slots: np.ndarray,
    comet_planet_ids: np.ndarray,
    angular_velocity: float,
    step_count: int,
    *,
    horizon: float,
    include_sun: bool = True,
    sun_radius: float = SUN_RADIUS,
) -> list[OrthogonalHitEvent]:
    """Orthogonal growing-circle hits for orbiting/static planets, polyline comets, and sun.

    Rotating planets and comets use chord polylines (all roots in ``[0, horizon]``).
    Set ``_USE_EXACT_CIRCULAR_ORBIT_ORTHOGONAL`` to use exact circular-orbit tangency instead.
    """

    from orbit_wars_pt.orthogonal_geometry_np import (
        orthogonal_hit_circular_orbit,
        orthogonal_hit_stationary,
        orthogonal_hits_polyline,
    )

    origin = np.asarray(origin_xy, dtype=np.float64)
    launch_off = float(origin_radius) + LAUNCH_RIM_OFFSET
    v = float(speed)
    t_hor = float(horizon)
    pa = np.asarray(planet_active, dtype=bool)
    ia = np.asarray(initial_active, dtype=bool)
    planets = np.asarray(planets, dtype=np.float64)
    init_pos = np.asarray(initial_planets, dtype=np.float64)[:, 2:4]
    delta = init_pos - np.asarray([CENTER, CENTER], dtype=np.float64)
    orbital_r = np.linalg.norm(delta, axis=1)
    initial_angle = np.arctan2(delta[:, 1], delta[:, 0])
    rotating = pa & ia & (orbital_r + planets[:, 4] < ROTATION_RADIUS_LIMIT)

    comet_slot_set: set[int] = set()
    for g in range(int(comet_group_active.shape[0])):
        if not comet_group_active[g]:
            continue
        for k in range(4):
            slot = int(comet_slots[g, k])
            if 0 <= slot < len(pa):
                comet_slot_set.add(slot)

    events: list[OrthogonalHitEvent] = []
    o_center = np.array([CENTER, CENTER], dtype=np.float64)

    for slot in range(int(planets.shape[0])):
        if not pa[slot]:
            continue
        r = float(planets[slot, 4])
        pos = planets[slot, 2:4]

        if slot in comet_slot_set:
            pts = _comet_chord_polyline_points(
                pos,
                slot,
                comet_paths=comet_paths,
                comet_path_lengths=comet_path_lengths,
                comet_group_active=comet_group_active,
                comet_path_index=comet_path_index,
                comet_slots=comet_slots,
                horizon=t_hor,
            )
            for t, lo, hi in orthogonal_hits_polyline(
                pts, r, origin, v, launch_off, t_hor
            ):
                events.append(OrthogonalHitEvent(t, lo, hi, slot, "planet"))
            continue

        if rotating[slot]:
            if _USE_EXACT_CIRCULAR_ORBIT_ORTHOGONAL:
                delta_pos = np.asarray(pos, dtype=np.float64) - o_center
                th0 = float(np.arctan2(delta_pos[1], delta_pos[0]))
                hit = orthogonal_hit_circular_orbit(
                    o_center,
                    float(orbital_r[slot]),
                    th0,
                    float(angular_velocity),
                    r,
                    origin,
                    v,
                    launch_off,
                    t_hor,
                )
                if hit is not None:
                    t, lo, hi = hit
                    events.append(OrthogonalHitEvent(t, lo, hi, slot, "planet"))
                continue

            pts = _rotating_planet_chord_polyline(
                pos,
                float(orbital_r[slot]),
                float(angular_velocity),
                horizon=int(t_hor),
                orbit_center=o_center,
            )
            if pts.shape[0] < 2:
                continue
            for t, lo, hi in orthogonal_hits_polyline(
                pts, r, origin, v, launch_off, t_hor
            ):
                events.append(OrthogonalHitEvent(t, lo, hi, slot, "planet"))
            continue

        hit = orthogonal_hit_stationary(pos, r, origin, v, launch_off, t_hor)
        if hit is not None:
            t, lo, hi = hit
            events.append(OrthogonalHitEvent(t, lo, hi, slot, "planet"))

    if include_sun:
        hit = orthogonal_hit_stationary(
            o_center,
            float(sun_radius),
            origin,
            v,
            launch_off,
            t_hor,
        )
        if hit is not None:
            t, lo, hi = hit
            events.append(OrthogonalHitEvent(t, lo, hi, -1, "sun"))

    return events


@dataclass(frozen=True)
class OcclusionWalkCache:
    """Pre-sorted event cones + board ticks for fast per-ray occlusion first-hit."""

    planet_count: int
    event_steps: tuple[tuple[tuple[AngleInterval, ...], str, int], ...]
    board_ticks: tuple[tuple[AngleInterval, ...], ...]


def build_occlusion_walk_cache(
    events: Sequence[OrthogonalHitEvent],
    object_order: Sequence[int],
    *,
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    horizon: int,
    board_size: float = BOARD_SIZE,
) -> OcclusionWalkCache:
    """Sort events once and pre-convert hit cones (matches ``first_hit_at_angle_orthogonal``)."""

    from orbit_wars_pt.orthogonal_geometry_np import cone_to_angle_intervals

    planets = (
        max(
            (int(e.slot) for e in events if e.kind == "planet" and e.slot >= 0),
            default=-1,
        )
        + 1
    )
    order_rank = {int(s): i for i, s in enumerate(object_order)}
    sorted_events = sorted(events, key=lambda e: _event_sort_key(e, order_rank))
    steps = tuple(
        (tuple(cone_to_angle_intervals(e.angle_lo, e.angle_hi)), e.kind, int(e.slot))
        for e in sorted_events
    )
    origin = np.asarray(origin_xy, dtype=np.float64)
    board_ticks = tuple(
        tuple(
            board_exit_angle_intervals(
                origin, origin_radius, speed, tick, board_size=board_size
            )
        )
        for tick in range(int(horizon))
    )
    return OcclusionWalkCache(
        planet_count=planets,
        event_steps=steps,
        board_ticks=board_ticks,
    )


def first_hit_signature_occlusion_walk(
    angle: float,
    cache: OcclusionWalkCache,
    *,
    include_sun: bool = True,
    include_board: bool = True,
) -> tuple[str, int]:
    """First-hit ``(kind, code)`` under event occlusion; tick omitted for comparisons."""

    theta = float(angle)
    blocked: list[AngleInterval] = []
    planets = int(cache.planet_count)

    for hit_t, kind, slot in cache.event_steps:
        hit = list(hit_t)
        if kind == "planet" and 0 <= slot < planets:
            if angle_in_intervals(theta, subtract_angle_intervals(hit, blocked)):
                return ("planet", int(slot))
        elif include_sun and kind == "sun":
            if angle_in_intervals(theta, subtract_angle_intervals(hit, blocked)):
                return ("sun", -1)
        blocked = union_angle_intervals([*blocked, *hit])

    if include_board:
        for b_hit_t in cache.board_ticks:
            if angle_in_intervals(theta, list(b_hit_t)):
                return ("board", -1)

    return ("none", -1)


def _event_sort_key(
    event: OrthogonalHitEvent,
    order_rank: dict[int, int],
) -> tuple[int, int]:
    """Sort key for occlusion sweep: integer tick, then Kaggle collision rank."""

    rank = order_rank.get(event.slot, 10_000) if event.kind == "planet" else 10_001
    tick = int(math.floor(event.t)) if math.isfinite(event.t) else 0
    return (tick, rank)


def sweep_interval_best_targets_from_events(
    events: Sequence[OrthogonalHitEvent],
    num_planets: int,
    *,
    object_order: Sequence[int] | None = None,
    include_board: bool = True,
    origin_xy: np.ndarray | None = None,
    origin_radius: float = 0.0,
    speed: float = 0.0,
    horizon: int = 24,
    board_size: float = BOARD_SIZE,
    p0_by_tick: np.ndarray | None = None,
    p1_by_tick: np.ndarray | None = None,
    radii: np.ndarray | None = None,
    active_by_tick: np.ndarray | None = None,
    occlusion_cache: "OcclusionWalkCache" | None = None,
    allow_edge_aim: bool = True,
    refine_boundaries: bool = True,
    debug_context: dict[str, Any] | None = None,
    selected_slots: set[int] | None = None,
    return_jobs: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, bool, np.ndarray, list[IntervalRefineJob | None]]:
    """Occlusion sweep over events sorted by ``floor(t)`` then collision rank.

    Per-planet aim uses first-contact bearing (clamped to post-occlusion cells), then
    optional edge-hugging heading when it hits on the same discrete tick.
    """

    from orbit_wars_pt.orthogonal_geometry_np import cone_to_angle_intervals

    order = list(range(num_planets)) if object_order is None else list(object_order)
    order_rank = {int(s): i for i, s in enumerate(order)}
    sorted_events = sorted(events, key=lambda e: _event_sort_key(e, order_rank))

    blocked: list[AngleInterval] = []
    best_angle = np.zeros((num_planets,), dtype=np.float64)
    best_width = np.zeros((num_planets,), dtype=np.float64)
    best_tick = np.full((num_planets,), -1, dtype=np.int32)
    best_jobs: list[IntervalRefineJob | None] = [None] * int(num_planets)
    overflow = False
    origin = None if origin_xy is None else np.asarray(origin_xy, dtype=np.float64)
    can_aim = (
        origin is not None
        and p0_by_tick is not None
        and p1_by_tick is not None
        and radii is not None
        and active_by_tick is not None
    )

    for event in sorted_events:
        hit = cone_to_angle_intervals(event.angle_lo, event.angle_hi)
        if event.kind == "planet" and 0 <= event.slot < num_planets:
            if selected_slots is not None and int(event.slot) not in selected_slots:
                blocked = union_angle_intervals([*blocked, *hit])
                continue
            cells = set_subtract_cells(hit, blocked)
            if can_aim:
                picked = _pick_planet_aim_from_visible_cells(
                    cells,
                    _event_first_contact_angle(event),
                    allow_edge_aim=allow_edge_aim,
                    refine_boundaries=refine_boundaries,
                    cache=occlusion_cache,
                    origin_xy=origin,
                    origin_radius=origin_radius,
                    speed=speed,
                    p0_by_tick=p0_by_tick,
                    p1_by_tick=p1_by_tick,
                    radii=radii,
                    active_by_tick=active_by_tick,
                    slot=int(event.slot),
                    object_order=order,
                    debug_context=debug_context,
                    return_job=return_jobs,
                )
                if return_jobs:
                    aim, tick, width, refine_job = picked
                else:
                    aim, tick, width = picked
                    refine_job = None
            else:
                aim, tick, width = None, -1, 0.0
                refine_job = None
                mid, w = _widest_cell_midpoint_and_width(cells)
                if mid is not None:
                    aim, tick, width = float(mid), 0, w
            if aim is not None and tick >= 0:
                slot = int(event.slot)
                prev = int(best_tick[slot])
                if (
                    prev < 0
                    or tick < prev
                    or (tick == prev and width > best_width[slot])
                ):
                    best_width[slot] = width
                    best_angle[slot] = aim
                    best_tick[slot] = tick
                    best_jobs[slot] = refine_job
        blocked = union_angle_intervals([*blocked, *hit])

    if include_board and origin is not None:
        for tick in range(int(horizon)):
            b_hit = board_exit_angle_intervals(
                origin, origin_radius, speed, tick, board_size=board_size
            )
            blocked = union_angle_intervals([*blocked, *b_hit])

    valid = (best_width > GEOM_EPS) & (best_tick >= 0)
    result = (
        best_angle.astype(np.float64),
        best_width.astype(np.float32),
        valid,
        overflow,
        best_tick,
    )
    if return_jobs:
        return (*result, best_jobs)
    return result


def first_hit_interval_best_targets_orthogonal_np(
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    planets: np.ndarray,
    planet_active: np.ndarray,
    initial_planets: np.ndarray,
    initial_active: np.ndarray,
    comet_paths: np.ndarray,
    comet_path_lengths: np.ndarray,
    comet_group_active: np.ndarray,
    comet_path_index: np.ndarray,
    comet_slots: np.ndarray,
    comet_planet_ids: np.ndarray,
    angular_velocity: float,
    step_count: int,
    object_p0_by_tick: np.ndarray,
    object_p1_by_tick: np.ndarray,
    object_active_by_tick: np.ndarray,
    *,
    object_order: Sequence[int] | None = None,
    include_board: bool = True,
    include_sun: bool = True,
    horizon: int = 24,
    board_size: float = BOARD_SIZE,
    sun_radius: float = SUN_RADIUS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, np.ndarray]:
    """Interval targets via orthogonal cones; per-planet ETA from single-target raycast."""

    events = collect_hit_events(
        origin_xy,
        origin_radius,
        speed,
        planets,
        planet_active,
        initial_planets,
        initial_active,
        comet_paths,
        comet_path_lengths,
        comet_group_active,
        comet_path_index,
        comet_slots,
        comet_planet_ids,
        angular_velocity,
        step_count,
        horizon=float(horizon),
        include_sun=include_sun,
        sun_radius=sun_radius,
    )
    radii = np.asarray(planets, dtype=np.float64)[:, 4]
    return sweep_interval_best_targets_from_events(
        events,
        int(planets.shape[0]),
        object_order=object_order,
        include_board=include_board,
        origin_xy=origin_xy,
        origin_radius=origin_radius,
        speed=speed,
        horizon=horizon,
        board_size=board_size,
        p0_by_tick=object_p0_by_tick,
        p1_by_tick=object_p1_by_tick,
        radii=radii,
        active_by_tick=object_active_by_tick,
    )


def first_hit_interval_best_targets_np(
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    object_p0_by_tick: np.ndarray,
    object_p1_by_tick: np.ndarray,
    object_radii: np.ndarray,
    object_active_by_tick: np.ndarray,
    *,
    object_order: Sequence[int] | None = None,
    include_board: bool = True,
    include_sun: bool = True,
    board_size: float = BOARD_SIZE,
    sun_radius: float = SUN_RADIUS,
    samples_per_span: int = 9,
    geometry: str = "sampled",
    planets: np.ndarray | None = None,
    planet_active: np.ndarray | None = None,
    initial_planets: np.ndarray | None = None,
    initial_active: np.ndarray | None = None,
    comet_paths: np.ndarray | None = None,
    comet_path_lengths: np.ndarray | None = None,
    comet_group_active: np.ndarray | None = None,
    comet_path_index: np.ndarray | None = None,
    comet_slots: np.ndarray | None = None,
    comet_planet_ids: np.ndarray | None = None,
    angular_velocity: float = 0.0,
    step_count: int = 0,
    horizon: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, np.ndarray]:
    """CPU equivalent of ``geometry_jax.first_hit_interval_best_targets_apply_jax``."""

    if geometry == "orthogonal":
        if planets is None or planet_active is None:
            raise ValueError("orthogonal geometry requires full planet/comet state arrays")
        return first_hit_interval_best_targets_orthogonal_np(
            origin_xy,
            origin_radius,
            speed,
            planets,
            planet_active,
            initial_planets if initial_planets is not None else planets,
            initial_active if initial_active is not None else planet_active,
            comet_paths if comet_paths is not None else np.zeros((0, 4, 0, 2)),
            comet_path_lengths if comet_path_lengths is not None else np.zeros((0, 4)),
            comet_group_active if comet_group_active is not None else np.zeros((0,), dtype=bool),
            comet_path_index if comet_path_index is not None else np.zeros((0,), dtype=np.int32),
            comet_slots if comet_slots is not None else np.zeros((0, 4), dtype=np.int32),
            comet_planet_ids if comet_planet_ids is not None else np.zeros((0, 4), dtype=np.int32),
            angular_velocity,
            step_count,
            object_p0_by_tick,
            object_p1_by_tick,
            object_active_by_tick,
            object_order=object_order,
            include_board=include_board,
            include_sun=include_sun,
            horizon=horizon,
            board_size=board_size,
            sun_radius=sun_radius,
        )

    pre = precompute_tick_planet_hits(
        origin_xy,
        origin_radius,
        speed,
        object_p0_by_tick,
        object_p1_by_tick,
        object_radii,
        object_active_by_tick,
        samples_per_span=samples_per_span,
    )
    return sweep_interval_best_targets(
        pre,
        object_order=object_order,
        include_board=include_board,
        include_sun=include_sun,
        origin_xy=origin_xy,
        origin_radius=origin_radius,
        speed=speed,
        board_size=board_size,
        sun_radius=sun_radius,
        samples_per_span=samples_per_span,
    )

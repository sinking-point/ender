"""Ray casts, sun/planet collision checks, and time-to-hit estimates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from orbit_wars_pt.constants import BOARD_SIZE, CENTER, ROTATION_RADIUS_LIMIT, SUN_RADIUS


TAU = 2.0 * math.pi


@dataclass(frozen=True)
class AngleInterval:
    """Closed launch-angle interval in radians.

    Intervals are stored in normalized ``[0, 2*pi)`` coordinates and may wrap
    by having ``lo > hi``. Most helpers below split wrapped intervals into
    non-wrapping pieces before doing set arithmetic.
    """

    lo: float
    hi: float


@dataclass(frozen=True)
class IntervalEvent:
    tick: int
    object_idx: int
    intervals: tuple[AngleInterval, ...]


def fleet_speed(ships: float, max_speed: float = 6.0) -> float:
    if ships <= 1.0:
        return 1.0
    return float(min(max_speed, 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5))


def point_to_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx, aby = bx - ax, by - ay
    ab2 = abx * abx + aby * aby
    if ab2 <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab2))
    qx, qy = ax + t * abx, ay + t * aby
    return math.hypot(px - qx, py - qy)


def segment_hits_sun(ax: float, ay: float, bx: float, by: float, sun_r: float = SUN_RADIUS) -> bool:
    return point_to_segment_distance(CENTER, CENTER, ax, ay, bx, by) < sun_r


def _norm_angle(a: float) -> float:
    return a % TAU


def _angle_diff(a: float, ref: float) -> float:
    """Return ``a`` shifted to the branch closest to ``ref``."""

    return ref + ((a - ref + math.pi) % TAU - math.pi)


def _split_interval(iv: AngleInterval) -> list[tuple[float, float]]:
    lo = _norm_angle(iv.lo)
    hi = _norm_angle(iv.hi)
    if math.isclose((hi - lo) % TAU, 0.0, abs_tol=1e-12) and iv.hi > iv.lo:
        return [(0.0, TAU)]
    if lo <= hi:
        return [(lo, hi)]
    return [(0.0, hi), (lo, TAU)]


def _merge_pieces(pieces: Sequence[tuple[float, float]], eps: float = 1e-9) -> list[tuple[float, float]]:
    clean = sorted((max(0.0, lo), min(TAU, hi)) for lo, hi in pieces if hi - lo > eps)
    if not clean:
        return []
    merged = [clean[0]]
    for lo, hi in clean[1:]:
        plo, phi = merged[-1]
        if lo <= phi + eps:
            merged[-1] = (plo, max(phi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _pieces_to_intervals(pieces: Sequence[tuple[float, float]]) -> list[AngleInterval]:
    out: list[AngleInterval] = []
    for lo, hi in _merge_pieces(pieces):
        if hi - lo >= TAU - 1e-9:
            out.append(AngleInterval(0.0, TAU))
        else:
            out.append(AngleInterval(_norm_angle(lo), _norm_angle(hi)))
    return out


def union_angle_intervals(intervals: Sequence[AngleInterval]) -> list[AngleInterval]:
    """Normalize and union angular intervals."""

    pieces: list[tuple[float, float]] = []
    for iv in intervals:
        pieces.extend(_split_interval(iv))
    merged = _merge_pieces(pieces)
    if len(merged) >= 2 and merged[0][0] <= 1e-9 and merged[-1][1] >= TAU - 1e-9:
        first = merged.pop(0)
        last = merged.pop()
        return [AngleInterval(last[0], first[1]), *_pieces_to_intervals(merged)]
    return _pieces_to_intervals(merged)


def subtract_angle_intervals(
    intervals: Sequence[AngleInterval], blocked: Sequence[AngleInterval]
) -> list[AngleInterval]:
    """Return ``intervals - blocked`` on the angle circle."""

    avail: list[tuple[float, float]] = []
    for iv in union_angle_intervals(intervals):
        avail.extend(_split_interval(iv))
    blockers: list[tuple[float, float]] = []
    for iv in union_angle_intervals(blocked):
        blockers.extend(_split_interval(iv))
    blockers = _merge_pieces(blockers)

    out: list[tuple[float, float]] = []
    for lo, hi in avail:
        spans = [(lo, hi)]
        for blo, bhi in blockers:
            next_spans: list[tuple[float, float]] = []
            for slo, shi in spans:
                if bhi <= slo or blo >= shi:
                    next_spans.append((slo, shi))
                    continue
                if blo > slo:
                    next_spans.append((slo, blo))
                if bhi < shi:
                    next_spans.append((bhi, shi))
            spans = next_spans
            if not spans:
                break
        out.extend(spans)
    return _pieces_to_intervals(out)


def intersect_angle_intervals(
    a_intervals: Sequence[AngleInterval], b_intervals: Sequence[AngleInterval]
) -> list[AngleInterval]:
    a_pieces: list[tuple[float, float]] = []
    b_pieces: list[tuple[float, float]] = []
    for iv in union_angle_intervals(a_intervals):
        a_pieces.extend(_split_interval(iv))
    for iv in union_angle_intervals(b_intervals):
        b_pieces.extend(_split_interval(iv))

    out: list[tuple[float, float]] = []
    for alo, ahi in a_pieces:
        for blo, bhi in b_pieces:
            lo = max(alo, blo)
            hi = min(ahi, bhi)
            if hi > lo:
                out.append((lo, hi))
    return _pieces_to_intervals(out)


def complement_angle_intervals(intervals: Sequence[AngleInterval]) -> list[AngleInterval]:
    pieces: list[tuple[float, float]] = []
    for iv in union_angle_intervals(intervals):
        pieces.extend(_split_interval(iv))
    pieces = _merge_pieces(pieces)
    if not pieces:
        return [AngleInterval(0.0, TAU)]
    out: list[tuple[float, float]] = []
    cursor = 0.0
    for lo, hi in pieces:
        if lo > cursor:
            out.append((cursor, lo))
        cursor = max(cursor, hi)
    if cursor < TAU:
        out.append((cursor, TAU))
    return union_angle_intervals(_pieces_to_intervals(out))


def widest_interval_midpoint(intervals: Sequence[AngleInterval]) -> float | None:
    best_width = -1.0
    best_mid: float | None = None
    for iv in union_angle_intervals(intervals):
        width = (iv.hi - iv.lo) % TAU
        if width <= 1e-12 and iv.hi > iv.lo:
            width = TAU
        if width > best_width:
            best_width = width
            best_mid = _norm_angle(iv.lo + 0.5 * width)
    return best_mid


def _shift_intervals(intervals: Sequence[AngleInterval], delta: float) -> list[AngleInterval]:
    return union_angle_intervals([AngleInterval(iv.lo + delta, iv.hi + delta) for iv in intervals])


def _cos_between_intervals(lo: float, hi: float) -> list[AngleInterval]:
    lo = max(-1.0, lo)
    hi = min(1.0, hi)
    if lo > hi:
        return []
    if lo <= -1.0 and hi >= 1.0:
        return [AngleInterval(0.0, TAU)]

    intervals = [AngleInterval(0.0, TAU)]
    if lo > -1.0:
        a = math.acos(lo)
        intervals = intersect_angle_intervals(intervals, [AngleInterval(-a, a)])
    if hi < 1.0:
        a = math.acos(hi)
        intervals = intersect_angle_intervals(intervals, [AngleInterval(a, TAU - a)])
    return intervals


def _sin_between_intervals(lo: float, hi: float) -> list[AngleInterval]:
    return _shift_intervals(_cos_between_intervals(lo, hi), math.pi / 2.0)


def board_exit_angle_intervals(
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    tick: int,
    *,
    board_size: float = BOARD_SIZE,
) -> list[AngleInterval]:
    """Angles whose fleet endpoint is out of bounds after this tick.

    The interpreter checks board exit after planet collisions for the tick, so
    these intervals are meant to block later ticks, not same-tick planet hits.
    """

    origin = np.asarray(origin_xy, dtype=np.float64)
    rho = float(origin_radius) + 0.1 + (float(tick) + 1.0) * float(speed)
    if rho <= 0.0:
        return []
    x_ok = _cos_between_intervals((0.0 - float(origin[0])) / rho, (board_size - float(origin[0])) / rho)
    y_ok = _sin_between_intervals((0.0 - float(origin[1])) / rho, (board_size - float(origin[1])) / rho)
    inside = intersect_angle_intervals(x_ok, y_ok)
    return complement_angle_intervals(inside)


def _circle_disk_angle_interval(
    origin_xy: np.ndarray,
    launch_offset: float,
    speed: float,
    tick: int,
    target_xy: np.ndarray,
    target_r: float,
    tau: float,
) -> AngleInterval | None:
    eps = 1e-12
    p = np.asarray(target_xy, dtype=np.float64) - np.asarray(origin_xy, dtype=np.float64)
    d = float(np.linalg.norm(p))
    rho = launch_offset + (float(tick) + tau) * speed
    if rho <= 0.0:
        return None
    if d <= 1e-12:
        if rho <= target_r + eps:
            return AngleInterval(0.0, TAU)
        return None
    if target_r >= d + rho - eps:
        return AngleInterval(0.0, TAU)
    if abs(d - rho) > target_r + eps:
        return None
    denom = 2.0 * d * rho
    if denom <= 1e-12:
        return None
    g = (d * d + rho * rho - target_r * target_r) / denom
    g = max(-1.0, min(1.0, g))
    alpha = math.acos(g)
    phi = math.atan2(float(p[1]), float(p[0]))
    return AngleInterval(_norm_angle(phi - alpha), _norm_angle(phi + alpha))


def _quadratic_roots(a: float, b: float, c: float, eps: float = 1e-12) -> list[float]:
    if abs(a) <= eps:
        if abs(b) <= eps:
            return []
        return [-c / b]
    disc = b * b - 4.0 * a * c
    if disc < -eps:
        return []
    sd = math.sqrt(max(0.0, disc))
    return [(-b - sd) / (2.0 * a), (-b + sd) / (2.0 * a)]


def _radial_active_windows(
    origin_xy: np.ndarray,
    launch_offset: float,
    speed: float,
    tick: int,
    object_p0: np.ndarray,
    object_p1: np.ndarray,
    object_radius: float,
    eps: float = 1e-10,
) -> list[tuple[float, float]]:
    """Tau spans where ``abs(|P(tau)-O| - rho(tau)) <= object_radius``."""

    origin = np.asarray(origin_xy, dtype=np.float64)
    p0 = np.asarray(object_p0, dtype=np.float64)
    p1 = np.asarray(object_p1, dtype=np.float64)
    q = p0 - origin
    d_vec = p1 - p0
    a_base = float(launch_offset) + float(tick) * float(speed)
    v = float(speed)
    radius = float(object_radius)

    def radial_ok(tau: float) -> bool:
        rho = a_base + v * tau
        if rho < -eps:
            return False
        dist = float(np.linalg.norm(q + tau * d_vec))
        return abs(dist - rho) <= radius + eps

    taus = [0.0, 1.0]
    dd = float(np.dot(d_vec, d_vec))
    qd = float(np.dot(q, d_vec))
    qq = float(np.dot(q, q))
    for c in (radius, -radius):
        ac = a_base + c
        qa = dd - v * v
        qb = 2.0 * (qd - v * ac)
        qc = qq - ac * ac
        for root in _quadratic_roots(qa, qb, qc):
            if eps < root < 1.0 - eps and radial_ok(root):
                taus.append(root)

    taus = sorted(set(round(t, 14) for t in taus))
    windows: list[tuple[float, float]] = []
    for a, b in zip(taus, taus[1:]):
        if b - a <= eps:
            continue
        mid = 0.5 * (a + b)
        if radial_ok(mid):
            windows.append((a, b))
    for tau in taus:
        if radial_ok(tau):
            pad = min(1e-8, max(0.0, tau), max(0.0, 1.0 - tau))
            if pad > 0.0:
                windows.append((tau - pad, tau + pad))
    return _merge_pieces(windows, eps=eps)


def _angular_hull(intervals: Sequence[AngleInterval]) -> list[AngleInterval]:
    """Smallest local hull for nearby intervals sampled from one smooth curve."""

    samples: list[float] = []
    for iv in intervals:
        mid = _norm_angle(iv.lo + 0.5 * ((iv.hi - iv.lo) % TAU))
        half = 0.5 * ((iv.hi - iv.lo) % TAU)
        if half <= 1e-12 and iv.hi > iv.lo:
            return [AngleInterval(0.0, TAU)]
        samples.extend([_angle_diff(iv.lo, mid), mid, _angle_diff(iv.hi, mid)])
    if not samples:
        return []
    ref = samples[len(samples) // 2]
    shifted = [_angle_diff(a, ref) for a in samples]
    return [AngleInterval(_norm_angle(min(shifted)), _norm_angle(max(shifted)))]


def tick_hit_angle_intervals(
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    tick: int,
    object_p0: np.ndarray,
    object_p1: np.ndarray,
    object_radius: float,
    *,
    max_depth: int = 8,
    min_depth: int = 4,
    linear_tol: float = 1e-3,
) -> list[AngleInterval]:
    """Approximate the analytic hit-angle union for one object during one tick.

    This implements the reduction described in the prompt: at each intra-tick
    ``tau``, intersect the fleet's radius-``rho`` circle around the origin with
    the moving object's disk, then adaptively hull the boundary intervals.
    """

    origin = np.asarray(origin_xy, dtype=np.float64)
    p0 = np.asarray(object_p0, dtype=np.float64)
    p1 = np.asarray(object_p1, dtype=np.float64)
    launch_offset = float(origin_radius) + 0.1
    radius = float(object_radius)
    active_windows = _radial_active_windows(
        origin,
        launch_offset,
        float(speed),
        int(tick),
        p0,
        p1,
        radius,
    )

    def at(tau: float) -> AngleInterval | None:
        return _circle_disk_angle_interval(
            origin,
            launch_offset,
            float(speed),
            int(tick),
            p0 + tau * (p1 - p0),
            radius,
            tau,
        )

    out: list[AngleInterval] = []

    def rec(a: float, b: float, depth: int) -> None:
        m = 0.5 * (a + b)
        iva, ivm, ivb = at(a), at(m), at(b)
        present = [iv for iv in (iva, ivm, ivb) if iv is not None]
        if not present:
            if depth < min_depth:
                rec(a, m, depth + 1)
                rec(m, b, depth + 1)
            return
        if depth >= max_depth:
            out.extend(_angular_hull(present))
            return
        if len(present) < 3:
            rec(a, m, depth + 1)
            rec(m, b, depth + 1)
            return

        mids = []
        halfs = []
        ref = None
        for iv in (iva, ivm, ivb):
            assert iv is not None
            width = (iv.hi - iv.lo) % TAU
            mid = _norm_angle(iv.lo + 0.5 * width)
            if ref is None:
                ref = mid
            mids.append(_angle_diff(mid, ref))
            halfs.append(0.5 * width)
        mid_lin_err = abs(mids[1] - 0.5 * (mids[0] + mids[2]))
        half_lin_err = abs(halfs[1] - 0.5 * (halfs[0] + halfs[2]))
        if max(mid_lin_err, half_lin_err) <= linear_tol:
            out.extend(_angular_hull(present))
            return
        rec(a, m, depth + 1)
        rec(m, b, depth + 1)

    for a, b in active_windows:
        rec(a, b, 0)
    return union_angle_intervals(out)


def first_hit_angle_intervals(
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    object_p0_by_tick: np.ndarray,
    object_p1_by_tick: np.ndarray,
    object_radii: np.ndarray,
    object_active_by_tick: np.ndarray,
    target_idx: int,
    *,
    object_order: Sequence[int] | None = None,
    max_depth: int = 8,
    min_depth: int = 4,
    include_board: bool = True,
    include_sun: bool = True,
    board_size: float = BOARD_SIZE,
    sun_radius: float = SUN_RADIUS,
) -> list[AngleInterval]:
    """Launch angles that hit ``target_idx`` before any earlier object.

    ``object_p0_by_tick`` and ``object_p1_by_tick`` are shaped ``[T, P, 2]``.
    ``object_active_by_tick`` is ``[T, P]`` and should already encode whether
    an object participates in collision checks for that tick. Same-tick ties
    are resolved by ``object_order``, matching the interpreter's planet-list
    order. Board exit and sun crossing are added only after all planets/comets
    for a tick, matching the installed Kaggle interpreter.
    """

    p0 = np.asarray(object_p0_by_tick, dtype=np.float64)
    p1 = np.asarray(object_p1_by_tick, dtype=np.float64)
    radii = np.asarray(object_radii, dtype=np.float64)
    active = np.asarray(object_active_by_tick, dtype=bool)
    ticks, objects = active.shape
    order = list(range(objects)) if object_order is None else list(object_order)
    blocked: list[AngleInterval] = []
    valid: list[AngleInterval] = []

    for tick in range(ticks):
        for obj_idx in order:
            if obj_idx < 0 or obj_idx >= objects or not active[tick, obj_idx]:
                continue
            hit = tick_hit_angle_intervals(
                origin_xy,
                origin_radius,
                speed,
                tick,
                p0[tick, obj_idx],
                p1[tick, obj_idx],
                radii[obj_idx],
                max_depth=max_depth,
                min_depth=min_depth,
            )
            if not hit:
                continue
            available = subtract_angle_intervals(hit, blocked)
            if obj_idx == target_idx:
                valid.extend(available)
            blocked = union_angle_intervals([*blocked, *hit])
        if include_board:
            blocked = union_angle_intervals(
                [
                    *blocked,
                    *board_exit_angle_intervals(
                        origin_xy,
                        origin_radius,
                        speed,
                        tick,
                        board_size=board_size,
                    ),
                ]
            )
        if include_sun:
            sun_xy = np.asarray([CENTER, CENTER], dtype=np.float64)
            sun_hit = tick_hit_angle_intervals(
                origin_xy,
                origin_radius,
                speed,
                tick,
                sun_xy,
                sun_xy,
                max(0.0, float(sun_radius) - 1e-9),
                max_depth=max_depth,
                min_depth=min_depth,
            )
            blocked = union_angle_intervals([*blocked, *sun_hit])
    return union_angle_intervals(valid)


def ray_circle_intersections(
    ox: float,
    oy: float,
    dx: float,
    dy: float,
    cx: float,
    cy: float,
    r: float,
) -> Tuple[list[float], list[float]]:
    """Intersections of ray o + t*d, t>=0 with circle center c radius r. d must be unit."""
    fx, fy = ox - cx, oy - cy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return [], []
    sd = math.sqrt(disc)
    t0 = (-b - sd) / 2.0
    t1 = (-b + sd) / 2.0
    ts = []
    for t in (t0, t1):
        if t >= 0.0:
            ts.append(t)
    return ts, []


def first_planet_hit_along_ray(
    ox: float,
    oy: float,
    angle: float,
    planet_xy: np.ndarray,
    planet_r: np.ndarray,
    planet_active: np.ndarray,
    exclude_idx: int | None = None,
) -> Tuple[float | None, int | None]:
    """Smallest positive ray distance to any active planet surface (center distance - r)."""

    dx, dy = math.cos(angle), math.sin(angle)
    best_t: float | None = None
    best_i: int | None = None
    n = planet_xy.shape[0]
    for i in range(n):
        if not planet_active[i] or i == exclude_idx:
            continue
        cx, cy = float(planet_xy[i, 0]), float(planet_xy[i, 1])
        r = float(planet_r[i])
        ts, _ = ray_circle_intersections(ox, oy, dx, dy, cx, cy, r)
        for t in ts:
            if best_t is None or t < best_t:
                best_t = t
                best_i = i
    return best_t, best_i


def launch_point(ox: float, oy: float, radius: float, dest_x: float, dest_y: float) -> Tuple[float, float]:
    vx, vy = dest_x - ox, dest_y - oy
    d = math.hypot(vx, vy)
    if d < 1e-6:
        return ox + radius + 0.1, oy
    ux, uy = vx / d, vy / d
    return ox + ux * (radius + 0.1), oy + uy * (radius + 0.1)


def estimate_time_to_hit(
    origin_x: float,
    origin_y: float,
    origin_r: float,
    dest_x: float,
    dest_y: float,
    dest_r: float,
    ships: float,
    max_speed: float = 6.0,
    max_steps: int = 600,
) -> float:
    """Discrete stepping until fleet reaches destination planet rim (approximate)."""

    sx, sy = launch_point(origin_x, origin_y, origin_r, dest_x, dest_y)
    angle = math.atan2(dest_y - origin_y, dest_x - origin_x)
    x, y = sx, sy
    for t in range(max_steps):
        dist_center = math.hypot(x - dest_x, y - dest_y)
        if dist_center <= dest_r + 0.05:
            return float(t)
        sp = fleet_speed(ships, max_speed)
        x += math.cos(angle) * sp
        y += math.sin(angle) * sp
        if x < 0 or x > BOARD_SIZE or y < 0 or y > BOARD_SIZE:
            return float("inf")
        if point_to_segment_distance(CENTER, CENTER, x - math.cos(angle) * sp, y - math.sin(angle) * sp, x, y) < SUN_RADIUS:
            return float("inf")
    return float("inf")


def path_hits_sun_or_other_planet_before_dest(
    origin_x: float,
    origin_y: float,
    origin_r: float,
    dest_x: float,
    dest_y: float,
    dest_r: float,
    ships: float,
    planet_xy: np.ndarray,
    planet_r: np.ndarray,
    planet_active: np.ndarray,
    origin_idx: int,
    dest_idx: int,
    max_speed: float = 6.0,
    max_steps: int = 600,
) -> bool:
    """True if straight-line motion hits sun or wrong planet before arriving at dest."""

    sx, sy = launch_point(origin_x, origin_y, origin_r, dest_x, dest_y)
    angle = math.atan2(dest_y - origin_y, dest_x - origin_x)
    x, y = sx, sy
    sp = fleet_speed(ships, max_speed)
    for _ in range(max_steps):
        px, py = x, y
        x += math.cos(angle) * sp
        y += math.sin(angle) * sp
        if x < 0 or x > BOARD_SIZE or y < 0 or y > BOARD_SIZE:
            return True
        if segment_hits_sun(px, py, x, y):
            return True
        dist_dest = math.hypot(x - dest_x, y - dest_y)
        if dist_dest <= dest_r + 0.05:
            return False
        n = planet_xy.shape[0]
        for i in range(n):
            if not planet_active[i] or i == origin_idx:
                continue
            d_pl = point_to_segment_distance(planet_xy[i, 0], planet_xy[i, 1], px, py, x, y)
            if d_pl < planet_r[i] + 0.02:
                if i != dest_idx:
                    return True
        sp = fleet_speed(ships, max_speed)
    return True


def planet_pred_velocity(
    init_xy: np.ndarray,
    cur_xy: np.ndarray,
    radius: float,
    angular_velocity: float,
    step_count: int,
    initial_active: bool,
    planet_active: bool,
) -> Tuple[float, float]:
    """Returns (vx, vy) per turn for rotating planets; static planets return (0,0)."""

    if not planet_active or not initial_active:
        return 0.0, 0.0
    dx0 = float(init_xy[0] - CENTER)
    dy0 = float(init_xy[1] - CENTER)
    orbital_r = math.hypot(dx0, dy0)
    if orbital_r < 1e-6:
        return 0.0, 0.0
    if orbital_r + radius >= ROTATION_RADIUS_LIMIT:
        return 0.0, 0.0
    th0 = math.atan2(dy0, dx0)
    th_next = th0 + angular_velocity * (step_count + 1)
    nx = CENTER + orbital_r * math.cos(th_next)
    ny = CENTER + orbital_r * math.sin(th_next)
    vx, vy = nx - float(cur_xy[0]), ny - float(cur_xy[1])
    return vx, vy

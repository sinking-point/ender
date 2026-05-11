"""Ray casts, sun/planet collision checks, and time-to-hit estimates."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from orbit_wars_pt.constants import BOARD_SIZE, CENTER, ROTATION_RADIUS_LIMIT, SUN_RADIUS


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

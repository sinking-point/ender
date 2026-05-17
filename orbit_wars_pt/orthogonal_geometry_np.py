"""Orthogonal growing-circle vs moving/stationary target circle (tangent cones).

Fleet model: launch from ``grow_center`` with rim offset ``launch_offset``, then
radius ``launch_offset + grow_rate * t``.  Hit when the growing circle meets the
target circle orthogonally (glancing / tangent rays bound the feasible launch cone).

Based on ``chatgpt-geometry.md``.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from orbit_wars_pt.geometry import AngleInterval, TAU, _norm_angle

GEOM_TOL = 1e-10


def _cone_from_point_to_circle(
    point: np.ndarray,
    circle_center: np.ndarray,
    circle_radius: float,
) -> tuple[float, float] | None:
    """Tangent cone from ``point`` to a circle; returns ``(angle_lo, angle_hi)``."""

    p = np.asarray(point, dtype=np.float64)
    c = np.asarray(circle_center, dtype=np.float64)
    r = float(circle_radius)
    ux = c[0] - p[0]
    uy = c[1] - p[1]
    d = math.hypot(ux, uy)
    if d <= r:
        return None
    centre_angle = math.atan2(uy, ux)
    half = math.asin(min(1.0, max(-1.0, r / d)))
    return (
        _norm_angle(centre_angle - half),
        _norm_angle(centre_angle + half),
    )


def _solve_quadratic_real(
    a: float,
    b: float,
    c: float,
    *,
    eps: float = 1e-14,
) -> list[float]:
    if abs(a) <= eps:
        if abs(b) <= eps:
            return [0.0] if abs(c) <= eps else []
        return [-c / b]
    disc = b * b - 4.0 * a * c
    if disc < -eps:
        return []
    if abs(disc) <= eps:
        return [-b / (2.0 * a)]
    sd = math.sqrt(disc)
    if b >= 0.0:
        q = -0.5 * (b + sd)
    else:
        q = -0.5 * (b - sd)
    if abs(q) <= eps:
        return [(-b - sd) / (2.0 * a), (-b + sd) / (2.0 * a)]
    return [q / a, c / q]


def orthogonal_hit_stationary(
    circle_center: np.ndarray,
    circle_radius: float,
    grow_center: np.ndarray,
    grow_rate: float,
    launch_offset: float,
    horizon: float,
    *,
    tol: float = GEOM_TOL,
) -> tuple[float, float, float] | None:
    """``(t, angle_lo, angle_hi)`` for a stationary target circle."""

    c = np.asarray(circle_center, dtype=np.float64)
    g = np.asarray(grow_center, dtype=np.float64)
    r = float(circle_radius)
    v = float(grow_rate)
    lo = float(launch_offset)
    t_max = float(horizon)
    if v <= 0.0 or t_max < 0.0:
        return None

    ux = c[0] - g[0]
    uy = c[1] - g[1]
    d = math.hypot(ux, uy)
    q = d * d - r * r
    if q < -tol:
        return None

    rad = math.sqrt(max(q, 0.0))
    if rad + tol < lo:
        return None
    if abs(rad - lo) <= tol:
        t = 0.0
    else:
        t = (rad - lo) / v
    if t < -tol or t > t_max + tol:
        return None
    t = min(max(t, 0.0), t_max)

    cone = _cone_from_point_to_circle(g, c, r)
    if cone is None:
        return None
    a0, a1 = cone
    return float(t), float(a0), float(a1)


def orthogonal_hit_circular_orbit(
    orbit_center: np.ndarray,
    orbit_radius: float,
    theta0: float,
    omega: float,
    circle_radius: float,
    grow_center: np.ndarray,
    grow_rate: float,
    launch_offset: float,
    horizon: float,
    *,
    tol: float = GEOM_TOL,
    max_newton: int = 4,
) -> tuple[float, float, float] | None:
    """``(t, angle_lo, angle_hi)`` for a centre moving on a circular orbit."""

    o = np.asarray(orbit_center, dtype=np.float64)
    g = np.asarray(grow_center, dtype=np.float64)
    rho = float(orbit_radius)
    th0 = float(theta0)
    w = float(omega)
    r = float(circle_radius)
    v = float(grow_rate)
    lo = float(launch_offset)
    t_max = float(horizon)

    if t_max < 0.0:
        return None

    dx, dy = o[0] - g[0], o[1] - g[1]
    c0, s0 = math.cos(th0), math.sin(th0)
    base = dx * dx + dy * dy + rho * rho - r * r - lo * lo

    def f(t: float) -> float:
        th = th0 + w * t
        grow = lo + v * t
        return base + 2.0 * rho * (dx * math.cos(th) + dy * math.sin(th)) - grow * grow

    def fp(t: float) -> float:
        th = th0 + w * t
        grow = lo + v * t
        return (
            2.0 * rho * w * (-dx * math.sin(th) + dy * math.cos(th))
            - 2.0 * v * grow
        )

    def output_at(t: float) -> tuple[float, float, float] | None:
        th = th0 + w * t
        cx = o[0] + rho * math.cos(th)
        cy = o[1] + rho * math.sin(th)
        cone = _cone_from_point_to_circle(g, np.array([cx, cy], dtype=np.float64), r)
        if cone is None:
            return None
        a0, a1 = cone
        return float(t), float(a0), float(a1)

    f0 = f(0.0)
    if abs(f0) <= tol:
        out = output_at(0.0)
        return out

    fT = f(t_max)
    if abs(fT) <= tol:
        return output_at(t_max)

    if f0 * fT > 0.0:
        return None

    f1 = fp(0.0)
    f2 = (
        -rho * w * w * (dx * c0 + dy * s0)
        - v * v
    )
    t: float | None = None
    if abs(f2) < 1e-30:
        if abs(f1) > 1e-30:
            t_lin = -f0 / f1
            if 0.0 <= t_lin <= t_max:
                t = t_lin
    else:
        disc = f1 * f1 - 4.0 * f2 * f0
        if disc >= 0.0:
            sd = math.sqrt(disc)
            for root in ((-f1 - sd) / (2.0 * f2), (-f1 + sd) / (2.0 * f2)):
                if 0.0 <= root <= t_max:
                    t = root if t is None else min(t, root)

    if t is None:
        t = 0.5 * t_max

    lo_t, hi_t = 0.0, t_max
    flo, fhi = f0, fT
    for _ in range(max_newton):
        ft = f(t)
        if abs(ft) <= tol:
            return output_at(t)
        if flo * ft <= 0.0:
            hi_t, fhi = t, ft
        else:
            lo_t, flo = t, ft
        dft = fp(t)
        if abs(dft) > 1e-30:
            t_new = t - ft / dft
            t = t_new if lo_t < t_new < hi_t else 0.5 * (lo_t + hi_t)
        else:
            t = 0.5 * (lo_t + hi_t)

    for _ in range(20):
        t = 0.5 * (lo_t + hi_t)
        ft = f(t)
        if abs(ft) <= tol or (hi_t - lo_t) <= tol:
            return output_at(t)
        if flo * ft <= 0.0:
            hi_t, fhi = t, ft
        else:
            lo_t, flo = t, ft

    return output_at(0.5 * (lo_t + hi_t))


def orthogonal_hits_polyline(
    points: np.ndarray,
    circle_radius: float,
    grow_center: np.ndarray,
    grow_rate: float,
    launch_offset: float,
    horizon: float,
    *,
    tol: float = GEOM_TOL,
) -> list[tuple[float, float, float]]:
    """All orthogonal hits along a piecewise-linear centre path (e.g. comets)."""

    pts = np.asarray(points, dtype=np.float64)
    g = np.asarray(grow_center, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        return []

    r = float(circle_radius)
    v = float(grow_rate)
    lo = float(launch_offset)
    t_max = min(float(horizon), float(len(pts) - 1))
    if t_max < 0.0:
        return []

    v2 = v * v
    r2 = r * r
    hits: list[tuple[float, float, float]] = []
    nseg = int(math.ceil(t_max))

    for i in range(nseg):
        seg_t0 = float(i)
        seg_t1 = min(float(i + 1), t_max)
        if seg_t1 < seg_t0:
            break
        u_lo, u_hi = 0.0, seg_t1 - seg_t0
        p0 = pts[i]
        p1 = pts[i + 1]
        a = p0 - g
        b = p1 - p0
        # |a + u*b|^2 - R^2 - (lo + v*(seg_t0+u))^2 = 0
        grow0 = lo + v * seg_t0
        a_coef = float(np.dot(b, b)) - v2
        b_coef = 2.0 * float(np.dot(a, b)) - 2.0 * v * grow0
        c_coef = float(np.dot(a, a)) - r2 - grow0 * grow0
        for u in _solve_quadratic_real(a_coef, b_coef, c_coef):
            if u_lo - tol <= u <= u_hi + tol:
                u = min(max(float(u), u_lo), u_hi)
                t = seg_t0 + u
                if hits and abs(t - hits[-1][0]) <= tol:
                    continue
                centre = p0 + u * b
                cone = _cone_from_point_to_circle(g, centre, r)
                if cone is None:
                    continue
                hits.append((float(t), float(cone[0]), float(cone[1])))

    return hits


def cone_to_angle_intervals(angle_lo: float, angle_hi: float) -> list[AngleInterval]:
    lo = _norm_angle(angle_lo)
    hi = _norm_angle(angle_hi)
    if lo <= hi + GEOM_TOL:
        return [AngleInterval(lo, hi)]
    return [AngleInterval(lo, TAU), AngleInterval(0.0, hi)]

"""Growing-circle external/internal tangency and overlap angular extrema.

Replaces orthogonal ``d² = (R+r)²`` with tangency ``|C-G| = R + r`` (external) or
``|C-G| = |R - r|`` (internal), where ``r = launch_offset + grow_rate * t``.

Shelved: exact circular-orbit tangency — use chord polylines (env-style) instead.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from orbit_wars_pt.constants import CENTER

TAU = 2.0 * math.pi
GEOM_EPS = 1e-5
GEOM_TOL = 1e-10


def _norm_angle(a: float) -> float:
    return float(a % TAU)


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


def _grow_radius(launch_offset: float, grow_rate: float, t: float) -> float:
    return float(launch_offset) + float(grow_rate) * float(t)


def _contact_angle(
    grow_center: np.ndarray,
    circle_center: np.ndarray,
    circle_radius: float,
    grow_radius: float,
    kind: str,
    *,
    tol: float = GEOM_TOL,
) -> float | None:
    g = np.asarray(grow_center, dtype=np.float64)
    c = np.asarray(circle_center, dtype=np.float64)
    ux = c[0] - g[0]
    uy = c[1] - g[1]
    d = math.hypot(ux, uy)
    if d <= tol:
        return None

    centre_angle = math.atan2(uy, ux)
    if kind == "external":
        return _norm_angle(centre_angle)
    if kind == "internal":
        if abs(circle_radius - grow_radius) <= tol:
            return None
        if grow_radius > circle_radius:
            return _norm_angle(centre_angle)
        return _norm_angle(centre_angle + math.pi)
    raise ValueError("kind must be 'external' or 'internal'")


def _verify_tangency(
    grow_center: np.ndarray,
    circle_center: np.ndarray,
    circle_radius: float,
    grow_radius: float,
    kind: str,
    *,
    tol: float = 1e-7,
) -> bool:
    d = float(np.linalg.norm(np.asarray(circle_center) - np.asarray(grow_center)))
    if kind == "external":
        return abs(d - (circle_radius + grow_radius)) <= tol
    if d <= tol:
        return False
    return abs(d - abs(circle_radius - grow_radius)) <= tol


def tangent_hit_time_stationary(
    circle_center: np.ndarray,
    circle_radius: float,
    grow_center: np.ndarray,
    grow_rate: float,
    launch_offset: float,
    horizon: float,
    *,
    mode: str = "both",
    tol: float = GEOM_TOL,
    return_all: bool = False,
) -> list[tuple[float, str, float]] | tuple[float, str, float] | None:
    """Stationary target circle; growing radius ``launch_offset + grow_rate * t``."""

    c = np.asarray(circle_center, dtype=np.float64)
    g = np.asarray(grow_center, dtype=np.float64)
    r_planet = float(circle_radius)
    v = float(grow_rate)
    lo = float(launch_offset)
    t_max = float(horizon)

    if v < 0.0 or t_max < 0.0:
        return [] if return_all else None

    d = float(np.linalg.norm(c - g))
    hits: list[tuple[float, str, float]] = []

    def add_candidate(t: float, kind: str) -> None:
        if t < -tol or t > t_max + tol:
            return
        t = min(max(float(t), 0.0), t_max)
        gr = _grow_radius(lo, v, t)
        if not _verify_tangency(g, c, r_planet, gr, kind, tol=1e-7):
            return
        ang = _contact_angle(g, c, r_planet, gr, kind, tol=tol)
        if ang is not None:
            hits.append((t, kind, ang))

    if mode in ("external", "both") and v > tol:
        add_candidate((d - r_planet - lo) / v, "external")

    if mode in ("internal", "both") and v > tol:
        add_candidate((r_planet - lo - d) / v, "internal")
        add_candidate((r_planet - lo + d) / v, "internal")
        add_candidate((r_planet + d - lo) / v, "internal")
        add_candidate((r_planet - d - lo) / v, "internal")

    hits.sort(key=lambda x: x[0])
    deduped: list[tuple[float, str, float]] = []
    for h in hits:
        if not deduped or abs(h[0] - deduped[-1][0]) > tol or h[1] != deduped[-1][1]:
            deduped.append(h)

    if return_all:
        return deduped
    return deduped[0] if deduped else None


def tangent_hit_time_polyline(
    points: np.ndarray,
    circle_radius: float,
    grow_center: np.ndarray,
    grow_rate: float,
    launch_offset: float,
    horizon: float,
    *,
    mode: str = "both",
    tol: float = GEOM_TOL,
    return_all: bool = False,
) -> list[tuple[float, str, float]] | tuple[float, str, float] | None:
    """Piecewise-linear centre path; one quadratic per segment per tangency kind."""

    pts = np.asarray(points, dtype=np.float64)
    g = np.asarray(grow_center, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        raise ValueError("points must have shape (N, 2) with N >= 2")

    r_planet = float(circle_radius)
    v = float(grow_rate)
    lo = float(launch_offset)
    t_cap = min(float(horizon), float(len(pts) - 1))
    if v < 0.0 or t_cap < 0.0:
        return [] if return_all else None

    hits: list[tuple[float, str, float]] = []
    kinds: list[str] = []
    if mode in ("external", "both"):
        kinds.append("external")
    if mode in ("internal", "both"):
        kinds.append("internal")

    nseg = int(math.ceil(t_cap))
    for i in range(nseg):
        seg_t0 = float(i)
        seg_t1 = min(float(i + 1), t_cap)
        if seg_t1 < seg_t0:
            break

        p0 = pts[i]
        p1 = pts[i + 1]
        b = p1 - p0
        a = p0 - g
        u_lo = 0.0
        u_hi = seg_t1 - seg_t0
        g0 = lo + v * seg_t0

        for kind in kinds:
            if kind == "external":
                quad_params = [(r_planet + g0, v)]
            else:
                # |R - r| = |(g0 + v*u) - R|  →  two squared branches.
                quad_params = [(r_planet - g0, -v), (g0 - r_planet, v)]

            for k0, k1 in quad_params:
                aq = float(np.dot(b, b) - k1 * k1)
                bq = 2.0 * float(np.dot(a, b) - k0 * k1)
                cq = float(np.dot(a, a) - k0 * k0)

                for u in _solve_quadratic_real(aq, bq, cq):
                    if u < u_lo - tol or u > u_hi + tol:
                        continue
                    u = min(max(float(u), u_lo), u_hi)
                    t = seg_t0 + u
                    centre = p0 + u * b
                    gr = _grow_radius(lo, v, t)
                    if not _verify_tangency(g, centre, r_planet, gr, kind, tol=1e-7):
                        continue
                    ang = _contact_angle(g, centre, r_planet, gr, kind, tol=tol)
                    if ang is None:
                        continue
                    hit = (float(t), kind, ang)
                    if not return_all:
                        return hit
                    if not hits or abs(hit[0] - hits[-1][0]) > tol or hit[1] != hits[-1][1]:
                        hits.append(hit)

    hits.sort(key=lambda x: x[0])
    if return_all:
        return hits
    return hits[0] if hits else None


def norm_angle_unwrap_near(a: float, ref: float) -> float:
    return float(ref + ((a - ref + math.pi) % TAU - math.pi))


def intersection_angles_from_grow_center(
    q: np.ndarray,
    circle_radius: float,
    grow_radius: float,
    *,
    eps: float = GEOM_EPS,
) -> tuple[float, float, float, float] | None:
    qx, qy = float(q[0]), float(q[1])
    d = math.hypot(qx, qy)
    if d <= eps or grow_radius <= eps:
        return None

    x = (grow_radius * grow_radius - circle_radius * circle_radius + d * d) / (
        2.0 * grow_radius * d
    )
    if x < -1.0 - 1e-9 or x > 1.0 + 1e-9:
        return None
    x = max(-1.0, min(1.0, x))
    alpha = math.atan2(qy, qx)
    beta = math.acos(x)
    return float(alpha - beta), float(alpha + beta), float(alpha), float(beta)


def intersection_angle_derivative(
    q: np.ndarray,
    qdot: np.ndarray,
    circle_radius: float,
    launch_offset: float,
    grow_rate: float,
    t: float,
    branch: int,
    *,
    eps: float = GEOM_EPS,
) -> float:
    qx, qy = float(q[0]), float(q[1])
    vx, vy = float(qdot[0]), float(qdot[1])
    r = _grow_radius(launch_offset, grow_rate, t)
    if r <= eps:
        return float("nan")

    d2 = qx * qx + qy * qy
    if d2 <= eps:
        return float("nan")

    d = math.sqrt(d2)
    alpha_dot = (qx * vy - qy * vx) / d2
    q_dot_qdot = qx * vx + qy * vy
    d_dot = q_dot_qdot / d
    r_dot = float(grow_rate)

    n = r * r - circle_radius * circle_radius + d * d
    den = 2.0 * r * d
    x = n / den
    if x <= -1.0 + eps or x >= 1.0 - eps:
        return float("nan")

    n_dot = 2.0 * r * r_dot + 2.0 * q_dot_qdot
    den_dot = 2.0 * (r_dot * d + r * d_dot)
    x_dot = (n_dot * den - n * den_dot) / (den * den)
    beta_dot = -x_dot / math.sqrt(max(1.0 - x * x, eps))
    return float(alpha_dot + branch * beta_dot)


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _trim_poly_desc(c: np.ndarray, *, eps: float = 1e-14) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    if c.size == 0:
        return c
    m = float(np.max(np.abs(c)))
    if m == 0.0:
        return np.array([0.0])
    keep = np.flatnonzero(np.abs(c) > eps * m)
    if keep.size == 0:
        return np.array([0.0])
    return c[keep[0] :]


def _poly_mul_asc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.convolve(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))


def _poly_add_asc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = max(len(a), len(b))
    out = np.zeros(n, dtype=np.float64)
    out[: len(a)] += a
    out[: len(b)] += b
    return out


def _poly_sub_asc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = max(len(a), len(b))
    out = np.zeros(n, dtype=np.float64)
    out[: len(a)] += a
    out[: len(b)] -= b
    return out


def _poly_trim_asc(c: np.ndarray, *, eps: float = 1e-14) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    if c.size == 0:
        return c
    m = float(np.max(np.abs(c)))
    if m == 0.0:
        return np.array([0.0])
    nz = np.flatnonzero(np.abs(c) > eps * m)
    if nz.size == 0:
        return np.array([0.0])
    return c[: int(nz[-1]) + 1]


_SQ_STATIONARY_COEFF_LAMBDAS: list | None = None


def _build_sq_stationary_coeff_lambdas() -> list:
    """Lazily build ``lambdify`` evaluators for the degree-6 squared stationary polynomial."""

    import sympy as sp

    t, vg, rp, r0s, a, bq, c, k, b0, b1 = sp.symbols(
        "t vg rp r0s a bq c k b0 b1", real=True
    )
    s = c + bq * t + a * t**2
    qdb = b0 + b1 * t
    r = r0s + vg * t
    n = r * r - rp * rp + s
    d = sp.sqrt(s)
    alpha_dot = k / s
    d_dot = qdb / d
    den = 2 * r * d
    x = n / den
    x_dot = sp.diff(x, t)
    num = sp.expand(sp.fraction(sp.together(alpha_dot**2 * (1 - x**2) - x_dot**2))[0])
    poly = sp.Poly(num, t)
    args = (a, bq, c, k, b0, b1, vg, rp, r0s)
    return [sp.lambdify(args, coeff, "numpy") for coeff in poly.all_coeffs()]


def _stationary_angle_sextic_coeffs(
    q0: np.ndarray,
    b: np.ndarray,
    r_planet: float,
    grow_rate: float,
    radius_at_zero: float,
) -> np.ndarray:
    """Ascending coeffs of the degree-≤6 squared stationary polynomial.

    For ``q(t)=q0+bt`` and ``r(t)=radius_at_zero+grow_rate*t``. Roots are candidates
    for ``dθ_±/dt=0`` (after branch filter / Newton polish).
    """

    global _SQ_STATIONARY_COEFF_LAMBDAS

    q0 = np.asarray(q0, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_coef = float(np.dot(b, b))
    bq = float(2.0 * np.dot(q0, b))
    c_coef = float(np.dot(q0, q0))
    k = float(_cross2(q0, b))
    b0 = float(np.dot(q0, b))
    b1 = a_coef
    v = float(grow_rate)
    rp = float(r_planet)
    r0s = float(radius_at_zero)

    if _SQ_STATIONARY_COEFF_LAMBDAS is None:
        _SQ_STATIONARY_COEFF_LAMBDAS = _build_sq_stationary_coeff_lambdas()

    args = (a_coef, bq, c_coef, k, b0, b1, v, rp, r0s)
    desc = np.array([float(fn(*args)) for fn in _SQ_STATIONARY_COEFF_LAMBDAS], dtype=np.float64)
    return desc[::-1]


def _intersection_angle_derivative_linear(
    q: np.ndarray,
    b: np.ndarray,
    r_planet: float,
    launch_offset: float,
    grow_rate: float,
    t: float,
    branch: int,
    *,
    eps: float = GEOM_EPS,
) -> float:
    """``d/dt`` of ``θ_±`` for ``q(t)=q+bt`` with ``r(t)=launch_offset+grow_rate*t``."""

    qx, qy = float(q[0]), float(q[1])
    bx, by = float(b[0]), float(b[1])
    r = float(launch_offset) + float(grow_rate) * float(t)
    if r <= eps:
        return float("nan")

    d2 = qx * qx + qy * qy
    if d2 <= eps:
        return float("nan")

    d = math.sqrt(d2)
    alpha_dot = (qx * by - qy * bx) / d2
    q_dot_b = qx * bx + qy * by
    d_dot = q_dot_b / d
    v = float(grow_rate)

    n = r * r - r_planet * r_planet + d2
    den = 2.0 * r * d
    x = n / den
    if x <= -1.0 + eps or x >= 1.0 - eps:
        return float("nan")

    n_dot = 2.0 * r * v + 2.0 * q_dot_b
    den_dot = 2.0 * (v * d + r * d_dot)
    x_dot = (n_dot * den - n * den_dot) / (den * den)
    beta_dot = -x_dot / math.sqrt(max(1.0 - x * x, eps))
    return float(alpha_dot + branch * beta_dot)


def _refine_stationary_time(
    t: float,
    q0: np.ndarray,
    b: np.ndarray,
    r_planet: float,
    launch_offset: float,
    grow_rate: float,
    branch: int,
    seg_lo: float,
    seg_hi: float,
    *,
    deriv_tol: float = 1e-7,
    max_iter: int = 12,
) -> float | None:
    """Newton polish on ``dθ_±/dt`` from a sextic seed (squaring introduces spurious roots)."""

    t = min(max(float(t), seg_lo), seg_hi)
    for _ in range(max_iter):
        q = q0 + b * t
        d = _intersection_angle_derivative_linear(
            q, b, r_planet, launch_offset, grow_rate, t, branch
        )
        if not math.isfinite(d):
            return None
        if abs(d) <= deriv_tol:
            return t

        eps = max(1e-7, 1e-6 * max(abs(t), 1.0))
        q_lo = q0 + b * (t - eps)
        q_hi = q0 + b * (t + eps)
        d_lo = _intersection_angle_derivative_linear(
            q_lo, b, r_planet, launch_offset, grow_rate, t - eps, branch
        )
        d_hi = _intersection_angle_derivative_linear(
            q_hi, b, r_planet, launch_offset, grow_rate, t + eps, branch
        )
        if not math.isfinite(d_lo) or not math.isfinite(d_hi):
            return None
        dd = (d_hi - d_lo) / (2.0 * eps)
        if abs(dd) <= 1e-14:
            return None
        t = min(max(t - d / dd, seg_lo), seg_hi)

    q = q0 + b * t
    d = _intersection_angle_derivative_linear(
        q, b, r_planet, launch_offset, grow_rate, t, branch
    )
    if math.isfinite(d) and abs(d) <= deriv_tol:
        return t
    return None


def bisect_derivative_root(
    deriv_fn: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    max_iter: int = 40,
    tol: float = GEOM_TOL,
) -> float | None:
    flo = deriv_fn(lo)
    fhi = deriv_fn(hi)
    if not math.isfinite(flo) or not math.isfinite(fhi):
        return None
    if abs(flo) <= tol:
        return lo
    if abs(fhi) <= tol:
        return hi
    if flo * fhi > 0.0:
        return None

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = deriv_fn(mid)
        if not math.isfinite(fm):
            return None
        if abs(fm) <= tol or hi - lo <= tol:
            return float(mid)
        if flo * fm <= 0.0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return float(0.5 * (lo + hi))


def angular_intersection_extrema(
    center_at: Callable[[float], np.ndarray],
    velocity_at: Callable[[float], np.ndarray],
    circle_radius: float,
    grow_center: np.ndarray,
    grow_rate: float,
    launch_offset: float,
    t_enter: float,
    t_exit: float,
    *,
    split_times: Sequence[float] = (),
    scan_per_piece: int = 8,
    tol: float = GEOM_TOL,
) -> dict[str, Any] | None:
    """Min/max bearing from ``grow_center`` to intersection points over ``[t_enter, t_exit]``."""

    g = np.asarray(grow_center, dtype=np.float64)
    r_planet = float(circle_radius)
    v = float(grow_rate)
    lo = float(launch_offset)
    t0 = float(t_enter)
    t1 = float(t_exit)
    if t1 < t0:
        raise ValueError("t_exit must be >= t_enter")

    if t1 == t0:
        c = np.asarray(center_at(t0), dtype=np.float64)
        gr = _grow_radius(lo, v, t0)
        angs = intersection_angles_from_grow_center(c - g, r_planet, gr)
        if angs is None:
            return None
        a_lo, a_hi, _, _ = angs
        return {
            "min": {
                "angle": a_lo,
                "time": t0,
                "point": g + gr * np.array([math.cos(a_lo), math.sin(a_lo)]),
                "branch": -1,
            },
            "max": {
                "angle": a_hi,
                "time": t0,
                "point": g + gr * np.array([math.cos(a_hi), math.sin(a_hi)]),
                "branch": 1,
            },
        }

    tm = 0.5 * (t0 + t1)
    cm = np.asarray(center_at(tm), dtype=np.float64)
    rm = _grow_radius(lo, v, tm)
    mid_angs = intersection_angles_from_grow_center(cm - g, r_planet, rm)
    ref = (
        0.5 * (mid_angs[0] + mid_angs[1])
        if mid_angs is not None
        else math.atan2(cm[1] - g[1], cm[0] - g[0])
    )

    candidates: list[dict[str, Any]] = []

    def add_candidate(t: float, branch: int) -> None:
        c = np.asarray(center_at(t), dtype=np.float64)
        gr = _grow_radius(lo, v, t)
        angs = intersection_angles_from_grow_center(c - g, r_planet, gr)
        if angs is None:
            return
        a_minus, a_plus, _, _ = angs
        a = a_minus if branch == -1 else a_plus
        a = norm_angle_unwrap_near(a, ref)
        candidates.append(
            {
                "angle": float(a),
                "time": float(t),
                "point": g + gr * np.array([math.cos(a), math.sin(a)]),
                "branch": branch,
            }
        )

    for branch in (-1, 1):
        add_candidate(t0, branch)
        add_candidate(t1, branch)

    cuts = sorted({t0, t1, *(float(s) for s in split_times if t0 < float(s) < t1)})
    for a, b in zip(cuts[:-1], cuts[1:]):
        eps_t = max(1e-9, 1e-9 * (b - a))
        aa, bb = a + eps_t, b - eps_t
        if bb <= aa:
            continue
        for branch in (-1, 1):

            def deriv(t: float, branch: int = branch) -> float:
                c = np.asarray(center_at(t), dtype=np.float64)
                vd = np.asarray(velocity_at(t), dtype=np.float64)
                return intersection_angle_derivative(
                    c - g, vd, r_planet, lo, v, t, branch
                )

            grid = np.linspace(aa, bb, scan_per_piece + 1)
            vals = [deriv(float(x)) for x in grid]
            for j in range(scan_per_piece):
                x0, x1 = float(grid[j]), float(grid[j + 1])
                y0, y1 = vals[j], vals[j + 1]
                if not math.isfinite(y0) or not math.isfinite(y1):
                    continue
                if abs(y0) <= tol:
                    add_candidate(x0, branch)
                    continue
                if y0 * y1 <= 0.0:
                    root = bisect_derivative_root(deriv, x0, x1, tol=tol)
                    if root is not None:
                        add_candidate(root, branch)

    if not candidates:
        return None
    return {
        "min": min(candidates, key=lambda x: x["angle"]),
        "max": max(candidates, key=lambda x: x["angle"]),
    }


def angular_intersection_extrema_stationary(
    circle_center: np.ndarray,
    circle_radius: float,
    grow_center: np.ndarray,
    grow_rate: float,
    launch_offset: float,
    t_enter: float,
    t_exit: float,
    *,
    tol: float = GEOM_TOL,
) -> dict[str, Any] | None:
    """Stationary target: angular extrema of growing/stationary circle intersections.

    For ``|C - G| > R``, the bearing spread is the tangent cone ``α ± arcsin(R/d)``,
    reached when ``r(t) = sqrt(d² - R²)`` (external tangent length). Window endpoints
    are included when the overlap interval is partial.
    """

    g = np.asarray(grow_center, dtype=np.float64)
    c = np.asarray(circle_center, dtype=np.float64)
    r_planet = float(circle_radius)
    v = float(grow_rate)
    lo = float(launch_offset)
    t0 = float(t_enter)
    t1 = float(t_exit)
    if t1 < t0:
        raise ValueError("t_exit must be >= t_enter")

    q = c - g
    d = math.hypot(q[0], q[1])
    if d <= tol:
        return None
    if d <= r_planet + tol:
        return None

    alpha = math.atan2(q[1], q[0])
    delta = math.asin(min(max(r_planet / d, -1.0), 1.0))
    a_minus = alpha - delta
    a_plus = alpha + delta

    r_tan = math.sqrt(max(d * d - r_planet * r_planet, 0.0))
    if v > tol:
        t_star = (r_tan - lo) / v
    else:
        t_star = t0
    ref = 0.5 * (
        norm_angle_unwrap_near(a_minus, alpha) + norm_angle_unwrap_near(a_plus, alpha)
    )

    def pack(t: float, branch: int, angle: float) -> dict[str, Any]:
        gr = _grow_radius(lo, v, t)
        a = norm_angle_unwrap_near(angle, ref)
        return {
            "angle": float(a),
            "time": float(t),
            "point": g + gr * np.array([math.cos(a), math.sin(a)]),
            "branch": int(branch),
        }

    candidates: list[dict[str, Any]] = []
    if t0 - tol <= t_star <= t1 + tol:
        t_use = min(max(t_star, t0), t1)
        candidates.extend(
            [
                pack(t_use, -1, a_minus),
                pack(t_use, 1, a_plus),
            ]
        )

    for t in (t0, t1):
        gr = _grow_radius(lo, v, t)
        angs = intersection_angles_from_grow_center(q, r_planet, gr)
        if angs is None:
            continue
        for branch, raw in ((-1, angs[0]), (1, angs[1])):
            candidates.append(pack(t, branch, raw))

    return {
        "min": min(candidates, key=lambda x: x["angle"]),
        "max": max(candidates, key=lambda x: x["angle"]),
    }


def angular_intersection_extrema_polyline_exact(
    points: np.ndarray,
    circle_radius: float,
    grow_center: np.ndarray,
    grow_rate: float,
    launch_offset: float,
    t_enter: float,
    t_exit: float,
    *,
    angle_ref: float | None = None,
    tol: float = GEOM_TOL,
    root_imag_tol: float = 1e-8,
    root_imag_polish_tol: float = 1e-6,
    deriv_tol: float = 1e-7,
) -> dict[str, Any] | None:
    """Exact polyline angular extrema: endpoints, knots, and sextic stationary roots.

    Sextic roots come from the squared ``dθ_±/dt=0`` condition; each ``(u, branch)``
    is kept only when ``|dθ_±/du|`` at that root is below ``deriv_tol`` (spurious roots
    from squaring are dropped).
    """

    pts = np.asarray(points, dtype=np.float64)
    g = np.asarray(grow_center, dtype=np.float64)
    r_planet = float(circle_radius)
    v = float(grow_rate)
    lo = float(launch_offset)
    t0 = float(t_enter)
    t1 = float(t_exit)

    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        raise ValueError("points must have shape (N, 2) with N >= 2")
    if t1 < t0:
        raise ValueError("t_exit must be >= t_enter")

    path_end = float(len(pts) - 1)
    t0 = max(t0, 0.0)
    t1 = min(t1, path_end)
    if t1 < t0:
        return None

    def center_at(t: float) -> np.ndarray:
        i = int(min(max(int(math.floor(t)), 0), len(pts) - 2))
        u = t - i
        return pts[i] + u * (pts[i + 1] - pts[i])

    if angle_ref is None:
        tm = 0.5 * (t0 + t1)
        cm = center_at(tm)
        rm = _grow_radius(lo, v, tm)
        mid_angs = intersection_angles_from_grow_center(cm - g, r_planet, rm)
        if mid_angs is not None:
            angle_ref = 0.5 * (mid_angs[0] + mid_angs[1])
        else:
            qm = cm - g
            angle_ref = float(math.atan2(qm[1], qm[0]))

    candidates: list[dict[str, Any]] = []

    def add_candidate(t: float, branch: int) -> None:
        c = center_at(t)
        gr = _grow_radius(lo, v, t)
        angs = intersection_angles_from_grow_center(c - g, r_planet, gr)
        if angs is None:
            return
        raw = angs[0] if branch == -1 else angs[1]
        angle = norm_angle_unwrap_near(raw, angle_ref)
        candidates.append(
            {
                "angle": float(angle),
                "time": float(t),
                "point": g + gr * np.array([math.cos(angle), math.sin(angle)]),
                "branch": int(branch),
            }
        )

    for br in (-1, 1):
        add_candidate(t0, br)
        add_candidate(t1, br)

    first_seg = int(min(max(int(math.floor(t0)), 0), len(pts) - 2))
    last_seg = int(min(max(int(math.floor(t1)), 0), len(pts) - 2))

    for i in range(first_seg, last_seg + 1):
        seg_lo = max(t0, float(i))
        seg_hi = min(t1, float(i + 1))
        if seg_hi <= seg_lo:
            continue

        for knot_t in (seg_lo, seg_hi):
            for br in (-1, 1):
                add_candidate(knot_t, br)

        p0 = pts[i]
        b = pts[i + 1] - p0
        # Local ``u`` is from ``seg_lo`` (not the integer knot): ``t = seg_lo + u``.
        u_off = seg_lo - float(i)
        q0 = p0 - g + b * u_off
        r0 = lo + v * seg_lo
        u_len = seg_hi - seg_lo

        poly_asc = _stationary_angle_sextic_coeffs(q0, b, r_planet, v, r0)
        poly_desc = _trim_poly_desc(poly_asc[::-1])
        if poly_desc.size > 1:
            for z in np.roots(poly_desc):
                if abs(z.imag) > root_imag_polish_tol:
                    continue
                u = float(z.real)
                if u < -tol or u > u_len + tol:
                    continue
                u = min(max(u, 0.0), u_len)
                for br in (-1, 1):
                    u_refined = _refine_stationary_time(
                        u,
                        q0,
                        b,
                        r_planet,
                        r0,
                        v,
                        br,
                        0.0,
                        u_len,
                        deriv_tol=deriv_tol,
                    )
                    if u_refined is not None:
                        if u_refined < -tol or u_refined > u_len + tol:
                            continue
                        u_refined = min(max(u_refined, 0.0), u_len)
                        q = q0 + b * u_refined
                        gr = r0 + v * u_refined
                        if intersection_angles_from_grow_center(q, r_planet, gr) is None:
                            continue
                        d = _intersection_angle_derivative_linear(
                            q, b, r_planet, r0, v, u_refined, br
                        )
                        if not math.isfinite(d) or abs(d) > deriv_tol:
                            continue
                        add_candidate(seg_lo + u_refined, br)

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x["time"], x["branch"], x["angle"]))
    deduped: list[dict[str, Any]] = []
    for c in candidates:
        if deduped:
            p = deduped[-1]
            if (
                abs(c["time"] - p["time"]) <= 1e-8
                and c["branch"] == p["branch"]
                and abs(c["angle"] - p["angle"]) <= 1e-8
            ):
                continue
        deduped.append(c)

    return {
        "min": min(deduped, key=lambda x: x["angle"]),
        "max": max(deduped, key=lambda x: x["angle"]),
        "candidates": deduped,
    }


def make_polyline_motion(
    points: np.ndarray,
) -> tuple[Callable[[float], np.ndarray], Callable[[float], np.ndarray], list[float]]:
    pts = np.asarray(points, dtype=np.float64)

    def center_at(t: float) -> np.ndarray:
        t = float(t)
        i = int(min(max(int(math.floor(t)), 0), len(pts) - 2))
        u = t - i
        return pts[i] + u * (pts[i + 1] - pts[i])

    def velocity_at(t: float) -> np.ndarray:
        t = float(t)
        i = int(min(max(int(math.floor(t)), 0), len(pts) - 2))
        return pts[i + 1] - pts[i]

    knot_times = [float(k) for k in range(1, len(pts) - 1)]
    return center_at, velocity_at, knot_times


def circles_overlap_at(
    center_at: Callable[[float], np.ndarray],
    grow_center: np.ndarray,
    circle_radius: float,
    launch_offset: float,
    grow_rate: float,
    t: float,
    *,
    eps: float = 1e-7,
) -> bool:
    g = np.asarray(grow_center, dtype=np.float64)
    c = np.asarray(center_at(t), dtype=np.float64)
    r = _grow_radius(launch_offset, grow_rate, t)
    d = float(np.linalg.norm(c - g))
    return d + eps < circle_radius + r and d > abs(circle_radius - r) - eps


def _windows_from_tangency_hits(
    hits: list[tuple[float, str, float]],
    center_at: Callable[[float], np.ndarray],
    grow_center: np.ndarray,
    circle_radius: float,
    launch_offset: float,
    grow_rate: float,
    horizon: float,
    *,
    eps: float = 1e-7,
) -> list[tuple[float, float]]:
    """Build overlap intervals by toggling at every tangency (external or internal)."""

    t_hor = float(horizon)
    if t_hor < 0.0:
        return []

    def overlaps(t: float) -> bool:
        return circles_overlap_at(
            center_at,
            grow_center,
            circle_radius,
            launch_offset,
            grow_rate,
            t,
            eps=eps,
        )

    # Stable order at equal times: external before internal.
    toggles = sorted(
        ((min(max(float(t), 0.0), t_hor), kind) for t, kind, _ in hits),
        key=lambda x: (x[0], 0 if x[1] == "external" else 1),
    )

    inside = overlaps(0.0)
    windows: list[tuple[float, float]] = []
    t_open: float | None = 0.0 if inside else None

    for t, _kind in toggles:
        if inside:
            if t_open is not None and t > t_open:
                windows.append((t_open, t))
            t_open = None
            inside = False
        else:
            t_open = t
            inside = True

    if inside and t_open is not None and t_hor > t_open:
        windows.append((t_open, t_hor))

    return windows


def intersection_windows(
    center_at: Callable[[float], np.ndarray],
    grow_center: np.ndarray,
    circle_radius: float,
    launch_offset: float,
    grow_rate: float,
    horizon: float,
    *,
    polyline_points: np.ndarray | None = None,
    stationary_center: np.ndarray | None = None,
    eps: float = 1e-7,
) -> list[tuple[float, float]]:
    """Contiguous ``[t_enter, t_exit]`` overlap intervals from tangency hits only."""

    has_poly = polyline_points is not None
    has_stat = stationary_center is not None
    if has_poly == has_stat:
        raise ValueError("exactly one of polyline_points or stationary_center is required")

    t_hor = float(horizon)
    g = np.asarray(grow_center, dtype=np.float64)
    r_planet = float(circle_radius)
    lo = float(launch_offset)
    v = float(grow_rate)

    if has_poly:
        hits = tangent_hit_time_polyline(
            np.asarray(polyline_points, dtype=np.float64),
            r_planet,
            g,
            v,
            lo,
            t_hor,
            mode="both",
            return_all=True,
        )
    else:
        hits = tangent_hit_time_stationary(
            np.asarray(stationary_center, dtype=np.float64),
            r_planet,
            g,
            v,
            lo,
            t_hor,
            mode="both",
            return_all=True,
        )

    assert isinstance(hits, list)
    return _windows_from_tangency_hits(
        hits, center_at, g, r_planet, lo, v, t_hor, eps=eps
    )


def angular_intersection_extrema_polyline(
    points: np.ndarray,
    circle_radius: float,
    grow_center: np.ndarray,
    grow_rate: float,
    launch_offset: float,
    t_enter: float,
    t_exit: float,
    *,
    scan_per_segment: int = 16,
) -> dict[str, Any] | None:
    """Polyline extrema: interval ends plus exact degree-6 stationary roots per segment."""

    del scan_per_segment
    return angular_intersection_extrema_polyline_exact(
        points,
        circle_radius,
        grow_center,
        grow_rate,
        launch_offset,
        t_enter,
        t_exit,
    )


def _rotating_chord_polyline(
    centre_xy: np.ndarray,
    orbital_radius: float,
    angular_velocity: float,
    horizon: int,
    *,
    orbit_center: np.ndarray | None = None,
) -> np.ndarray:
    o = np.asarray(
        [CENTER, CENTER] if orbit_center is None else orbit_center,
        dtype=np.float64,
    )
    p0 = np.asarray(centre_xy, dtype=np.float64)
    rho = float(orbital_radius)
    w = float(angular_velocity)
    th0 = math.atan2(p0[1] - o[1], p0[0] - o[0])
    rows = [p0.copy()]
    for dt in range(1, int(horizon) + 1):
        th = th0 + w * float(dt)
        rows.append(o + rho * np.array([math.cos(th), math.sin(th)], dtype=np.float64))
    return np.stack(rows, axis=0)

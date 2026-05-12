"""Validate symbolic tick-hit replacement through first-hit occlusion.

This test patches ``orbit_wars_pt.geometry_jax.tick_hit_intervals_jax`` to a
symbolic tick-hit interval builder (based on the user's proposal), then
verifies that ``first_hit_intervals_jax`` membership matches an exact
per-angle first-hit predicate.

Correctness scope:
* First-hit interval membership for a *single target object*.
* Exact predicate checks swept segment vs disk intersection + board bounds
  + sun blocking, matching the interpreter ordering used in the existing
  ``check_geometry_jax.py`` audit.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

import jax
import jax.numpy as jnp

import orbit_wars_pt.geometry_jax as gj


TAU = 2.0 * float(np.pi)


def _swept_pair_hit_np(a0, a1, p0, p1, radius: float) -> bool:
    # Check whether the moving segment intersects the radius-R disk.
    d0 = a0 - p0
    dv = (a1 - a0) - (p1 - p0)
    qa = float(np.dot(dv, dv))
    qb = 2.0 * float(np.dot(d0, dv))
    qc = float(np.dot(d0, d0)) - radius * radius
    if qa < 1e-12:
        return qc <= 0.0
    disc = qb * qb - 4.0 * qa * qc
    if disc < 0.0:
        return False
    sd = float(np.sqrt(max(disc, 0.0)))
    t1 = (-qb - sd) / (2.0 * qa)
    t2 = (-qb + sd) / (2.0 * qa)
    return t2 >= 0.0 and t1 <= 1.0


def _point_to_segment_distance_np(point, start, end) -> float:
    delta = end - start
    l2 = float(np.dot(delta, delta))
    if l2 <= 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.dot(point - start, delta) / l2)
    t = max(0.0, min(1.0, t))
    projection = start + t * delta
    return float(np.linalg.norm(point - projection))


def _exact_first_hit_np(
    origin_xy,
    origin_radius,
    speed,
    p0,
    p1,
    radii,
    active,
    angle,
    target_idx: int,
    *,
    board_size: float = 100.0,
    sun_radius: float = 10.0,
) -> bool:
    direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
    launch_offset = float(origin_radius) + 0.1
    origin = np.asarray(origin_xy, dtype=np.float64)
    for tick in range(active.shape[0]):
        a0 = origin + (launch_offset + tick * speed) * direction
        a1 = origin + (launch_offset + (tick + 1) * speed) * direction
        for obj_idx in range(active.shape[1]):
            if not active[tick, obj_idx]:
                continue
            if _swept_pair_hit_np(
                a0,
                a1,
                np.asarray(p0[tick, obj_idx], dtype=np.float64),
                np.asarray(p1[tick, obj_idx], dtype=np.float64),
                float(radii[obj_idx]),
            ):
                return obj_idx == target_idx
        # Board exit check after planet list processing.
        if not (0.0 <= a1[0] <= board_size and 0.0 <= a1[1] <= board_size):
            return False
        # Sun blocking check.
        if _point_to_segment_distance_np(np.array([50.0, 50.0]), a0, a1) < sun_radius:
            return False
    return False


# ---------------------------------------------------------------------------
# Symbolic tick-hit replacement (user proposal), embedded here for testing.
# ---------------------------------------------------------------------------

ROOT_EPS = 1e-4


def _cross2(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return a[0] * b[1] - a[1] * b[0]


def _swept_hit_theta_sym(
    theta: jnp.ndarray,
    q: jnp.ndarray,
    d_vec: jnp.ndarray,
    a_base: jnp.ndarray,
    speed: jnp.ndarray,
    radius: jnp.ndarray,
    eps: float = float(gj.GEOM_EPS),
) -> jnp.ndarray:
    u = jnp.stack([jnp.cos(theta), jnp.sin(theta)])
    r0 = q - a_base * u
    s = d_vec - speed * u

    aa = jnp.dot(s, s)
    bb = 2.0 * jnp.dot(r0, s)
    cc = jnp.dot(r0, r0) - radius * radius

    disc = bb * bb - 4.0 * aa * cc

    stationary = aa <= eps
    stationary_hit = cc <= 0.0

    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    t1 = (-bb - sqrt_disc) / jnp.maximum(2.0 * aa, eps)
    t2 = (-bb + sqrt_disc) / jnp.maximum(2.0 * aa, eps)

    moving_hit = (disc >= -eps) & (t2 >= -eps) & (t1 <= 1.0 + eps)
    return jnp.where(stationary, stationary_hit, moving_hit)


def _norm_angle_sym(a: jnp.ndarray) -> jnp.ndarray:
    return jnp.mod(a, gj.TAU)


def _endpoint_boundary_angles_sym(
    p: jnp.ndarray, rho: jnp.ndarray, radius: jnp.ndarray, eps: float = float(gj.GEOM_EPS)
) -> tuple[jnp.ndarray, jnp.ndarray]:
    d = jnp.linalg.norm(p)
    denom = 2.0 * rho
    c = (jnp.dot(p, p) + rho * rho - radius * radius) / jnp.maximum(denom, eps)
    rhs = c / jnp.maximum(d, eps)
    valid = (rho > eps) & (d > eps) & (jnp.abs(rhs) <= 1.0 + eps)
    phi = jnp.arctan2(p[1], p[0])
    delta = jnp.arccos(jnp.clip(rhs, -1.0, 1.0))
    angles = _norm_angle_sym(jnp.stack([phi - delta, phi + delta]))
    valids = jnp.stack([valid, valid])
    return angles, valids


def _linear_trig_roots_sym(
    c0: jnp.ndarray,
    cc: jnp.ndarray,
    cs: jnp.ndarray,
    eps: float = float(gj.GEOM_EPS),
) -> tuple[jnp.ndarray, jnp.ndarray]:
    amp = jnp.sqrt(cc * cc + cs * cs)
    rhs = -c0 / jnp.maximum(amp, eps)
    valid = (amp > eps) & (jnp.abs(rhs) <= 1.0 + eps)
    phi = jnp.arctan2(cs, cc)
    delta = jnp.arccos(jnp.clip(rhs, -1.0, 1.0))
    angles = _norm_angle_sym(jnp.stack([phi - delta, phi + delta]))
    valids = jnp.stack([valid, valid])
    return angles, valids


def _interior_tangent_poly_coeffs_sym(
    q: jnp.ndarray,
    d_vec: jnp.ndarray,
    a_base: jnp.ndarray,
    speed: jnp.ndarray,
    radius: jnp.ndarray,
) -> jnp.ndarray:
    qx, qy = q[0], q[1]
    dx, dy = d_vec[0], d_vec[1]
    v = speed
    A = a_base
    R = radius

    h0 = qx * dy - qy * dx
    hc = v * qy - A * dy
    hs = -v * qx + A * dx

    a0 = dx * dx + dy * dy + v * v
    ac = -2.0 * v * dx
    ass = -2.0 * v * dy

    H0 = h0 + hc
    H1 = 2.0 * hs
    H2 = h0 - hc

    S0 = a0 + ac
    S1 = 2.0 * ass
    S2 = a0 - ac

    c0 = H0 * H0 - R * R * S0
    c1 = 2.0 * H0 * H1 - R * R * S1
    c2 = H1 * H1 + 2.0 * H0 * H2 - R * R * (S0 + S2)
    c3 = 2.0 * H1 * H2 - R * R * S1
    c4 = H2 * H2 - R * R * S2
    return jnp.stack([c0, c1, c2, c3, c4])


def _interior_tangent_residual_sym(
    theta: jnp.ndarray,
    q: jnp.ndarray,
    d_vec: jnp.ndarray,
    a_base: jnp.ndarray,
    speed: jnp.ndarray,
    radius: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    u = jnp.stack([jnp.cos(theta), jnp.sin(theta)])
    r0 = q - a_base * u
    s = d_vec - speed * u
    ss = jnp.dot(s, s)
    cross = _cross2(r0, s)
    residual = cross * cross - radius * radius * ss
    tau_star = -jnp.dot(r0, s) / jnp.maximum(ss, float(gj.GEOM_EPS))
    return residual, tau_star, ss


def _angles_from_quartic_roots_sym(
    coeff_asc: jnp.ndarray,
    q: jnp.ndarray,
    d_vec: jnp.ndarray,
    a_base: jnp.ndarray,
    speed: jnp.ndarray,
    radius: jnp.ndarray,
    eps: float = ROOT_EPS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    coeff_desc_z = coeff_asc[::-1]
    coeff_desc_y = coeff_asc

    roots_z = jnp.roots(coeff_desc_z, strip_zeros=False)
    roots_y = jnp.roots(coeff_desc_y, strip_zeros=False)

    z_real = jnp.real(roots_z)
    z_valid0 = jnp.isfinite(z_real) & (jnp.abs(jnp.imag(roots_z)) <= eps)
    theta_z = _norm_angle_sym(2.0 * jnp.arctan(z_real))

    y_real = jnp.real(roots_y)
    y_valid0 = jnp.isfinite(y_real) & (jnp.abs(jnp.imag(roots_y)) <= eps)
    theta_y = _norm_angle_sym(2.0 * jnp.arctan2(jnp.ones_like(y_real), y_real))

    theta = jnp.concatenate([theta_z, theta_y])
    valid0 = jnp.concatenate([z_valid0, y_valid0])

    def check(th):
        residual, tau_star, ss = _interior_tangent_residual_sym(
            th, q, d_vec, a_base, speed, radius
        )
        scale = radius * radius * ss + 1.0
        ok_res = jnp.abs(residual) <= 100.0 * eps * scale
        ok_tau = (tau_star > float(gj.GEOM_EPS)) & (tau_star < 1.0 - float(gj.GEOM_EPS))
        ok_ss = ss > float(gj.GEOM_EPS)
        return ok_res & ok_tau & ok_ss

    valid = valid0 & jax.vmap(check)(theta)
    return theta, valid


def _projection_switch_angles_sym(
    q: jnp.ndarray, d_vec: jnp.ndarray, a_base: jnp.ndarray, speed: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    qx, qy = q[0], q[1]
    dx, dy = d_vec[0], d_vec[1]
    A = a_base
    v = speed

    c0_0 = qx * dx + qy * dy + A * v
    cc_0 = -v * qx - A * dx
    cs_0 = -v * qy - A * dy
    a0, v0 = _linear_trig_roots_sym(c0_0, cc_0, cs_0)

    Ap = A + v
    c0_1 = (qx + dx) * dx + (qy + dy) * dy + Ap * v
    cc_1 = -v * (qx + dx) - Ap * dx
    cs_1 = -v * (qy + dy) - Ap * dy
    a1, v1 = _linear_trig_roots_sym(c0_1, cc_1, cs_1)

    return jnp.concatenate([a0, a1]), jnp.concatenate([v0, v1])


def _merge_sorted_cells_sym(
    lo: jnp.ndarray, hi: jnp.ndarray, valid: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Merging isn't required for membership checks; interval_membership
    # already computes the union over valid cells.
    return lo, hi, valid


@jax.jit
def tick_hit_intervals_symbolic_jax(
    origin_xy: jnp.ndarray,
    origin_radius: jnp.ndarray,
    speed: jnp.ndarray,
    tick: jnp.ndarray,
    object_p0: jnp.ndarray,
    object_p1: jnp.ndarray,
    object_radius: jnp.ndarray,
    object_active: jnp.ndarray = jnp.asarray(True),
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    q = object_p0 - origin_xy
    d_vec = object_p1 - object_p0
    a_base = origin_radius + 0.1 + tick.astype(origin_xy.dtype) * speed
    R = object_radius

    ep0_angles, ep0_valid = _endpoint_boundary_angles_sym(q, a_base, R)
    ep1_angles, ep1_valid = _endpoint_boundary_angles_sym(q + d_vec, a_base + speed, R)

    coeff_asc = _interior_tangent_poly_coeffs_sym(q, d_vec, a_base, speed, R)
    tan_angles, tan_valid = _angles_from_quartic_roots_sym(coeff_asc, q, d_vec, a_base, speed, R)

    proj_angles, proj_valid = _projection_switch_angles_sym(q, d_vec, a_base, speed)

    fixed_angles = jnp.asarray([jnp.pi], dtype=origin_xy.dtype)
    fixed_valid = jnp.asarray([True], dtype=bool)

    boundary_angles = jnp.concatenate([ep0_angles, ep1_angles, tan_angles, proj_angles, fixed_angles])
    boundary_valid = jnp.concatenate([ep0_valid, ep1_valid, tan_valid, proj_valid, fixed_valid]) & object_active

    endpoints = jnp.concatenate(
        [
            jnp.asarray([0.0, gj.TAU], dtype=origin_xy.dtype),
            jnp.where(boundary_valid, _norm_angle_sym(boundary_angles), 0.0),
        ]
    )
    endpoints = jnp.sort(jnp.clip(endpoints, 0.0, gj.TAU))
    lo = endpoints[:-1]
    hi = endpoints[1:]
    mid = 0.5 * (lo + hi)

    # Midpoint membership with exact swept predicate.
    cell_hit = jax.vmap(lambda th: _swept_hit_theta_sym(th, q, d_vec, a_base, speed, R))(mid)

    valid = object_active & cell_hit & (hi - lo > float(gj.GEOM_EPS))
    return _merge_sorted_cells_sym(lo, hi, valid)


def tick_hit_intervals_symbolic_wrapper(
    origin_xy: jnp.ndarray,
    origin_radius: jnp.ndarray,
    speed: jnp.ndarray,
    tick: jnp.ndarray,
    object_p0: jnp.ndarray,
    object_p1: jnp.ndarray,
    object_radius: jnp.ndarray,
    object_active: jnp.ndarray = jnp.asarray(True),
    *,
    samples_per_span: int = 9,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Signature-compatible wrapper (ignores samples_per_span).
    return tick_hit_intervals_symbolic_jax(
        origin_xy, origin_radius, speed, tick, object_p0, object_p1, object_radius, object_active
    )


def main() -> None:
    # Patch BEFORE the first call to first_hit_intervals_jax.
    gj.tick_hit_intervals_jax = tick_hit_intervals_symbolic_wrapper  # type: ignore[assignment]

    batch = 6
    ticks = 6
    objects = 8
    num_angles = 360

    rng = np.random.default_rng(1)

    origin_xy = rng.uniform(18.0, 82.0, size=(batch, 2)).astype(np.float32)
    origin_radius = rng.uniform(1.0, 4.0, size=(batch,)).astype(np.float32)
    speed = rng.uniform(1.0, 6.0, size=(batch,)).astype(np.float32)

    p0 = rng.uniform(5.0, 95.0, size=(batch, ticks, objects, 2)).astype(np.float32)
    drift = rng.normal(0.0, 2.5, size=(batch, ticks, objects, 2)).astype(np.float32)
    p1 = np.clip(p0 + drift, -5.0, 105.0).astype(np.float32)

    radii = rng.uniform(0.8, 4.5, size=(batch, objects)).astype(np.float32)
    active = rng.random(size=(batch, ticks, objects)) > 0.15

    target_idx = rng.integers(0, objects, size=(batch,), dtype=np.int32)

    # Ensure at least one "reachable" easy target per case (not required but helps).
    for b in range(batch):
        obj = int(target_idx[b])
        tick = int(rng.integers(1, max(2, ticks // 2 + 1)))
        angle = float(rng.uniform(0.0, TAU))
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        rho = float(origin_radius[b]) + 0.1 + (tick + 0.5) * float(speed[b])
        center = origin_xy[b] + rho * direction
        p0[b, tick, obj] = center
        p1[b, tick, obj] = center + rng.normal(0.0, 0.5, size=(2,))
        radii[b, obj] = 2.5
        active[b, tick, obj] = True

    angles = np.asarray(gj.probe_angle_grid(num_angles), dtype=np.float32)

    # JAX intervals for the single target per case.
    jax_fn = jax.jit(
        jax.vmap(
            lambda ox, orad, sp, p0b, p1b, rb, ab, tb: gj.first_hit_intervals_jax(
                ox,
                orad,
                sp,
                p0b,
                p1b,
                rb,
                ab,
                tb,
                samples_per_span=9,
                max_block_intervals=64,
                max_valid_intervals=16,
            )
        )
    )

    args = (
        jnp.asarray(origin_xy),
        jnp.asarray(origin_radius),
        jnp.asarray(speed),
        jnp.asarray(p0),
        jnp.asarray(p1),
        jnp.asarray(radii),
        jnp.asarray(active),
        jnp.asarray(target_idx),
    )

    lo, hi, valid, _overflow = jax_fn(*args)

    # Membership from interval tensors.
    sym_mask = jax.vmap(gj.interval_membership, in_axes=(None, 0, 0, 0))(
        jnp.asarray(angles), lo, hi, valid
    )
    sym_mask_np = np.asarray(sym_mask, dtype=bool)

    # Exact per-angle first-hit predicate.
    mism = 0
    total = batch * num_angles
    for b in range(batch):
        ti = int(target_idx[b])
        for ai in range(num_angles):
            exact = _exact_first_hit_np(
                origin_xy[b],
                float(origin_radius[b]),
                float(speed[b]),
                p0[b],
                p1[b],
                radii[b],
                active[b],
                float(angles[ai]),
                ti,
            )
            if exact != bool(sym_mask_np[b, ai]):
                mism += 1

    print(f"symbolic tick -> first-hit mismatch: {mism}/{total}")
    if mism:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


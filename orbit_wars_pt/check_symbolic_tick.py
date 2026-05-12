"""Validate proposed symbolic tick-hit intervals.

This test checks *tick_hit_intervals_symbolic_jax* interval membership against
an exact swept-segment-vs-disk predicate on a dense angle grid.

It does not touch rollout / first-hit occlusion logic; it validates the
foundation: for one tick and one moving disk, does the interval set returned
by the symbolic boundary construction match the exact predicate?
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from orbit_wars_pt.geometry_jax import GEOM_EPS, interval_membership


TAU = 2.0 * jnp.pi


# ---------------------------------------------------------------------------
# Symbolic tick-hit code (adapted from the user's proposal)
# ---------------------------------------------------------------------------

ROOT_EPS = 1e-4


def _cross2(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return a[0] * b[1] - a[1] * b[0]


def _norm_angle(a: jnp.ndarray) -> jnp.ndarray:
    return jnp.mod(a, TAU)


def _compact_same_capacity(
    lo: jnp.ndarray,
    hi: jnp.ndarray,
    valid: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    n = lo.shape[0]
    idx = jnp.cumsum(valid.astype(jnp.int32)) - 1
    slots = jnp.arange(n, dtype=jnp.int32)
    write = valid[:, None] & (idx[:, None] == slots[None, :])

    has = jnp.any(write, axis=0)
    out_lo = jnp.sum(jnp.where(write, lo[:, None], 0.0), axis=0)
    out_hi = jnp.sum(jnp.where(write, hi[:, None], 0.0), axis=0)
    return out_lo, out_hi, has


def _merge_sorted_cells(
    lo: jnp.ndarray,
    hi: jnp.ndarray,
    valid: jnp.ndarray,
    eps: float = float(GEOM_EPS),
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Assumes sorted non-wrapping cells over [0, 2pi].
    n = lo.shape[0]

    prev_valid = jnp.concatenate([jnp.asarray([False]), valid[:-1]])
    prev_hi = jnp.concatenate([jnp.asarray([0.0], dtype=hi.dtype), hi[:-1]])
    next_valid = jnp.concatenate([valid[1:], jnp.asarray([False])])
    next_lo = jnp.concatenate([lo[1:], jnp.asarray([0.0], dtype=lo.dtype)])

    contiguous_prev = valid & prev_valid & (jnp.abs(lo - prev_hi) <= eps)
    contiguous_next = valid & next_valid & (jnp.abs(hi - next_lo) <= eps)

    start_mask = valid & (~contiguous_prev)
    end_mask = valid & (~contiguous_next)

    start_lo, _, start_valid = _compact_same_capacity(lo, hi, start_mask)
    _, end_hi, end_valid = _compact_same_capacity(lo, hi, end_mask)

    out_valid = start_valid & end_valid
    # Capacity preserved; invalid cells keep arbitrary lo/hi but membership gates on valid.
    return start_lo, end_hi, out_valid


def _swept_hit_theta(
    theta: jnp.ndarray,
    q: jnp.ndarray,
    d_vec: jnp.ndarray,
    a_base: jnp.ndarray,
    speed: jnp.ndarray,
    radius: jnp.ndarray,
    eps: float = float(GEOM_EPS),
) -> jnp.ndarray:
    """Exact swept-pair hit predicate for one launch angle.

    Equivalent to checking whether the relative segment
        q - A u  ->  q + D - (A+v) u
    intersects the radius-R disk around zero, where u is the unit direction at theta
    and tau in [0,1] parametrizes the target motion.
    """

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


def _endpoint_boundary_angles(
    p: jnp.ndarray,
    rho: jnp.ndarray,
    radius: jnp.ndarray,
    eps: float = float(GEOM_EPS),
) -> tuple[jnp.ndarray, jnp.ndarray]:
    d = jnp.linalg.norm(p)
    denom = 2.0 * rho
    c = (jnp.dot(p, p) + rho * rho - radius * radius) / jnp.maximum(denom, eps)
    rhs = c / jnp.maximum(d, eps)

    valid = (rho > eps) & (d > eps) & (jnp.abs(rhs) <= 1.0 + eps)

    phi = jnp.arctan2(p[1], p[0])
    delta = jnp.arccos(jnp.clip(rhs, -1.0, 1.0))
    angles = _norm_angle(jnp.stack([phi - delta, phi + delta]))
    valids = jnp.stack([valid, valid])
    return angles, valids


def _linear_trig_roots(
    c0: jnp.ndarray,
    cc: jnp.ndarray,
    cs: jnp.ndarray,
    eps: float = float(GEOM_EPS),
) -> tuple[jnp.ndarray, jnp.ndarray]:
    amp = jnp.sqrt(cc * cc + cs * cs)
    rhs = -c0 / jnp.maximum(amp, eps)
    valid = (amp > eps) & (jnp.abs(rhs) <= 1.0 + eps)

    phi = jnp.arctan2(cs, cc)
    delta = jnp.arccos(jnp.clip(rhs, -1.0, 1.0))
    angles = _norm_angle(jnp.stack([phi - delta, phi + delta]))
    valids = jnp.stack([valid, valid])
    return angles, valids


def _interior_tangent_poly_coeffs(
    q: jnp.ndarray,
    d_vec: jnp.ndarray,
    a_base: jnp.ndarray,
    speed: jnp.ndarray,
    radius: jnp.ndarray,
) -> jnp.ndarray:
    """Quartic coefficients for interior tangent boundary (z = tan(theta/2))."""

    qx, qy = q[0], q[1]
    dx, dy = d_vec[0], d_vec[1]
    v = speed
    A = a_base
    R = radius

    # cross(q - A u, D - v u) expressed via sin/cos.
    h0 = qx * dy - qy * dx
    hc = v * qy - A * dy
    hs = -v * qx + A * dx

    # ||D - v u||^2 = a0 + ac*cos(theta) + ass*sin(theta)
    a0 = dx * dx + dy * dy + v * v
    ac = -2.0 * v * dx
    ass = -2.0 * v * dy

    # tan-half substitution:
    # cos = (1-z^2)/(1+z^2)
    # sin = 2z/(1+z^2)
    H0 = h0 + hc
    H1 = 2.0 * hs
    H2 = h0 - hc

    S0 = a0 + ac
    S1 = 2.0 * ass
    S2 = a0 - ac

    # H(z)^2 - R^2 S(z) (1+z^2) = 0
    c0 = H0 * H0 - R * R * S0
    c1 = 2.0 * H0 * H1 - R * R * S1
    c2 = H1 * H1 + 2.0 * H0 * H2 - R * R * (S0 + S2)
    c3 = 2.0 * H1 * H2 - R * R * S1
    c4 = H2 * H2 - R * R * S2
    return jnp.stack([c0, c1, c2, c3, c4])


def _interior_tangent_residual(
    theta: jnp.ndarray,
    q: jnp.ndarray,
    d_vec: jnp.ndarray,
    a_base: jnp.ndarray,
    speed: jnp.ndarray,
    radius: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Tangent residual, projection tau*, and s length squared."""

    u = jnp.stack([jnp.cos(theta), jnp.sin(theta)])
    r0 = q - a_base * u
    s = d_vec - speed * u

    ss = jnp.dot(s, s)
    cross = _cross2(r0, s)
    residual = cross * cross - radius * radius * ss
    tau_star = -jnp.dot(r0, s) / jnp.maximum(ss, GEOM_EPS)
    return residual, tau_star, ss


def _angles_from_quartic_roots(
    coeff_asc: jnp.ndarray,
    q: jnp.ndarray,
    d_vec: jnp.ndarray,
    a_base: jnp.ndarray,
    speed: jnp.ndarray,
    radius: jnp.ndarray,
    eps: float = ROOT_EPS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve interior tangent quartic and return candidate boundary angles."""

    coeff_desc_z = coeff_asc[::-1]
    coeff_desc_y = coeff_asc

    roots_z = jnp.roots(coeff_desc_z, strip_zeros=False)
    roots_y = jnp.roots(coeff_desc_y, strip_zeros=False)

    z_real = jnp.real(roots_z)
    z_valid0 = jnp.isfinite(z_real) & (jnp.abs(jnp.imag(roots_z)) <= eps)
    theta_z = _norm_angle(2.0 * jnp.arctan(z_real))

    y_real = jnp.real(roots_y)
    y_valid0 = jnp.isfinite(y_real) & (jnp.abs(jnp.imag(roots_y)) <= eps)
    theta_y = _norm_angle(2.0 * jnp.arctan2(jnp.ones_like(y_real), y_real))

    theta = jnp.concatenate([theta_z, theta_y])
    valid0 = jnp.concatenate([z_valid0, y_valid0])

    def check(th):
        residual, tau_star, ss = _interior_tangent_residual(
            th, q, d_vec, a_base, speed, radius
        )
        scale = radius * radius * ss + 1.0
        ok_res = jnp.abs(residual) <= 100.0 * eps * scale
        ok_tau = (tau_star > GEOM_EPS) & (tau_star < 1.0 - GEOM_EPS)
        ok_ss = ss > GEOM_EPS
        return ok_res & ok_tau & ok_ss

    valid = valid0 & jax.vmap(check)(theta)
    return theta, valid


def _projection_switch_angles(
    q: jnp.ndarray,
    d_vec: jnp.ndarray,
    a_base: jnp.ndarray,
    speed: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Roots where the closest-point projection switches endpoint/interior."""

    qx, qy = q[0], q[1]
    dx, dy = d_vec[0], d_vec[1]
    A = a_base
    v = speed

    # tau* = 0 boundary: (q - A u) · (D - v u) = 0
    c0_0 = qx * dx + qy * dy + A * v
    cc_0 = -v * qx - A * dx
    cs_0 = -v * qy - A * dy
    a0, v0 = _linear_trig_roots(c0_0, cc_0, cs_0)

    # tau* = 1 boundary: r1 · s = 0
    Ap = A + v
    c0_1 = (qx + dx) * dx + (qy + dy) * dy + Ap * v
    cc_1 = -v * (qx + dx) - Ap * dx
    cs_1 = -v * (qy + dy) - Ap * dy
    a1, v1 = _linear_trig_roots(c0_1, cc_1, cs_1)

    return jnp.concatenate([a0, a1]), jnp.concatenate([v0, v1])


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
    """Exact swept-disk angular intervals for one moving disk (per tick)."""

    q = object_p0 - origin_xy
    d_vec = object_p1 - object_p0
    a_base = origin_radius + 0.1 + tick.astype(origin_xy.dtype) * speed
    R = object_radius

    ep0_angles, ep0_valid = _endpoint_boundary_angles(q, a_base, R)
    ep1_angles, ep1_valid = _endpoint_boundary_angles(q + d_vec, a_base + speed, R)

    coeff_asc = _interior_tangent_poly_coeffs(q, d_vec, a_base, speed, R)
    tan_angles, tan_valid = _angles_from_quartic_roots(
        coeff_asc, q, d_vec, a_base, speed, R
    )

    proj_angles, proj_valid = _projection_switch_angles(q, d_vec, a_base, speed)

    fixed_angles = jnp.asarray([jnp.pi], dtype=origin_xy.dtype)
    fixed_valid = jnp.asarray([True], dtype=bool)

    boundary_angles = jnp.concatenate(
        [ep0_angles, ep1_angles, tan_angles, proj_angles, fixed_angles]
    )
    boundary_valid = jnp.concatenate(
        [ep0_valid, ep1_valid, tan_valid, proj_valid, fixed_valid]
    ) & object_active

    endpoints = jnp.concatenate(
        [
            jnp.asarray([0.0, TAU], dtype=origin_xy.dtype),
            jnp.where(boundary_valid, _norm_angle(boundary_angles), 0.0),
        ]
    )
    endpoints = jnp.sort(jnp.clip(endpoints, 0.0, TAU))
    lo = endpoints[:-1]
    hi = endpoints[1:]
    mid = 0.5 * (lo + hi)

    cell_hit = jax.vmap(lambda th: _swept_hit_theta(th, q, d_vec, a_base, speed, R))(mid)
    valid = object_active & cell_hit & (hi - lo > GEOM_EPS)
    return _merge_sorted_cells(lo, hi, valid)


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------


def _exact_tick_predicate(
    origin_xy: np.ndarray,
    origin_radius: float,
    speed: float,
    tick: int,
    p0: np.ndarray,
    p1: np.ndarray,
    radius: float,
    angles: np.ndarray,
) -> np.ndarray:
    # Compute exact membership for each angle by direct quadratic segment/disk check.
    origin_xy_t = jnp.asarray(origin_xy, dtype=jnp.float32)
    q = jnp.asarray(p0, dtype=jnp.float32) - origin_xy_t
    d_vec = jnp.asarray(p1, dtype=jnp.float32) - jnp.asarray(p0, dtype=jnp.float32)
    a_base = jnp.asarray(origin_radius + 0.1 + tick * speed, dtype=jnp.float32)
    speed_t = jnp.asarray(speed, dtype=jnp.float32)
    R = jnp.asarray(radius, dtype=jnp.float32)

    v = jax.vmap(
        lambda th: _swept_hit_theta(th, q, d_vec, a_base, speed_t, R)
    )(jnp.asarray(angles, dtype=jnp.float32))
    return np.asarray(v, dtype=bool)


def main() -> None:
    rng = np.random.default_rng(0)

    batch = 20
    num_angles = 720

    angles = np.linspace(0.0, float(2.0 * np.pi), num_angles, endpoint=False, dtype=np.float32)

    # Random geometry; ensure we have enough variation.
    origin_xy = rng.uniform(10.0, 90.0, size=(batch, 2)).astype(np.float32)
    origin_radius = rng.uniform(1.0, 4.0, size=(batch,)).astype(np.float32)
    speed = rng.uniform(1.0, 6.0, size=(batch,)).astype(np.float32)
    tick = rng.integers(0, 5, size=(batch,), dtype=np.int32)

    p0 = rng.uniform(0.0, 100.0, size=(batch, 2)).astype(np.float32)
    drift = rng.normal(0.0, 3.0, size=(batch, 2)).astype(np.float32)
    p1 = np.clip(p0 + drift, -5.0, 105.0).astype(np.float32)

    radius = rng.uniform(0.5, 5.0, size=(batch,)).astype(np.float32)
    active = rng.random(size=(batch,)).astype(bool)

    # Evaluate symbolic intervals.
    lo, hi, valid = jax.vmap(
        lambda ox, orad, sp, ti, x0, x1, rr, aa: tick_hit_intervals_symbolic_jax(
            ox,
            orad,
            sp,
            ti,
            x0,
            x1,
            rr,
            aa,
        )
    )(
        jnp.asarray(origin_xy),
        jnp.asarray(origin_radius),
        jnp.asarray(speed),
        jnp.asarray(tick, dtype=jnp.int32),
        jnp.asarray(p0),
        jnp.asarray(p1),
        jnp.asarray(radius),
        jnp.asarray(active),
    )

    # Membership via interval set.
    sym_mask = jax.vmap(interval_membership, in_axes=(None, 0, 0, 0))(
        jnp.asarray(angles, dtype=jnp.float32),
        lo,
        hi,
        valid,
    )
    sym_mask_np = np.asarray(sym_mask, dtype=bool)

    # Membership via exact predicate.
    exact_masks = []
    for b in range(batch):
        exact_masks.append(
            _exact_tick_predicate(
                origin_xy[b],
                float(origin_radius[b]),
                float(speed[b]),
                int(tick[b]),
                p0[b],
                p1[b],
                float(radius[b]),
                angles,
            )
        )
    exact_masks = np.stack(exact_masks, axis=0)

    mism = (sym_mask_np != exact_masks)
    mism_count = int(mism.sum())
    total = int(mism.size)
    print(f"symbolic tick membership mismatches: {mism_count}/{total}")
    if mism_count:
        # Print first few mismatches for inspection.
        idx = np.argwhere(mism)
        for row_i in idx[:5]:
            b, a_i = int(row_i[0]), int(row_i[1])
            th = float(angles[a_i])
            print(f"  mismatch at case={b}, theta_idx={a_i}, theta={th:.6f}")


if __name__ == "__main__":
    main()


"""Fixed-capacity JAX angular interval geometry for Orbit Wars.

The public helpers in this module return interval tensors ``(lo, hi, valid)``
rather than sampled angle masks. The construction mirrors ``geometry.py``:
solve radial active tau windows analytically, hull angular disk intersections
inside those windows, then apply interpreter-ordered interval subtraction.
"""

from __future__ import annotations

from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
from jax import lax

from orbit_wars_pt.constants import BOARD_SIZE, CENTER, SUN_RADIUS


TAU = 2.0 * jnp.pi
GEOM_EPS = 1e-5
ANGLE_PAD = 1e-4


def probe_angle_grid(num_angles: int, dtype=jnp.float32) -> jnp.ndarray:
    """Utility for tests/benchmarks; not used by the interval generator."""

    return jnp.linspace(0.0, TAU, num_angles, endpoint=False, dtype=dtype)


def _norm_angle(a: jnp.ndarray) -> jnp.ndarray:
    return jnp.mod(a, TAU)


def _angle_diff(a: jnp.ndarray, ref: jnp.ndarray) -> jnp.ndarray:
    return ref + jnp.mod(a - ref + jnp.pi, TAU) - jnp.pi


def _empty(capacity: int, dtype=jnp.float32) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return (
        jnp.zeros((capacity,), dtype=dtype),
        jnp.zeros((capacity,), dtype=dtype),
        jnp.zeros((capacity,), dtype=bool),
    )


def _append(
    store_lo: jnp.ndarray,
    store_hi: jnp.ndarray,
    store_valid: jnp.ndarray,
    count: jnp.ndarray,
    new_lo: jnp.ndarray,
    new_hi: jnp.ndarray,
    new_valid: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    cap = store_lo.shape[0]
    n = new_lo.shape[0]
    add_count = jnp.sum(new_valid.astype(jnp.int32))
    overflow = count + add_count > cap
    compact_idx = count + jnp.cumsum(new_valid.astype(jnp.int32)) - 1
    slots = jnp.arange(cap, dtype=jnp.int32)
    write = new_valid[:, None] & (compact_idx[:, None] == slots[None, :]) & (compact_idx[:, None] < cap)
    has_write = jnp.any(write, axis=0)
    write_lo = jnp.sum(jnp.where(write, new_lo[:, None], 0.0), axis=0)
    write_hi = jnp.sum(jnp.where(write, new_hi[:, None], 0.0), axis=0)
    store_lo = jnp.where(has_write, write_lo, store_lo)
    store_hi = jnp.where(has_write, write_hi, store_hi)
    store_valid = has_write | store_valid
    return store_lo, store_hi, store_valid, count + add_count, overflow


def split_wrapped_intervals(
    lo: jnp.ndarray, hi: jnp.ndarray, valid: jnp.ndarray, eps: float = GEOM_EPS
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Split possibly wrapping intervals into non-wrapping pieces."""

    safe_lo = jnp.where(valid, lo, 0.0)
    safe_hi = jnp.where(valid, hi, 0.0)
    lo_n = _norm_angle(safe_lo)
    hi_n = _norm_angle(safe_hi)
    full = valid & (hi - lo >= TAU - eps)
    nowrap = valid & (~full) & (lo_n <= hi_n) & (hi_n - lo_n > eps)
    wrap = valid & (~full) & (lo_n > hi_n)

    lo1 = jnp.where(full, 0.0, lo_n)
    hi1 = jnp.where(full, TAU, jnp.where(wrap, TAU, hi_n))
    v1 = full | nowrap | wrap

    lo2 = jnp.zeros_like(lo_n)
    hi2 = hi_n
    v2 = wrap & (hi_n > eps)
    return jnp.concatenate([lo1, lo2]), jnp.concatenate([hi1, hi2]), jnp.concatenate([v1, v2])


def interval_membership(
    angles: jnp.ndarray, lo: jnp.ndarray, hi: jnp.ndarray, valid: jnp.ndarray
) -> jnp.ndarray:
    """Return ``[A]`` membership of probe angles in a non-wrapping interval set."""

    return jnp.any(
        valid[:, None] & (angles[None, :] >= lo[:, None]) & (angles[None, :] <= hi[:, None]),
        axis=0,
    )


def _set_subtract_cells(
    hit_lo: jnp.ndarray,
    hit_hi: jnp.ndarray,
    hit_valid: jnp.ndarray,
    block_lo: jnp.ndarray,
    block_hi: jnp.ndarray,
    block_valid: jnp.ndarray,
    eps: float = GEOM_EPS,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return elementary non-wrapping cells in ``hit - blocked``."""

    endpoints = jnp.concatenate(
        [
            jnp.asarray([0.0, TAU], dtype=hit_lo.dtype),
            jnp.where(hit_valid, hit_lo, TAU),
            jnp.where(hit_valid, hit_hi, TAU),
            jnp.where(block_valid, block_lo, TAU),
            jnp.where(block_valid, block_hi, TAU),
        ]
    )
    endpoints = jnp.sort(jnp.clip(endpoints, 0.0, TAU))
    lo = endpoints[:-1]
    hi = endpoints[1:]
    mid = 0.5 * (lo + hi)
    in_hit = jnp.any(hit_valid[:, None] & (mid[None, :] >= hit_lo[:, None]) & (mid[None, :] <= hit_hi[:, None]), axis=0)
    in_block = jnp.any(block_valid[:, None] & (mid[None, :] >= block_lo[:, None]) & (mid[None, :] <= block_hi[:, None]), axis=0)
    valid = in_hit & (~in_block) & (hi - lo > eps)
    return lo, hi, valid


def _full_interval(dtype=jnp.float32) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return (
        jnp.asarray([0.0], dtype=dtype),
        jnp.asarray([TAU], dtype=dtype),
        jnp.asarray([True], dtype=bool),
    )


def _cos_between_intervals(lo: jnp.ndarray, hi: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    dtype = lo.dtype
    full_lo, full_hi, full_valid = _full_interval(dtype)
    none_lo, none_hi, none_valid = _empty(1, dtype)

    ge_valid = lo <= -1.0
    ge_none = lo > 1.0
    a_ge = jnp.arccos(jnp.clip(lo, -1.0, 1.0))
    ge_lo, ge_hi, ge_v = split_wrapped_intervals(
        jnp.asarray([-a_ge], dtype=dtype),
        jnp.asarray([a_ge], dtype=dtype),
        jnp.asarray([~ge_none], dtype=bool),
    )
    ge_lo = jnp.where(ge_valid, jnp.pad(full_lo, (0, ge_lo.shape[0] - 1)), ge_lo)
    ge_hi = jnp.where(ge_valid, jnp.pad(full_hi, (0, ge_hi.shape[0] - 1)), ge_hi)
    ge_v = jnp.where(ge_valid, jnp.pad(full_valid, (0, ge_v.shape[0] - 1)), ge_v)

    le_valid = hi >= 1.0
    le_none = hi < -1.0
    a_le = jnp.arccos(jnp.clip(hi, -1.0, 1.0))
    le_lo = jnp.asarray([a_le], dtype=dtype)
    le_hi = jnp.asarray([TAU - a_le], dtype=dtype)
    le_v = jnp.asarray([~le_none], dtype=bool)
    le_lo = jnp.where(le_valid, full_lo, le_lo)
    le_hi = jnp.where(le_valid, full_hi, le_hi)
    le_v = jnp.where(le_valid, full_valid, le_v)

    out_lo, out_hi, out_v = _set_subtract_cells(ge_lo, ge_hi, ge_v, none_lo, none_hi, none_valid)
    return _set_intersect_cells(out_lo, out_hi, out_v, le_lo, le_hi, le_v)


def _set_intersect_cells(
    a_lo: jnp.ndarray,
    a_hi: jnp.ndarray,
    a_valid: jnp.ndarray,
    b_lo: jnp.ndarray,
    b_hi: jnp.ndarray,
    b_valid: jnp.ndarray,
    eps: float = GEOM_EPS,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    endpoints = jnp.concatenate(
        [
            jnp.asarray([0.0, TAU], dtype=a_lo.dtype),
            jnp.where(a_valid, a_lo, TAU),
            jnp.where(a_valid, a_hi, TAU),
            jnp.where(b_valid, b_lo, TAU),
            jnp.where(b_valid, b_hi, TAU),
        ]
    )
    endpoints = jnp.sort(jnp.clip(endpoints, 0.0, TAU))
    lo = endpoints[:-1]
    hi = endpoints[1:]
    mid = 0.5 * (lo + hi)
    in_a = jnp.any(a_valid[:, None] & (mid[None, :] >= a_lo[:, None]) & (mid[None, :] <= a_hi[:, None]), axis=0)
    in_b = jnp.any(b_valid[:, None] & (mid[None, :] >= b_lo[:, None]) & (mid[None, :] <= b_hi[:, None]), axis=0)
    valid = in_a & in_b & (hi - lo > eps)
    return lo, hi, valid


def _shift_intervals(
    lo: jnp.ndarray, hi: jnp.ndarray, valid: jnp.ndarray, delta: float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return split_wrapped_intervals(lo + delta, hi + delta, valid)


def _quadratic_roots(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray, eps: float = GEOM_EPS):
    disc = b * b - 4.0 * a * c
    sd = jnp.sqrt(jnp.maximum(disc, 0.0))
    quad_valid = jnp.abs(a) > eps
    lin_valid = (~quad_valid) & (jnp.abs(b) > eps)
    r0 = jnp.where(quad_valid, (-b - sd) / (2.0 * a), jnp.where(lin_valid, -c / b, 0.0))
    r1 = jnp.where(quad_valid, (-b + sd) / (2.0 * a), r0)
    valid = (quad_valid & (disc >= -eps)) | lin_valid
    return jnp.stack([r0, r1]), jnp.stack([valid, valid])


TANGENT_MODE_EXTERNAL = 1
TANGENT_MODE_INTERNAL = 2
TANGENT_MODE_BOTH = TANGENT_MODE_EXTERNAL | TANGENT_MODE_INTERNAL
TANGENT_KIND_EXTERNAL = 0
TANGENT_KIND_INTERNAL = 1
TANGENT_BRANCH_MINUS = -1
TANGENT_BRANCH_PLUS = 1


def _sort_and_dedup_tangent_candidates(
    times: jnp.ndarray,
    kinds: jnp.ndarray,
    valid: jnp.ndarray,
    *,
    tol: float = 1e-7,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    dedup_tol = max(tol, 1e-5)
    inf = jnp.asarray(jnp.inf, dtype=times.dtype)
    # Secondary key by kind keeps ordering deterministic for equal times.
    key = jnp.where(valid, times + 1e-8 * kinds.astype(times.dtype), inf)
    order = jnp.argsort(key)
    t_sorted = times[order]
    k_sorted = kinds[order]
    v_sorted = valid[order]
    prev_t = jnp.concatenate([jnp.asarray([-jnp.inf], dtype=t_sorted.dtype), t_sorted[:-1]])
    prev_k = jnp.concatenate([jnp.asarray([-1], dtype=k_sorted.dtype), k_sorted[:-1]])
    dedup = v_sorted & ~((jnp.abs(t_sorted - prev_t) <= dedup_tol) & (k_sorted == prev_k))
    return t_sorted, k_sorted, dedup


def _poly_pad_asc(coeff: jnp.ndarray, size: int) -> jnp.ndarray:
    coeff = jnp.asarray(coeff)
    if coeff.shape[0] >= size:
        return coeff[:size]
    return jnp.pad(coeff, (0, size - coeff.shape[0]))


def _poly_add_asc_jax(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    size = max(a.shape[0], b.shape[0])
    return _poly_pad_asc(a, size) + _poly_pad_asc(b, size)


def _poly_sub_asc_jax(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    size = max(a.shape[0], b.shape[0])
    return _poly_pad_asc(a, size) - _poly_pad_asc(b, size)


def _poly_mul_asc_jax(a: jnp.ndarray, b: jnp.ndarray, *, size: int) -> jnp.ndarray:
    return _poly_pad_asc(jnp.convolve(jnp.asarray(a), jnp.asarray(b)), size)


def _poly_deriv_asc_jax(a: jnp.ndarray) -> jnp.ndarray:
    a = jnp.asarray(a)
    if a.shape[0] <= 1:
        return jnp.zeros((1,), dtype=a.dtype)
    idx = jnp.arange(1, a.shape[0], dtype=a.dtype)
    return idx * a[1:]


def _trim_degree6_desc(
    coeff_desc: jnp.ndarray,
    *,
    eps: float = 1e-12,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Trim descending coeffs to degree<=6 and left-align into a fixed size tensor."""

    c = jnp.asarray(coeff_desc)
    if c.shape[0] != 7:
        raise ValueError("degree-6 descending coefficients must have shape (7,)")
    scale = jnp.max(jnp.abs(c))
    nz = jnp.abs(c) > jnp.maximum(eps, eps * scale)
    first = jnp.argmax(nz.astype(jnp.int32))
    any_nz = jnp.any(nz)
    degree = jnp.where(any_nz, 6 - first, 0).astype(jnp.int32)
    shift = first.astype(jnp.int32)
    idx = jnp.arange(7, dtype=jnp.int32)
    src = jnp.minimum(idx + shift, 6)
    shifted = c[src]
    shifted = jnp.where(idx <= degree, shifted, jnp.asarray(0.0, dtype=c.dtype))
    return shifted, degree


def stationary_angle_sextic_coeffs_jax(
    q0: jnp.ndarray,
    b: jnp.ndarray,
    r_planet: jnp.ndarray,
    grow_rate: jnp.ndarray,
    radius_at_zero: jnp.ndarray,
) -> jnp.ndarray:
    """Descending degree-6 coeffs mirroring ``_stationary_angle_sextic_coeffs``.

    This builds the squared stationary-root numerator directly with fixed-size
    polynomial arithmetic in ascending powers of ``t`` and then drops the
    guaranteed-negligible degree-7/8 terms after trimming.
    """

    q0 = jnp.asarray(q0)
    b = jnp.asarray(b, dtype=q0.dtype)
    a_coef = jnp.sum(b * b)
    bq = 2.0 * jnp.sum(q0 * b)
    c_coef = jnp.sum(q0 * q0)
    k = q0[0] * b[1] - q0[1] * b[0]
    v = jnp.asarray(grow_rate, dtype=q0.dtype)
    rp = jnp.asarray(r_planet, dtype=q0.dtype)
    r0s = jnp.asarray(radius_at_zero, dtype=q0.dtype)

    size = 9
    s = _poly_pad_asc(jnp.asarray([c_coef, bq, a_coef], dtype=q0.dtype), size)
    r = _poly_pad_asc(jnp.asarray([r0s, v], dtype=q0.dtype), size)
    r2 = _poly_mul_asc_jax(r, r, size=size)
    n = _poly_add_asc_jax(r2, s)
    n = n.at[0].add(-(rp * rp))
    qdb = _poly_deriv_asc_jax(0.5 * s)
    qdb = _poly_pad_asc(qdb, size)
    n_dot = _poly_deriv_asc_jax(n)
    n_dot = _poly_pad_asc(n_dot, size)

    four_r2s = 4.0 * _poly_mul_asc_jax(r2, s, size=size)
    n_sq = _poly_mul_asc_jax(n, n, size=size)
    left_inner = _poly_sub_asc_jax(four_r2s, n_sq)
    left = (k * k) * _poly_mul_asc_jax(r2, left_inner, size=size)

    r_ndot = _poly_mul_asc_jax(r, n_dot, size=size)
    v_n = v * n
    term_a = _poly_mul_asc_jax(s, _poly_sub_asc_jax(r_ndot, v_n), size=size)
    nrq = _poly_mul_asc_jax(_poly_mul_asc_jax(n, r, size=size), qdb, size=size)
    term = _poly_sub_asc_jax(term_a, nrq)
    right = _poly_mul_asc_jax(term, term, size=size)

    poly_asc = 4.0 * _poly_sub_asc_jax(left, right)
    # The symbolic reference cancels to degree <= 6. In finite precision the top
    # two degrees can retain tiny residue; drop them after scale-aware trimming.
    scale = jnp.max(jnp.abs(poly_asc))
    poly_asc = jnp.where(jnp.abs(poly_asc) <= jnp.maximum(1e-20, 1e-14 * scale), 0.0, poly_asc)
    poly_asc_deg6 = poly_asc[:7]
    return poly_asc_deg6[::-1]


def _companion_matrix_degree6_desc_jax(
    coeff_desc: jnp.ndarray,
    *,
    eps: float = 1e-12,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build a 6×6 companion matrix and effective degree for descending coeffs."""

    coeff, degree = _trim_degree6_desc(coeff_desc, eps=eps)
    lead = coeff[0]
    safe_lead = jnp.where(jnp.abs(lead) > eps, lead, jnp.asarray(1.0, dtype=coeff.dtype))
    monic_tail = coeff[1:] / safe_lead
    n = 6
    comp = jnp.zeros((n, n), dtype=coeff.dtype)
    comp = comp.at[1:, :-1].set(jnp.eye(n - 1, dtype=coeff.dtype))
    comp = comp.at[:, -1].set(-monic_tail[::-1])
    return comp, degree


def poly_roots_degree6_desc_jax(
    coeff_desc: jnp.ndarray,
    *,
    eps: float = 1e-12,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Companion-matrix roots for a real degree-<=6 polynomial in descending order.

    Returns ``(roots[6], valid[6])``. For lower-degree polynomials, extra roots from the
    padded companion matrix may appear; they are left to later physics/derivative filters.
    """

    comp, degree = _companion_matrix_degree6_desc_jax(coeff_desc, eps=eps)
    roots = jnp.linalg.eigvals(comp)
    valid = degree > 0
    return roots, jnp.full((6,), valid, dtype=bool)


def poly_roots_degree6_desc_batch_jax(
    coeff_desc_batch: jnp.ndarray,
    *,
    eps: float = 1e-12,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Batched companion-matrix roots: one ``eigvals`` over ``[..., 6, 6]`` companions."""

    coeff_desc_batch = jnp.asarray(coeff_desc_batch)
    if coeff_desc_batch.shape[-1] != 7:
        raise ValueError("coeff_desc_batch must have trailing shape (7,)")
    batch_shape = coeff_desc_batch.shape[:-1]
    flat = coeff_desc_batch.reshape(-1, 7)
    companions, degrees = jax.vmap(
        lambda c: _companion_matrix_degree6_desc_jax(c, eps=eps),
        in_axes=0,
    )(flat)
    roots_flat = jnp.linalg.eigvals(companions)
    valid_flat = jnp.broadcast_to((degrees > 0)[:, None], roots_flat.shape)
    return (
        roots_flat.reshape(batch_shape + (6,)),
        valid_flat.reshape(batch_shape + (6,)),
    )


def poly_roots_degree6_desc_batch_impl_jax(
    coeff_desc_batch: jnp.ndarray,
    *,
    implementation: lax.linalg.EigImplementation,
    eps: float = 1e-12,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Batched companion-matrix roots using an explicit eig backend."""

    coeff_desc_batch = jnp.asarray(coeff_desc_batch)
    if coeff_desc_batch.shape[-1] != 7:
        raise ValueError("coeff_desc_batch must have trailing shape (7,)")
    batch_shape = coeff_desc_batch.shape[:-1]
    flat = coeff_desc_batch.reshape(-1, 7)
    companions, degrees = jax.vmap(
        lambda c: _companion_matrix_degree6_desc_jax(c, eps=eps),
        in_axes=0,
    )(flat)
    roots_flat = lax.linalg.eig(
        companions,
        compute_left_eigenvectors=False,
        compute_right_eigenvectors=False,
        implementation=implementation,
    )[0]
    valid_flat = jnp.broadcast_to((degrees > 0)[:, None], roots_flat.shape)
    return (
        roots_flat.reshape(batch_shape + (6,)),
        valid_flat.reshape(batch_shape + (6,)),
    )


def _intersection_angles_exist_jax(
    q: jnp.ndarray,
    circle_radius: jnp.ndarray,
    grow_radius: jnp.ndarray,
    *,
    eps: float = GEOM_EPS,
) -> jnp.ndarray:
    q = jnp.asarray(q)
    d = jnp.linalg.norm(q)
    denom = 2.0 * jnp.maximum(grow_radius * d, eps)
    x = (grow_radius * grow_radius - circle_radius * circle_radius + d * d) / denom
    return (d > eps) & (grow_radius > eps) & (x >= -1.0 - 1e-6) & (x <= 1.0 + 1e-6)


def _intersection_angle_derivative_linear_jax(
    q: jnp.ndarray,
    b: jnp.ndarray,
    r_planet: jnp.ndarray,
    launch_offset: jnp.ndarray,
    grow_rate: jnp.ndarray,
    t: jnp.ndarray,
    branch: jnp.ndarray,
    *,
    eps: float = GEOM_EPS,
) -> jnp.ndarray:
    """JAX port of ``_intersection_angle_derivative_linear``."""

    q = jnp.asarray(q)
    b = jnp.asarray(b, dtype=q.dtype)
    r = launch_offset + grow_rate * t
    d2 = jnp.sum(q * q)
    d = jnp.sqrt(jnp.maximum(d2, eps))
    alpha_dot = (q[0] * b[1] - q[1] * b[0]) / jnp.maximum(d2, eps)
    q_dot_b = jnp.sum(q * b)
    d_dot = q_dot_b / d
    n = r * r - r_planet * r_planet + d2
    den = 2.0 * r * d
    x = n / jnp.maximum(den, eps)
    n_dot = 2.0 * r * grow_rate + 2.0 * q_dot_b
    den_dot = 2.0 * (grow_rate * d + r * d_dot)
    x_dot = (n_dot * den - n * den_dot) / jnp.maximum(den * den, eps)
    beta_dot = -x_dot / jnp.sqrt(jnp.maximum(1.0 - x * x, eps))
    deriv = alpha_dot + branch.astype(q.dtype) * beta_dot
    valid = (r > eps) & (d2 > eps) & (x > -1.0 + eps) & (x < 1.0 - eps)
    return jnp.where(valid, deriv, jnp.asarray(jnp.nan, dtype=q.dtype))


def _refine_stationary_time_jax(
    u0: jnp.ndarray,
    q0: jnp.ndarray,
    b: jnp.ndarray,
    r_planet: jnp.ndarray,
    launch_offset: jnp.ndarray,
    grow_rate: jnp.ndarray,
    branch: jnp.ndarray,
    seg_lo: jnp.ndarray,
    seg_hi: jnp.ndarray,
    *,
    deriv_tol: float = 1e-5,
    max_iter: int = 12,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Newton polish mirroring ``_refine_stationary_time``."""

    dtype = q0.dtype
    t0 = jnp.clip(u0, seg_lo, seg_hi)

    def body(_, carry):
        t, ok = carry
        q = q0 + b * t
        d = _intersection_angle_derivative_linear_jax(
            q, b, r_planet, launch_offset, grow_rate, t, branch
        )
        finite = jnp.isfinite(d)
        close = finite & (jnp.abs(d) <= deriv_tol)
        eps_t = jnp.maximum(jnp.asarray(1e-7, dtype=dtype), jnp.asarray(1e-6, dtype=dtype) * jnp.maximum(jnp.abs(t), 1.0))
        q_lo = q0 + b * (t - eps_t)
        q_hi = q0 + b * (t + eps_t)
        d_lo = _intersection_angle_derivative_linear_jax(
            q_lo, b, r_planet, launch_offset, grow_rate, t - eps_t, branch
        )
        d_hi = _intersection_angle_derivative_linear_jax(
            q_hi, b, r_planet, launch_offset, grow_rate, t + eps_t, branch
        )
        finite2 = finite & jnp.isfinite(d_lo) & jnp.isfinite(d_hi)
        dd = (d_hi - d_lo) / (2.0 * eps_t)
        step_ok = finite2 & (jnp.abs(dd) > 1e-14)
        t_new = jnp.clip(t - d / dd, seg_lo, seg_hi)
        t_out = jnp.where(close | (~step_ok), t, t_new)
        ok_out = ok & (close | step_ok)
        return t_out, ok_out

    t1, ok1 = jax.lax.fori_loop(0, max_iter, body, (t0, jnp.asarray(True)))
    q1 = q0 + b * t1
    d1 = _intersection_angle_derivative_linear_jax(
        q1, b, r_planet, launch_offset, grow_rate, t1, branch
    )
    final_ok = ok1 & jnp.isfinite(d1) & (jnp.abs(d1) <= deriv_tol)
    return t1, final_ok


def _sextic_stationary_root_candidates_from_roots_jax(
    roots: jnp.ndarray,
    roots_valid: jnp.ndarray,
    q0: jnp.ndarray,
    b: jnp.ndarray,
    r_planet: jnp.ndarray,
    launch_offset: jnp.ndarray,
    grow_rate: jnp.ndarray,
    u_len: jnp.ndarray,
    *,
    tol: float = GEOM_EPS,
    root_imag_polish_tol: float = 1e-6,
    deriv_tol: float = 1e-5,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Filter/refine stationary times given precomputed eigenvalue roots (length 6)."""
    u_raw = jnp.real(roots)
    imag_ok = jnp.abs(jnp.imag(roots)) <= root_imag_polish_tol
    in_range = (u_raw >= -tol) & (u_raw <= u_len + tol)
    u_clamped = jnp.clip(u_raw, 0.0, u_len)

    root_idx = jnp.arange(6, dtype=jnp.int32)
    branch_vals = jnp.asarray([TANGENT_BRANCH_MINUS, TANGENT_BRANCH_PLUS], dtype=jnp.int32)
    root_grid = jnp.repeat(root_idx, 2)
    branch_grid = jnp.tile(branch_vals, 6)
    u_grid = jnp.repeat(u_clamped, 2)
    valid_grid = jnp.repeat(roots_valid & imag_ok & in_range, 2)

    def one_candidate(u, branch, valid0):
        u_refined, ok = _refine_stationary_time_jax(
            u,
            q0,
            b,
            r_planet,
            launch_offset,
            grow_rate,
            branch,
            jnp.asarray(0.0, dtype=q0.dtype),
            u_len,
            deriv_tol=deriv_tol,
        )
        q = q0 + b * u_refined
        gr = launch_offset + grow_rate * u_refined
        exists = _intersection_angles_exist_jax(q, r_planet, gr)
        d = _intersection_angle_derivative_linear_jax(
            q, b, r_planet, launch_offset, grow_rate, u_refined, branch
        )
        ok = valid0 & ok & exists & jnp.isfinite(d) & (jnp.abs(d) <= deriv_tol)
        return u_refined, branch, ok

    times, branches, valid = jax.vmap(one_candidate)(u_grid, branch_grid, valid_grid)
    # Sort by (time, branch) for easier comparison with NumPy.
    inf = jnp.asarray(jnp.inf, dtype=times.dtype)
    key = jnp.where(valid, times + 1e-8 * branches.astype(times.dtype), inf)
    order = jnp.argsort(key)
    t_sorted = times[order]
    b_sorted = branches[order]
    v_sorted = valid[order]
    prev_t = jnp.concatenate([jnp.asarray([-jnp.inf], dtype=t_sorted.dtype), t_sorted[:-1]])
    prev_b = jnp.concatenate([jnp.asarray([0], dtype=b_sorted.dtype), b_sorted[:-1]])
    dedup = v_sorted & ~((jnp.abs(t_sorted - prev_t) <= 5e-5) & (b_sorted == prev_b))
    return t_sorted, b_sorted, dedup


def sextic_stationary_root_candidates_jax(
    coeff_desc: jnp.ndarray,
    q0: jnp.ndarray,
    b: jnp.ndarray,
    r_planet: jnp.ndarray,
    launch_offset: jnp.ndarray,
    grow_rate: jnp.ndarray,
    u_len: jnp.ndarray,
    *,
    tol: float = GEOM_EPS,
    root_imag_polish_tol: float = 1e-6,
    deriv_tol: float = 1e-5,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Prototype JAX port of the per-segment sextic root filtering in the NumPy reference."""

    roots, roots_valid = poly_roots_degree6_desc_jax(coeff_desc)
    return _sextic_stationary_root_candidates_from_roots_jax(
        roots,
        roots_valid,
        q0,
        b,
        r_planet,
        launch_offset,
        grow_rate,
        u_len,
        tol=tol,
        root_imag_polish_tol=root_imag_polish_tol,
        deriv_tol=deriv_tol,
    )


def sextic_stationary_root_candidates_batch_jax(
    coeff_desc_batch: jnp.ndarray,
    q0_batch: jnp.ndarray,
    b_batch: jnp.ndarray,
    r_planet: jnp.ndarray,
    launch_offset_batch: jnp.ndarray,
    grow_rate: jnp.ndarray,
    u_len_batch: jnp.ndarray,
    *,
    tol: float = GEOM_EPS,
    root_imag_polish_tol: float = 1e-6,
    deriv_tol: float = 1e-5,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Batched sextic roots: one ``eigvals`` over all rows, then per-row Newton/filter."""

    coeff_desc_batch = jnp.asarray(coeff_desc_batch)
    batch_shape = coeff_desc_batch.shape[:-1]
    flat_coeff = coeff_desc_batch.reshape(-1, 7)
    flat_q0 = jnp.asarray(q0_batch).reshape(-1, 2)
    flat_b = jnp.asarray(b_batch).reshape(-1, 2)
    flat_r0 = jnp.asarray(launch_offset_batch).reshape(-1)
    flat_u_len = jnp.asarray(u_len_batch).reshape(-1)
    def _broadcast_batch(x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x)
        if x.shape == batch_shape:
            return x.reshape(-1)
        pad = len(batch_shape) - x.ndim
        if pad < 0:
            raise ValueError(f"cannot broadcast {x.shape} to batch shape {batch_shape}")
        return jnp.broadcast_to(x[(...,) + (None,) * pad], batch_shape).reshape(-1)

    flat_r_planet = _broadcast_batch(r_planet)
    flat_grow_rate = _broadcast_batch(grow_rate)

    roots_batch, roots_valid_batch = poly_roots_degree6_desc_batch_jax(flat_coeff)

    def one_row(roots, roots_valid, q0, b, launch_offset, u_len, rp, gr):
        return _sextic_stationary_root_candidates_from_roots_jax(
            roots,
            roots_valid,
            q0,
            b,
            rp,
            launch_offset,
            gr,
            u_len,
            tol=tol,
            root_imag_polish_tol=root_imag_polish_tol,
            deriv_tol=deriv_tol,
        )

    t_flat, br_flat, v_flat = jax.vmap(one_row)(
        roots_batch,
        roots_valid_batch,
        flat_q0,
        flat_b,
        flat_r0,
        flat_u_len,
        flat_r_planet,
        flat_grow_rate,
    )
    return (
        t_flat.reshape(batch_shape + (12,)),
        br_flat.reshape(batch_shape + (12,)),
        v_flat.reshape(batch_shape + (12,)),
    )


def sextic_stationary_root_candidates_from_roots_batch_jax(
    roots_batch: jnp.ndarray,
    roots_valid_batch: jnp.ndarray,
    q0_batch: jnp.ndarray,
    b_batch: jnp.ndarray,
    r_planet: jnp.ndarray,
    launch_offset_batch: jnp.ndarray,
    grow_rate: jnp.ndarray,
    u_len_batch: jnp.ndarray,
    *,
    tol: float = GEOM_EPS,
    root_imag_polish_tol: float = 1e-6,
    deriv_tol: float = 1e-5,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Batched Newton/filter stage for precomputed sextic eigenvalue roots."""

    roots_batch = jnp.asarray(roots_batch)
    roots_valid_batch = jnp.asarray(roots_valid_batch)
    batch_shape = roots_batch.shape[:-1]
    flat_roots = roots_batch.reshape(-1, roots_batch.shape[-1])
    flat_roots_valid = roots_valid_batch.reshape(-1, roots_valid_batch.shape[-1])
    flat_q0 = jnp.asarray(q0_batch).reshape(-1, 2)
    flat_b = jnp.asarray(b_batch).reshape(-1, 2)
    flat_r0 = jnp.asarray(launch_offset_batch).reshape(-1)
    flat_u_len = jnp.asarray(u_len_batch).reshape(-1)

    def _broadcast_batch(x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x)
        if x.shape == batch_shape:
            return x.reshape(-1)
        pad = len(batch_shape) - x.ndim
        if pad < 0:
            raise ValueError(f"cannot broadcast {x.shape} to batch shape {batch_shape}")
        return jnp.broadcast_to(x[(...,) + (None,) * pad], batch_shape).reshape(-1)

    flat_r_planet = _broadcast_batch(r_planet)
    flat_grow_rate = _broadcast_batch(grow_rate)

    def one_row(roots, roots_valid, q0, b, launch_offset, u_len, rp, gr):
        return _sextic_stationary_root_candidates_from_roots_jax(
            roots,
            roots_valid,
            q0,
            b,
            rp,
            launch_offset,
            gr,
            u_len,
            tol=tol,
            root_imag_polish_tol=root_imag_polish_tol,
            deriv_tol=deriv_tol,
        )

    t_flat, br_flat, v_flat = jax.vmap(one_row)(
        flat_roots,
        flat_roots_valid,
        flat_q0,
        flat_b,
        flat_r0,
        flat_u_len,
        flat_r_planet,
        flat_grow_rate,
    )
    return (
        t_flat.reshape(batch_shape + (12,)),
        br_flat.reshape(batch_shape + (12,)),
        v_flat.reshape(batch_shape + (12,)),
    )


def tangent_hit_times_stationary_jax(
    circle_center: jnp.ndarray,
    circle_radius: jnp.ndarray,
    grow_center: jnp.ndarray,
    grow_rate: jnp.ndarray,
    launch_offset: jnp.ndarray,
    horizon: jnp.ndarray,
    *,
    mode_mask: int = TANGENT_MODE_BOTH,
    tol: float = 1e-7,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return fixed-capacity stationary tangency candidates sorted by time."""

    verify_tol = 1e-5
    c = jnp.asarray(circle_center)
    g = jnp.asarray(grow_center)
    r = jnp.asarray(circle_radius, dtype=c.dtype)
    v = jnp.asarray(grow_rate, dtype=c.dtype)
    lo = jnp.asarray(launch_offset, dtype=c.dtype)
    t_max = jnp.asarray(horizon, dtype=c.dtype)
    d = jnp.linalg.norm(c - g)

    times = jnp.asarray(
        [
            (d - r - lo) / jnp.maximum(v, tol),
            (r - lo - d) / jnp.maximum(v, tol),
            (r - lo + d) / jnp.maximum(v, tol),
            (r + d - lo) / jnp.maximum(v, tol),
            (r - d - lo) / jnp.maximum(v, tol),
        ],
        dtype=c.dtype,
    )
    kinds = jnp.asarray(
        [
            TANGENT_KIND_EXTERNAL,
            TANGENT_KIND_INTERNAL,
            TANGENT_KIND_INTERNAL,
            TANGENT_KIND_INTERNAL,
            TANGENT_KIND_INTERNAL,
        ],
        dtype=jnp.int32,
    )
    ext_enabled = bool(mode_mask & TANGENT_MODE_EXTERNAL)
    int_enabled = bool(mode_mask & TANGENT_MODE_INTERNAL)
    mode_valid = jnp.asarray(
        [ext_enabled, int_enabled, int_enabled, int_enabled, int_enabled], dtype=bool
    )
    in_range = (times >= -tol) & (times <= t_max + tol) & (v > tol)
    times = jnp.clip(times, 0.0, t_max)
    gr = lo + v * times
    ext_err = jnp.abs(d - (r + gr))
    int_err = jnp.abs(d - jnp.abs(r - gr))
    geom_err = jnp.where(kinds == TANGENT_KIND_EXTERNAL, ext_err, int_err)
    valid = mode_valid & in_range & (geom_err <= verify_tol)
    return _sort_and_dedup_tangent_candidates(times, kinds, valid, tol=tol)


def tangent_hit_times_polyline_jax(
    points: jnp.ndarray,
    circle_radius: jnp.ndarray,
    grow_center: jnp.ndarray,
    grow_rate: jnp.ndarray,
    launch_offset: jnp.ndarray,
    horizon: jnp.ndarray,
    *,
    mode_mask: int = TANGENT_MODE_BOTH,
    tol: float = 1e-7,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return fixed-capacity polyline tangency candidates sorted by time."""

    verify_tol = 1e-4
    pts = jnp.asarray(points)
    g = jnp.asarray(grow_center, dtype=pts.dtype)
    r = jnp.asarray(circle_radius, dtype=pts.dtype)
    v = jnp.asarray(grow_rate, dtype=pts.dtype)
    lo = jnp.asarray(launch_offset, dtype=pts.dtype)
    t_cap = jnp.minimum(jnp.asarray(horizon, dtype=pts.dtype), pts.shape[0] - 1.0)

    seg_t0 = jnp.arange(pts.shape[0] - 1, dtype=pts.dtype)
    seg_t1 = jnp.minimum(seg_t0 + 1.0, t_cap)
    seg_valid = (v >= 0.0) & (t_cap >= 0.0) & (seg_t1 >= seg_t0)

    p0 = pts[:-1]
    p1 = pts[1:]
    b = p1 - p0
    a = p0 - g
    u_hi = seg_t1 - seg_t0
    g0 = lo + v * seg_t0

    branch_k0 = jnp.stack([r + g0, r - g0, g0 - r], axis=1)
    branch_k1 = jnp.broadcast_to(
        jnp.asarray([v, -v, v], dtype=pts.dtype)[None, :],
        branch_k0.shape,
    )
    branch_kind = jnp.asarray(
        [TANGENT_KIND_EXTERNAL, TANGENT_KIND_INTERNAL, TANGENT_KIND_INTERNAL], dtype=jnp.int32
    )
    ext_enabled = bool(mode_mask & TANGENT_MODE_EXTERNAL)
    int_enabled = bool(mode_mask & TANGENT_MODE_INTERNAL)
    branch_enabled = jnp.asarray([ext_enabled, int_enabled, int_enabled], dtype=bool)

    dd = jnp.sum(b * b, axis=1)[:, None]
    qd = jnp.sum(a * b, axis=1)[:, None]
    qq = jnp.sum(a * a, axis=1)[:, None]
    aq = dd - branch_k1 * branch_k1
    bq = 2.0 * (qd - branch_k0 * branch_k1)
    cq = qq - branch_k0 * branch_k0

    roots, roots_valid = _quadratic_roots(aq, bq, cq, eps=tol)
    roots = jnp.moveaxis(roots, 0, -1)
    roots_valid = jnp.moveaxis(roots_valid, 0, -1)

    u = jnp.clip(roots, 0.0, u_hi[:, None, None])
    t = seg_t0[:, None, None] + u
    centre = p0[:, None, None, :] + u[..., None] * b[:, None, None, :]
    gr = lo + v * t
    d = jnp.linalg.norm(centre - g, axis=-1)
    ext_err = jnp.abs(d - (r + gr))
    int_err = jnp.abs(d - jnp.abs(r - gr))
    kind_grid = jnp.broadcast_to(branch_kind[None, :, None], roots.shape)
    geom_err = jnp.where(kind_grid == TANGENT_KIND_EXTERNAL, ext_err, int_err)

    valid = (
        roots_valid
        & seg_valid[:, None, None]
        & branch_enabled[None, :, None]
        & (roots >= -tol)
        & (roots <= u_hi[:, None, None] + tol)
        & (geom_err <= verify_tol)
    )

    return _sort_and_dedup_tangent_candidates(
        t.reshape(-1),
        kind_grid.reshape(-1),
        valid.reshape(-1),
        tol=tol,
    )


def _circles_overlap_at_t0_jax(
    circle_center: jnp.ndarray,
    circle_radius: jnp.ndarray,
    grow_center: jnp.ndarray,
    launch_offset: jnp.ndarray,
    *,
    eps: float = 1e-7,
) -> jnp.ndarray:
    """Strict overlap-at-zero test matching ``circles_overlap_at(..., t=0)``."""

    d = jnp.linalg.norm(jnp.asarray(circle_center) - jnp.asarray(grow_center))
    r = jnp.asarray(launch_offset, dtype=d.dtype)
    rp = jnp.asarray(circle_radius, dtype=d.dtype)
    return (d + eps < rp + r) & (d > jnp.abs(rp - r) - eps)


def toggle_intersection_windows_jax(
    hit_times: jnp.ndarray,
    hit_valid: jnp.ndarray,
    inside0: jnp.ndarray,
    horizon: jnp.ndarray,
    *,
    max_windows: int | None = None,
    eps: float = 1e-7,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build fixed-capacity overlap windows by toggling on each valid tangency time.

    Tangency kind is intentionally ignored. The first tangency opens or closes a
    window depending on whether the circles overlap at ``t=0``; subsequent
    tangencies alternate.

    When ``max_windows`` is set, output arrays have shape ``(max_windows,)`` and at
  most that many windows are stored (extra tangencies are still processed but
    cannot append further windows).
    """

    times = jnp.clip(jnp.asarray(hit_times), 0.0, jnp.asarray(horizon, dtype=jnp.asarray(hit_times).dtype))
    valid = jnp.asarray(hit_valid, dtype=bool)
    hor = jnp.asarray(horizon, dtype=times.dtype)
    cap = int(max_windows) if max_windows is not None else times.shape[0] + 1

    def body(carry, x):
        inside, t_open, count, lo_store, hi_store, v_store = carry
        t, is_valid = x

        def on_valid(args):
            inside0, t_open0, count0, lo0, hi0, vv0 = args

            def close_window(args2):
                t_cur, t_open1, count1, lo1, hi1, vv1 = args2
                can_write = (count1 < cap) & (t_cur > t_open1 + eps)
                lo1 = jax.lax.cond(can_write, lambda a: a.at[count1].set(t_open1), lambda a: a, lo1)
                hi1 = jax.lax.cond(can_write, lambda a: a.at[count1].set(t_cur), lambda a: a, hi1)
                vv1 = jax.lax.cond(can_write, lambda a: a.at[count1].set(True), lambda a: a, vv1)
                count1 = count1 + can_write.astype(jnp.int32)
                return (jnp.asarray(False), t_open1, count1, lo1, hi1, vv1)

            def open_window(args2):
                _t_cur, _t_open1, count1, lo1, hi1, vv1 = args2
                return (jnp.asarray(True), t, count1, lo1, hi1, vv1)

            return jax.lax.cond(
                inside0,
                close_window,
                open_window,
                (t, t_open0, count0, lo0, hi0, vv0),
            )

        new_carry = jax.lax.cond(
            is_valid,
            on_valid,
            lambda args: args,
            (inside, t_open, count, lo_store, hi_store, v_store),
        )
        return new_carry, None

    lo0 = jnp.zeros((cap,), dtype=times.dtype)
    hi0 = jnp.zeros((cap,), dtype=times.dtype)
    v0 = jnp.zeros((cap,), dtype=bool)
    t_open0 = jnp.where(inside0, jnp.asarray(0.0, dtype=times.dtype), jnp.asarray(0.0, dtype=times.dtype))
    (inside_f, t_open_f, count_f, lo_f, hi_f, v_f), _ = jax.lax.scan(
        body,
        (jnp.asarray(inside0, dtype=bool), t_open0, jnp.asarray(0, dtype=jnp.int32), lo0, hi0, v0),
        (times, valid),
    )

    can_tail = inside_f & (t_open_f + eps < hor) & (count_f < cap)
    lo_f = jax.lax.cond(can_tail, lambda a: a.at[count_f].set(t_open_f), lambda a: a, lo_f)
    hi_f = jax.lax.cond(can_tail, lambda a: a.at[count_f].set(hor), lambda a: a, hi_f)
    v_f = jax.lax.cond(can_tail, lambda a: a.at[count_f].set(True), lambda a: a, v_f)
    return lo_f, hi_f, v_f


def intersection_windows_stationary_jax(
    circle_center: jnp.ndarray,
    circle_radius: jnp.ndarray,
    grow_center: jnp.ndarray,
    grow_rate: jnp.ndarray,
    launch_offset: jnp.ndarray,
    horizon: jnp.ndarray,
    *,
    tol: float = 1e-7,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Stationary overlap windows from tangency-hit toggles."""

    hit_t, _hit_kind, hit_valid = tangent_hit_times_stationary_jax(
        circle_center,
        circle_radius,
        grow_center,
        grow_rate,
        launch_offset,
        horizon,
        tol=tol,
    )
    inside0 = _circles_overlap_at_t0_jax(circle_center, circle_radius, grow_center, launch_offset, eps=tol)
    return toggle_intersection_windows_jax(hit_t, hit_valid, inside0, horizon, eps=tol)


def intersection_windows_polyline_jax(
    points: jnp.ndarray,
    circle_radius: jnp.ndarray,
    grow_center: jnp.ndarray,
    grow_rate: jnp.ndarray,
    launch_offset: jnp.ndarray,
    horizon: jnp.ndarray,
    *,
    tol: float = 1e-7,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Polyline overlap windows from tangency-hit toggles."""

    hit_t, _hit_kind, hit_valid = tangent_hit_times_polyline_jax(
        points,
        circle_radius,
        grow_center,
        grow_rate,
        launch_offset,
        horizon,
        tol=tol,
    )
    inside0 = _circles_overlap_at_t0_jax(points[0], circle_radius, grow_center, launch_offset, eps=tol)
    return toggle_intersection_windows_jax(hit_t, hit_valid, inside0, horizon, eps=tol)


def intersection_windows_polyline_capped_jax(
    points: jnp.ndarray,
    circle_radius: jnp.ndarray,
    grow_center: jnp.ndarray,
    grow_rate: jnp.ndarray,
    launch_offset: jnp.ndarray,
    horizon: jnp.ndarray,
    *,
    max_tangency_hits: int = 4,
    max_windows: int = 2,
    tol: float = 1e-7,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Polyline overlap windows using the first ``max_tangency_hits`` tangencies, capped at ``max_windows``."""

    hit_t, _hit_kind, hit_valid = tangent_hit_times_polyline_jax(
        points,
        circle_radius,
        grow_center,
        grow_rate,
        launch_offset,
        horizon,
        tol=tol,
    )
    hit_t = hit_t[:max_tangency_hits]
    hit_valid = hit_valid[:max_tangency_hits]
    inside0 = _circles_overlap_at_t0_jax(points[0], circle_radius, grow_center, launch_offset, eps=tol)
    return toggle_intersection_windows_jax(
        hit_t,
        hit_valid,
        inside0,
        horizon,
        max_windows=max_windows,
        eps=tol,
    )


def _radial_active_spans(
    origin_xy: jnp.ndarray,
    launch_offset: jnp.ndarray,
    speed: jnp.ndarray,
    tick: jnp.ndarray,
    object_p0: jnp.ndarray,
    object_p1: jnp.ndarray,
    object_radius: jnp.ndarray,
    eps: float = GEOM_EPS,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return up to five tau spans where radial circle/disk intersection exists."""

    q = object_p0 - origin_xy
    d_vec = object_p1 - object_p0
    a_base = launch_offset + tick.astype(origin_xy.dtype) * speed

    def radial_ok(tau):
        rho = a_base + speed * tau
        dist = jnp.linalg.norm(q + tau * d_vec)
        return (rho >= -eps) & (jnp.abs(dist - rho) <= object_radius + eps)

    dd = jnp.dot(d_vec, d_vec)
    qd = jnp.dot(q, d_vec)
    qq = jnp.dot(q, q)
    roots = []
    root_valids = []
    for c in (object_radius, -object_radius):
        ac = a_base + c
        qa = dd - speed * speed
        qb = 2.0 * (qd - speed * ac)
        qc = qq - ac * ac
        r, rv = _quadratic_roots(qa, qb, qc)
        roots.append(r)
        root_valids.append(rv)
    root = jnp.concatenate(roots)
    root_valid = jnp.concatenate(root_valids) & (root > eps) & (root < 1.0 - eps)
    root_valid = root_valid & jax.vmap(radial_ok)(root)
    taus = jnp.sort(jnp.concatenate([jnp.asarray([0.0, 1.0], dtype=origin_xy.dtype), jnp.where(root_valid, root, 0.0)]))
    lo = taus[:-1]
    hi = taus[1:]
    mid = 0.5 * (lo + hi)
    valid = (hi - lo > eps) & jax.vmap(radial_ok)(mid)
    return lo, hi, valid


def _circle_disk_interval_at_tau(
    origin_xy: jnp.ndarray,
    launch_offset: jnp.ndarray,
    speed: jnp.ndarray,
    tick: jnp.ndarray,
    target_xy: jnp.ndarray,
    target_radius: jnp.ndarray,
    tau: jnp.ndarray,
    eps: float = GEOM_EPS,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    p = target_xy - origin_xy
    d = jnp.linalg.norm(p)
    rho = launch_offset + (tick.astype(origin_xy.dtype) + tau) * speed
    full = target_radius >= d + rho - eps
    radial = jnp.abs(d - rho) <= target_radius + eps
    valid = (rho > 0.0) & (full | ((d > eps) & radial))
    phi = jnp.arctan2(p[1], p[0])
    denom = jnp.maximum(2.0 * d * rho, eps)
    g = jnp.clip((d * d + rho * rho - target_radius * target_radius) / denom, -1.0, 1.0)
    alpha = jnp.arccos(g)
    lo = jnp.where(full, 0.0, _norm_angle(phi - alpha))
    hi = jnp.where(full, TAU, _norm_angle(phi + alpha))
    return lo, hi, valid, full


@partial(jax.jit, static_argnames=("samples_per_span",))
def tick_hit_intervals_jax(
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
    """Return fixed-capacity non-wrapping angle pieces for one object/tick.

    Capacity is ``10``: five radial spans, split at most once for wraparound.
    """

    launch_offset = origin_radius + 0.1
    span_lo, span_hi, span_valid = _radial_active_spans(
        origin_xy, launch_offset, speed, tick, object_p0, object_p1, object_radius
    )
    frac = jnp.linspace(0.0, 1.0, samples_per_span, dtype=origin_xy.dtype)
    tau = span_lo[:, None] + (span_hi - span_lo)[:, None] * frac[None, :]
    target = object_p0[None, None, :] + tau[..., None] * (object_p1 - object_p0)[None, None, :]
    lo, hi, valid, full = jax.vmap(
        jax.vmap(
            lambda xy, tt: _circle_disk_interval_at_tau(
                origin_xy, launch_offset, speed, tick, xy, object_radius, tt
            )
        )
    )(target, tau)
    valid = valid & span_valid[:, None] & object_active

    width = jnp.mod(hi - lo, TAU)
    mid = _norm_angle(lo + 0.5 * width)
    any_full = jnp.any(valid & full, axis=1)
    any_valid = jnp.any(valid, axis=1)
    ref_idx = jnp.argmax(valid.astype(jnp.int32), axis=1)
    ref = jnp.take_along_axis(mid, ref_idx[:, None], axis=1)
    lo_s = _angle_diff(lo, ref)
    hi_s = _angle_diff(hi, ref)
    mid_s = _angle_diff(mid, ref)
    pts = jnp.stack([lo_s, mid_s, hi_s], axis=-1)
    hull_lo = jnp.min(jnp.where(valid[..., None], pts, jnp.inf), axis=(1, 2))
    hull_hi = jnp.max(jnp.where(valid[..., None], pts, -jnp.inf), axis=(1, 2))
    out_lo = jnp.where(any_valid, jnp.where(any_full, 0.0, _norm_angle(hull_lo - ANGLE_PAD)), 0.0)
    out_hi = jnp.where(any_valid, jnp.where(any_full, TAU, _norm_angle(hull_hi + ANGLE_PAD)), 0.0)
    out_valid = any_valid
    return split_wrapped_intervals(out_lo, out_hi, out_valid)


def _all_planet_tick_hits(
    origin_xy: jnp.ndarray,
    origin_radius: jnp.ndarray,
    speed: jnp.ndarray,
    tick: jnp.ndarray,
    object_p0_row: jnp.ndarray,
    object_p1_row: jnp.ndarray,
    object_radii: jnp.ndarray,
    object_active_row: jnp.ndarray,
    *,
    samples_per_span: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Batched ``tick_hit_intervals_jax`` for every object in one tick row.

    One radial-disk solve per planet, fused as a single ``vmap`` instead of
    ``P`` sequential calls inside the occlusion loop.
    """

    def one(p0j: jnp.ndarray, p1j: jnp.ndarray, rj: jnp.ndarray, aj: jnp.ndarray):
        return tick_hit_intervals_jax(
            origin_xy,
            origin_radius,
            speed,
            tick,
            p0j,
            p1j,
            rj,
            aj,
            samples_per_span=samples_per_span,
        )

    return jax.vmap(one)(object_p0_row, object_p1_row, object_radii, object_active_row)


def _board_exit_intervals_jax(
    origin_xy: jnp.ndarray,
    origin_radius: jnp.ndarray,
    speed: jnp.ndarray,
    tick: jnp.ndarray,
    board_size: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Exact endpoint out-of-bounds intervals for the end of this tick."""

    rho = origin_radius + 0.1 + (tick.astype(origin_xy.dtype) + 1.0) * speed
    x_lo, x_hi, x_valid = _cos_between_intervals(
        (0.0 - origin_xy[0]) / rho, (board_size - origin_xy[0]) / rho
    )
    y_lo, y_hi, y_valid = _cos_between_intervals(
        (0.0 - origin_xy[1]) / rho, (board_size - origin_xy[1]) / rho
    )
    y_lo, y_hi, y_valid = _shift_intervals(y_lo, y_hi, y_valid, jnp.pi / 2.0)
    inside_lo, inside_hi, inside_valid = _set_intersect_cells(
        x_lo, x_hi, x_valid, y_lo, y_hi, y_valid
    )
    full_lo, full_hi, full_valid = _full_interval(origin_xy.dtype)
    return _set_subtract_cells(full_lo, full_hi, full_valid, inside_lo, inside_hi, inside_valid)


@partial(
    jax.jit,
    static_argnames=(
        "object_order",
        "include_board",
        "include_sun",
        "board_size",
        "sun_radius",
        "samples_per_span",
        "max_block_intervals",
        "max_valid_intervals",
    ),
)
def first_hit_intervals_jax(
    origin_xy: jnp.ndarray,
    origin_radius: jnp.ndarray,
    speed: jnp.ndarray,
    object_p0_by_tick: jnp.ndarray,
    object_p1_by_tick: jnp.ndarray,
    object_radii: jnp.ndarray,
    object_active_by_tick: jnp.ndarray,
    target_idx: int | jnp.ndarray,
    *,
    object_order: Sequence[int] | None = None,
    include_board: bool = True,
    include_sun: bool = True,
    board_size: float = BOARD_SIZE,
    sun_radius: float = SUN_RADIUS,
    samples_per_span: int = 9,
    max_block_intervals: int = 32,
    max_valid_intervals: int = 8,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return fixed-capacity non-wrapping intervals that hit target first.

    The output is ``(lo, hi, valid)``. Planet/comet hits are processed in list
    order; board and sun blockers are appended after each tick.
    """

    ticks = int(object_active_by_tick.shape[0])
    objects = int(object_active_by_tick.shape[1])
    order = tuple(range(objects)) if object_order is None else tuple(object_order)
    num_order = len(order)
    order_arr = jnp.asarray(order, dtype=jnp.int32)
    target_idx_j = jnp.asarray(target_idx)

    ti_vec = jnp.arange(ticks, dtype=jnp.int32)

    def hits_one_tick(ti: jnp.ndarray):
        return _all_planet_tick_hits(
            origin_xy,
            origin_radius,
            speed,
            ti,
            object_p0_by_tick[ti],
            object_p1_by_tick[ti],
            object_radii,
            object_active_by_tick[ti],
            samples_per_span=samples_per_span,
        )

    all_hits_lo, all_hits_hi, all_hits_valid = jax.vmap(hits_one_tick)(ti_vec)

    block_lo, block_hi, block_valid = _empty(max_block_intervals, origin_xy.dtype)
    valid_lo, valid_hi, valid_valid = _empty(max_valid_intervals, origin_xy.dtype)
    block_count = jnp.asarray(0, dtype=jnp.int32)
    valid_count = jnp.asarray(0, dtype=jnp.int32)
    overflow = jnp.asarray(False)

    def outer_body(tick_i, carry):
        block_lo, block_hi, block_valid, block_count, valid_lo, valid_hi, valid_valid, valid_count, overflow = (
            carry
        )

        tick = jnp.asarray(tick_i, dtype=jnp.int32)
        all_lo = all_hits_lo[tick_i]
        all_hi = all_hits_hi[tick_i]
        all_valid = all_hits_valid[tick_i]

        def inner_body(k, inner):
            (
                block_lo,
                block_hi,
                block_valid,
                block_count,
                valid_lo,
                valid_hi,
                valid_valid,
                valid_count,
                overflow,
            ) = inner
            obj_idx = order_arr[k]
            hit_lo = all_lo[obj_idx]
            hit_hi = all_hi[obj_idx]
            hit_valid = all_valid[obj_idx]
            avail_lo, avail_hi, avail_valid = _set_subtract_cells(
                hit_lo, hit_hi, hit_valid, block_lo, block_hi, block_valid
            )
            is_target = target_idx_j == obj_idx
            valid_lo, valid_hi, valid_valid, valid_count, valid_overflow = _append(
                valid_lo,
                valid_hi,
                valid_valid,
                valid_count,
                avail_lo,
                avail_hi,
                avail_valid & is_target,
            )
            block_lo, block_hi, block_valid, block_count, block_overflow = _append(
                block_lo, block_hi, block_valid, block_count, hit_lo, hit_hi, hit_valid
            )
            overflow = overflow | valid_overflow | block_overflow
            return (
                block_lo,
                block_hi,
                block_valid,
                block_count,
                valid_lo,
                valid_hi,
                valid_valid,
                valid_count,
                overflow,
            )

        inner_init = (
            block_lo,
            block_hi,
            block_valid,
            block_count,
            valid_lo,
            valid_hi,
            valid_valid,
            valid_count,
            overflow,
        )
        (
            block_lo,
            block_hi,
            block_valid,
            block_count,
            valid_lo,
            valid_hi,
            valid_valid,
            valid_count,
            overflow,
        ) = jax.lax.fori_loop(0, num_order, inner_body, inner_init)

        if include_board:
            b_lo, b_hi, b_valid = _board_exit_intervals_jax(
                origin_xy, origin_radius, speed, tick, board_size
            )
            block_lo, block_hi, block_valid, block_count, block_overflow = _append(
                block_lo, block_hi, block_valid, block_count, b_lo, b_hi, b_valid
            )
            overflow = overflow | block_overflow
        if include_sun:
            sun_xy = jnp.asarray([CENTER, CENTER], dtype=origin_xy.dtype)
            s_lo, s_hi, s_valid = tick_hit_intervals_jax(
                origin_xy,
                origin_radius,
                speed,
                tick,
                sun_xy,
                sun_xy,
                jnp.asarray(sun_radius - 1e-9, dtype=origin_xy.dtype),
                jnp.asarray(True),
                samples_per_span=samples_per_span,
            )
            block_lo, block_hi, block_valid, block_count, block_overflow = _append(
                block_lo, block_hi, block_valid, block_count, s_lo, s_hi, s_valid
            )
            overflow = overflow | block_overflow
        return (
            block_lo,
            block_hi,
            block_valid,
            block_count,
            valid_lo,
            valid_hi,
            valid_valid,
            valid_count,
            overflow,
        )

    init_carry = (
        block_lo,
        block_hi,
        block_valid,
        block_count,
        valid_lo,
        valid_hi,
        valid_valid,
        valid_count,
        overflow,
    )
    (
        block_lo,
        block_hi,
        block_valid,
        block_count,
        valid_lo,
        valid_hi,
        valid_valid,
        valid_count,
        overflow,
    ) = jax.lax.fori_loop(0, ticks, outer_body, init_carry)
    return valid_lo, valid_hi, valid_valid, overflow


def _precompute_all_tick_planet_hits_for_best_targets_jax(
    origin_xy: jnp.ndarray,
    origin_radius: jnp.ndarray,
    speed: jnp.ndarray,
    object_p0_by_tick: jnp.ndarray,
    object_p1_by_tick: jnp.ndarray,
    object_radii: jnp.ndarray,
    object_active_by_tick: jnp.ndarray,
    samples_per_span: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """``[T, P, S]`` hit interval tensors for every tick and planet (no occlusion)."""

    ticks = int(object_active_by_tick.shape[0])
    ti_vec = jnp.arange(ticks, dtype=jnp.int32)

    def hits_one_tick(ti: jnp.ndarray):
        return _all_planet_tick_hits(
            origin_xy,
            origin_radius,
            speed,
            ti,
            object_p0_by_tick[ti],
            object_p1_by_tick[ti],
            object_radii,
            object_active_by_tick[ti],
            samples_per_span=samples_per_span,
        )

    return jax.vmap(hits_one_tick)(ti_vec)


def _precompute_all_tick_planet_and_sun_hits_jax(
    origin_xy: jnp.ndarray,
    origin_radius: jnp.ndarray,
    speed: jnp.ndarray,
    object_p0_by_tick: jnp.ndarray,
    object_p1_by_tick: jnp.ndarray,
    object_radii: jnp.ndarray,
    object_active_by_tick: jnp.ndarray,
    samples_per_span: int,
    sun_radius: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """``[T, P+1, S]`` planet rows plus one sun row per tick (no occlusion)."""

    p_lo, p_hi, p_val = _precompute_all_tick_planet_hits_for_best_targets_jax(
        origin_xy,
        origin_radius,
        speed,
        object_p0_by_tick,
        object_p1_by_tick,
        object_radii,
        object_active_by_tick,
        samples_per_span,
    )
    ticks = int(object_active_by_tick.shape[0])
    dtype = origin_xy.dtype
    sun_xy = jnp.asarray([CENTER, CENTER], dtype=dtype)
    sr = jnp.asarray(sun_radius - 1e-9, dtype=dtype)

    def sun_for_tick(ti: jnp.ndarray):
        return tick_hit_intervals_jax(
            origin_xy,
            origin_radius,
            speed,
            ti,
            sun_xy,
            sun_xy,
            sr,
            jnp.asarray(True),
            samples_per_span=samples_per_span,
        )

    ti_vec = jnp.arange(ticks, dtype=jnp.int32)
    s_lo, s_hi, s_val = jax.vmap(sun_for_tick)(ti_vec)
    s_lo = s_lo[:, None, :]
    s_hi = s_hi[:, None, :]
    s_val = s_val[:, None, :]
    return (
        jnp.concatenate([p_lo, s_lo], axis=1),
        jnp.concatenate([p_hi, s_hi], axis=1),
        jnp.concatenate([p_val, s_val], axis=1),
    )


def _hit_intervals_to_bin_mask(
    lo: jnp.ndarray,
    hi: jnp.ndarray,
    valid: jnp.ndarray,
    *,
    n_bins: int,
) -> jnp.ndarray:
    """``[n_bins]`` bool: any angular bin overlaps a valid ``[lo, hi]`` elementary cell.

    Overlap, rather than center membership, keeps sub-bin-width target cones visible.
    """

    dtype = lo.dtype
    dtheta = TAU / jnp.asarray(float(n_bins), dtype=dtype)
    bin_lo = jnp.arange(n_bins, dtype=dtype) * dtheta
    bin_hi = bin_lo + dtheta
    inside = valid[:, None] & (bin_hi[None, :] >= lo[:, None]) & (bin_lo[None, :] <= hi[:, None])
    return jnp.any(inside, axis=0)


def _max_circular_true_run_length_and_start(mask: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Longest consecutive True on a ring of length ``n``; returns ``(length, start)``."""

    n = int(mask.shape[0])
    m2 = jnp.concatenate([mask, mask], axis=0)
    # Reverse scan gives the number of consecutive True values starting at each
    # position in the doubled mask; clipping to ``n`` handles the circular wrap.
    _, rev_lengths = jax.lax.scan(
        lambda acc, x: (
            jnp.where(x, acc + jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32)),
            jnp.where(x, acc + jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32)),
        ),
        jnp.asarray(0, dtype=jnp.int32),
        m2[::-1],
    )
    lengths2 = rev_lengths[::-1]
    lengths = jnp.minimum(lengths2[:n], jnp.asarray(n, dtype=jnp.int32))
    max_len = jnp.max(lengths)
    best_start = jnp.argmax(lengths)
    return max_len, best_start


def first_hit_union_scan_bins_apply_jax(
    origin_xy: jnp.ndarray,
    origin_radius: jnp.ndarray,
    speed: jnp.ndarray,
    object_p0_by_tick: jnp.ndarray,
    object_p1_by_tick: jnp.ndarray,
    object_radii: jnp.ndarray,
    object_active_by_tick: jnp.ndarray,
    *,
    samples_per_span: int,
    n_block_bins: int,
    sun_radius: float = SUN_RADIUS,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Experimental first-hit style angles via **binned** prefix-OR over ticks (no board).

    Pipeline:

    1. Precompute planet + sun hit intervals ``[T, P+1, S]``.
    2. Map each object's intervals to a fixed ``n_block_bins`` angular bitmask.
    3. Per tick, ``E_t = OR_j mask[t,j]``; ``associative_scan(bitwise_or)`` yields cumulative
       blocked union through tick ``t`` inclusive; **blocked before tick** ``t`` is the
       prior prefix (empty at ``t=0``).
    4. For each ``(t, planet)``, available bins = planet mask minus blocked-before;
       widest True run on the **discretized** circle; best over ``t`` is the reported width.

    This ignores board edges, moves sun into precompute, and approximates interval geometry
    on a uniform bin grid — **not** bit-identical with ``first_hit_best_targets_apply_jax``.
    Intended for benchmarking scan-shaped parallelism.

    Returns ``(angle[P], width[P], valid[P], overflow)`` with ``overflow`` always false.
    """

    planets = int(object_radii.shape[0])
    dtype = origin_xy.dtype
    dtheta = TAU / jnp.asarray(float(n_block_bins), dtype=dtype)

    all_lo, all_hi, all_va = _precompute_all_tick_planet_and_sun_hits_jax(
        origin_xy,
        origin_radius,
        speed,
        object_p0_by_tick,
        object_p1_by_tick,
        object_radii,
        object_active_by_tick,
        samples_per_span,
        sun_radius,
    )
    ticks = int(all_lo.shape[0])

    def hit_to_mask(row_lo: jnp.ndarray, row_hi: jnp.ndarray, row_va: jnp.ndarray) -> jnp.ndarray:
        return _hit_intervals_to_bin_mask(row_lo, row_hi, row_va, n_bins=n_block_bins)

    hit_mask_obj = jax.vmap(jax.vmap(hit_to_mask))(all_lo, all_hi, all_va)
    E_t = jnp.any(hit_mask_obj, axis=1)
    cum_or = jax.lax.associative_scan(jnp.bitwise_or, E_t, axis=0)
    zeros = jnp.zeros((n_block_bins,), dtype=jnp.bool_)
    blocked_before = jnp.concatenate([zeros[None, :], cum_or[:-1]], axis=0)

    jp = jnp.arange(planets, dtype=jnp.int32)
    jt = jnp.arange(ticks, dtype=jnp.int32)

    def one_tj(t: jnp.ndarray, j: jnp.ndarray):
        mobs = hit_mask_obj[t, j]
        bb = blocked_before[t]
        avail = mobs & (~bb)
        mlen, mstart = _max_circular_true_run_length_and_start(avail)
        width = mlen.astype(dtype) * dtheta
        angle = _norm_angle((mstart.astype(dtype) + 0.5 * mlen.astype(dtype)) * dtheta)
        ok = mlen > 0
        return width, angle, ok

    def row_t(t: jnp.ndarray):
        return jax.vmap(lambda j: one_tj(t, j))(jp)

    w_tab, ang_tab, val_tab = jax.vmap(row_t)(jt)
    best_w = jnp.max(w_tab, axis=0)
    rr = jnp.arange(planets, dtype=jnp.int32)
    best_t = jnp.argmax(w_tab, axis=0)
    best_angle = ang_tab[best_t, rr]
    best_valid = (best_w > GEOM_EPS) & val_tab[best_t, rr]
    overflow = jnp.asarray(False)
    return best_angle, best_w, best_valid, overflow


def first_hit_union_scan_bins_best_targets_apply_jax(
    origin_xy: jnp.ndarray,
    origin_radius: jnp.ndarray,
    speed: jnp.ndarray,
    object_p0_by_tick: jnp.ndarray,
    object_p1_by_tick: jnp.ndarray,
    object_radii: jnp.ndarray,
    object_active_by_tick: jnp.ndarray,
    policy_object_mask: jnp.ndarray | None = None,
    *,
    samples_per_span: int,
    n_block_bins: int,
    include_sun: bool = True,
    include_board: bool = True,
    board_size: float = BOARD_SIZE,
    sun_radius: float = SUN_RADIUS,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fast rollout-only interval-bin first-hit candidate generator.

    This is deliberately approximate: hit intervals are rasterized to angular bins,
    blockers are prefix-OR'd across ticks, and same-tick planet ordering is ignored.
    It preserves narrow analytic target cones better than sparse rays while avoiding
    the ``rays x planets`` swept collision matrix in the hot rollout path.
    """

    planets = int(object_radii.shape[0])
    dtype = origin_xy.dtype
    dtheta = TAU / jnp.asarray(float(n_block_bins), dtype=dtype)

    all_lo, all_hi, all_va = _precompute_all_tick_planet_and_sun_hits_jax(
        origin_xy,
        origin_radius,
        speed,
        object_p0_by_tick,
        object_p1_by_tick,
        object_radii,
        object_active_by_tick,
        samples_per_span,
        sun_radius if include_sun else 0.0,
    )
    if not include_sun:
        all_va = all_va.at[:, planets, :].set(False)

    ticks = int(all_lo.shape[0])

    def hit_to_mask(row_lo: jnp.ndarray, row_hi: jnp.ndarray, row_va: jnp.ndarray) -> jnp.ndarray:
        return _hit_intervals_to_bin_mask(row_lo, row_hi, row_va, n_bins=n_block_bins)

    hit_mask_obj = jax.vmap(jax.vmap(hit_to_mask))(all_lo, all_hi, all_va)

    if policy_object_mask is None:
        policy_mask = jnp.ones((planets,), dtype=jnp.bool_)
    else:
        policy_mask = policy_object_mask.astype(jnp.bool_)
    hit_mask_planets = hit_mask_obj[:, :planets, :] & policy_mask[None, :, None]
    blocker_obj = hit_mask_obj

    if include_board:
        ti_vec = jnp.arange(ticks, dtype=jnp.int32)

        def board_for_tick(ti: jnp.ndarray):
            b_lo, b_hi, b_va = _board_exit_intervals_jax(
                origin_xy, origin_radius, speed, ti, board_size
            )
            return _hit_intervals_to_bin_mask(b_lo, b_hi, b_va, n_bins=n_block_bins)

        board_mask = jax.vmap(board_for_tick)(ti_vec)
        blocker_tick = jnp.any(blocker_obj, axis=1) | board_mask
    else:
        blocker_tick = jnp.any(blocker_obj, axis=1)

    cum_or = jax.lax.associative_scan(jnp.bitwise_or, blocker_tick, axis=0)
    zeros = jnp.zeros((n_block_bins,), dtype=jnp.bool_)
    blocked_before = jnp.concatenate([zeros[None, :], cum_or[:-1]], axis=0)

    jp = jnp.arange(planets, dtype=jnp.int32)
    jt = jnp.arange(ticks, dtype=jnp.int32)

    def one_tj(t: jnp.ndarray, j: jnp.ndarray):
        avail = hit_mask_planets[t, j] & (~blocked_before[t])
        mlen, mstart = _max_circular_true_run_length_and_start(avail)
        width = mlen.astype(dtype) * dtheta
        angle = _norm_angle((mstart.astype(dtype) + 0.5 * mlen.astype(dtype)) * dtheta)
        ok = mlen > 0
        return width, angle, ok

    def row_t(t: jnp.ndarray):
        return jax.vmap(lambda j: one_tj(t, j))(jp)

    w_tab, ang_tab, val_tab = jax.vmap(row_t)(jt)
    iinf = jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
    t_scores = jnp.where(val_tab, jt[:, None], iinf)
    best_score = jnp.min(t_scores, axis=0)
    best_t = jnp.argmin(t_scores, axis=0).astype(jnp.int32)
    best_angle = ang_tab[best_t, jp]
    best_w = w_tab[best_t, jp]
    best_valid = (best_score < iinf) & (best_w > GEOM_EPS) & val_tab[best_t, jp]
    hit_tick = jnp.where(best_valid, best_t.astype(dtype), jnp.asarray(0.0, dtype=dtype))
    true_planet = jnp.where(best_valid, jp, jnp.asarray(-1, dtype=jnp.int32))
    true_tick = jnp.where(best_valid, hit_tick, jnp.asarray(500.0, dtype=dtype))
    overflow = jnp.asarray(False)
    return best_angle, best_w, best_valid, overflow, hit_tick, true_planet, true_tick


def _game_point_to_segment_distance(point: jnp.ndarray, start: jnp.ndarray, end: jnp.ndarray) -> jnp.ndarray:
    """Closest distance from ``point`` to segment ``start→end`` (same as ``jax_orbit_wars``)."""

    delta = end - start
    l2 = jnp.sum(delta * delta, axis=-1, keepdims=True)
    t = jnp.where(l2 == 0.0, 0.0, jnp.sum((point - start) * delta, axis=-1, keepdims=True) / l2)
    t = jnp.clip(t, 0.0, 1.0)
    projection = start + t * delta
    return jnp.linalg.norm(point - projection, axis=-1)


def first_hit_brute_rays_baseline_apply_jax(
    origin_xy: jnp.ndarray,
    origin_radius: jnp.ndarray,
    speed: jnp.ndarray,
    object_p0_by_tick: jnp.ndarray,
    object_p1_by_tick: jnp.ndarray,
    object_radii: jnp.ndarray,
    object_active_by_tick: jnp.ndarray,
    *,
    n_rays: int = 2048,
    n_substeps_per_tick: int = 8,
    ray_angles: jnp.ndarray | None = None,
    include_sun: bool = True,
    include_board: bool = True,
    sun_radius: float = SUN_RADIUS,
    board_size: float = BOARD_SIZE,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Brute forward check: ``n_rays`` **virtual fleets**, one per launch heading, in parallel.

    Each ray angle ``θ_k = 2π k / n_rays`` is a separate fleet with heading ``u(θ_k)``, same
    speed and rim offset as the real game. The fleets **do not block each other** (no
    fleet–fleet geometry); only planets, sun, and board affect each path independently.

    Per virtual fleet, motion matches ``jax_orbit_wars._move_fleets_and_collect_combats``:

    * Launch from ``origin + (origin_radius + 0.1) u(θ)`` (same rim offset as the env).
    * Each tick: fleet segment ``a0 → a1`` with ``a1 = a0 + speed * u`` (straight motion).
    * Planet hit: ``_swept_pair_hit`` on ``(a0,a1)`` vs ``(p0[t,j], p1[t,j])`` with planet
      radius, masked by ``object_active_by_tick``. Among planet hits, the lowest index
      ``j`` wins (``jnp.argmax`` on the hit mask), matching ``_first_true``.
    * Sun: distance from board center to the **fleet segment** is ``< sun_radius`` (not a
      moving disk in the planet list).
    * Out of bounds: endpoint ``a1`` outside ``[0, board_size]^2`` (same as the env).
    * Within one tick, precedence if multiple apply: **planet** (lowest ``j``) else
      **sun** else **board**, matching combat attribution when a planet is also hit.

    ``n_substeps_per_tick`` is ignored (kept for benchmark CLI compatibility).

    If ``ray_angles`` is provided (shape ``[n_rays]``), those headings are used instead of
    the uniform grid ``2π k / n_rays``.

    Returns ``(first_event_lex[n_rays], hit_any[n_rays])``. Lex packs
    ``tick * stride + code`` with ``code ∈ [0, P)`` planet index, ``code = P`` for sun
    (if ``include_sun``), ``code = P+1`` for board when both sun and board are on, etc.;
    ``stride = P + int(include_sun) + int(include_board)``. No hit on the horizon is
    ``2**31 - 1``.
    """

    _ = n_substeps_per_tick
    dtype = origin_xy.dtype
    ticks = int(object_active_by_tick.shape[0])
    planets = int(object_radii.shape[0])
    launch_off = origin_radius + jnp.asarray(0.1, dtype=dtype)

    if ray_angles is None:
        kvec = jnp.arange(n_rays, dtype=dtype)
        theta = kvec * (TAU / jnp.asarray(float(n_rays), dtype=dtype))
    else:
        theta = ray_angles
    u = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)

    tvec = jnp.arange(ticks, dtype=dtype)
    t3 = tvec[:, None, None]
    factor = launch_off + speed * t3
    a0 = origin_xy[None, None, :] + u[None, :, :] * factor
    a1 = a0 + speed * u[None, :, :]

    p0 = object_p0_by_tick
    p1 = object_p1_by_tick
    d0 = a0[:, :, None, :] - p0[:, None, :, :]
    dv = (a1[:, :, None, :] - a0[:, :, None, :]) - (p1[:, None, :, :] - p0[:, None, :, :])
    qa = jnp.sum(dv * dv, axis=-1)
    qb = 2.0 * jnp.sum(d0 * dv, axis=-1)
    qc = jnp.sum(d0 * d0, axis=-1) - object_radii[None, None, :] ** 2
    disc = qb * qb - 4.0 * qa * qc
    static_hit = qc <= 0.0
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    qa_safe = jnp.where(qa < jnp.asarray(1e-12, dtype=dtype), 1.0, qa)
    t1 = (-qb - sqrt_disc) / (2.0 * qa_safe)
    t2 = (-qb + sqrt_disc) / (2.0 * qa_safe)
    moving_hit = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
    qa_small = jnp.asarray(1e-12, dtype=dtype)
    hit_planet_raw = jnp.where(qa < qa_small, static_hit, moving_hit)
    act = object_active_by_tick[:, None, :]
    hit_planet = hit_planet_raw & act
    any_planet = jnp.any(hit_planet, axis=-1)
    planet_idx = jnp.argmax(hit_planet.astype(jnp.int32), axis=-1)

    sun_xy = jnp.asarray([CENTER, CENTER], dtype=dtype)
    sun_dist = _game_point_to_segment_distance(sun_xy, a0, a1)
    sun_hit = sun_dist < jnp.asarray(sun_radius, dtype=dtype)
    sun_hit_eff = sun_hit & jnp.asarray(include_sun, dtype=jnp.bool_)

    in_bounds = (
        (a1[..., 0] >= 0.0)
        & (a1[..., 0] <= jnp.asarray(board_size, dtype=dtype))
        & (a1[..., 1] >= 0.0)
        & (a1[..., 1] <= jnp.asarray(board_size, dtype=dtype))
    )
    oob = ~in_bounds & jnp.asarray(include_board, dtype=jnp.bool_)

    sun_code = planets
    board_code = planets + int(include_sun)
    stride = planets + int(include_sun) + int(include_board)
    event_code = jnp.where(
        any_planet,
        planet_idx.astype(jnp.int32),
        jnp.where(
            sun_hit_eff,
            jnp.asarray(sun_code, dtype=jnp.int32),
            jnp.where(oob, jnp.asarray(board_code, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32)),
        ),
    )
    had_event = any_planet | sun_hit_eff | oob

    t_col = jnp.arange(ticks, dtype=jnp.int32)[:, None]
    tick_lex = t_col * jnp.asarray(stride, dtype=jnp.int32) + event_code
    big = jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
    masked = jnp.where(had_event, tick_lex, big)
    first_lex = jnp.min(masked, axis=0).astype(jnp.int32)
    hit_any = first_lex < big
    return first_lex, hit_any


def first_hit_brute_rays_stream_apply_jax(
    origin_xy: jnp.ndarray,
    origin_radius: jnp.ndarray,
    speed: jnp.ndarray,
    object_p0_by_tick: jnp.ndarray,
    object_p1_by_tick: jnp.ndarray,
    object_radii: jnp.ndarray,
    object_active_by_tick: jnp.ndarray,
    policy_object_mask: jnp.ndarray | None = None,
    *,
    n_rays: int = 2048,
    ray_angles: jnp.ndarray | None = None,
    include_sun: bool = True,
    include_board: bool = True,
    sun_radius: float = SUN_RADIUS,
    board_size: float = BOARD_SIZE,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Streaming version of ``first_hit_brute_rays_baseline_apply_jax``.

    It computes one tick's ``[rays, planets]`` sweep at a time instead of
    materializing the full ``[ticks, rays, planets]`` tensor, and skips the
    expensive tick body once every ray has already found its first event.
    """

    dtype = origin_xy.dtype
    ticks = int(object_active_by_tick.shape[0])
    planets = int(object_radii.shape[0])
    launch_off = origin_radius + jnp.asarray(0.1, dtype=dtype)

    if ray_angles is None:
        kvec = jnp.arange(n_rays, dtype=dtype)
        theta = kvec * (TAU / jnp.asarray(float(n_rays), dtype=dtype))
    else:
        theta = ray_angles
    u = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)

    sun_xy = jnp.asarray([CENTER, CENTER], dtype=dtype)
    sun_r2 = jnp.asarray(sun_radius, dtype=dtype) ** 2
    board = jnp.asarray(board_size, dtype=dtype)
    qa_small = jnp.asarray(1e-12, dtype=dtype)
    sun_code = planets
    board_code = planets + int(include_sun)
    stride = planets + int(include_sun) + int(include_board)
    stride_j = jnp.asarray(stride, dtype=jnp.int32)
    big = jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)

    first0 = jnp.full((n_rays,), big, dtype=jnp.int32)
    done0 = jnp.zeros((n_rays,), dtype=jnp.bool_)
    if policy_object_mask is None:
        policy_object_mask = jnp.ones((planets,), dtype=jnp.bool_)
    else:
        policy_object_mask = policy_object_mask.astype(jnp.bool_)

    def tick_body(tick_i, carry):
        policy_first_lex, policy_done, true_first_lex, true_done = carry

        def compute_tick(carry_inner):
            policy_first_lex, policy_done, true_first_lex, true_done = carry_inner
            tick_f = jnp.asarray(tick_i, dtype=dtype)
            factor = launch_off + speed * tick_f
            a0 = origin_xy[None, :] + u * factor
            a1 = a0 + speed * u
            p0 = object_p0_by_tick[tick_i]
            p1 = object_p1_by_tick[tick_i]

            d0 = a0[:, None, :] - p0[None, :, :]
            dv = (a1[:, None, :] - a0[:, None, :]) - (p1[None, :, :] - p0[None, :, :])
            qa = jnp.sum(dv * dv, axis=-1)
            qb = 2.0 * jnp.sum(d0 * dv, axis=-1)
            qc = jnp.sum(d0 * d0, axis=-1) - object_radii[None, :] ** 2
            disc = qb * qb - 4.0 * qa * qc
            static_hit = qc <= 0.0
            sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
            qa_safe = jnp.where(qa < qa_small, 1.0, qa)
            t1 = (-qb - sqrt_disc) / (2.0 * qa_safe)
            t2 = (-qb + sqrt_disc) / (2.0 * qa_safe)
            moving_hit = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
            hit_planet_raw = jnp.where(qa < qa_small, static_hit, moving_hit)
            hit_planet_true = hit_planet_raw & object_active_by_tick[tick_i][None, :]
            any_planet_true = jnp.any(hit_planet_true, axis=-1)
            planet_idx_true = jnp.argmax(hit_planet_true.astype(jnp.int32), axis=-1)
            hit_planet_policy = hit_planet_true & policy_object_mask[None, :]
            any_planet_policy = jnp.any(hit_planet_policy, axis=-1)
            planet_idx_policy = jnp.argmax(hit_planet_policy.astype(jnp.int32), axis=-1)

            delta = a1 - a0
            l2 = jnp.sum(delta * delta, axis=-1, keepdims=True)
            proj_t = jnp.where(
                l2 == 0.0,
                0.0,
                jnp.sum((sun_xy - a0) * delta, axis=-1, keepdims=True) / l2,
            )
            proj_t = jnp.clip(proj_t, 0.0, 1.0)
            projection = a0 + proj_t * delta
            sun_dist2 = jnp.sum((sun_xy - projection) * (sun_xy - projection), axis=-1)
            sun_hit_eff = (sun_dist2 < sun_r2) & jnp.asarray(include_sun, dtype=jnp.bool_)

            in_bounds = (
                (a1[..., 0] >= 0.0)
                & (a1[..., 0] <= board)
                & (a1[..., 1] >= 0.0)
                & (a1[..., 1] <= board)
            )
            oob = (~in_bounds) & jnp.asarray(include_board, dtype=jnp.bool_)
            policy_event_code = jnp.where(
                any_planet_policy,
                planet_idx_policy.astype(jnp.int32),
                jnp.where(
                    sun_hit_eff,
                    jnp.asarray(sun_code, dtype=jnp.int32),
                    jnp.where(oob, jnp.asarray(board_code, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32)),
                ),
            )
            true_event_code = jnp.where(
                any_planet_true,
                planet_idx_true.astype(jnp.int32),
                jnp.where(
                    sun_hit_eff,
                    jnp.asarray(sun_code, dtype=jnp.int32),
                    jnp.where(oob, jnp.asarray(board_code, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32)),
                ),
            )
            had_policy_event = any_planet_policy | sun_hit_eff | oob
            had_true_event = any_planet_true | sun_hit_eff | oob
            tick = jnp.asarray(tick_i, dtype=jnp.int32)
            policy_tick_lex = tick * stride_j + policy_event_code
            true_tick_lex = tick * stride_j + true_event_code
            new_policy_event = (~policy_done) & had_policy_event
            new_true_event = (~true_done) & had_true_event
            policy_first_lex = jnp.where(new_policy_event, policy_tick_lex, policy_first_lex)
            true_first_lex = jnp.where(new_true_event, true_tick_lex, true_first_lex)
            policy_done = policy_done | had_policy_event
            true_done = true_done | had_true_event
            return policy_first_lex, policy_done, true_first_lex, true_done

        return jax.lax.cond(
            jnp.all(policy_done & true_done),
            lambda x: x,
            compute_tick,
            (policy_first_lex, policy_done, true_first_lex, true_done),
        )

    policy_first_lex, policy_done, true_first_lex, true_done = jax.lax.fori_loop(
        0, ticks, tick_body, (first0, done0, first0, done0)
    )
    return policy_first_lex, policy_done, true_first_lex, true_done


def first_hit_brute_best_targets_from_rays_apply_jax(
    origin_xy: jnp.ndarray,
    origin_radius: jnp.ndarray,
    speed: jnp.ndarray,
    object_p0_by_tick: jnp.ndarray,
    object_p1_by_tick: jnp.ndarray,
    object_radii: jnp.ndarray,
    object_active_by_tick: jnp.ndarray,
    policy_object_mask: jnp.ndarray | None = None,
    *,
    n_rays: int = 2048,
    include_sun: bool = True,
    include_board: bool = True,
    board_size: float = BOARD_SIZE,
    sun_radius: float = SUN_RADIUS,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-planet launch heading from discrete rays (same sweep as ``first_hit_brute_rays_baseline_apply_jax``).

    For each planet ``j``, keep only rays whose **first** terminal event is hitting ``j`` (same
    precedence as the baseline: planet index, then sun, then board within a tick; earliest
    tick wins). Among those rays, pick the one with the **smallest hit tick** (earliest time);
    ties use the **smallest ray index**. The reported angle is that ray's heading; width is
    one bin ``2π / n_rays``. If no ray hits ``j`` first, ``valid[j]`` is false.

    Returns ``(angle[P], width[P], valid[P], overflow, policy_hit_tick[P],
    true_hit_planet[P], true_hit_tick[P])`` with ``overflow`` always false.
    ``policy_hit_tick`` is the visible terminal-event tick for the selected ray.
    ``true_hit_planet`` / ``true_hit_tick`` keep the unmasked first planet hit for
    env bookkeeping, so a hidden future comet can receive incoming ships even
    though it was ignored for policy target selection.
    """

    dtype = origin_xy.dtype
    planets = int(object_radii.shape[0])
    first_lex, hit_any, true_first_lex, true_hit_any = first_hit_brute_rays_stream_apply_jax(
        origin_xy,
        origin_radius,
        speed,
        object_p0_by_tick,
        object_p1_by_tick,
        object_radii,
        object_active_by_tick,
        policy_object_mask,
        n_rays=n_rays,
        include_sun=include_sun,
        include_board=include_board,
        board_size=board_size,
        sun_radius=sun_radius,
    )
    stride = planets + int(include_sun) + int(include_board)
    stride_j = jnp.asarray(stride, dtype=jnp.int32)
    iinf = jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
    codes = jnp.where(hit_any, first_lex % stride_j, jnp.asarray(-1, dtype=jnp.int32))
    hit_ticks = jnp.where(hit_any, first_lex // stride_j, iinf)
    true_codes = jnp.where(true_hit_any, true_first_lex % stride_j, jnp.asarray(-1, dtype=jnp.int32))
    true_ticks = jnp.where(true_hit_any, true_first_lex // stride_j, iinf)

    kvec = jnp.arange(n_rays, dtype=dtype)
    theta = kvec * (TAU / jnp.asarray(float(n_rays), dtype=dtype))
    dtheta = TAU / jnp.asarray(float(n_rays), dtype=dtype)
    jp = jnp.arange(planets, dtype=jnp.int32)
    # One (R,P) mat over rays×planets — fuses better than vmap(one_planet) on small P.
    mask = hit_any[:, None] & (codes[:, None] == jp[None, :])
    scores_rp = jnp.where(mask, hit_ticks[:, None], iinf)
    br = jnp.argmin(scores_rp, axis=0).astype(jnp.int32)
    ok = jnp.any(mask, axis=0)
    ang = jnp.where(ok, _norm_angle(theta[br]), jnp.asarray(0.0, dtype=dtype))
    wid = jnp.where(ok, dtheta, jnp.asarray(0.0, dtype=dtype))
    tick = jnp.where(ok, hit_ticks[br].astype(dtype), jnp.asarray(0.0, dtype=dtype))
    true_planet = jnp.where(ok & (true_codes[br] < planets), true_codes[br], jnp.asarray(-1, dtype=jnp.int32))
    true_tick = jnp.where(ok & (true_codes[br] < planets), true_ticks[br].astype(dtype), jnp.asarray(500.0, dtype=dtype))
    overflow = jnp.asarray(False)
    return ang, wid, ok, overflow, tick, true_planet, true_tick


def first_hit_brute_best_targets_from_rays_chunked_apply_jax(
    origin_xy: jnp.ndarray,
    origin_radius: jnp.ndarray,
    speed: jnp.ndarray,
    object_p0_by_tick: jnp.ndarray,
    object_p1_by_tick: jnp.ndarray,
    object_radii: jnp.ndarray,
    object_active_by_tick: jnp.ndarray,
    policy_object_mask: jnp.ndarray | None = None,
    *,
    n_rays: int = 2048,
    ray_chunk_size: int = 256,
    include_sun: bool = True,
    include_board: bool = True,
    board_size: float = BOARD_SIZE,
    sun_radius: float = SUN_RADIUS,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Chunked version of ``first_hit_brute_best_targets_from_rays_apply_jax``.

    The result matches the full ray grid semantics, but the largest temporary
    raycasting tensors use ``ray_chunk_size`` rays rather than ``n_rays``.
    """

    dtype = origin_xy.dtype
    planets = int(object_radii.shape[0])
    chunk = max(1, int(ray_chunk_size))
    chunks = (int(n_rays) + chunk - 1) // chunk
    stride = planets + int(include_sun) + int(include_board)
    stride_j = jnp.asarray(stride, dtype=jnp.int32)
    iinf = jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
    # Score sorts by earliest hit tick, then by global ray index for ties.
    score_inf = jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
    best_score0 = jnp.full((planets,), score_inf, dtype=jnp.int32)
    best_ray0 = jnp.zeros((planets,), dtype=jnp.int32)
    best_tick0 = jnp.full((planets,), iinf, dtype=jnp.int32)
    best_true_planet0 = jnp.full((planets,), -1, dtype=jnp.int32)
    best_true_tick0 = jnp.full((planets,), iinf, dtype=jnp.int32)
    jp = jnp.arange(planets, dtype=jnp.int32)
    dtheta = TAU / jnp.asarray(float(n_rays), dtype=dtype)

    def body(ci, carry):
        best_score, best_ray, best_tick, best_true_planet, best_true_tick = carry
        start = ci * chunk
        ray_i = start + jnp.arange(chunk, dtype=jnp.int32)
        ray_valid = ray_i < jnp.asarray(n_rays, dtype=jnp.int32)
        theta = ray_i.astype(dtype) * dtheta
        first_lex, hit_any, true_first_lex, true_hit_any = first_hit_brute_rays_stream_apply_jax(
            origin_xy,
            origin_radius,
            speed,
            object_p0_by_tick,
            object_p1_by_tick,
            object_radii,
            object_active_by_tick,
            policy_object_mask,
            n_rays=chunk,
            ray_angles=theta,
            include_sun=include_sun,
            include_board=include_board,
            board_size=board_size,
            sun_radius=sun_radius,
        )
        hit_any = hit_any & ray_valid
        codes = jnp.where(hit_any, first_lex % stride_j, jnp.asarray(-1, dtype=jnp.int32))
        hit_ticks = jnp.where(hit_any, first_lex // stride_j, iinf)
        true_codes = jnp.where(true_hit_any, true_first_lex % stride_j, jnp.asarray(-1, dtype=jnp.int32))
        true_ticks = jnp.where(true_hit_any, true_first_lex // stride_j, iinf)
        mask = hit_any[:, None] & (codes[:, None] == jp[None, :])
        # ``n_rays + 1`` is plenty for the tie-break index and keeps the score int32.
        ray_score = hit_ticks[:, None] * jnp.asarray(n_rays + 1, dtype=jnp.int32) + ray_i[:, None]
        scores_rp = jnp.where(mask, ray_score, score_inf)
        local_score = jnp.min(scores_rp, axis=0)
        local_ray_idx = jnp.argmin(scores_rp, axis=0).astype(jnp.int32)
        local_ray = ray_i[local_ray_idx]
        local_tick = hit_ticks[local_ray_idx]
        local_true_code = true_codes[local_ray_idx]
        local_true_planet = jnp.where(local_true_code < planets, local_true_code, jnp.asarray(-1, dtype=jnp.int32))
        local_true_tick = true_ticks[local_ray_idx]
        update = local_score < best_score
        return (
            jnp.where(update, local_score, best_score),
            jnp.where(update, local_ray, best_ray),
            jnp.where(update, local_tick, best_tick),
            jnp.where(update, local_true_planet, best_true_planet),
            jnp.where(update, local_true_tick, best_true_tick),
        )

    best_score, best_ray, best_tick, best_true_planet, best_true_tick = jax.lax.fori_loop(
        0, chunks, body, (best_score0, best_ray0, best_tick0, best_true_planet0, best_true_tick0)
    )
    ok = best_score < score_inf
    ang = jnp.where(ok, _norm_angle(best_ray.astype(dtype) * dtheta), jnp.asarray(0.0, dtype=dtype))
    wid = jnp.where(ok, dtheta, jnp.asarray(0.0, dtype=dtype))
    tick = jnp.where(ok, best_tick.astype(dtype), jnp.asarray(0.0, dtype=dtype))
    true_planet = jnp.where(ok, best_true_planet, jnp.asarray(-1, dtype=jnp.int32))
    true_tick = jnp.where(ok, best_true_tick.astype(dtype), jnp.asarray(500.0, dtype=dtype))
    overflow = jnp.asarray(False)
    return ang, wid, ok, overflow, tick, true_planet, true_tick


def ray_segments_by_tick_jax(
    origin_xy: jnp.ndarray,
    origin_radius: jnp.ndarray,
    speed: jnp.ndarray,
    *,
    ticks: int,
    n_rays: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``(theta[R], dtheta, a0[T,R,2], a1[T,R,2])`` for straight launch rays."""

    dtype = origin_xy.dtype
    kvec = jnp.arange(n_rays, dtype=dtype)
    theta = kvec * (TAU / jnp.asarray(float(n_rays), dtype=dtype))
    u = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    launch_off = origin_radius + jnp.asarray(0.1, dtype=dtype)
    tvec = jnp.arange(ticks, dtype=dtype)[:, None, None]
    factor = launch_off + speed * tvec
    a0 = origin_xy[None, None, :] + u[None, :, :] * factor
    a1 = a0 + speed * u[None, :, :]
    dtheta = TAU / jnp.asarray(float(n_rays), dtype=dtype)
    return theta, dtheta, a0, a1


def _swept_disk_hits_core_jax(
    a0: jnp.ndarray,
    a1: jnp.ndarray,
    p0: jnp.ndarray,
    p1: jnp.ndarray,
    radii: jnp.ndarray,
    active: jnp.ndarray,
) -> jnp.ndarray:
    """Swept segment-vs-moving-disk hit mask ``[T, R, P]``."""

    d0 = a0[:, :, None, :] - p0[:, None, :, :]
    dv = (a1[:, :, None, :] - a0[:, :, None, :]) - (p1[:, None, :, :] - p0[:, None, :, :])
    qa = jnp.sum(dv * dv, axis=-1)
    qb = 2.0 * jnp.sum(d0 * dv, axis=-1)
    qc = jnp.sum(d0 * d0, axis=-1) - radii[None, None, :] ** 2
    disc = qb * qb - 4.0 * qa * qc
    static_hit = qc <= 0.0
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    qa_small = jnp.asarray(1e-12, dtype=a0.dtype)
    qa_safe = jnp.where(qa < qa_small, 1.0, qa)
    t1 = (-qb - sqrt_disc) / (2.0 * qa_safe)
    t2 = (-qb + sqrt_disc) / (2.0 * qa_safe)
    moving_hit = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
    hit_raw = jnp.where(qa < qa_small, static_hit, moving_hit)
    return hit_raw & active[:, None, :]


def stationary_hits_by_tick_jax(
    a0: jnp.ndarray,
    a1: jnp.ndarray,
    centers: jnp.ndarray,
    radii: jnp.ndarray,
    valid: jnp.ndarray,
) -> jnp.ndarray:
    """Specialized swept hits for stationary targets.

    Solve one ray--circle intersection per ``(ray, target)`` from the launch point,
    convert first contact distance to ``tick = floor(dist / speed)``, then emit a
    one-hot ``[T, R, P]`` event tensor.
    """

    dtype = a0.dtype
    ticks = a0.shape[0]
    launch = a0[0]  # [R, 2]
    ray_delta = a1[0] - a0[0]  # [R, 2]
    speed = jnp.linalg.norm(ray_delta, axis=-1)  # [R]
    speed_safe = jnp.maximum(speed, jnp.asarray(1e-12, dtype=dtype))
    u = ray_delta / speed_safe[:, None]

    m = launch[:, None, :] - centers[None, :, :]  # [R, P, 2]
    b = 2.0 * jnp.sum(m * u[:, None, :], axis=-1)  # [R, P]
    c = jnp.sum(m * m, axis=-1) - radii[None, :] ** 2  # [R, P]
    disc = b * b - 4.0 * c
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    root_lo = (-b - sqrt_disc) * 0.5
    root_hi = (-b + sqrt_disc) * 0.5

    inside = c <= 0.0
    has_forward = root_hi >= 0.0
    s_hit = jnp.where(
        inside,
        jnp.asarray(0.0, dtype=dtype),
        jnp.where(root_lo >= 0.0, root_lo, root_hi),
    )
    hit_ok = valid[None, :] & (disc >= 0.0) & has_forward
    tick_f = jnp.floor(s_hit / speed_safe[:, None])
    tick_i = tick_f.astype(jnp.int32)
    tick_ok = (tick_i >= 0) & (tick_i < ticks)
    hit_ok = hit_ok & tick_ok

    tick_axis = jnp.arange(ticks, dtype=jnp.int32)[:, None, None]
    return hit_ok[None, :, :] & (tick_axis == tick_i[None, :, :])


def stationary_hit_codes_jax(
    a0: jnp.ndarray,
    a1: jnp.ndarray,
    centers: jnp.ndarray,
    radii: jnp.ndarray,
    valid: jnp.ndarray,
    object_idx: jnp.ndarray,
    *,
    object_count: int,
) -> jnp.ndarray:
    """Earliest hit code ``[R, P]`` for stationary targets.

    Codes are ``tick * object_count + object_id``; missing hits are ``int32 max``.
    """

    dtype = a0.dtype
    ticks = a0.shape[0]
    launch = a0[0]  # [R, 2]
    ray_delta = a1[0] - a0[0]  # [R, 2]
    speed = jnp.linalg.norm(ray_delta, axis=-1)  # [R]
    speed_safe = jnp.maximum(speed, jnp.asarray(1e-12, dtype=dtype))
    u = ray_delta / speed_safe[:, None]

    m = launch[:, None, :] - centers[None, :, :]  # [R, P, 2]
    b = 2.0 * jnp.sum(m * u[:, None, :], axis=-1)  # [R, P]
    c = jnp.sum(m * m, axis=-1) - radii[None, :] ** 2  # [R, P]
    disc = b * b - 4.0 * c
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    root_lo = (-b - sqrt_disc) * 0.5
    root_hi = (-b + sqrt_disc) * 0.5

    inside = c <= 0.0
    has_forward = root_hi >= 0.0
    s_hit = jnp.where(
        inside,
        jnp.asarray(0.0, dtype=dtype),
        jnp.where(root_lo >= 0.0, root_lo, root_hi),
    )
    tick_i = jnp.floor(s_hit / speed_safe[:, None]).astype(jnp.int32)
    hit_ok = valid[None, :] & (disc >= 0.0) & has_forward & (tick_i >= 0) & (tick_i < ticks)
    code = tick_i * jnp.asarray(object_count, dtype=jnp.int32) + object_idx[None, :]
    return jnp.where(hit_ok, code, jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32))


def rotating_hits_by_tick_jax(
    a0: jnp.ndarray,
    a1: jnp.ndarray,
    init_xy: jnp.ndarray,
    radii: jnp.ndarray,
    valid: jnp.ndarray,
    angular_velocity: jnp.ndarray,
    step_count: jnp.ndarray,
) -> jnp.ndarray:
    """Specialized swept hits for orbiting targets, projected inline from initial state."""

    delta = init_xy - CENTER
    orbital_r = jnp.linalg.norm(delta, axis=1)
    initial_angle = jnp.arctan2(delta[:, 1], delta[:, 0])
    tick = jnp.arange(a0.shape[0], dtype=jnp.float32)[:, None]
    step_f = step_count.astype(jnp.float32)
    # Mirror the env timing exactly: the state at ``step_count`` still stores the
    # position reached after the previous move, so forecast segment ``t`` is
    # ``(step_count + t - 1) -> (step_count + t)``, clamped at 0 for the first turn.
    t0 = jnp.maximum(step_f + tick - 1.0, 0.0)
    t1 = step_f + tick
    ang0 = initial_angle[None, :] + angular_velocity * t0
    ang1 = initial_angle[None, :] + angular_velocity * t1
    active = jnp.broadcast_to(valid[None, :], (a0.shape[0], valid.shape[0]))
    p0x = CENTER + orbital_r[None, :] * jnp.cos(ang0)
    p0y = CENTER + orbital_r[None, :] * jnp.sin(ang0)
    p1x = CENTER + orbital_r[None, :] * jnp.cos(ang1)
    p1y = CENTER + orbital_r[None, :] * jnp.sin(ang1)

    d0x = a0[..., 0][:, :, None] - p0x[:, None, :]
    d0y = a0[..., 1][:, :, None] - p0y[:, None, :]
    dvx = (a1[..., 0] - a0[..., 0])[:, :, None] - (p1x - p0x)[:, None, :]
    dvy = (a1[..., 1] - a0[..., 1])[:, :, None] - (p1y - p0y)[:, None, :]
    qa = dvx * dvx + dvy * dvy
    qb = 2.0 * (d0x * dvx + d0y * dvy)
    qc = d0x * d0x + d0y * d0y - radii[None, None, :] ** 2
    disc = qb * qb - 4.0 * qa * qc
    static_hit = qc <= 0.0
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    qa_small = jnp.asarray(1e-12, dtype=a0.dtype)
    qa_safe = jnp.where(qa < qa_small, 1.0, qa)
    root_lo = (-qb - sqrt_disc) / (2.0 * qa_safe)
    root_hi = (-qb + sqrt_disc) / (2.0 * qa_safe)
    moving_hit = (disc >= 0.0) & (root_hi >= 0.0) & (root_lo <= 1.0)
    hit_raw = jnp.where(qa < qa_small, static_hit, moving_hit)
    return hit_raw & active[:, None, :]


def rotating_hit_codes_jax(
    a0: jnp.ndarray,
    a1: jnp.ndarray,
    init_xy: jnp.ndarray,
    radii: jnp.ndarray,
    valid: jnp.ndarray,
    object_idx: jnp.ndarray,
    angular_velocity: jnp.ndarray,
    step_count: jnp.ndarray,
    *,
    object_count: int,
) -> jnp.ndarray:
    """Earliest hit code ``[R, P]`` for orbiting targets."""

    tick_i = jnp.arange(a0.shape[0], dtype=jnp.int32)
    tick = tick_i.astype(jnp.float32)[:, None]
    delta = init_xy - CENTER
    orbital_r = jnp.linalg.norm(delta, axis=1)
    initial_angle = jnp.arctan2(delta[:, 1], delta[:, 0])
    step_f = step_count.astype(jnp.float32)
    t0 = jnp.maximum(step_f + tick - 1.0, 0.0)
    t1 = step_f + tick
    ang0 = initial_angle[None, :] + angular_velocity * t0
    ang1 = initial_angle[None, :] + angular_velocity * t1
    active = jnp.broadcast_to(valid[None, :], (a0.shape[0], valid.shape[0]))
    p0x = CENTER + orbital_r[None, :] * jnp.cos(ang0)
    p0y = CENTER + orbital_r[None, :] * jnp.sin(ang0)
    p1x = CENTER + orbital_r[None, :] * jnp.cos(ang1)
    p1y = CENTER + orbital_r[None, :] * jnp.sin(ang1)

    d0x = a0[..., 0][:, :, None] - p0x[:, None, :]
    d0y = a0[..., 1][:, :, None] - p0y[:, None, :]
    dvx = (a1[..., 0] - a0[..., 0])[:, :, None] - (p1x - p0x)[:, None, :]
    dvy = (a1[..., 1] - a0[..., 1])[:, :, None] - (p1y - p0y)[:, None, :]
    qa = dvx * dvx + dvy * dvy
    qb = 2.0 * (d0x * dvx + d0y * dvy)
    qc = d0x * d0x + d0y * d0y - radii[None, None, :] ** 2
    disc = qb * qb - 4.0 * qa * qc
    static_hit = qc <= 0.0
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    qa_small = jnp.asarray(1e-12, dtype=a0.dtype)
    qa_safe = jnp.where(qa < qa_small, 1.0, qa)
    root_lo = (-qb - sqrt_disc) / (2.0 * qa_safe)
    root_hi = (-qb + sqrt_disc) / (2.0 * qa_safe)
    moving_hit = (disc >= 0.0) & (root_hi >= 0.0) & (root_lo <= 1.0)
    hit_ok = jnp.where(qa < qa_small, static_hit, moving_hit) & active[:, None, :]
    tick_codes = jnp.where(
        hit_ok,
        tick_i[:, None, None],
        jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32),
    )
    first_tick = jnp.min(tick_codes, axis=0)
    code = first_tick * jnp.asarray(object_count, dtype=jnp.int32) + object_idx[None, :]
    return jnp.where(first_tick < jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32), code, first_tick)


def comet_hits_by_tick_jax(
    a0: jnp.ndarray,
    a1: jnp.ndarray,
    cur_xy: jnp.ndarray,
    radii: jnp.ndarray,
    group_for_slot: jnp.ndarray,
    comet_k: jnp.ndarray,
    group_active: jnp.ndarray,
    path_idx0: jnp.ndarray,
    path_len: jnp.ndarray,
    comet_paths: jnp.ndarray,
    valid: jnp.ndarray,
) -> jnp.ndarray:
    """Specialized swept hits for already-active comet slots using direct path slices."""

    tick = jnp.arange(a0.shape[0], dtype=jnp.int32)[:, None]
    p0_idx = path_idx0[None, :] + tick
    p1_idx = p0_idx + 1
    active = (
        valid[None, :]
        & group_active[None, :]
        & (path_idx0[None, :] >= 0)
        & (p0_idx < path_len[None, :])
    )
    p0_safe = jnp.clip(p0_idx, 0, comet_paths.shape[2] - 1)
    p1_safe = jnp.clip(p1_idx, 0, comet_paths.shape[2] - 1)
    path_p0x = comet_paths[group_for_slot[None, :], comet_k[None, :], p0_safe, 0]
    path_p0y = comet_paths[group_for_slot[None, :], comet_k[None, :], p0_safe, 1]
    path_p1x = comet_paths[group_for_slot[None, :], comet_k[None, :], p1_safe, 0]
    path_p1y = comet_paths[group_for_slot[None, :], comet_k[None, :], p1_safe, 1]
    p0x = jnp.where(tick == 0, cur_xy[None, :, 0], path_p0x)
    p0y = jnp.where(tick == 0, cur_xy[None, :, 1], path_p0y)
    p1x = jnp.where(p1_idx < path_len[None, :], path_p1x, path_p0x)
    p1y = jnp.where(p1_idx < path_len[None, :], path_p1y, path_p0y)

    d0x = a0[..., 0][:, :, None] - p0x[:, None, :]
    d0y = a0[..., 1][:, :, None] - p0y[:, None, :]
    dvx = (a1[..., 0] - a0[..., 0])[:, :, None] - (p1x - p0x)[:, None, :]
    dvy = (a1[..., 1] - a0[..., 1])[:, :, None] - (p1y - p0y)[:, None, :]
    qa = dvx * dvx + dvy * dvy
    qb = 2.0 * (d0x * dvx + d0y * dvy)
    qc = d0x * d0x + d0y * d0y - radii[None, None, :] ** 2
    disc = qb * qb - 4.0 * qa * qc
    static_hit = qc <= 0.0
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    qa_small = jnp.asarray(1e-12, dtype=a0.dtype)
    qa_safe = jnp.where(qa < qa_small, 1.0, qa)
    root_lo = (-qb - sqrt_disc) / (2.0 * qa_safe)
    root_hi = (-qb + sqrt_disc) / (2.0 * qa_safe)
    moving_hit = (disc >= 0.0) & (root_hi >= 0.0) & (root_lo <= 1.0)
    hit_raw = jnp.where(qa < qa_small, static_hit, moving_hit)
    return hit_raw & active[:, None, :]


def scheduled_comet_hits_by_tick_jax(
    a0: jnp.ndarray,
    a1: jnp.ndarray,
    comet_paths: jnp.ndarray,
    comet_path_lengths: jnp.ndarray,
    step_count: jnp.ndarray,
    reserve_valid: jnp.ndarray,
    *,
    spawn_steps: tuple[int, ...],
) -> jnp.ndarray:
    """Specialized swept hits for comets derived directly from schedule/path tables.

    Slots are the stable reserve comet slots ``k=0..3``. Group ``g`` contributes to slot
    ``k`` on tick ``t`` when ``rel = step_count + t - spawn_steps[g]`` satisfies
    ``0 <= rel`` and ``rel + 1 < path_len[g, k]``. Different comet groups do not overlap
    in time, so OR over groups is valid.
    """

    dtype = a0.dtype
    ticks = a0.shape[0]
    groups = comet_paths.shape[0]
    comet_k = comet_paths.shape[1]

    tick = jnp.arange(ticks, dtype=jnp.int32)[:, None, None, None]  # [T,1,1,1]
    spawn = jnp.asarray(spawn_steps, dtype=jnp.int32)[None, :, None, None]  # [1,G,1,1]
    rel = step_count.astype(jnp.int32) + tick - spawn  # [T,G,1,1]
    lengths = comet_path_lengths[None, :, :, None]  # [1,G,K,1]
    valid = reserve_valid[None, None, :, None]  # [1,1,K,1]
    active = (valid & (rel >= 0) & ((rel + 1) < lengths))[..., 0]  # [T,G,K]

    rel0 = jnp.clip(rel[..., 0], 0, comet_paths.shape[2] - 1)  # [T,G,1]
    rel1 = jnp.clip(rel[..., 0] + 1, 0, comet_paths.shape[2] - 1)
    p0x = comet_paths[None, :, :, :, 0]
    p0y = comet_paths[None, :, :, :, 1]
    # Gather per (T,G,K)
    path0x = jnp.take_along_axis(p0x, rel0[:, :, None, :], axis=3)[..., 0]  # [T,G,K]
    path0y = jnp.take_along_axis(p0y, rel0[:, :, None, :], axis=3)[..., 0]
    path1x = jnp.take_along_axis(p0x, rel1[:, :, None, :], axis=3)[..., 0]
    path1y = jnp.take_along_axis(p0y, rel1[:, :, None, :], axis=3)[..., 0]

    ax0 = a0[..., 0][:, :, None, None]
    ay0 = a0[..., 1][:, :, None, None]
    dax = (a1[..., 0] - a0[..., 0])[:, :, None, None]
    day = (a1[..., 1] - a0[..., 1])[:, :, None, None]
    dvx = dax - (path1x[:, None, :, :] - path0x[:, None, :, :])
    dvy = day - (path1y[:, None, :, :] - path0y[:, None, :, :])
    d0x = ax0 - path0x[:, None, :, :]
    d0y = ay0 - path0y[:, None, :, :]
    qa = dvx * dvx + dvy * dvy
    qb = 2.0 * (d0x * dvx + d0y * dvy)
    qc = d0x * d0x + d0y * d0y - (jnp.asarray(1.0, dtype=dtype) ** 2)
    disc = qb * qb - 4.0 * qa * qc
    static_hit = qc <= 0.0
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    qa_small = jnp.asarray(1e-12, dtype=dtype)
    qa_safe = jnp.where(qa < qa_small, 1.0, qa)
    root_lo = (-qb - sqrt_disc) / (2.0 * qa_safe)
    root_hi = (-qb + sqrt_disc) / (2.0 * qa_safe)
    moving_hit = (disc >= 0.0) & (root_hi >= 0.0) & (root_lo <= 1.0)
    hit_raw = jnp.where(qa < qa_small, static_hit, moving_hit)
    return jnp.any(hit_raw & active[:, None, :, :], axis=2)


def scheduled_comet_hit_codes_jax(
    a0: jnp.ndarray,
    a1: jnp.ndarray,
    comet_paths: jnp.ndarray,
    comet_path_lengths: jnp.ndarray,
    step_count: jnp.ndarray,
    reserve_valid: jnp.ndarray,
    object_idx: jnp.ndarray,
    *,
    spawn_steps: tuple[int, ...],
    object_count: int,
) -> jnp.ndarray:
    """Earliest scheduled-comet hit code ``[R, K]`` for reserve comet slots."""

    dtype = a0.dtype
    ticks = a0.shape[0]
    tick_i = jnp.arange(ticks, dtype=jnp.int32)
    groups = comet_paths.shape[0]
    comet_k = comet_paths.shape[1]

    tick = tick_i[:, None, None, None]  # [T,1,1,1]
    spawn = jnp.asarray(spawn_steps, dtype=jnp.int32)[None, :, None, None]  # [1,G,1,1]
    rel = step_count.astype(jnp.int32) + tick - spawn  # [T,G,1,1]
    lengths = comet_path_lengths[None, :, :, None]  # [1,G,K,1]
    valid = reserve_valid[None, None, :, None]  # [1,1,K,1]
    active = (valid & (rel >= 0) & ((rel + 1) < lengths))[..., 0]  # [T,G,K]

    rel0 = jnp.clip(rel[..., 0], 0, comet_paths.shape[2] - 1)  # [T,G,1]
    rel1 = jnp.clip(rel[..., 0] + 1, 0, comet_paths.shape[2] - 1)
    p0x = comet_paths[None, :, :, :, 0]
    p0y = comet_paths[None, :, :, :, 1]
    path0x = jnp.take_along_axis(p0x, rel0[:, :, None, :], axis=3)[..., 0]  # [T,G,K]
    path0y = jnp.take_along_axis(p0y, rel0[:, :, None, :], axis=3)[..., 0]
    path1x = jnp.take_along_axis(p0x, rel1[:, :, None, :], axis=3)[..., 0]
    path1y = jnp.take_along_axis(p0y, rel1[:, :, None, :], axis=3)[..., 0]

    ax0 = a0[..., 0][:, :, None, None]
    ay0 = a0[..., 1][:, :, None, None]
    dax = (a1[..., 0] - a0[..., 0])[:, :, None, None]
    day = (a1[..., 1] - a0[..., 1])[:, :, None, None]
    dvx = dax - (path1x[:, None, :, :] - path0x[:, None, :, :])
    dvy = day - (path1y[:, None, :, :] - path0y[:, None, :, :])
    d0x = ax0 - path0x[:, None, :, :]
    d0y = ay0 - path0y[:, None, :, :]
    qa = dvx * dvx + dvy * dvy
    qb = 2.0 * (d0x * dvx + d0y * dvy)
    qc = d0x * d0x + d0y * d0y - (jnp.asarray(1.0, dtype=dtype) ** 2)
    disc = qb * qb - 4.0 * qa * qc
    static_hit = qc <= 0.0
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    qa_small = jnp.asarray(1e-12, dtype=dtype)
    qa_safe = jnp.where(qa < qa_small, 1.0, qa)
    root_lo = (-qb - sqrt_disc) / (2.0 * qa_safe)
    root_hi = (-qb + sqrt_disc) / (2.0 * qa_safe)
    moving_hit = (disc >= 0.0) & (root_hi >= 0.0) & (root_lo <= 1.0)
    hit_ok = jnp.where(qa < qa_small, static_hit, moving_hit) & active[:, None, :, :]
    tick_codes = jnp.where(
        hit_ok,
        tick_i[:, None, None, None],
        jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32),
    )
    first_tick = jnp.min(tick_codes, axis=(0, 2))
    code = first_tick * jnp.asarray(object_count, dtype=jnp.int32) + object_idx[None, :]
    return jnp.where(first_tick < jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32), code, first_tick)


def swept_disk_hits_from_positions_jax(
    a0: jnp.ndarray,
    a1: jnp.ndarray,
    p0: jnp.ndarray,
    p1: jnp.ndarray,
    radii: jnp.ndarray,
    active: jnp.ndarray,
) -> jnp.ndarray:
    """Compatibility wrapper for callers that already have explicit positions."""

    return _swept_disk_hits_core_jax(a0, a1, p0, p1, radii, active)


def sun_board_hits_by_tick_jax(
    a0: jnp.ndarray,
    a1: jnp.ndarray,
    *,
    include_sun: bool = True,
    include_board: bool = True,
    board_size: float = BOARD_SIZE,
    sun_radius: float = SUN_RADIUS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(sun_hit[T,R], board_hit[T,R])`` for ray segments."""

    dtype = a0.dtype
    sun_xy = jnp.asarray([CENTER, CENTER], dtype=dtype)
    sun_hit = _game_point_to_segment_distance(sun_xy, a0, a1) < jnp.asarray(sun_radius, dtype=dtype)
    sun_hit = sun_hit & jnp.asarray(include_sun, dtype=jnp.bool_)
    board = jnp.asarray(board_size, dtype=dtype)
    in_bounds = (
        (a1[..., 0] >= 0.0)
        & (a1[..., 0] <= board)
        & (a1[..., 1] >= 0.0)
        & (a1[..., 1] <= board)
    )
    board_hit = (~in_bounds) & jnp.asarray(include_board, dtype=jnp.bool_)
    return sun_hit, board_hit


def reduce_ray_planet_hits_to_targets_jax(
    object_hits: jnp.ndarray,
    hidden_object_mask: jnp.ndarray,
    *,
    n_rays: int,
    num_targets: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Reduce per-ray per-tick object hits into per-target best headings.

    ``object_hits`` may contain non-target occluders in slots ``[num_targets, ...)``.
    ``hidden_object_mask`` marks hits that env sees but policy skips. In this path that
    should only be future comet reserve slots; sun is not hidden and still occludes.
    The reduction is sequential only over ticks: once a ray sees any event on a tick,
    later ticks are ignored for that ray. Within a tick, lowest object index wins.
    """

    ticks, _, object_count = object_hits.shape
    dtype = jnp.float32
    iinf = jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
    hidden_object_mask = hidden_object_mask.astype(jnp.bool_)

    def body(tick_i, carry):
        policy_done, policy_first_object, policy_first_tick, true_done, true_first_object, true_first_tick = carry
        hits_t = object_hits[tick_i]

        true_object_hits = hits_t & (~true_done[:, None])
        any_true_object = jnp.any(true_object_hits, axis=1)
        true_object_idx = jnp.argmax(true_object_hits.astype(jnp.int32), axis=1)
        new_true = (~true_done) & any_true_object
        true_first_object = jnp.where(new_true, true_object_idx, true_first_object)
        true_first_tick = jnp.where(new_true, jnp.asarray(tick_i, dtype=jnp.int32), true_first_tick)
        true_done = true_done | new_true

        policy_effective_hits = hits_t & (~policy_done[:, None]) & (~hidden_object_mask[None, :])
        any_policy_blocker = jnp.any(policy_effective_hits, axis=1)
        policy_target_hits = policy_effective_hits[:, :num_targets]
        any_policy_target = jnp.any(policy_target_hits, axis=1)
        policy_object_idx = jnp.argmax(policy_target_hits.astype(jnp.int32), axis=1)
        new_policy = (~policy_done) & any_policy_blocker
        policy_first_object = jnp.where(new_policy & any_policy_target, policy_object_idx, policy_first_object)
        policy_first_tick = jnp.where(new_policy, jnp.asarray(tick_i, dtype=jnp.int32), policy_first_tick)
        policy_done = policy_done | new_policy
        return (
            policy_done,
            policy_first_object,
            policy_first_tick,
            true_done,
            true_first_object,
            true_first_tick,
        )

    init = (
        jnp.zeros((n_rays,), dtype=jnp.bool_),
        jnp.full((n_rays,), -1, dtype=jnp.int32),
        jnp.full((n_rays,), iinf, dtype=jnp.int32),
        jnp.zeros((n_rays,), dtype=jnp.bool_),
        jnp.full((n_rays,), -1, dtype=jnp.int32),
        jnp.full((n_rays,), iinf, dtype=jnp.int32),
    )
    (
        _policy_done,
        policy_first_object,
        policy_first_tick,
        _true_done,
        true_first_object,
        true_first_tick,
    ) = jax.lax.fori_loop(0, ticks, body, init)

    kvec = jnp.arange(n_rays, dtype=dtype)
    dtheta = TAU / jnp.asarray(float(n_rays), dtype=dtype)
    theta = kvec * dtheta
    jp = jnp.arange(num_targets, dtype=jnp.int32)
    hit_any = policy_first_object >= 0
    score_inf = jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
    ray_idx = jnp.arange(n_rays, dtype=jnp.int32)
    mask = hit_any[:, None] & (policy_first_object[:, None] == jp[None, :])
    scores = jnp.where(
        mask,
        policy_first_tick[:, None] * jnp.asarray(n_rays + 1, dtype=jnp.int32) + ray_idx[:, None],
        score_inf,
    )
    best_score = jnp.min(scores, axis=0)
    best_ray = jnp.argmin(scores, axis=0).astype(jnp.int32)
    valid = best_score < score_inf
    angle = jnp.where(valid, _norm_angle(theta[best_ray]), jnp.asarray(0.0, dtype=dtype))
    width = jnp.where(valid, dtheta, jnp.asarray(0.0, dtype=dtype))
    hit_tick = jnp.where(valid, policy_first_tick[best_ray].astype(dtype), jnp.asarray(0.0, dtype=dtype))
    true_planet = jnp.where(
        valid & (true_first_object[best_ray] < num_targets),
        true_first_object[best_ray],
        jnp.asarray(-1, dtype=jnp.int32),
    )
    true_tick = jnp.where(
        valid & (true_planet >= 0),
        true_first_tick[best_ray].astype(dtype),
        jnp.asarray(500.0, dtype=dtype),
    )
    overflow = jnp.asarray(False)
    return angle, width, valid, overflow, hit_tick, true_planet, true_tick


def reduce_ray_object_codes_to_targets_jax(
    object_codes: jnp.ndarray,
    hidden_object_mask: jnp.ndarray,
    *,
    n_rays: int,
    num_targets: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Reduce per-ray object hit codes into per-target best headings.

    ``object_codes`` is ``[R, C]`` with scores ``tick * object_count + object_id``
    and ``int32 max`` for misses.
    """

    dtype = jnp.float32
    score_inf = jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
    object_count = int(hidden_object_mask.shape[0])
    object_id = jnp.mod(object_codes, jnp.asarray(object_count, dtype=jnp.int32))
    hidden = jnp.take(hidden_object_mask.astype(jnp.bool_), jnp.clip(object_id, 0, object_count - 1))

    true_code = jnp.min(object_codes, axis=1)
    policy_codes = jnp.where(hidden | (object_codes == score_inf), score_inf, object_codes)
    policy_code = jnp.min(policy_codes, axis=1)

    policy_first_tick = policy_code // jnp.asarray(object_count, dtype=jnp.int32)
    policy_first_object = jnp.mod(policy_code, jnp.asarray(object_count, dtype=jnp.int32))
    true_first_tick = true_code // jnp.asarray(object_count, dtype=jnp.int32)
    true_first_object = jnp.mod(true_code, jnp.asarray(object_count, dtype=jnp.int32))

    kvec = jnp.arange(n_rays, dtype=dtype)
    dtheta = TAU / jnp.asarray(float(n_rays), dtype=dtype)
    theta = kvec * dtheta
    jp = jnp.arange(num_targets, dtype=jnp.int32)
    hit_any = (policy_code < score_inf) & (policy_first_object < num_targets)
    ray_idx = jnp.arange(n_rays, dtype=jnp.int32)
    mask = hit_any[:, None] & (policy_first_object[:, None] == jp[None, :])
    scores = jnp.where(
        mask,
        policy_first_tick[:, None] * jnp.asarray(n_rays + 1, dtype=jnp.int32) + ray_idx[:, None],
        score_inf,
    )
    best_score = jnp.min(scores, axis=0)
    best_ray = jnp.argmin(scores, axis=0).astype(jnp.int32)
    valid = best_score < score_inf
    angle = jnp.where(valid, _norm_angle(theta[best_ray]), jnp.asarray(0.0, dtype=dtype))
    width = jnp.where(valid, dtheta, jnp.asarray(0.0, dtype=dtype))
    hit_tick = jnp.where(valid, policy_first_tick[best_ray].astype(dtype), jnp.asarray(0.0, dtype=dtype))
    true_planet = jnp.where(
        valid & (true_code[best_ray] < score_inf) & (true_first_object[best_ray] < num_targets),
        true_first_object[best_ray],
        jnp.asarray(-1, dtype=jnp.int32),
    )
    true_tick = jnp.where(
        valid & (true_planet >= 0),
        true_first_tick[best_ray].astype(dtype),
        jnp.asarray(500.0, dtype=dtype),
    )
    overflow = jnp.asarray(False)
    return angle, width, valid, overflow, hit_tick, true_planet, true_tick


def _sweep_best_targets_from_precomputed_hits_jax(
    origin_xy: jnp.ndarray,
    origin_radius: jnp.ndarray,
    speed: jnp.ndarray,
    all_hits_lo: jnp.ndarray,
    all_hits_hi: jnp.ndarray,
    all_hits_valid: jnp.ndarray,
    *,
    object_order: Sequence[int] | None = None,
    include_board: bool = True,
    include_sun: bool = True,
    board_size: float = BOARD_SIZE,
    sun_radius: float = SUN_RADIUS,
    samples_per_span: int = 9,
    max_block_intervals: int = 32,
    same_tick_planets_parallel: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Occlusion sweep + best-angle recording given precomputed per-tick planet hits.

    When ``same_tick_planets_parallel`` is False (default), planets in ``object_order``
    are processed sequentially within each tick: each planet subtracts the blocker
    union that already includes all earlier planets that tick — same-tick ties match
    ``geometry.first_hit_angle_intervals`` / the Kaggle interpreter.

    When True, every planet in a tick subtracts only the blocker union carried in from
    **previous** ticks; widest cells and bests are computed in parallel with ``vmap``,
    then all planets' hit intervals are appended in ``object_order`` before board/sun.
    That changes angular tie semantics within a tick but removes the sequential
    ``P`` dependence for the subtract / argmax portion.
    """

    ticks = int(all_hits_lo.shape[0])
    objects = int(all_hits_lo.shape[1])
    order = tuple(range(objects)) if object_order is None else tuple(object_order)
    num_order = len(order)
    order_arr = jnp.asarray(order, dtype=jnp.int32)

    block_lo, block_hi, block_valid = _empty(max_block_intervals, origin_xy.dtype)
    block_count = jnp.asarray(0, dtype=jnp.int32)
    best_angle = jnp.zeros((objects,), dtype=origin_xy.dtype)
    best_width = jnp.zeros((objects,), dtype=origin_xy.dtype)
    overflow = jnp.asarray(False)

    def outer_body(tick_i, carry):
        block_lo, block_hi, block_valid, block_count, best_angle, best_width, overflow = carry

        tick = jnp.asarray(tick_i, dtype=jnp.int32)
        all_lo = all_hits_lo[tick_i]
        all_hi = all_hits_hi[tick_i]
        all_valid = all_hits_valid[tick_i]

        if same_tick_planets_parallel:
            bl0, bh0, bv0, bc0 = block_lo, block_hi, block_valid, block_count
            hit_lo_o = all_lo[order_arr]
            hit_hi_o = all_hi[order_arr]
            hit_va_o = all_valid[order_arr]

            def subtract_one(hl, hh, hv):
                return _set_subtract_cells(hl, hh, hv, bl0, bh0, bv0)

            avail_lo, avail_hi, avail_valid = jax.vmap(subtract_one)(hit_lo_o, hit_hi_o, hit_va_o)
            widths = jnp.where(avail_valid, avail_hi - avail_lo, -1.0)
            idx = jnp.argmax(widths, axis=-1)
            width = jnp.max(widths, axis=-1)
            row = jnp.arange(num_order, dtype=jnp.int32)
            midpoint = _norm_angle(0.5 * (avail_lo[row, idx] + avail_hi[row, idx]))
            prev_w = best_width[order_arr]
            upd = width > prev_w
            new_w = jnp.where(upd, width, prev_w)
            new_a = jnp.where(upd, midpoint, best_angle[order_arr])
            best_width = best_width.at[order_arr].set(new_w)
            best_angle = best_angle.at[order_arr].set(new_a)

            def append_slot(k, inner):
                bl, bh, bv, bc, ov = inner
                oi = order_arr[k]
                hl, hh, hv = all_lo[oi], all_hi[oi], all_valid[oi]
                bl, bh, bv, bc, bo = _append(bl, bh, bv, bc, hl, hh, hv)
                return (bl, bh, bv, bc, ov | bo)

            block_lo, block_hi, block_valid, block_count, overflow = jax.lax.fori_loop(
                0, num_order, append_slot, (bl0, bh0, bv0, bc0, overflow)
            )
        else:

            def inner_body(k, inner):
                block_lo, block_hi, block_valid, block_count, best_angle, best_width, overflow = inner
                obj_idx = order_arr[k]
                hit_lo = all_lo[obj_idx]
                hit_hi = all_hi[obj_idx]
                hit_valid = all_valid[obj_idx]
                avail_lo, avail_hi, avail_valid = _set_subtract_cells(
                    hit_lo, hit_hi, hit_valid, block_lo, block_hi, block_valid
                )
                widths = jnp.where(avail_valid, avail_hi - avail_lo, -1.0)
                idx = jnp.argmax(widths)
                width = widths[idx]
                update = width > best_width[obj_idx]
                midpoint = _norm_angle(0.5 * (avail_lo[idx] + avail_hi[idx]))
                best_angle = best_angle.at[obj_idx].set(jnp.where(update, midpoint, best_angle[obj_idx]))
                best_width = best_width.at[obj_idx].set(jnp.where(update, width, best_width[obj_idx]))

                block_lo, block_hi, block_valid, block_count, block_overflow = _append(
                    block_lo, block_hi, block_valid, block_count, hit_lo, hit_hi, hit_valid
                )
                overflow = overflow | block_overflow
                return (block_lo, block_hi, block_valid, block_count, best_angle, best_width, overflow)

            inner_init = (block_lo, block_hi, block_valid, block_count, best_angle, best_width, overflow)
            block_lo, block_hi, block_valid, block_count, best_angle, best_width, overflow = jax.lax.fori_loop(
                0, num_order, inner_body, inner_init
            )

        if include_board:
            b_lo, b_hi, b_valid = _board_exit_intervals_jax(
                origin_xy, origin_radius, speed, tick, board_size
            )
            block_lo, block_hi, block_valid, block_count, block_overflow = _append(
                block_lo, block_hi, block_valid, block_count, b_lo, b_hi, b_valid
            )
            overflow = overflow | block_overflow
        if include_sun:
            sun_xy = jnp.asarray([CENTER, CENTER], dtype=origin_xy.dtype)
            s_lo, s_hi, s_valid = tick_hit_intervals_jax(
                origin_xy,
                origin_radius,
                speed,
                tick,
                sun_xy,
                sun_xy,
                jnp.asarray(sun_radius - 1e-9, dtype=origin_xy.dtype),
                jnp.asarray(True),
                samples_per_span=samples_per_span,
            )
            block_lo, block_hi, block_valid, block_count, block_overflow = _append(
                block_lo, block_hi, block_valid, block_count, s_lo, s_hi, s_valid
            )
            overflow = overflow | block_overflow
        return (block_lo, block_hi, block_valid, block_count, best_angle, best_width, overflow)

    init_carry = (block_lo, block_hi, block_valid, block_count, best_angle, best_width, overflow)
    block_lo, block_hi, block_valid, block_count, best_angle, best_width, overflow = jax.lax.fori_loop(
        0, ticks, outer_body, init_carry
    )

    valid = best_width > GEOM_EPS
    return best_angle, best_width, valid, overflow


def first_hit_interval_best_targets_apply_jax(
    origin_xy: jnp.ndarray,
    origin_radius: jnp.ndarray,
    speed: jnp.ndarray,
    object_p0_by_tick: jnp.ndarray,
    object_p1_by_tick: jnp.ndarray,
    object_radii: jnp.ndarray,
    object_active_by_tick: jnp.ndarray,
    *,
    object_order: Sequence[int] | None = None,
    include_board: bool = True,
    include_sun: bool = True,
    board_size: float = BOARD_SIZE,
    sun_radius: float = SUN_RADIUS,
    samples_per_span: int = 9,
    max_block_intervals: int = 32,
    same_tick_planets_parallel: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Angular-interval sweep + occlusion (legacy training geometry; for benchmarks / audits)."""

    all_hits_lo, all_hits_hi, all_hits_valid = _precompute_all_tick_planet_hits_for_best_targets_jax(
        origin_xy,
        origin_radius,
        speed,
        object_p0_by_tick,
        object_p1_by_tick,
        object_radii,
        object_active_by_tick,
        samples_per_span,
    )
    return _sweep_best_targets_from_precomputed_hits_jax(
        origin_xy,
        origin_radius,
        speed,
        all_hits_lo,
        all_hits_hi,
        all_hits_valid,
        object_order=object_order,
        include_board=include_board,
        include_sun=include_sun,
        board_size=board_size,
        sun_radius=sun_radius,
        samples_per_span=samples_per_span,
        max_block_intervals=max_block_intervals,
        same_tick_planets_parallel=same_tick_planets_parallel,
    )


def first_hit_best_targets_apply_jax(
    origin_xy: jnp.ndarray,
    origin_radius: jnp.ndarray,
    speed: jnp.ndarray,
    object_p0_by_tick: jnp.ndarray,
    object_p1_by_tick: jnp.ndarray,
    object_radii: jnp.ndarray,
    object_active_by_tick: jnp.ndarray,
    policy_object_mask: jnp.ndarray | None = None,
    *,
    object_order: Sequence[int] | None = None,
    include_board: bool = True,
    include_sun: bool = True,
    board_size: float = BOARD_SIZE,
    sun_radius: float = SUN_RADIUS,
    samples_per_span: int = 9,
    max_block_intervals: int = 32,
    same_tick_planets_parallel: bool = False,
    n_rays: int = 2048,
    ray_chunk_size: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Discrete-ray first-hit targets (``first_hit_brute_best_targets_from_rays_apply_jax``).

    Per planet: among rays whose first event is that planet, use the ray with the **earliest
    hit tick** (ties: smallest ray index); angle = that ray; width = one bin ``2π / n_rays``.
    The final return value is that selected first-hit tick.

    Prefer this inside a single outer ``jit(vmap(...))``. Interval-sweep kwargs
    ``object_order``, ``samples_per_span``, ``max_block_intervals``, and
    ``same_tick_planets_parallel`` are accepted for API compatibility but ignored.
    ``n_rays`` controls angular resolution and GPU memory (fewer rays = cheaper).
    """

    del object_order, samples_per_span, max_block_intervals, same_tick_planets_parallel
    if int(ray_chunk_size) > 0 and int(ray_chunk_size) < int(n_rays):
        return first_hit_brute_best_targets_from_rays_chunked_apply_jax(
            origin_xy,
            origin_radius,
            speed,
            object_p0_by_tick,
            object_p1_by_tick,
            object_radii,
            object_active_by_tick,
            policy_object_mask,
            n_rays=n_rays,
            ray_chunk_size=ray_chunk_size,
            include_sun=include_sun,
            include_board=include_board,
            board_size=board_size,
            sun_radius=sun_radius,
        )
    return first_hit_brute_best_targets_from_rays_apply_jax(
        origin_xy,
        origin_radius,
        speed,
        object_p0_by_tick,
        object_p1_by_tick,
        object_radii,
        object_active_by_tick,
        policy_object_mask,
        n_rays=n_rays,
        include_sun=include_sun,
        include_board=include_board,
        board_size=board_size,
        sun_radius=sun_radius,
    )


@partial(
    jax.jit,
    static_argnames=(
        "object_order",
        "include_board",
        "include_sun",
        "board_size",
        "sun_radius",
        "samples_per_span",
        "max_block_intervals",
        "same_tick_planets_parallel",
        "n_rays",
        "ray_chunk_size",
    ),
)
def first_hit_best_targets_jax(
    origin_xy: jnp.ndarray,
    origin_radius: jnp.ndarray,
    speed: jnp.ndarray,
    object_p0_by_tick: jnp.ndarray,
    object_p1_by_tick: jnp.ndarray,
    object_radii: jnp.ndarray,
    object_active_by_tick: jnp.ndarray,
    *,
    object_order: Sequence[int] | None = None,
    include_board: bool = True,
    include_sun: bool = True,
    board_size: float = BOARD_SIZE,
    sun_radius: float = SUN_RADIUS,
    samples_per_span: int = 9,
    max_block_intervals: int = 32,
    same_tick_planets_parallel: bool = False,
    n_rays: int = 2048,
    ray_chunk_size: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Discrete-ray first-hit targets (delegates to ``first_hit_best_targets_apply_jax``).

    Returns ``(angle[P], width[P], valid[P], overflow, hit_tick[P])`` — per planet, ray with earliest
    first-hit tick among headings that hit that planet first (game swept segments).
    Interval-sweep kwargs are ignored but kept so this ``@jit`` entry point stays drop-in compatible.
    """

    return first_hit_best_targets_apply_jax(
        origin_xy,
        origin_radius,
        speed,
        object_p0_by_tick,
        object_p1_by_tick,
        object_radii,
        object_active_by_tick,
        object_order=object_order,
        include_board=include_board,
        include_sun=include_sun,
        board_size=board_size,
        sun_radius=sun_radius,
        samples_per_span=samples_per_span,
        max_block_intervals=max_block_intervals,
        same_tick_planets_parallel=same_tick_planets_parallel,
        n_rays=n_rays,
        ray_chunk_size=ray_chunk_size,
    )


batched_first_hit_intervals_jax = jax.vmap(
    first_hit_intervals_jax,
    in_axes=(0, 0, 0, 0, 0, 0, 0, 0),
)

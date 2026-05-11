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
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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

    ticks = object_active_by_tick.shape[0]
    objects = object_active_by_tick.shape[1]
    order = tuple(range(objects)) if object_order is None else tuple(object_order)
    target_idx_j = jnp.asarray(target_idx)

    block_lo, block_hi, block_valid = _empty(max_block_intervals, origin_xy.dtype)
    valid_lo, valid_hi, valid_valid = _empty(max_valid_intervals, origin_xy.dtype)
    block_count = jnp.asarray(0, dtype=jnp.int32)
    valid_count = jnp.asarray(0, dtype=jnp.int32)
    overflow = jnp.asarray(False)

    for tick_i in range(ticks):
        for obj_idx in order:
            hit_lo, hit_hi, hit_valid = tick_hit_intervals_jax(
                origin_xy,
                origin_radius,
                speed,
                jnp.asarray(tick_i, dtype=jnp.int32),
                object_p0_by_tick[tick_i, obj_idx],
                object_p1_by_tick[tick_i, obj_idx],
                object_radii[obj_idx],
                object_active_by_tick[tick_i, obj_idx],
                samples_per_span=samples_per_span,
            )
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
        if include_board:
            b_lo, b_hi, b_valid = _board_exit_intervals_jax(
                origin_xy, origin_radius, speed, jnp.asarray(tick_i, dtype=jnp.int32), board_size
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
                jnp.asarray(tick_i, dtype=jnp.int32),
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
    return valid_lo, valid_hi, valid_valid, overflow


batched_first_hit_intervals_jax = jax.vmap(
    first_hit_intervals_jax,
    in_axes=(0, 0, 0, 0, 0, 0, 0, 0),
)

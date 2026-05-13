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
    """``[n_bins]`` bool: any bin center lies in a valid ``[lo, hi]`` elementary cell."""

    dtype = lo.dtype
    centers = (jnp.arange(n_bins, dtype=dtype) + 0.5) * (TAU / jnp.asarray(float(n_bins), dtype=dtype))
    inside = valid[:, None] & (centers[None, :] >= lo[:, None]) & (centers[None, :] <= hi[:, None])
    return jnp.any(inside, axis=0)


def _max_circular_true_run_length_and_start(mask: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Longest consecutive True on a ring of length ``n``; returns ``(length, start)``."""

    n = int(mask.shape[0])
    m2 = jnp.concatenate([mask, mask], axis=0)
    idx = jnp.arange(n, dtype=jnp.int32)

    def length_from_start(s):
        window = jax.lax.dynamic_slice(m2, (s,), (n,))
        all_true = jnp.all(window)
        first_false = jnp.argmax(~window)
        return jnp.where(all_true, jnp.asarray(n, dtype=jnp.int32), first_false)

    lengths = jax.vmap(length_from_start)(idx)
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
    *,
    n_rays: int = 2048,
    ray_angles: jnp.ndarray | None = None,
    include_sun: bool = True,
    include_board: bool = True,
    sun_radius: float = SUN_RADIUS,
    board_size: float = BOARD_SIZE,
) -> tuple[jnp.ndarray, jnp.ndarray]:
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

    def tick_body(tick_i, carry):
        first_lex, done = carry

        def compute_tick(carry_inner):
            first_lex, done = carry_inner
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
            hit_planet = hit_planet_raw & object_active_by_tick[tick_i][None, :]
            any_planet = jnp.any(hit_planet, axis=-1)
            planet_idx = jnp.argmax(hit_planet.astype(jnp.int32), axis=-1)

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
            tick_lex = jnp.asarray(tick_i, dtype=jnp.int32) * stride_j + event_code
            new_event = (~done) & had_event
            first_lex = jnp.where(new_event, tick_lex, first_lex)
            done = done | had_event
            return first_lex, done

        return jax.lax.cond(jnp.all(done), lambda x: x, compute_tick, (first_lex, done))

    first_lex, done = jax.lax.fori_loop(0, ticks, tick_body, (first0, done0))
    return first_lex, done


def first_hit_brute_best_targets_from_rays_apply_jax(
    origin_xy: jnp.ndarray,
    origin_radius: jnp.ndarray,
    speed: jnp.ndarray,
    object_p0_by_tick: jnp.ndarray,
    object_p1_by_tick: jnp.ndarray,
    object_radii: jnp.ndarray,
    object_active_by_tick: jnp.ndarray,
    *,
    n_rays: int = 2048,
    include_sun: bool = True,
    include_board: bool = True,
    board_size: float = BOARD_SIZE,
    sun_radius: float = SUN_RADIUS,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-planet launch heading from discrete rays (same sweep as ``first_hit_brute_rays_baseline_apply_jax``).

    For each planet ``j``, keep only rays whose **first** terminal event is hitting ``j`` (same
    precedence as the baseline: planet index, then sun, then board within a tick; earliest
    tick wins). Among those rays, pick the one with the **smallest hit tick** (earliest time);
    ties use the **smallest ray index**. The reported angle is that ray's heading; width is
    one bin ``2π / n_rays``. If no ray hits ``j`` first, ``valid[j]`` is false.

    Returns ``(angle[P], width[P], valid[P], overflow, hit_tick[P])`` with
    ``overflow`` always false. ``hit_tick`` is the first terminal-event tick for
    the selected ray.
    """

    dtype = origin_xy.dtype
    planets = int(object_radii.shape[0])
    first_lex, hit_any = first_hit_brute_rays_stream_apply_jax(
        origin_xy,
        origin_radius,
        speed,
        object_p0_by_tick,
        object_p1_by_tick,
        object_radii,
        object_active_by_tick,
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
    overflow = jnp.asarray(False)
    return ang, wid, ok, overflow, tick


def first_hit_brute_best_targets_from_rays_chunked_apply_jax(
    origin_xy: jnp.ndarray,
    origin_radius: jnp.ndarray,
    speed: jnp.ndarray,
    object_p0_by_tick: jnp.ndarray,
    object_p1_by_tick: jnp.ndarray,
    object_radii: jnp.ndarray,
    object_active_by_tick: jnp.ndarray,
    *,
    n_rays: int = 2048,
    ray_chunk_size: int = 256,
    include_sun: bool = True,
    include_board: bool = True,
    board_size: float = BOARD_SIZE,
    sun_radius: float = SUN_RADIUS,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    jp = jnp.arange(planets, dtype=jnp.int32)
    dtheta = TAU / jnp.asarray(float(n_rays), dtype=dtype)

    def body(ci, carry):
        best_score, best_ray, best_tick = carry
        start = ci * chunk
        ray_i = start + jnp.arange(chunk, dtype=jnp.int32)
        ray_valid = ray_i < jnp.asarray(n_rays, dtype=jnp.int32)
        theta = ray_i.astype(dtype) * dtheta
        first_lex, hit_any = first_hit_brute_rays_stream_apply_jax(
            origin_xy,
            origin_radius,
            speed,
            object_p0_by_tick,
            object_p1_by_tick,
            object_radii,
            object_active_by_tick,
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
        mask = hit_any[:, None] & (codes[:, None] == jp[None, :])
        # ``n_rays + 1`` is plenty for the tie-break index and keeps the score int32.
        ray_score = hit_ticks[:, None] * jnp.asarray(n_rays + 1, dtype=jnp.int32) + ray_i[:, None]
        scores_rp = jnp.where(mask, ray_score, score_inf)
        local_score = jnp.min(scores_rp, axis=0)
        local_ray_idx = jnp.argmin(scores_rp, axis=0).astype(jnp.int32)
        local_ray = ray_i[local_ray_idx]
        local_tick = hit_ticks[local_ray_idx]
        update = local_score < best_score
        return (
            jnp.where(update, local_score, best_score),
            jnp.where(update, local_ray, best_ray),
            jnp.where(update, local_tick, best_tick),
        )

    best_score, best_ray, best_tick = jax.lax.fori_loop(
        0, chunks, body, (best_score0, best_ray0, best_tick0)
    )
    ok = best_score < score_inf
    ang = jnp.where(ok, _norm_angle(best_ray.astype(dtype) * dtheta), jnp.asarray(0.0, dtype=dtype))
    wid = jnp.where(ok, dtheta, jnp.asarray(0.0, dtype=dtype))
    tick = jnp.where(ok, best_tick.astype(dtype), jnp.asarray(0.0, dtype=dtype))
    overflow = jnp.asarray(False)
    return ang, wid, ok, overflow, tick


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
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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

"""Stage-by-stage real-map benchmark for tangent geometry components.

This is meant to answer a very specific question:

    is the cost coming from one component, or from composing all of them?

It mirrors rollout-style inputs:

- start from batched JAX env state
- forecast future planet/comet paths with the same path-only replay used in rollout
- choose one launch origin per env from the current state
- flatten to per-target tensors with fixed shapes

Then it benchmarks each stage independently:

- forecast
- rays baseline
- target packing / scatter moving targets to ``[env, MAX_MOVING]``
- stationary tangency hits / windows / extrema (flat planet slots + mask)
- polyline tangency hits on packed moving targets
- polyline windows (first 4 tangencies → at most 2 windows per target)
- polyline extrema endpoints / sextic coeffs / sextic roots / full
- polyline root inputs / eigvals / Newton-filter

Moving-target stages use fixed ``[num_envs, MAX_MOVING, …]`` shapes instead of
``[num_envs * MAX_PLANETS, 145 windows, …]``.

The benchmark does not create ``jax.jit`` wrappers inside the timed loop.

Example:

    ./.venv/bin/python -m orbit_wars_pt.bench_tangent_components_real --platform cuda --num-envs 64
"""

from __future__ import annotations

import argparse
import os
import time
from functools import partial

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--platform", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--n-rays", type=int, default=256)
    p.add_argument("--ray-chunk-size", type=int, default=64)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-fleets", type=int, default=128)
    p.add_argument("--ship-speed", type=float, default=6.0)
    p.add_argument("--fraction", type=float, default=0.75)
    return p.parse_args()


ARGS = parse_args()
if ARGS.platform == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"
elif ARGS.platform == "cuda":
    os.environ["JAX_PLATFORMS"] = "cuda,cpu"

import jax
import jax.numpy as jnp
from jax import lax

import jax_orbit_wars as jow
from jax_orbit_wars import PLANET_OWNER, PLANET_RADIUS, PLANET_SHIPS, PLANET_X, PLANET_Y, empty_actions
from orbit_wars_pt.batched_env import stack_initial_states
from orbit_wars_pt.env_wrapper import OrbitWarsEnvConfig
from orbit_wars_pt.constants import (
    CENTER,
    FRACTIONS,
    MAX_MOVING_TARGETS,
    MAX_POLYLINE_INTERSECTION_WINDOWS,
    MAX_POLYLINE_TANGENCY_HITS,
    ROTATION_RADIUS_LIMIT,
)
from orbit_wars_pt.geometry_jax import (
    first_hit_best_targets_apply_jax,
    intersection_windows_polyline_capped_jax,
    intersection_windows_stationary_jax,
    poly_roots_degree6_desc_batch_jax,
    poly_roots_degree6_desc_batch_impl_jax,
    sextic_stationary_root_candidates_batch_jax,
    sextic_stationary_root_candidates_from_roots_batch_jax,
    stationary_angle_sextic_coeffs_jax,
    tangent_hit_times_polyline_jax,
    tangent_hit_times_stationary_jax,
)
from orbit_wars_pt.jax_setup import configure_jax_for_training
from orbit_wars_pt.micro_jax import (
    _fleet_speed_jax,
    _forecast_planet_paths_one_tick,
    selected_origin_fraction_targets_batched,
)


TAU = 2.0 * jnp.pi
ROOT_CAP = 12
SAMPLED_EXTREMA_SAMPLES = 8


def _unwrap_near(a: jnp.ndarray, ref: jnp.ndarray) -> jnp.ndarray:
    return ref + jnp.mod(a - ref + jnp.pi, TAU) - jnp.pi


def _intersection_angles(
    q: jnp.ndarray,
    circle_radius: jnp.ndarray,
    grow_radius: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    qx, qy = q[0], q[1]
    d = jnp.sqrt(jnp.maximum(qx * qx + qy * qy, 1e-12))
    x = (grow_radius * grow_radius - circle_radius * circle_radius + d * d) / jnp.maximum(
        2.0 * grow_radius * d, 1e-12
    )
    valid = (d > 1e-6) & (grow_radius > 1e-6) & (x >= -1.0 - 1e-6) & (x <= 1.0 + 1e-6)
    x = jnp.clip(x, -1.0, 1.0)
    alpha = jnp.arctan2(qy, qx)
    beta = jnp.arccos(x)
    return alpha - beta, alpha + beta, valid


def _stationary_window_extrema(
    circle_center: jnp.ndarray,
    circle_radius: jnp.ndarray,
    grow_center: jnp.ndarray,
    grow_rate: jnp.ndarray,
    launch_offset: jnp.ndarray,
    win_lo: jnp.ndarray,
    win_hi: jnp.ndarray,
    win_valid: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    q = circle_center - grow_center
    d = jnp.linalg.norm(q)
    alpha = jnp.arctan2(q[1], q[0])
    delta = jnp.arcsin(jnp.clip(circle_radius / jnp.maximum(d, 1e-6), -1.0, 1.0))
    a_minus = alpha - delta
    a_plus = alpha + delta
    r_tan = jnp.sqrt(jnp.maximum(d * d - circle_radius * circle_radius, 0.0))
    ref = 0.5 * (_unwrap_near(a_minus, alpha) + _unwrap_near(a_plus, alpha))

    def one_window(lo, hi, valid):
        cand = jnp.zeros((6,), dtype=circle_center.dtype)
        cand_valid = jnp.zeros((6,), dtype=bool)
        t_star = jnp.where(grow_rate > 1e-6, (r_tan - launch_offset) / grow_rate, lo)

        def set_one(cand_in, valid_in, idx, angle, ok):
            cand_in = cand_in.at[idx].set(_unwrap_near(angle, ref))
            valid_in = valid_in.at[idx].set(ok)
            return cand_in, valid_in

        ok_star = valid & (t_star >= lo - 1e-6) & (t_star <= hi + 1e-6)
        cand, cand_valid = set_one(cand, cand_valid, 0, a_minus, ok_star)
        cand, cand_valid = set_one(cand, cand_valid, 1, a_plus, ok_star)

        gr0 = launch_offset + grow_rate * lo
        am0, ap0, ok0 = _intersection_angles(q, circle_radius, gr0)
        cand, cand_valid = set_one(cand, cand_valid, 2, am0, valid & ok0)
        cand, cand_valid = set_one(cand, cand_valid, 3, ap0, valid & ok0)

        gr1 = launch_offset + grow_rate * hi
        am1, ap1, ok1 = _intersection_angles(q, circle_radius, gr1)
        cand, cand_valid = set_one(cand, cand_valid, 4, am1, valid & ok1)
        cand, cand_valid = set_one(cand, cand_valid, 5, ap1, valid & ok1)

        big = jnp.asarray(1e9, dtype=circle_center.dtype)
        amin = jnp.min(jnp.where(cand_valid, cand, big))
        amax = jnp.max(jnp.where(cand_valid, cand, -big))
        return amin, amax, jnp.any(cand_valid)

    return jax.vmap(one_window)(win_lo, win_hi, win_valid)


def _vmap_windows(fn):
    """``fn(lo, hi, valid)`` over a trailing window axis."""

    def batched(win_lo, win_hi, win_valid):
        return jax.vmap(fn)(win_lo, win_hi, win_valid)

    return batched


def _polyline_window_segment_geometry(
    points: jnp.ndarray,
    lo: jnp.ndarray,
    hi: jnp.ndarray,
    valid: jnp.ndarray,
    grow_center: jnp.ndarray,
    grow_rate: jnp.ndarray,
    launch_offset: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Vectorized per-segment geometry inside one polyline window."""

    horizon = points.shape[0] - 1
    i = jnp.arange(horizon, dtype=points.dtype)
    seg_lo = jnp.maximum(lo, i)
    seg_hi = jnp.minimum(hi, i + 1.0)
    use = valid & (seg_hi > seg_lo + 1e-7)
    p0 = points[:-1]
    b = points[1:] - p0
    u_off = seg_lo - i
    q0 = p0 - grow_center + b * u_off[:, None]
    r0 = launch_offset + grow_rate * seg_lo
    u_len = seg_hi - seg_lo
    return q0, b, r0, u_len, use


def _grow_center_for_points(grow_center: jnp.ndarray, points: jnp.ndarray) -> jnp.ndarray:
    """``[E, 2]`` or ``[E, M, 2]`` → ``[E, M, 1, 1, 2]`` for packed segment geometry."""

    if grow_center.ndim == 2:
        return grow_center[:, None, None, None, :]
    return grow_center[:, :, None, None, :]


def _polyline_packed_segment_geometry(
    points: jnp.ndarray,
    win_lo: jnp.ndarray,
    win_hi: jnp.ndarray,
    win_valid: jnp.ndarray,
    grow_center: jnp.ndarray,
    grow_rate: jnp.ndarray,
    launch_offset: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """All envs × moving × windows × segments: ``[..., 2]`` / ``[..., S]`` tensors."""

    horizon = points.shape[-2] - 1
    seg_i = jnp.arange(horizon, dtype=points.dtype)
    seg_lo = jnp.maximum(win_lo[..., None], seg_i)
    seg_hi = jnp.minimum(win_hi[..., None], seg_i + 1.0)
    use = win_valid[..., None] & (seg_hi > seg_lo + 1e-7)

    p0 = points[..., :-1, :]
    b_seg = points[..., 1:, :] - p0
    gc = _grow_center_for_points(grow_center, points)
    u_off = seg_lo - seg_i
    q0 = p0[..., None, :, :] - gc + b_seg[..., None, :, :] * u_off[..., None]
    r0 = launch_offset[..., None, None] + grow_rate[..., None, None] * seg_lo
    u_len = seg_hi - seg_lo
    return q0, b_seg, r0, u_len, use


def _polyline_packed_sextic_candidates(
    points: jnp.ndarray,
    radii: jnp.ndarray,
    grow_center: jnp.ndarray,
    grow_rate: jnp.ndarray,
    launch_offset: jnp.ndarray,
    win_lo: jnp.ndarray,
    win_hi: jnp.ndarray,
    win_valid: jnp.ndarray,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """One ``eigvals`` over all ``E × M × W × segments`` rows, then per-row Newton/filter."""

    q0, b_seg, r0, u_len, use = _polyline_packed_segment_geometry(
        points, win_lo, win_hi, win_valid, grow_center, grow_rate, launch_offset
    )
    emws_shape = q0.shape
    b_emws = jnp.broadcast_to(b_seg[..., None, :, :], emws_shape)
    coeff_emws = jax.vmap(
        lambda q0_i, b_i, r0_i, rp, gr: stationary_angle_sextic_coeffs_jax(
            q0_i, b_i, rp, gr, r0_i
        )
    )(
        q0.reshape(-1, 2),
        b_emws.reshape(-1, 2),
        r0.reshape(-1),
        jnp.broadcast_to(radii[..., None, None], emws_shape[:4]).reshape(-1),
        jnp.broadcast_to(grow_rate[..., None, None], emws_shape[:4]).reshape(-1),
    ).reshape(emws_shape[:4] + (7,))
    t_ws, br_ws, rv_ws = sextic_stationary_root_candidates_batch_jax(
        coeff_emws,
        q0,
        b_emws,
        radii,
        r0,
        grow_rate,
        u_len,
    )
    return t_ws, br_ws, rv_ws, q0, b_seg, r0, u_len, use


def _polyline_packed_root_inputs(
    points: jnp.ndarray,
    radii: jnp.ndarray,
    grow_center: jnp.ndarray,
    grow_rate: jnp.ndarray,
    launch_offset: jnp.ndarray,
    win_lo: jnp.ndarray,
    win_hi: jnp.ndarray,
    win_valid: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    q0, b_seg, r0, u_len, use = _polyline_packed_segment_geometry(
        points, win_lo, win_hi, win_valid, grow_center, grow_rate, launch_offset
    )
    emws_shape = q0.shape
    b_emws = jnp.broadcast_to(b_seg[..., None, :, :], emws_shape)
    coeff_emws = jax.vmap(
        lambda q0_i, b_i, r0_i, rp, gr: stationary_angle_sextic_coeffs_jax(
            q0_i, b_i, rp, gr, r0_i
        )
    )(
        q0.reshape(-1, 2),
        b_emws.reshape(-1, 2),
        r0.reshape(-1),
        jnp.broadcast_to(radii[..., None, None], emws_shape[:4]).reshape(-1),
        jnp.broadcast_to(grow_rate[..., None, None], emws_shape[:4]).reshape(-1),
    ).reshape(emws_shape[:4] + (7,))
    return coeff_emws, q0, b_emws, b_seg, r0, u_len, use, radii


def _polyline_packed_sextic_eigvals_only(
    coeff_emws: jnp.ndarray,
    use: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    roots_ws, roots_valid_ws = poly_roots_degree6_desc_batch_jax(coeff_emws)
    metric = jnp.sum(
        jnp.where(
            use[..., None],
            jnp.where(roots_valid_ws, jnp.abs(roots_ws.real) + jnp.abs(roots_ws.imag), 0.0),
            0.0,
        )
    )
    return roots_ws, roots_valid_ws, metric


def _polyline_packed_sextic_eigvals_only_impl(
    coeff_emws: jnp.ndarray,
    use: jnp.ndarray,
    implementation: lax.linalg.EigImplementation,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    roots_ws, roots_valid_ws = poly_roots_degree6_desc_batch_impl_jax(
        coeff_emws,
        implementation=implementation,
    )
    metric = jnp.sum(
        jnp.where(
            use[..., None],
            jnp.where(roots_valid_ws, jnp.abs(roots_ws.real) + jnp.abs(roots_ws.imag), 0.0),
            0.0,
        )
    )
    return roots_ws, roots_valid_ws, metric


def _polyline_packed_sextic_newton_only(
    roots_ws: jnp.ndarray,
    roots_valid_ws: jnp.ndarray,
    q0: jnp.ndarray,
    b_emws: jnp.ndarray,
    radii: jnp.ndarray,
    r0: jnp.ndarray,
    grow_rate: jnp.ndarray,
    u_len: jnp.ndarray,
    use: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    t_ws, br_ws, rv_ws = sextic_stationary_root_candidates_from_roots_batch_jax(
        roots_ws,
        roots_valid_ws,
        q0,
        b_emws,
        radii,
        r0,
        grow_rate,
        u_len,
    )
    metric = jnp.sum(
        jnp.where(
            use[..., None],
            jnp.where(rv_ws, jnp.abs(t_ws) + jnp.abs(br_ws.astype(t_ws.dtype)), 0.0),
            0.0,
        )
    )
    return t_ws, br_ws, rv_ws, metric


def _polyline_packed_extrema_from_precomputed_roots(
    points: jnp.ndarray,
    radii: jnp.ndarray,
    grow_center: jnp.ndarray,
    grow_rate: jnp.ndarray,
    launch_offset: jnp.ndarray,
    win_lo: jnp.ndarray,
    win_hi: jnp.ndarray,
    win_valid: jnp.ndarray,
    t_ws: jnp.ndarray,
    br_ws: jnp.ndarray,
    rv_ws: jnp.ndarray,
    q0: jnp.ndarray,
    b_seg: jnp.ndarray,
    r0: jnp.ndarray,
    use: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    def per_target(
        pts,
        r,
        ox,
        sp,
        loff,
        wlo,
        whi,
        wval,
        tw,
        brw,
        rvw,
        q0w,
        b_s,
        r0w,
        usew,
    ):
        nwin = wlo.shape[0]
        b_ws = jnp.broadcast_to(b_s[None, :, :], (nwin,) + b_s.shape)
        return _polyline_target_extrema_from_segments(
            pts,
            r,
            ox,
            sp,
            loff,
            wlo,
            whi,
            wval,
            tw,
            brw,
            rvw,
            q0w,
            b_ws,
            r0w,
            usew,
        )

    return jax.vmap(jax.vmap(per_target))(
        points,
        radii,
        grow_center,
        grow_rate,
        launch_offset,
        win_lo,
        win_hi,
        win_valid,
        t_ws,
        br_ws,
        rv_ws,
        q0,
        b_seg,
        r0,
        use,
    )


def _polyline_packed_roots_with_impl(
    coeff_emws: jnp.ndarray,
    q0: jnp.ndarray,
    b_emws: jnp.ndarray,
    radii: jnp.ndarray,
    r0: jnp.ndarray,
    grow_rate: jnp.ndarray,
    u_len: jnp.ndarray,
    use: jnp.ndarray,
    implementation: lax.linalg.EigImplementation,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    roots_ws, roots_valid_ws, _ = _polyline_packed_sextic_eigvals_only_impl(
        coeff_emws, use, implementation
    )
    t_ws, br_ws, rv_ws = sextic_stationary_root_candidates_from_roots_batch_jax(
        roots_ws,
        roots_valid_ws,
        q0,
        b_emws,
        radii,
        r0,
        grow_rate,
        u_len,
    )
    metric = jnp.sum(
        jnp.where(
            use[..., None],
            jnp.where(rv_ws, jnp.abs(t_ws) + jnp.abs(br_ws.astype(t_ws.dtype)), 0.0),
            0.0,
        )
    )
    return t_ws, br_ws, rv_ws, metric


def _polyline_packed_extrema_with_impl(
    points: jnp.ndarray,
    radii: jnp.ndarray,
    grow_center: jnp.ndarray,
    grow_rate: jnp.ndarray,
    launch_offset: jnp.ndarray,
    win_lo: jnp.ndarray,
    win_hi: jnp.ndarray,
    win_valid: jnp.ndarray,
    implementation: lax.linalg.EigImplementation,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    coeff_emws, q0, b_emws, b_seg, r0, u_len, use, radii_out = _polyline_packed_root_inputs(
        points, radii, grow_center, grow_rate, launch_offset, win_lo, win_hi, win_valid
    )
    t_ws, br_ws, rv_ws, _ = _polyline_packed_roots_with_impl(
        coeff_emws, q0, b_emws, radii_out, r0, grow_rate, u_len, use, implementation
    )
    return _polyline_packed_extrema_from_precomputed_roots(
        points,
        radii_out,
        grow_center,
        grow_rate,
        launch_offset,
        win_lo,
        win_hi,
        win_valid,
        t_ws,
        br_ws,
        rv_ws,
        q0,
        b_seg,
        r0,
        use,
    )


def _polyline_target_extrema_from_segments(
    points: jnp.ndarray,
    circle_radius: jnp.ndarray,
    grow_center: jnp.ndarray,
    grow_rate: jnp.ndarray,
    launch_offset: jnp.ndarray,
    win_lo: jnp.ndarray,
    win_hi: jnp.ndarray,
    win_valid: jnp.ndarray,
    t_ws: jnp.ndarray,
    br_ws: jnp.ndarray,
    rv_ws: jnp.ndarray,
    q0_ws: jnp.ndarray,
    b_ws: jnp.ndarray,
    r0_ws: jnp.ndarray,
    use_ws: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extrema for one target given precomputed segment roots (no eig)."""

    horizon = points.shape[0] - 1
    cand_cap = 4 + horizon * ROOT_CAP

    def one_window(lo, hi, valid, q0_b, b_b, r0_b, use_b, t_s, br_s, rv_s):
        tm = 0.5 * (lo + hi)
        i_mid = jnp.clip(jnp.floor(tm).astype(jnp.int32), 0, points.shape[0] - 2)
        u_mid = tm - i_mid.astype(tm.dtype)
        c_mid = points[i_mid] + u_mid * (points[i_mid + 1] - points[i_mid])
        r_mid = launch_offset + grow_rate * tm
        am_mid, ap_mid, ok_mid = _intersection_angles(c_mid - grow_center, circle_radius, r_mid)
        ref = jnp.where(
            ok_mid,
            0.5 * (am_mid + ap_mid),
            jnp.arctan2(c_mid[1] - grow_center[1], c_mid[0] - grow_center[0]),
        )

        cand = jnp.zeros((cand_cap,), dtype=points.dtype)
        cand_valid = jnp.zeros((cand_cap,), dtype=bool)

        def set_one(cand_in, valid_in, idx, angle, ok):
            cand_in = cand_in.at[idx].set(_unwrap_near(angle, ref))
            valid_in = valid_in.at[idx].set(ok)
            return cand_in, valid_in

        def add_endpoint(cand_in, valid_in, base, t):
            i0 = jnp.clip(jnp.floor(t).astype(jnp.int32), 0, points.shape[0] - 2)
            u0 = t - i0.astype(t.dtype)
            c0 = points[i0] + u0 * (points[i0 + 1] - points[i0])
            gr0 = launch_offset + grow_rate * t
            am0, ap0, ok0 = _intersection_angles(c0 - grow_center, circle_radius, gr0)
            cand_in, valid_in = set_one(cand_in, valid_in, base, am0, valid & ok0)
            cand_in, valid_in = set_one(cand_in, valid_in, base + 1, ap0, valid & ok0)
            return cand_in, valid_in

        cand, cand_valid = add_endpoint(cand, cand_valid, 0, lo)
        cand, cand_valid = add_endpoint(cand, cand_valid, 2, hi)

        def scatter_seg(i, carry):
            cand_in, valid_in = carry
            use = use_b[i]
            q0 = q0_b[i]
            b = b_b[i]
            r0 = r0_b[i]
            u_roots = t_s[i]
            branches = br_s[i]
            root_valid = rv_s[i]
            q_roots = q0[None, :] + u_roots[:, None] * b[None, :]
            gr_roots = r0 + grow_rate * u_roots
            amr, apr, okr = jax.vmap(_intersection_angles, in_axes=(0, None, 0))(
                q_roots, circle_radius, gr_roots
            )
            raw = jnp.where(branches < 0, amr, apr)
            ang = _unwrap_near(raw, ref)
            base = 4 + i * ROOT_CAP
            idxs = base + jnp.arange(ROOT_CAP, dtype=jnp.int32)
            val = use & root_valid & okr
            cand_in = cand_in.at[idxs].set(ang)
            valid_in = valid_in.at[idxs].set(val)
            return cand_in, valid_in

        cand, cand_valid = lax.fori_loop(0, horizon, scatter_seg, (cand, cand_valid))
        big = jnp.asarray(1e9, dtype=points.dtype)
        amin = jnp.min(jnp.where(cand_valid, cand, big))
        amax = jnp.max(jnp.where(cand_valid, cand, -big))
        return amin, amax, jnp.any(cand_valid)

    return jax.vmap(one_window)(
        win_lo, win_hi, win_valid, q0_ws, b_ws, r0_ws, use_ws, t_ws, br_ws, rv_ws
    )


def _polyline_target_extrema(
    points: jnp.ndarray,
    circle_radius: jnp.ndarray,
    grow_center: jnp.ndarray,
    grow_rate: jnp.ndarray,
    launch_offset: jnp.ndarray,
    win_lo: jnp.ndarray,
    win_hi: jnp.ndarray,
    win_valid: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    q0_ws, b_ws, r0_ws, u_len_ws, use_ws = jax.vmap(
        lambda lo, hi, valid: _polyline_window_segment_geometry(
            points, lo, hi, valid, grow_center, grow_rate, launch_offset
        )
    )(win_lo, win_hi, win_valid)
    coeff_ws = jax.vmap(
        jax.vmap(
            lambda q0, b, r0: stationary_angle_sextic_coeffs_jax(
                q0, b, circle_radius, grow_rate, r0
            )
        )
    )(q0_ws, b_ws, r0_ws)
    t_ws, br_ws, rv_ws = sextic_stationary_root_candidates_batch_jax(
        coeff_ws, q0_ws, b_ws, circle_radius, r0_ws, grow_rate, u_len_ws
    )
    return _polyline_target_extrema_from_segments(
        points,
        circle_radius,
        grow_center,
        grow_rate,
        launch_offset,
        win_lo,
        win_hi,
        win_valid,
        t_ws,
        br_ws,
        rv_ws,
        q0_ws,
        b_ws,
        r0_ws,
        use_ws,
    )


def _polyline_packed_sextic_roots_only(
    points: jnp.ndarray,
    radii: jnp.ndarray,
    grow_center: jnp.ndarray,
    grow_rate: jnp.ndarray,
    launch_offset: jnp.ndarray,
    win_lo: jnp.ndarray,
    win_hi: jnp.ndarray,
    win_valid: jnp.ndarray,
) -> jnp.ndarray:
    t_ws, br_ws, rv_ws, _q0, _b, _r0, _u_len, use = _polyline_packed_sextic_candidates(
        points, radii, grow_center, grow_rate, launch_offset, win_lo, win_hi, win_valid
    )
    return jnp.sum(
        jnp.where(
            use[..., None],
            jnp.where(
                rv_ws,
                jnp.abs(t_ws) + jnp.abs(br_ws.astype(points.dtype)),
                0.0,
            ),
            0.0,
        ),
        axis=-2,
    )


def _polyline_packed_extrema(
    points: jnp.ndarray,
    radii: jnp.ndarray,
    grow_center: jnp.ndarray,
    grow_rate: jnp.ndarray,
    launch_offset: jnp.ndarray,
    win_lo: jnp.ndarray,
    win_hi: jnp.ndarray,
    win_valid: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    t_ws, br_ws, rv_ws, q0, b_seg, r0, u_len, use = _polyline_packed_sextic_candidates(
        points, radii, grow_center, grow_rate, launch_offset, win_lo, win_hi, win_valid
    )

    def per_target(
        pts,
        r,
        ox,
        sp,
        loff,
        wlo,
        whi,
        wval,
        tw,
        brw,
        rvw,
        q0w,
        b_s,
        r0w,
        usew,
    ):
        nwin = wlo.shape[0]
        b_ws = jnp.broadcast_to(b_s[None, :, :], (nwin,) + b_s.shape)
        return _polyline_target_extrema_from_segments(
            pts,
            r,
            ox,
            sp,
            loff,
            wlo,
            whi,
            wval,
            tw,
            brw,
            rvw,
            q0w,
            b_ws,
            r0w,
            usew,
        )

    return jax.vmap(jax.vmap(per_target))(
        points,
        radii,
        grow_center,
        grow_rate,
        launch_offset,
        win_lo,
        win_hi,
        win_valid,
        t_ws,
        br_ws,
        rv_ws,
        q0,
        b_seg,
        r0,
        use,
    )


def _polyline_window_extrema(
    points: jnp.ndarray,
    circle_radius: jnp.ndarray,
    grow_center: jnp.ndarray,
    grow_rate: jnp.ndarray,
    launch_offset: jnp.ndarray,
    win_lo: jnp.ndarray,
    win_hi: jnp.ndarray,
    win_valid: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return _polyline_target_extrema(
        points,
        circle_radius,
        grow_center,
        grow_rate,
        launch_offset,
        win_lo,
        win_hi,
        win_valid,
    )


def _polyline_window_extrema_endpoints_only(
    points: jnp.ndarray,
    circle_radius: jnp.ndarray,
    grow_center: jnp.ndarray,
    grow_rate: jnp.ndarray,
    launch_offset: jnp.ndarray,
    win_lo: jnp.ndarray,
    win_hi: jnp.ndarray,
    win_valid: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    def one_window(lo, hi, valid):
        tm = 0.5 * (lo + hi)
        i_mid = jnp.clip(jnp.floor(tm).astype(jnp.int32), 0, points.shape[0] - 2)
        u_mid = tm - i_mid.astype(tm.dtype)
        c_mid = points[i_mid] + u_mid * (points[i_mid + 1] - points[i_mid])
        r_mid = launch_offset + grow_rate * tm
        am_mid, ap_mid, ok_mid = _intersection_angles(c_mid - grow_center, circle_radius, r_mid)
        ref = jnp.where(
            ok_mid,
            0.5 * (am_mid + ap_mid),
            jnp.arctan2(c_mid[1] - grow_center[1], c_mid[0] - grow_center[0]),
        )

        cand = jnp.zeros((4,), dtype=points.dtype)
        cand_valid = jnp.zeros((4,), dtype=bool)

        def set_one(cand_in, valid_in, idx, angle, ok):
            cand_in = cand_in.at[idx].set(_unwrap_near(angle, ref))
            valid_in = valid_in.at[idx].set(ok)
            return cand_in, valid_in

        def add_endpoint(cand_in, valid_in, base, t):
            i0 = jnp.clip(jnp.floor(t).astype(jnp.int32), 0, points.shape[0] - 2)
            u0 = t - i0.astype(t.dtype)
            c0 = points[i0] + u0 * (points[i0 + 1] - points[i0])
            gr0 = launch_offset + grow_rate * t
            am0, ap0, ok0 = _intersection_angles(c0 - grow_center, circle_radius, gr0)
            cand_in, valid_in = set_one(cand_in, valid_in, base, am0, valid & ok0)
            cand_in, valid_in = set_one(cand_in, valid_in, base + 1, ap0, valid & ok0)
            return cand_in, valid_in

        cand, cand_valid = add_endpoint(cand, cand_valid, 0, lo)
        cand, cand_valid = add_endpoint(cand, cand_valid, 2, hi)
        big = jnp.asarray(1e9, dtype=points.dtype)
        amin = jnp.min(jnp.where(cand_valid, cand, big))
        amax = jnp.max(jnp.where(cand_valid, cand, -big))
        return amin, amax, jnp.any(cand_valid)

    return _vmap_windows(one_window)(win_lo, win_hi, win_valid)


def _polyline_window_extrema_sampled(
    points: jnp.ndarray,
    circle_radius: jnp.ndarray,
    grow_center: jnp.ndarray,
    grow_rate: jnp.ndarray,
    launch_offset: jnp.ndarray,
    win_lo: jnp.ndarray,
    win_hi: jnp.ndarray,
    win_valid: jnp.ndarray,
    *,
    n_samples: int = SAMPLED_EXTREMA_SAMPLES,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    sample_tau = jnp.linspace(0.0, 1.0, n_samples, dtype=points.dtype)

    def one_window(lo, hi, valid):
        tm = 0.5 * (lo + hi)
        i_mid = jnp.clip(jnp.floor(tm).astype(jnp.int32), 0, points.shape[0] - 2)
        u_mid = tm - i_mid.astype(tm.dtype)
        c_mid = points[i_mid] + u_mid * (points[i_mid + 1] - points[i_mid])
        r_mid = launch_offset + grow_rate * tm
        am_mid, ap_mid, ok_mid = _intersection_angles(c_mid - grow_center, circle_radius, r_mid)
        ref = jnp.where(
            ok_mid,
            0.5 * (am_mid + ap_mid),
            jnp.arctan2(c_mid[1] - grow_center[1], c_mid[0] - grow_center[0]),
        )

        ts = lo + (hi - lo) * sample_tau
        i0 = jnp.clip(jnp.floor(ts).astype(jnp.int32), 0, points.shape[0] - 2)
        u0 = ts - i0.astype(ts.dtype)
        c0 = points[i0] + u0[:, None] * (points[i0 + 1] - points[i0])
        gr0 = launch_offset + grow_rate * ts
        am0, ap0, ok0 = jax.vmap(_intersection_angles, in_axes=(0, None, 0))(
            c0 - grow_center[None, :], circle_radius, gr0
        )
        ang_l = _unwrap_near(am0, ref)
        ang_r = _unwrap_near(ap0, ref)
        cand = jnp.concatenate([ang_l, ang_r], axis=0)
        cand_valid = jnp.concatenate([valid & ok0, valid & ok0], axis=0)
        big = jnp.asarray(1e9, dtype=points.dtype)
        amin = jnp.min(jnp.where(cand_valid, cand, big))
        amax = jnp.max(jnp.where(cand_valid, cand, -big))
        return amin, amax, jnp.any(cand_valid)

    return _vmap_windows(one_window)(win_lo, win_hi, win_valid)


def _polyline_target_sextic_coeffs_only(
    points: jnp.ndarray,
    circle_radius: jnp.ndarray,
    grow_center: jnp.ndarray,
    grow_rate: jnp.ndarray,
    launch_offset: jnp.ndarray,
    win_lo: jnp.ndarray,
    win_hi: jnp.ndarray,
    win_valid: jnp.ndarray,
) -> jnp.ndarray:
    q0_ws, b_ws, r0_ws, _u_len_ws, use_ws = jax.vmap(
        lambda lo, hi, valid: _polyline_window_segment_geometry(
            points, lo, hi, valid, grow_center, grow_rate, launch_offset
        )
    )(win_lo, win_hi, win_valid)
    coeff_ws = jax.vmap(
        jax.vmap(
            lambda q0, b, r0: stationary_angle_sextic_coeffs_jax(
                q0, b, circle_radius, grow_rate, r0
            )
        )
    )(q0_ws, b_ws, r0_ws)
    return jnp.sum(jnp.where(use_ws[:, :, None], jnp.abs(coeff_ws), 0.0), axis=-2)


def _polyline_window_sextic_coeffs_only(
    points: jnp.ndarray,
    circle_radius: jnp.ndarray,
    grow_center: jnp.ndarray,
    grow_rate: jnp.ndarray,
    launch_offset: jnp.ndarray,
    win_lo: jnp.ndarray,
    win_hi: jnp.ndarray,
    win_valid: jnp.ndarray,
) -> jnp.ndarray:
    return _polyline_target_sextic_coeffs_only(
        points,
        circle_radius,
        grow_center,
        grow_rate,
        launch_offset,
        win_lo,
        win_hi,
        win_valid,
    )


def _polyline_target_sextic_roots_only(
    points: jnp.ndarray,
    circle_radius: jnp.ndarray,
    grow_center: jnp.ndarray,
    grow_rate: jnp.ndarray,
    launch_offset: jnp.ndarray,
    win_lo: jnp.ndarray,
    win_hi: jnp.ndarray,
    win_valid: jnp.ndarray,
) -> jnp.ndarray:
    q0_ws, b_ws, r0_ws, u_len_ws, use_ws = jax.vmap(
        lambda lo, hi, valid: _polyline_window_segment_geometry(
            points, lo, hi, valid, grow_center, grow_rate, launch_offset
        )
    )(win_lo, win_hi, win_valid)
    coeff_ws = jax.vmap(
        jax.vmap(
            lambda q0, b, r0: stationary_angle_sextic_coeffs_jax(
                q0, b, circle_radius, grow_rate, r0
            )
        )
    )(q0_ws, b_ws, r0_ws)
    u_roots_ws, branches_ws, root_valid_ws = sextic_stationary_root_candidates_batch_jax(
        coeff_ws, q0_ws, b_ws, circle_radius, r0_ws, grow_rate, u_len_ws
    )
    return jnp.sum(
        jnp.where(
            use_ws[:, :, None],
            jnp.where(
                root_valid_ws,
                jnp.abs(u_roots_ws)
                + jnp.abs(branches_ws.astype(points.dtype)),
                0.0,
            ),
            0.0,
        ),
        axis=1,
    )


def _polyline_window_sextic_roots_only(
    points: jnp.ndarray,
    circle_radius: jnp.ndarray,
    grow_center: jnp.ndarray,
    grow_rate: jnp.ndarray,
    launch_offset: jnp.ndarray,
    win_lo: jnp.ndarray,
    win_hi: jnp.ndarray,
    win_valid: jnp.ndarray,
) -> jnp.ndarray:
    return _polyline_target_sextic_roots_only(
        points,
        circle_radius,
        grow_center,
        grow_rate,
        launch_offset,
        win_lo,
        win_hi,
        win_valid,
    )


def _pack_moving_along_targets(
    x: jnp.ndarray,
    moving_mask: jnp.ndarray,
    max_moving: int,
) -> jnp.ndarray:
    """Gather up to ``max_moving`` moving targets per env into a fixed leading axis."""

    envs, planets = moving_mask.shape
    slot = jnp.arange(planets, dtype=jnp.int32)
    rank = jnp.where(moving_mask, slot[None, :], (planets + slot)[None, :])
    order = jnp.argsort(rank, axis=1)
    idx = order[:, :max_moving]
    if x.ndim == 2:
        return jnp.take_along_axis(x, idx, axis=1)
    idx_exp = idx[(...,) + (None,) * (x.ndim - 2)]
    return jnp.take_along_axis(x, idx_exp, axis=1)


@jax.jit
def _pack_moving_targets(
    points: jnp.ndarray,
    radii: jnp.ndarray,
    origin_xy: jnp.ndarray,
    speed: jnp.ndarray,
    launch_offset: jnp.ndarray,
    horizon: jnp.ndarray,
    moving_mask: jnp.ndarray,
    *,
    max_moving: int = MAX_MOVING_TARGETS,
) -> dict[str, jnp.ndarray]:
    packed_mask = _pack_moving_along_targets(moving_mask, moving_mask, max_moving)
    return {
        "points": _pack_moving_along_targets(points, moving_mask, max_moving),
        "radii": _pack_moving_along_targets(radii, moving_mask, max_moving),
        "origin_xy": _pack_moving_along_targets(origin_xy, moving_mask, max_moving),
        "speed": _pack_moving_along_targets(speed, moving_mask, max_moving),
        "launch_offset": _pack_moving_along_targets(launch_offset, moving_mask, max_moving),
        "horizon": _pack_moving_along_targets(horizon, moving_mask, max_moving),
        "moving_mask": packed_mask,
    }


def _choose_origin_idx(state) -> jnp.ndarray:
    owned = (
        state.planet_active
        & (state.planets[:, :, PLANET_OWNER] == 0.0)
        & (state.planets[:, :, PLANET_SHIPS] >= 1.0)
    )
    return jnp.argmax(owned.astype(jnp.int32), axis=1).astype(jnp.int32)


@partial(jax.jit, static_argnames=("horizon",))
def _forecast_batched(state, horizon: int):
    def one_env(s):
        def scan_step(carry, _):
            return _forecast_planet_paths_one_tick(carry)

        _, (p0, p1, active) = lax.scan(scan_step, s, None, length=horizon)
        return p0, p1, active

    return jax.vmap(one_env)(state)


@jax.jit
def _prepare_targets(
    p0_b: jnp.ndarray,
    p1_b: jnp.ndarray,
    active_b: jnp.ndarray,
    origin_xy: jnp.ndarray,
    speed: jnp.ndarray,
    launch_offset: jnp.ndarray,
    object_radii: jnp.ndarray,
):
    envs, ticks, planets = active_b.shape
    active_any = jnp.any(active_b, axis=1)
    seg_move = jnp.linalg.norm(p1_b - p0_b, axis=-1)
    max_seg_move = jnp.max(jnp.where(active_b, seg_move, 0.0), axis=1)
    stationary_mask = active_any & (max_seg_move <= 1e-6)
    moving_mask = active_any & (~stationary_mask)

    points = jnp.concatenate(
        [
            jnp.transpose(p0_b, (0, 2, 1, 3)),
            jnp.transpose(p1_b[:, -1:, :, :], (0, 2, 1, 3)),
        ],
        axis=2,
    )
    centers = points[:, :, 0, :]
    origin_rep = jnp.broadcast_to(origin_xy[:, None, :], (envs, planets, 2))
    speed_rep = jnp.broadcast_to(speed[:, None], (envs, planets))
    launch_rep = jnp.broadcast_to(launch_offset[:, None], (envs, planets))
    horizon_rep = jnp.full((envs, planets), float(ticks), dtype=points.dtype)

    return {
        "envs": envs,
        "planets": planets,
        "centers": centers.reshape(envs * planets, 2),
        "points": points,
        "points_flat": points.reshape(envs * planets, ticks + 1, 2),
        "radii": object_radii.reshape(envs * planets),
        "radii_ep": object_radii,
        "origin_xy": origin_rep.reshape(envs * planets, 2),
        "origin_xy_ep": origin_rep,
        "speed": speed_rep.reshape(envs * planets),
        "speed_ep": speed_rep,
        "launch_offset": launch_rep.reshape(envs * planets),
        "launch_offset_ep": launch_rep,
        "horizon": horizon_rep.reshape(envs * planets),
        "horizon_ep": horizon_rep,
        "stationary_mask": stationary_mask.reshape(envs * planets),
        "stationary_mask_ep": stationary_mask,
        "moving_mask": moving_mask.reshape(envs * planets),
        "moving_mask_ep": moving_mask,
        "active_any": active_any.reshape(envs * planets),
    }


def _vmap_moving(fn):
    return jax.vmap(jax.vmap(fn))


def _potential_moving_mask_from_state(state) -> jnp.ndarray:
    init_pos = state.initial_planets[:, :, PLANET_X : PLANET_Y + 1]
    delta = init_pos - CENTER
    orbital_r = jnp.linalg.norm(delta, axis=-1)
    rotating = (
        state.planet_active
        & state.initial_active
        & (orbital_r + state.planets[:, :, PLANET_RADIUS] < ROTATION_RADIUS_LIMIT)
    )
    comet_slots = ~state.initial_active
    return rotating | comet_slots


@partial(jax.jit, static_argnames=("horizon",))
def _forecast_packed_potential_moving_batched(state, *, horizon: int):
    moving_mask = _potential_moving_mask_from_state(state)
    envs, planets = moving_mask.shape
    slot = jnp.arange(planets, dtype=jnp.int32)
    rank = jnp.where(moving_mask, slot[None, :], (planets + slot)[None, :])
    order = jnp.argsort(rank, axis=1)
    idx = order[:, :MAX_MOVING_TARGETS]

    def one_env(s, idx_env):
        def gather_xy(xy):
            idx_exp = idx_env[:, None]
            return jnp.take_along_axis(xy, idx_exp, axis=0)

        def gather_bool(x):
            return jnp.take_along_axis(x, idx_env, axis=0)

        def scan_step(carry, _):
            s_next, (old_pos, new_pos, collision_enabled) = _forecast_planet_paths_one_tick(carry)
            return s_next, (gather_xy(old_pos), gather_xy(new_pos), gather_bool(collision_enabled))

        _, (p0, p1, active) = lax.scan(scan_step, s, None, length=horizon)
        return p0, p1, active

    p0, p1, active = jax.vmap(one_env)(state, idx)
    return {
        "idx": idx,
        "mask": jnp.take_along_axis(moving_mask, idx, axis=1),
        "p0": p0,
        "p1": p1,
        "active": active,
    }


@partial(jax.jit, static_argnames=("horizon",))
def _sampled_extrema_packedforecast_fused_from_state(state, *, horizon: int):
    origin_xy, _origin_radius, launch_offset, speed, _object_radii, _policy_mask = _launch_context_from_state(
        state
    )
    packed_forecast = _forecast_packed_potential_moving_batched(state, horizon=horizon)
    moving_mask = packed_forecast["mask"]

    stat_init_pos = state.planets[:, :, PLANET_X : PLANET_Y + 1]
    delta = state.initial_planets[:, :, PLANET_X : PLANET_Y + 1] - CENTER
    orbital_r = jnp.linalg.norm(delta, axis=-1)
    rotating = (
        state.planet_active
        & state.initial_active
        & (orbital_r + state.planets[:, :, PLANET_RADIUS] < ROTATION_RADIUS_LIMIT)
    )
    stationary_mask = state.planet_active & state.initial_active & (~rotating)
    envs, planets = stationary_mask.shape
    centers = stat_init_pos.reshape(envs * planets, 2)
    radii = state.planets[:, :, PLANET_RADIUS].reshape(envs * planets)
    origin_rep = jnp.broadcast_to(origin_xy[:, None, :], (envs, planets, 2)).reshape(envs * planets, 2)
    speed_rep = jnp.broadcast_to(speed[:, None], (envs, planets)).reshape(envs * planets)
    launch_rep = jnp.broadcast_to(launch_offset[:, None], (envs, planets)).reshape(envs * planets)
    horizon_rep = jnp.full((envs * planets,), float(horizon), dtype=state.planets.dtype)
    stat_w_lo, stat_w_hi, stat_w_valid = STAT_WINDOWS_JIT(
        centers, radii, origin_rep, speed_rep, launch_rep, horizon_rep
    )
    stat_w_valid = stat_w_valid & stationary_mask.reshape(envs * planets)[:, None]
    stat_ext = STAT_EXTREMA_JIT(
        centers, radii, origin_rep, speed_rep, launch_rep, stat_w_lo, stat_w_hi, stat_w_valid
    )

    packed_radii = jnp.take_along_axis(state.planets[:, :, PLANET_RADIUS], packed_forecast["idx"], axis=1)
    packed_origin_xy = jnp.broadcast_to(origin_xy[:, None, :], (envs, MAX_MOVING_TARGETS, 2))
    packed_speed = jnp.broadcast_to(speed[:, None], (envs, MAX_MOVING_TARGETS))
    packed_launch = jnp.broadcast_to(launch_offset[:, None], (envs, MAX_MOVING_TARGETS))
    packed_horizon = jnp.full((envs, MAX_MOVING_TARGETS), float(horizon), dtype=state.planets.dtype)

    packed_points = jnp.concatenate(
        [
            jnp.transpose(packed_forecast["p0"], (0, 2, 1, 3)),
            jnp.transpose(packed_forecast["p1"][:, -1:, :, :], (0, 2, 1, 3)),
        ],
        axis=2,
    )

    move_w_lo, move_w_hi, move_w_valid = MOVE_WINDOWS_PACKED_JIT(
        packed_points,
        packed_radii,
        packed_origin_xy,
        packed_speed,
        packed_launch,
        packed_horizon,
    )
    move_w_valid = move_w_valid & moving_mask[:, :, None]
    move_ext = MOVE_EXTREMA_SAMPLED_PACKED_JIT(
        packed_points,
        packed_radii,
        packed_origin_xy,
        packed_speed,
        packed_launch,
        move_w_lo,
        move_w_hi,
        move_w_valid,
    )
    return stat_ext, move_ext


def _sampled_extrema_packedforecast_staged_from_state(state, *, horizon: int):
    origin_xy, _origin_radius, launch_offset, speed, _object_radii, _policy_mask = _launch_context_from_state(
        state
    )
    packed_forecast = _forecast_packed_potential_moving_batched(state, horizon=horizon)
    moving_mask = packed_forecast["mask"]

    stat_init_pos = state.planets[:, :, PLANET_X : PLANET_Y + 1]
    delta = state.initial_planets[:, :, PLANET_X : PLANET_Y + 1] - CENTER
    orbital_r = jnp.linalg.norm(delta, axis=-1)
    rotating = (
        state.planet_active
        & state.initial_active
        & (orbital_r + state.planets[:, :, PLANET_RADIUS] < ROTATION_RADIUS_LIMIT)
    )
    stationary_mask = state.planet_active & state.initial_active & (~rotating)
    envs, planets = stationary_mask.shape
    centers = stat_init_pos.reshape(envs * planets, 2)
    radii = state.planets[:, :, PLANET_RADIUS].reshape(envs * planets)
    origin_rep = jnp.broadcast_to(origin_xy[:, None, :], (envs, planets, 2)).reshape(envs * planets, 2)
    speed_rep = jnp.broadcast_to(speed[:, None], (envs, planets)).reshape(envs * planets)
    launch_rep = jnp.broadcast_to(launch_offset[:, None], (envs, planets)).reshape(envs * planets)
    horizon_rep = jnp.full((envs * planets,), float(horizon), dtype=state.planets.dtype)
    stat_w_lo, stat_w_hi, stat_w_valid = STAT_WINDOWS_JIT(
        centers, radii, origin_rep, speed_rep, launch_rep, horizon_rep
    )
    stat_w_valid = stat_w_valid & stationary_mask.reshape(envs * planets)[:, None]
    stat_ext = STAT_EXTREMA_JIT(
        centers, radii, origin_rep, speed_rep, launch_rep, stat_w_lo, stat_w_hi, stat_w_valid
    )

    packed_points = jnp.concatenate(
        [
            jnp.transpose(packed_forecast["p0"], (0, 2, 1, 3)),
            jnp.transpose(packed_forecast["p1"][:, -1:, :, :], (0, 2, 1, 3)),
        ],
        axis=2,
    )
    packed_radii = jnp.take_along_axis(state.planets[:, :, PLANET_RADIUS], packed_forecast["idx"], axis=1)
    packed_origin_xy = jnp.broadcast_to(origin_xy[:, None, :], (envs, MAX_MOVING_TARGETS, 2))
    packed_speed = jnp.broadcast_to(speed[:, None], (envs, MAX_MOVING_TARGETS))
    packed_launch = jnp.broadcast_to(launch_offset[:, None], (envs, MAX_MOVING_TARGETS))
    packed_horizon = jnp.full((envs, MAX_MOVING_TARGETS), float(horizon), dtype=state.planets.dtype)
    move_w_lo, move_w_hi, move_w_valid = MOVE_WINDOWS_PACKED_JIT(
        packed_points,
        packed_radii,
        packed_origin_xy,
        packed_speed,
        packed_launch,
        packed_horizon,
    )
    move_w_valid = move_w_valid & moving_mask[:, :, None]
    move_ext = MOVE_EXTREMA_SAMPLED_PACKED_JIT(
        packed_points,
        packed_radii,
        packed_origin_xy,
        packed_speed,
        packed_launch,
        move_w_lo,
        move_w_hi,
        move_w_valid,
    )
    return stat_ext, move_ext


def _launch_context_from_state(
    state,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    origin_idx = _choose_origin_idx(state)
    ships = state.planets[jnp.arange(state.planets.shape[0]), origin_idx, PLANET_SHIPS]
    send = jnp.floor(jnp.asarray(float(ARGS.fraction), dtype=jnp.float32) * ships)
    speed = _fleet_speed_jax(send, float(ARGS.ship_speed))
    origin_xy = state.planets[jnp.arange(state.planets.shape[0]), origin_idx][:, PLANET_X : PLANET_Y + 1]
    origin_radius = state.planets[jnp.arange(state.planets.shape[0]), origin_idx, PLANET_RADIUS]
    launch_offset = origin_radius + jnp.asarray(0.1, dtype=jnp.float32)
    object_radii = state.planets[:, :, PLANET_RADIUS]
    policy_mask = state.planet_active
    return origin_xy, origin_radius, launch_offset, speed, object_radii, policy_mask


def _fraction_index_for_args() -> int:
    vals = np.asarray(FRACTIONS, dtype=np.float32)
    return int(np.argmin(np.abs(vals - np.float32(ARGS.fraction))))


@partial(jax.jit, static_argnames=("horizon", "n_rays", "ray_chunk_size"))
def _rays_fused_from_state(state, *, horizon: int, n_rays: int, ray_chunk_size: int):
    p0_b, p1_b, active_b = _forecast_batched(state, horizon)
    origin_xy, origin_radius, _launch_offset, speed, object_radii, policy_mask = _launch_context_from_state(state)

    return jax.vmap(
        lambda ox, orad, sp, p0, p1, rr, act, pm: first_hit_best_targets_apply_jax(
            ox,
            orad,
            sp,
            p0,
            p1,
            rr,
            act,
            pm,
            n_rays=n_rays,
            ray_chunk_size=ray_chunk_size,
        )
    )(origin_xy, origin_radius, speed, p0_b, p1_b, object_radii, active_b, policy_mask)


@partial(jax.jit, static_argnames=("horizon", "n_rays"))
def _category_rays_fused_from_state(state, *, horizon: int, n_rays: int):
    origin_idx = _choose_origin_idx(state)
    frac_idx = jnp.full_like(origin_idx, _fraction_index_for_args())
    return selected_origin_fraction_targets_batched(
        state,
        origin_idx,
        frac_idx,
        horizon=horizon,
        ship_speed=float(ARGS.ship_speed),
        n_rays=n_rays,
        ray_chunk_size=0,
        first_hit_method="category-rays",
    )


@partial(jax.jit, static_argnames=("horizon",))
def _sampled_extrema_fused_from_state(state, *, horizon: int):
    p0_b, p1_b, active_b = _forecast_batched(state, horizon)
    origin_xy, _origin_radius, launch_offset, speed, object_radii, _policy_mask = _launch_context_from_state(state)
    prepared = _prepare_targets(
        p0_b,
        p1_b,
        active_b,
        origin_xy,
        speed,
        launch_offset,
        object_radii,
    )
    packed = _pack_moving_targets(
        prepared["points"],
        prepared["radii_ep"],
        prepared["origin_xy_ep"],
        prepared["speed_ep"],
        prepared["launch_offset_ep"],
        prepared["horizon_ep"],
        prepared["moving_mask_ep"],
    )

    stat_w_lo, stat_w_hi, stat_w_valid = STAT_WINDOWS_JIT(
        prepared["centers"],
        prepared["radii"],
        prepared["origin_xy"],
        prepared["speed"],
        prepared["launch_offset"],
        prepared["horizon"],
    )
    move_w_lo, move_w_hi, move_w_valid = MOVE_WINDOWS_PACKED_JIT(
        packed["points"],
        packed["radii"],
        packed["origin_xy"],
        packed["speed"],
        packed["launch_offset"],
        packed["horizon"],
    )
    stat_w_valid = stat_w_valid & prepared["stationary_mask"][:, None]
    move_w_valid = move_w_valid & packed["moving_mask"][:, :, None]

    stat_ext = STAT_EXTREMA_JIT(
        prepared["centers"],
        prepared["radii"],
        prepared["origin_xy"],
        prepared["speed"],
        prepared["launch_offset"],
        stat_w_lo,
        stat_w_hi,
        stat_w_valid,
    )
    move_ext = MOVE_EXTREMA_SAMPLED_PACKED_JIT(
        packed["points"],
        packed["radii"],
        packed["origin_xy"],
        packed["speed"],
        packed["launch_offset"],
        move_w_lo,
        move_w_hi,
        move_w_valid,
    )
    return stat_ext, move_ext


def _rays_staged_from_state(state, *, horizon: int, n_rays: int, ray_chunk_size: int):
    p0_b, p1_b, active_b = _forecast_batched(state, horizon)
    origin_xy, origin_radius, _launch_offset, speed, object_radii, policy_mask = _launch_context_from_state(state)
    return jax.vmap(
        lambda ox, orad, sp, p0, p1, rr, act, pm: first_hit_best_targets_apply_jax(
            ox,
            orad,
            sp,
            p0,
            p1,
            rr,
            act,
            pm,
            n_rays=n_rays,
            ray_chunk_size=ray_chunk_size,
        )
    )(origin_xy, origin_radius, speed, p0_b, p1_b, object_radii, active_b, policy_mask)


def _category_rays_staged_from_state(state, *, horizon: int, n_rays: int):
    origin_idx = _choose_origin_idx(state)
    frac_idx = jnp.full_like(origin_idx, _fraction_index_for_args())
    return selected_origin_fraction_targets_batched(
        state,
        origin_idx,
        frac_idx,
        horizon=horizon,
        ship_speed=float(ARGS.ship_speed),
        n_rays=n_rays,
        ray_chunk_size=0,
        first_hit_method="category-rays",
    )


def _sampled_extrema_staged_from_state(state, *, horizon: int):
    p0_b, p1_b, active_b = _forecast_batched(state, horizon)
    origin_xy, _origin_radius, launch_offset, speed, object_radii, _policy_mask = _launch_context_from_state(state)
    prepared = _prepare_targets(
        p0_b,
        p1_b,
        active_b,
        origin_xy,
        speed,
        launch_offset,
        object_radii,
    )
    packed = _pack_moving_targets(
        prepared["points"],
        prepared["radii_ep"],
        prepared["origin_xy_ep"],
        prepared["speed_ep"],
        prepared["launch_offset_ep"],
        prepared["horizon_ep"],
        prepared["moving_mask_ep"],
    )
    stat_w_lo, stat_w_hi, stat_w_valid = STAT_WINDOWS_JIT(
        prepared["centers"],
        prepared["radii"],
        prepared["origin_xy"],
        prepared["speed"],
        prepared["launch_offset"],
        prepared["horizon"],
    )
    move_w_lo, move_w_hi, move_w_valid = MOVE_WINDOWS_PACKED_JIT(
        packed["points"],
        packed["radii"],
        packed["origin_xy"],
        packed["speed"],
        packed["launch_offset"],
        packed["horizon"],
    )
    stat_w_valid = stat_w_valid & prepared["stationary_mask"][:, None]
    move_w_valid = move_w_valid & packed["moving_mask"][:, :, None]
    stat_ext = STAT_EXTREMA_JIT(
        prepared["centers"],
        prepared["radii"],
        prepared["origin_xy"],
        prepared["speed"],
        prepared["launch_offset"],
        stat_w_lo,
        stat_w_hi,
        stat_w_valid,
    )
    move_ext = MOVE_EXTREMA_SAMPLED_PACKED_JIT(
        packed["points"],
        packed["radii"],
        packed["origin_xy"],
        packed["speed"],
        packed["launch_offset"],
        move_w_lo,
        move_w_hi,
        move_w_valid,
    )
    return stat_ext, move_ext


STAT_HITS_JIT = jax.jit(jax.vmap(tangent_hit_times_stationary_jax))
STAT_WINDOWS_JIT = jax.jit(jax.vmap(intersection_windows_stationary_jax))
STAT_EXTREMA_JIT = jax.jit(jax.vmap(_stationary_window_extrema))

MOVE_HITS_PACKED_JIT = jax.jit(_vmap_moving(tangent_hit_times_polyline_jax))
MOVE_WINDOWS_PACKED_JIT = jax.jit(
    _vmap_moving(
        lambda pts, r, ox, sp, loff, hor: intersection_windows_polyline_capped_jax(
            pts,
            r,
            ox,
            sp,
            loff,
            hor,
            max_tangency_hits=MAX_POLYLINE_TANGENCY_HITS,
            max_windows=MAX_POLYLINE_INTERSECTION_WINDOWS,
        )
    )
)
MOVE_EXTREMA_ENDPOINTS_PACKED_JIT = jax.jit(_vmap_moving(_polyline_window_extrema_endpoints_only))
MOVE_EXTREMA_SAMPLED_PACKED_JIT = jax.jit(_vmap_moving(_polyline_window_extrema_sampled))
MOVE_EXTREMA_COEFFS_PACKED_JIT = jax.jit(_vmap_moving(_polyline_window_sextic_coeffs_only))
MOVE_EXTREMA_ROOT_INPUTS_PACKED_JIT = jax.jit(_polyline_packed_root_inputs)
MOVE_EXTREMA_EIGVALS_LAPACK_PACKED_JIT = jax.jit(
    lambda coeff_emws, use: _polyline_packed_sextic_eigvals_only_impl(
        coeff_emws, use, lax.linalg.EigImplementation.LAPACK
    )
)
MOVE_EXTREMA_EIGVALS_MAGMA_PACKED_JIT = jax.jit(
    lambda coeff_emws, use: _polyline_packed_sextic_eigvals_only_impl(
        coeff_emws, use, lax.linalg.EigImplementation.MAGMA
    )
)
MOVE_EXTREMA_NEWTON_PACKED_JIT = jax.jit(_polyline_packed_sextic_newton_only)
MOVE_EXTREMA_ROOTS_LAPACK_PACKED_JIT = jax.jit(
    lambda coeff_emws, q0, b_emws, radii, r0, grow_rate, u_len, use: _polyline_packed_roots_with_impl(
        coeff_emws,
        q0,
        b_emws,
        radii,
        r0,
        grow_rate,
        u_len,
        use,
        lax.linalg.EigImplementation.LAPACK,
    )
)
MOVE_EXTREMA_ROOTS_MAGMA_PACKED_JIT = jax.jit(
    lambda coeff_emws, q0, b_emws, radii, r0, grow_rate, u_len, use: _polyline_packed_roots_with_impl(
        coeff_emws,
        q0,
        b_emws,
        radii,
        r0,
        grow_rate,
        u_len,
        use,
        lax.linalg.EigImplementation.MAGMA,
    )
)
MOVE_EXTREMA_LAPACK_PACKED_JIT = jax.jit(
    lambda points, radii, grow_center, grow_rate, launch_offset, win_lo, win_hi, win_valid: _polyline_packed_extrema_with_impl(
        points,
        radii,
        grow_center,
        grow_rate,
        launch_offset,
        win_lo,
        win_hi,
        win_valid,
        lax.linalg.EigImplementation.LAPACK,
    )
)
MOVE_EXTREMA_MAGMA_PACKED_JIT = jax.jit(
    lambda points, radii, grow_center, grow_rate, launch_offset, win_lo, win_hi, win_valid: _polyline_packed_extrema_with_impl(
        points,
        radii,
        grow_center,
        grow_rate,
        launch_offset,
        win_lo,
        win_hi,
        win_valid,
        lax.linalg.EigImplementation.MAGMA,
    )
)
PACK_MOVING_JIT = jax.jit(_pack_moving_targets)


def _format_row(row: dict[str, object]) -> str:
    return (
        f"{row['label']:>20} "
        f"{row['compile_run_s']:>13.4f} "
        f"{row['mean_run_s']:>10.4f} "
        f"{row['min_run_s']:>9.4f} "
        f"{1000.0 * float(row['mean_run_s']) / float(ARGS.num_envs):>10.3f}"
    )


def _bench(label: str, fn, *args, **kwargs):
    print(f"[bench] starting {label}...", flush=True)
    jax.block_until_ready(args)
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    jax.block_until_ready(out)
    compile_run_s = time.perf_counter() - t0

    times = []
    for _ in range(ARGS.repeat):
        jax.block_until_ready(args)
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)

    row = {
        "label": label,
        "compile_run_s": compile_run_s,
        "mean_run_s": float(np.mean(times)),
        "min_run_s": float(np.min(times)),
        "out": out,
    }
    print("[bench] finished " + _format_row(row), flush=True)
    return row


def _bench_arg_sequence(label: str, fn, arg_seq, *, kwargs: dict[str, object] | None = None):
    kwargs = {} if kwargs is None else dict(kwargs)
    print(f"[bench] starting {label}...", flush=True)
    jax.block_until_ready(arg_seq[0])
    t0 = time.perf_counter()
    out = fn(*arg_seq[0], **kwargs)
    jax.block_until_ready(out)
    compile_run_s = time.perf_counter() - t0

    times = []
    for i in range(1, len(arg_seq)):
        jax.block_until_ready(arg_seq[i])
        t0 = time.perf_counter()
        out = fn(*arg_seq[i], **kwargs)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)

    row = {
        "label": label,
        "compile_run_s": compile_run_s,
        "mean_run_s": float(np.mean(times)),
        "min_run_s": float(np.min(times)),
        "out": out,
    }
    print("[bench] finished " + _format_row(row), flush=True)
    return row


def _batched_noop_actions(num_envs: int, num_agents: int) -> jnp.ndarray:
    one = empty_actions(num_agents, 1)
    return jnp.broadcast_to(one, (num_envs,) + one.shape)


_BATCHED_STEP_NOOP = jax.jit(jax.vmap(lambda s, a: jow.jit_step(s, a, jow.OrbitWarsConfig())))


def _tree_where(mask: jnp.ndarray, new_tree, old_tree):
    return jax.tree.map(
        lambda n, o: jnp.where(
            mask.reshape((mask.shape[0],) + (1,) * (n.ndim - 1)),
            n,
            o,
        ),
        new_tree,
        old_tree,
    )


def _decorrelate_state_batch(state_b, *, horizon: int, num_agents: int):
    envs = int(state_b.planets.shape[0])
    spread = max(1, int(horizon))
    offsets = jnp.arange(envs, dtype=jnp.int32) % jnp.asarray(spread, dtype=jnp.int32)
    noop = _batched_noop_actions(envs, num_agents)
    cur = state_b
    for t in range(spread - 1):
        stepped = _BATCHED_STEP_NOOP(cur, noop)
        take_step = offsets > jnp.asarray(t, dtype=jnp.int32)
        cur = _tree_where(take_step, stepped, cur)
    return cur


def _state_sequence(state_b, *, repeats: int, num_agents: int):
    noop = _batched_noop_actions(int(state_b.planets.shape[0]), num_agents)
    seq = [state_b]
    cur = state_b
    for _ in range(repeats):
        cur = _BATCHED_STEP_NOOP(cur, noop)
        seq.append(cur)
    return seq


def _build_case_inputs(state_b, *, horizon: int):
    p0_b, p1_b, active_b = _forecast_batched(state_b, horizon)
    origin_idx = _choose_origin_idx(state_b)
    ships = state_b.planets[jnp.arange(state_b.planets.shape[0]), origin_idx, PLANET_SHIPS]
    send = jnp.floor(jnp.asarray(float(ARGS.fraction), dtype=jnp.float32) * ships)
    speed = _fleet_speed_jax(send, float(ARGS.ship_speed))
    origin_xy = state_b.planets[jnp.arange(state_b.planets.shape[0]), origin_idx][:, PLANET_X : PLANET_Y + 1]
    origin_radius = state_b.planets[jnp.arange(state_b.planets.shape[0]), origin_idx, PLANET_RADIUS]
    launch_offset = origin_radius + jnp.asarray(0.1, dtype=jnp.float32)
    object_radii = state_b.planets[:, :, PLANET_RADIUS]
    policy_mask = state_b.planet_active

    prepared = _prepare_targets(
        p0_b,
        p1_b,
        active_b,
        origin_xy,
        speed,
        launch_offset,
        object_radii,
    )
    packed = PACK_MOVING_JIT(
        prepared["points"],
        prepared["radii_ep"],
        prepared["origin_xy_ep"],
        prepared["speed_ep"],
        prepared["launch_offset_ep"],
        prepared["horizon_ep"],
        prepared["moving_mask_ep"],
    )
    stat_w_lo, stat_w_hi, stat_w_valid = STAT_WINDOWS_JIT(
        prepared["centers"],
        prepared["radii"],
        prepared["origin_xy"],
        prepared["speed"],
        prepared["launch_offset"],
        prepared["horizon"],
    )
    move_w_lo, move_w_hi, move_w_valid = MOVE_WINDOWS_PACKED_JIT(
        packed["points"],
        packed["radii"],
        packed["origin_xy"],
        packed["speed"],
        packed["launch_offset"],
        packed["horizon"],
    )
    stat_w_valid = stat_w_valid & prepared["stationary_mask"][:, None]
    move_w_valid = move_w_valid & packed["moving_mask"][:, :, None]
    root_inputs = MOVE_EXTREMA_ROOT_INPUTS_PACKED_JIT(
        packed["points"],
        packed["radii"],
        packed["origin_xy"],
        packed["speed"],
        packed["launch_offset"],
        move_w_lo,
        move_w_hi,
        move_w_valid,
    )
    return {
        "state": state_b,
        "forecast": (p0_b, p1_b, active_b),
        "origin_idx": origin_idx,
        "origin_xy": origin_xy,
        "origin_radius": origin_radius,
        "launch_offset": launch_offset,
        "speed": speed,
        "object_radii": object_radii,
        "policy_mask": policy_mask,
        "prepared": prepared,
        "packed": packed,
        "stat_windows": (stat_w_lo, stat_w_hi, stat_w_valid),
        "move_windows": (move_w_lo, move_w_hi, move_w_valid),
        "root_inputs": root_inputs,
    }


def main() -> None:
    configure_jax_for_training(prefer_gpu=True, verbose=False)
    print(f"devices {jax.devices()}", flush=True)
    print(
        "name               compile+run_s mean_run_s min_run_s per_env_ms",
        flush=True,
    )

    cfg = OrbitWarsEnvConfig(
        num_agents=2,
        max_fleets=int(ARGS.max_fleets),
        episode_seed=int(ARGS.seed),
    )
    state_b, _ = stack_initial_states(cfg, int(ARGS.num_envs), int(ARGS.seed))
    state_b = _decorrelate_state_batch(
        state_b,
        horizon=int(ARGS.horizon),
        num_agents=int(cfg.num_agents),
    )
    state_seq = _state_sequence(
        state_b,
        repeats=int(ARGS.repeat),
        num_agents=int(cfg.num_agents),
    )
    case_seq = [_build_case_inputs(sb, horizon=int(ARGS.horizon)) for sb in state_seq]

    forecast_row = _bench_arg_sequence(
        "forecast",
        _forecast_batched,
        [(case["state"], int(ARGS.horizon)) for case in case_seq],
    )
    p0_b, p1_b, active_b = forecast_row["out"]

    rays_fused_row = _bench_arg_sequence(
        "rays_fused",
        _rays_fused_from_state,
        [(case["state"],) for case in case_seq],
        kwargs={
            "horizon": int(ARGS.horizon),
            "n_rays": int(ARGS.n_rays),
            "ray_chunk_size": int(ARGS.ray_chunk_size),
        },
    )
    category_rays_fused_row = _bench_arg_sequence(
        "category_rays_fused",
        _category_rays_fused_from_state,
        [(case["state"],) for case in case_seq],
        kwargs={
            "horizon": int(ARGS.horizon),
            "n_rays": int(ARGS.n_rays),
        },
    )
    sampled_fused_row = _bench_arg_sequence(
        "sampled_fused",
        _sampled_extrema_fused_from_state,
        [(case["state"],) for case in case_seq],
        kwargs={"horizon": int(ARGS.horizon)},
    )
    sampled_packed_fused_row = _bench_arg_sequence(
        "sampled_packed_fused",
        _sampled_extrema_packedforecast_fused_from_state,
        [(case["state"],) for case in case_seq],
        kwargs={"horizon": int(ARGS.horizon)},
    )
    rays_staged_row = _bench_arg_sequence(
        "rays_staged",
        _rays_staged_from_state,
        [(case["state"],) for case in case_seq],
        kwargs={
            "horizon": int(ARGS.horizon),
            "n_rays": int(ARGS.n_rays),
            "ray_chunk_size": int(ARGS.ray_chunk_size),
        },
    )
    category_rays_staged_row = _bench_arg_sequence(
        "category_rays_staged",
        _category_rays_staged_from_state,
        [(case["state"],) for case in case_seq],
        kwargs={
            "horizon": int(ARGS.horizon),
            "n_rays": int(ARGS.n_rays),
        },
    )
    sampled_staged_row = _bench_arg_sequence(
        "sampled_staged",
        _sampled_extrema_staged_from_state,
        [(case["state"],) for case in case_seq],
        kwargs={"horizon": int(ARGS.horizon)},
    )
    sampled_packed_staged_row = _bench_arg_sequence(
        "sampled_packed_staged",
        _sampled_extrema_packedforecast_staged_from_state,
        [(case["state"],) for case in case_seq],
        kwargs={"horizon": int(ARGS.horizon)},
    )

    origin_xy = case_seq[0]["origin_xy"]
    origin_radius = case_seq[0]["origin_radius"]
    speed = case_seq[0]["speed"]
    object_radii = case_seq[0]["object_radii"]
    policy_mask = case_seq[0]["policy_mask"]

    rays_fn = jax.jit(
        jax.vmap(
            lambda ox, orad, sp, p0, p1, rr, act, pm: first_hit_best_targets_apply_jax(
                ox,
                orad,
                sp,
                p0,
                p1,
                rr,
                act,
                pm,
                n_rays=int(ARGS.n_rays),
                ray_chunk_size=int(ARGS.ray_chunk_size),
            )
        )
    )
    ray_row = _bench_arg_sequence(
        "rays",
        rays_fn,
        [
            (
                case["origin_xy"],
                case["origin_radius"],
                case["speed"],
                case["forecast"][0],
                case["forecast"][1],
                case["object_radii"],
                case["forecast"][2],
                case["policy_mask"],
            )
            for case in case_seq
        ],
    )
    category_ray_row = _bench_arg_sequence(
        "category_rays",
        selected_origin_fraction_targets_batched,
        [
            (
                case["state"],
                case["origin_idx"],
                jnp.full_like(case["origin_idx"], _fraction_index_for_args()),
            )
            for case in case_seq
        ],
        kwargs={
            "horizon": int(ARGS.horizon),
            "ship_speed": float(ARGS.ship_speed),
            "n_rays": int(ARGS.n_rays),
            "ray_chunk_size": 0,
            "first_hit_method": "category-rays",
        },
    )

    prepared_row = _bench_arg_sequence(
        "prepare_targets",
        _prepare_targets,
        [
            (
                case["forecast"][0],
                case["forecast"][1],
                case["forecast"][2],
                case["origin_xy"],
                case["speed"],
                case["launch_offset"],
                case["object_radii"],
            )
            for case in case_seq
        ],
    )
    prepared = prepared_row["out"]

    pack_row = _bench_arg_sequence(
        "pack_moving",
        PACK_MOVING_JIT,
        [
            (
                case["prepared"]["points"],
                case["prepared"]["radii_ep"],
                case["prepared"]["origin_xy_ep"],
                case["prepared"]["speed_ep"],
                case["prepared"]["launch_offset_ep"],
                case["prepared"]["horizon_ep"],
                case["prepared"]["moving_mask_ep"],
            )
            for case in case_seq
        ],
    )
    packed = pack_row["out"]

    stat_hits_row = _bench_arg_sequence(
        "stationary_hits",
        STAT_HITS_JIT,
        [
            (
                case["prepared"]["centers"],
                case["prepared"]["radii"],
                case["prepared"]["origin_xy"],
                case["prepared"]["speed"],
                case["prepared"]["launch_offset"],
                case["prepared"]["horizon"],
            )
            for case in case_seq
        ],
    )
    move_hits_row = _bench_arg_sequence(
        "polyline_hits",
        MOVE_HITS_PACKED_JIT,
        [
            (
                case["packed"]["points"],
                case["packed"]["radii"],
                case["packed"]["origin_xy"],
                case["packed"]["speed"],
                case["packed"]["launch_offset"],
                case["packed"]["horizon"],
            )
            for case in case_seq
        ],
    )
    stat_windows_row = _bench_arg_sequence(
        "stationary_windows",
        STAT_WINDOWS_JIT,
        [
            (
                case["prepared"]["centers"],
                case["prepared"]["radii"],
                case["prepared"]["origin_xy"],
                case["prepared"]["speed"],
                case["prepared"]["launch_offset"],
                case["prepared"]["horizon"],
            )
            for case in case_seq
        ],
    )
    move_windows_row = _bench_arg_sequence(
        "polyline_windows",
        MOVE_WINDOWS_PACKED_JIT,
        [
            (
                case["packed"]["points"],
                case["packed"]["radii"],
                case["packed"]["origin_xy"],
                case["packed"]["speed"],
                case["packed"]["launch_offset"],
                case["packed"]["horizon"],
            )
            for case in case_seq
        ],
    )

    stat_w_lo, stat_w_hi, stat_w_valid = case_seq[0]["stat_windows"]
    move_w_lo, move_w_hi, move_w_valid = case_seq[0]["move_windows"]

    stat_ext_row = _bench_arg_sequence(
        "stationary_extrema",
        STAT_EXTREMA_JIT,
        [
            (
                case["prepared"]["centers"],
                case["prepared"]["radii"],
                case["prepared"]["origin_xy"],
                case["prepared"]["speed"],
                case["prepared"]["launch_offset"],
                case["stat_windows"][0],
                case["stat_windows"][1],
                case["stat_windows"][2],
            )
            for case in case_seq
        ],
    )
    move_ext_endpoints_row = _bench_arg_sequence(
        "poly_ext_endpoints",
        MOVE_EXTREMA_ENDPOINTS_PACKED_JIT,
        [
            (
                case["packed"]["points"],
                case["packed"]["radii"],
                case["packed"]["origin_xy"],
                case["packed"]["speed"],
                case["packed"]["launch_offset"],
                case["move_windows"][0],
                case["move_windows"][1],
                case["move_windows"][2],
            )
            for case in case_seq
        ],
    )
    move_ext_sampled_row = _bench_arg_sequence(
        "poly_ext_sampled",
        MOVE_EXTREMA_SAMPLED_PACKED_JIT,
        [
            (
                case["packed"]["points"],
                case["packed"]["radii"],
                case["packed"]["origin_xy"],
                case["packed"]["speed"],
                case["packed"]["launch_offset"],
                case["move_windows"][0],
                case["move_windows"][1],
                case["move_windows"][2],
            )
            for case in case_seq
        ],
    )
    move_ext_coeffs_row = _bench_arg_sequence(
        "poly_ext_coeffs",
        MOVE_EXTREMA_COEFFS_PACKED_JIT,
        [
            (
                case["packed"]["points"],
                case["packed"]["radii"],
                case["packed"]["origin_xy"],
                case["packed"]["speed"],
                case["packed"]["launch_offset"],
                case["move_windows"][0],
                case["move_windows"][1],
                case["move_windows"][2],
            )
            for case in case_seq
        ],
    )
    move_ext_root_inputs_row = _bench_arg_sequence(
        "poly_ext_root_inputs",
        MOVE_EXTREMA_ROOT_INPUTS_PACKED_JIT,
        [
            (
                case["packed"]["points"],
                case["packed"]["radii"],
                case["packed"]["origin_xy"],
                case["packed"]["speed"],
                case["packed"]["launch_offset"],
                case["move_windows"][0],
                case["move_windows"][1],
                case["move_windows"][2],
            )
            for case in case_seq
        ],
    )
    coeff_emws, q0_ws, b_emws, b_seg_ws, r0_ws, u_len_ws, use_ws, radii_ws = case_seq[0]["root_inputs"]
    move_ext_eigvals_lapack_row = _bench_arg_sequence(
        "poly_ext_eig_lapack",
        MOVE_EXTREMA_EIGVALS_LAPACK_PACKED_JIT,
        [(case["root_inputs"][0], case["root_inputs"][6]) for case in case_seq],
    )
    move_ext_eigvals_magma_row = _bench_arg_sequence(
        "poly_ext_eig_magma",
        MOVE_EXTREMA_EIGVALS_MAGMA_PACKED_JIT,
        [(case["root_inputs"][0], case["root_inputs"][6]) for case in case_seq],
    )
    move_ext_roots_lapack_row = _bench_arg_sequence(
        "poly_ext_root_lapack",
        MOVE_EXTREMA_ROOTS_LAPACK_PACKED_JIT,
        [
            (
                case["root_inputs"][0],
                case["root_inputs"][1],
                case["root_inputs"][2],
                case["root_inputs"][7],
                case["root_inputs"][4],
                case["packed"]["speed"],
                case["root_inputs"][5],
                case["root_inputs"][6],
            )
            for case in case_seq
        ],
    )
    move_ext_roots_magma_row = _bench_arg_sequence(
        "poly_ext_root_magma",
        MOVE_EXTREMA_ROOTS_MAGMA_PACKED_JIT,
        [
            (
                case["root_inputs"][0],
                case["root_inputs"][1],
                case["root_inputs"][2],
                case["root_inputs"][7],
                case["root_inputs"][4],
                case["packed"]["speed"],
                case["root_inputs"][5],
                case["root_inputs"][6],
            )
            for case in case_seq
        ],
    )
    move_ext_lapack_row = _bench_arg_sequence(
        "poly_extrema_lapack",
        MOVE_EXTREMA_LAPACK_PACKED_JIT,
        [
            (
                case["packed"]["points"],
                case["packed"]["radii"],
                case["packed"]["origin_xy"],
                case["packed"]["speed"],
                case["packed"]["launch_offset"],
                case["move_windows"][0],
                case["move_windows"][1],
                case["move_windows"][2],
            )
            for case in case_seq
        ],
    )
    move_ext_magma_row = _bench_arg_sequence(
        "poly_extrema_magma",
        MOVE_EXTREMA_MAGMA_PACKED_JIT,
        [
            (
                case["packed"]["points"],
                case["packed"]["radii"],
                case["packed"]["origin_xy"],
                case["packed"]["speed"],
                case["packed"]["launch_offset"],
                case["move_windows"][0],
                case["move_windows"][1],
                case["move_windows"][2],
            )
            for case in case_seq
        ],
    )

    stationary_mask_np = np.asarray(jax.device_get(case_seq[0]["prepared"]["stationary_mask"]))
    moving_mask_np = np.asarray(jax.device_get(case_seq[0]["prepared"]["moving_mask"]))
    packed_moving_np = np.asarray(jax.device_get(case_seq[0]["packed"]["moving_mask"]))
    stat_windows_np = np.asarray(jax.device_get(case_seq[0]["stat_windows"][2]))
    move_windows_np = np.asarray(jax.device_get(case_seq[0]["move_windows"][2]))

    rows = [
        forecast_row,
        rays_fused_row,
        category_rays_fused_row,
        sampled_fused_row,
        sampled_packed_fused_row,
        rays_staged_row,
        category_rays_staged_row,
        sampled_staged_row,
        sampled_packed_staged_row,
        prepared_row,
        pack_row,
        ray_row,
        category_ray_row,
        stat_hits_row,
        move_hits_row,
        stat_windows_row,
        move_windows_row,
        stat_ext_row,
        move_ext_endpoints_row,
        move_ext_sampled_row,
        move_ext_coeffs_row,
        move_ext_root_inputs_row,
        move_ext_eigvals_lapack_row,
        move_ext_eigvals_magma_row,
        move_ext_roots_lapack_row,
        move_ext_roots_magma_row,
        move_ext_lapack_row,
        move_ext_magma_row,
    ]

    print(
        f"real tangent-components benchmark envs={ARGS.num_envs} horizon={ARGS.horizon} "
        f"moving_cap={MAX_MOVING_TARGETS} windows_cap={MAX_POLYLINE_INTERSECTION_WINDOWS} "
        f"tangency_cap={MAX_POLYLINE_TANGENCY_HITS} rays={ARGS.n_rays} repeat={ARGS.repeat}"
    )
    step_counts_np = np.asarray(jax.device_get(case_seq[0]["state"].step_count))
    print(
        f"decorrelated start steps min={int(step_counts_np.min())} "
        f"max={int(step_counts_np.max())} unique={int(np.unique(step_counts_np).size)}"
    )
    print(
        f"targets stationary={int(stationary_mask_np.sum())} "
        f"moving={int(moving_mask_np.sum())} "
        f"packed_moving_slots={int(packed_moving_np.sum())} "
        f"total={int((stationary_mask_np | moving_mask_np).sum())}"
    )
    print(
        f"windows stationary={int(stat_windows_np.sum())} "
        f"moving_packed={int(move_windows_np.sum())} "
        f"total={int(stat_windows_np.sum() + move_windows_np.sum())}"
    )
    print("name compile+run_s mean_run_s min_run_s per_env_ms")
    for row in rows:
        print(_format_row(row))


if __name__ == "__main__":
    main()

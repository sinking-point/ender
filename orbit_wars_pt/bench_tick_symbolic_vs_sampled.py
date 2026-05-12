"""Benchmark sampled vs symbolic tick-hit interval builders.

Measures:
  * JIT compile latency (first call)
  * steady-state runtime (second call)

This is a *micro-benchmark* for tick-level geometry cost. The batched section also
runs ``first_hit_interval_best_targets_apply_jax`` and a **stage breakdown**:
precompute-only (``vmap`` over ticks of planet ``tick_hit``), sweep-only (occlusion
``fori_loop`` plus board / sun ``tick_hit`` on precomputed tensors), and the **fused**
interval kernel. Split-stage runtimes need not sum to the full kernel because XLA may
fuse precompute+sweep in the monolithic compile.

Use ``--batch 128`` to mirror rollout-style vmap over environments (default 128).

Use ``--sweep-breakdown`` (with ``--skip-single-env``) to split sampled sweep time
into planets-only vs board vs sun (same precomputed hits, separate JIT per config).

Use ``--parallel-vs-sequential-sweep`` for ``same_tick_planets_parallel`` False vs True
(``jit(vmap(first_hit_interval_best_targets_apply_jax))``, sampled ``tick_hit`` only).

``--union-scan-benchmark`` compares ``first_hit_union_scan_bins_apply_jax`` (binned
prefix-OR, no board, sun in pre) to the default interval sweep (``--n-block-bins``).

``--brute-ray-baseline`` runs ``first_hit_brute_rays_baseline_apply_jax`` (2048 headings
by default; game-accurate swept segments per tick) vs ``first_hit_interval_best_targets_apply_jax``
(see ``--n-brute-rays``). ``--n-brute-substeps`` is ignored but kept for CLI compatibility.
"""

from __future__ import annotations

import argparse
import os
import sys
from functools import partial
from time import perf_counter


def _early_platform_from_argv() -> None:
    """Set ``JAX_PLATFORMS`` before importing JAX (``--platform cpu|cuda``)."""

    argv = sys.argv[1:]
    for i, tok in enumerate(argv):
        if tok == "--platform" and i + 1 < len(argv):
            os.environ["JAX_PLATFORMS"] = argv[i + 1]
            return
        if tok.startswith("--platform="):
            os.environ["JAX_PLATFORMS"] = tok.split("=", 1)[1]
            return
    os.environ.setdefault("JAX_PLATFORMS", "cpu")


_early_platform_from_argv()

import jax
import jax.numpy as jnp
import numpy as np

import orbit_wars_pt.geometry_jax as gj
from orbit_wars_pt.geometry_jax import GEOM_EPS, tick_hit_intervals_jax
from orbit_wars_pt.constants import BOARD_SIZE, MAX_PLANETS


TAU = 2.0 * float(np.pi)
ROOT_EPS = 1e-4


def _cross2(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return a[0] * b[1] - a[1] * b[0]


def _norm_angle(a: jnp.ndarray) -> jnp.ndarray:
    return jnp.mod(a, TAU)


def _swept_hit_theta_exact(
    theta: jnp.ndarray,
    q: jnp.ndarray,
    d_vec: jnp.ndarray,
    a_base: jnp.ndarray,
    speed: jnp.ndarray,
    radius: jnp.ndarray,
    eps: float = float(GEOM_EPS),
) -> jnp.ndarray:
    """Exact swept segment vs disk predicate for one launch angle."""

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


def _angles_from_quartic_roots(
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
    theta_z = _norm_angle(2.0 * jnp.arctan(z_real))

    y_real = jnp.real(roots_y)
    y_valid0 = jnp.isfinite(y_real) & (jnp.abs(jnp.imag(roots_y)) <= eps)
    theta_y = _norm_angle(2.0 * jnp.arctan2(jnp.ones_like(y_real), y_real))

    theta = jnp.concatenate([theta_z, theta_y])
    valid0 = jnp.concatenate([z_valid0, y_valid0])

    def check(th):
        u = jnp.stack([jnp.cos(th), jnp.sin(th)])
        r0 = q - a_base * u
        s = d_vec - speed * u
        ss = jnp.dot(s, s)
        cross = _cross2(r0, s)
        residual = cross * cross - radius * radius * ss
        scale = radius * radius * ss + 1.0
        ok_res = jnp.abs(residual) <= 100.0 * eps * scale
        tau_star = -jnp.dot(r0, s) / jnp.maximum(ss, float(GEOM_EPS))
        ok_tau = (tau_star > float(GEOM_EPS)) & (tau_star < 1.0 - float(GEOM_EPS))
        ok_ss = ss > float(GEOM_EPS)
        return ok_res & ok_tau & ok_ss

    valid = valid0 & jax.vmap(check)(theta)
    return theta, valid


def _projection_switch_angles(
    q: jnp.ndarray,
    d_vec: jnp.ndarray,
    a_base: jnp.ndarray,
    speed: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    qx, qy = q[0], q[1]
    dx, dy = d_vec[0], d_vec[1]
    A = a_base
    v = speed

    c0_0 = qx * dx + qy * dy + A * v
    cc_0 = -v * qx - A * dx
    cs_0 = -v * qy - A * dy
    a0, v0 = _linear_trig_roots(c0_0, cc_0, cs_0)

    Ap = A + v
    c0_1 = (qx + dx) * dx + (qy + dy) * dy + Ap * v
    cc_1 = -v * (qx + dx) - Ap * dx
    cs_1 = -v * (qy + dy) - Ap * dy
    a1, v1 = _linear_trig_roots(c0_1, cc_1, cs_1)

    return jnp.concatenate([a0, a1]), jnp.concatenate([v0, v1])


@jax.jit
def tick_hit_symbolic_compat_for_bench(
    origin_xy,
    origin_radius,
    speed,
    tick,
    object_p0,
    object_p1,
    object_radius,
    object_active=jnp.asarray(True),
    *,
    samples_per_span: int = 9,
):
    del samples_per_span
    return tick_hit_intervals_symbolic_jax_one(
        origin_xy, origin_radius, speed, tick, object_p0, object_p1, object_radius, object_active
    )


def tick_hit_intervals_symbolic_jax_one(
    origin_xy: jnp.ndarray,
    origin_radius: jnp.ndarray,
    speed: jnp.ndarray,
    tick: jnp.ndarray,
    object_p0: jnp.ndarray,
    object_p1: jnp.ndarray,
    object_radius: jnp.ndarray,
    object_active: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    q = object_p0 - origin_xy
    d_vec = object_p1 - object_p0
    a_base = origin_radius + 0.1 + tick.astype(origin_xy.dtype) * speed
    R = object_radius

    ep0_angles, ep0_valid = _endpoint_boundary_angles(q, a_base, R)
    ep1_angles, ep1_valid = _endpoint_boundary_angles(q + d_vec, a_base + speed, R)

    coeff_asc = _interior_tangent_poly_coeffs(q, d_vec, a_base, speed, R)
    tan_angles, tan_valid = _angles_from_quartic_roots(coeff_asc, q, d_vec, a_base, speed, R)

    proj_angles, proj_valid = _projection_switch_angles(q, d_vec, a_base, speed)

    fixed_angles = jnp.asarray([jnp.pi], dtype=origin_xy.dtype)
    fixed_valid = jnp.asarray([True], dtype=bool)

    boundary_angles = jnp.concatenate([ep0_angles, ep1_angles, tan_angles, proj_angles, fixed_angles])
    boundary_valid = jnp.concatenate([ep0_valid, ep1_valid, tan_valid, proj_valid, fixed_valid]) & object_active

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

    cell_hit = jax.vmap(lambda th: _swept_hit_theta_exact(th, q, d_vec, a_base, speed, R))(mid)
    valid = object_active & cell_hit & (hi - lo > float(GEOM_EPS))
    return lo, hi, valid


def _make_random_case(batch_objects: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    origin_xy = rng.uniform(10.0, 90.0, size=(2,)).astype(np.float32)
    origin_radius = float(rng.uniform(1.0, 4.0))
    speed = float(rng.uniform(1.0, 6.0))
    tick = int(rng.integers(0, 24))

    object_p0 = rng.uniform(0.0, BOARD_SIZE, size=(batch_objects, 2)).astype(np.float32)
    drift = rng.normal(0.0, 3.0, size=(batch_objects, 2)).astype(np.float32)
    object_p1 = np.clip(object_p0 + drift, -5.0, BOARD_SIZE + 5.0).astype(np.float32)
    object_radius = rng.uniform(0.8, 4.5, size=(batch_objects,)).astype(np.float32)
    object_active = (rng.random(size=(batch_objects,)) > 0.15).astype(np.bool_)
    return origin_xy, origin_radius, speed, tick, object_p0, object_p1, object_radius, object_active


def _make_batched_envs(batch: int, P: int, ticks: int, seed: int) -> dict:
    """Random tensors shaped like ``batch`` independent envs."""
    rng = np.random.default_rng(seed)
    origin_xy = rng.uniform(10.0, 90.0, size=(batch, 2)).astype(np.float32)
    origin_radius = rng.uniform(1.0, 4.0, size=(batch,)).astype(np.float32)
    speed = rng.uniform(1.0, 6.0, size=(batch,)).astype(np.float32)
    tick = rng.integers(0, min(24, max(1, ticks)), size=(batch,), dtype=np.int32)

    object_p0 = rng.uniform(0.0, BOARD_SIZE, size=(batch, P, 2)).astype(np.float32)
    drift = rng.normal(0.0, 3.0, size=(batch, P, 2)).astype(np.float32)
    object_p1 = np.clip(object_p0 + drift, -5.0, BOARD_SIZE + 5.0).astype(np.float32)
    object_radius = rng.uniform(0.8, 4.5, size=(batch, P)).astype(np.float32)
    object_active = (rng.random(size=(batch, P)) > 0.15).astype(np.bool_)

    p0_by_tick = rng.uniform(0.0, BOARD_SIZE, size=(batch, ticks, P, 2)).astype(np.float32)
    drift_t = rng.normal(0.0, 3.0, size=(batch, ticks, P, 2)).astype(np.float32)
    p1_by_tick = np.clip(p0_by_tick + drift_t, -5.0, BOARD_SIZE + 5.0).astype(np.float32)
    active_by_tick = (rng.random(size=(batch, ticks, P)) > 0.15).astype(np.bool_)

    return {
        "origin_xy": jnp.asarray(origin_xy),
        "origin_radius": jnp.asarray(origin_radius),
        "speed": jnp.asarray(speed),
        "tick": jnp.asarray(tick, dtype=jnp.int32),
        "p0": jnp.asarray(object_p0),
        "p1": jnp.asarray(object_p1),
        "r": jnp.asarray(object_radius),
        "act": jnp.asarray(object_active),
        "p0_by_tick": jnp.asarray(p0_by_tick),
        "p1_by_tick": jnp.asarray(p1_by_tick),
        "active_by_tick": jnp.asarray(active_by_tick),
    }


def bench_tick_vmap_over_envs(batch: int, P: int, samples_per_span: int, seed: int) -> None:
    """One tick row: vmap(batch) inner vmap(P) — like precomputing all planet hits for one tick."""
    d = _make_batched_envs(batch, P, ticks=24, seed=seed)

    def one_env_sampled(ox, orad, sp, ti, p0r, p1r, rr, ar):
        return jax.vmap(
            lambda p0i, p1i, ri, ai: tick_hit_intervals_jax(
                ox, orad, sp, ti, p0i, p1i, ri, ai, samples_per_span=samples_per_span
            )
        )(p0r, p1r, rr, ar)

    def one_env_symbolic(ox, orad, sp, ti, p0r, p1r, rr, ar):
        return jax.vmap(
            lambda p0i, p1i, ri, ai: tick_hit_intervals_symbolic_jax_one(
                ox, orad, sp, ti, p0i, p1i, ri, ai
            )
        )(p0r, p1r, rr, ar)

    in_axes = (0, 0, 0, 0, 0, 0, 0, 0)
    sampled_batched = jax.jit(jax.vmap(one_env_sampled, in_axes=in_axes))
    symbolic_batched = jax.jit(jax.vmap(one_env_symbolic, in_axes=in_axes))

    batched_args = (
        d["origin_xy"],
        d["origin_radius"],
        d["speed"],
        d["tick"],
        d["p0"],
        d["p1"],
        d["r"],
        d["act"],
    )

    t0 = perf_counter()
    out_s = sampled_batched(*batched_args)
    _ = jax.device_get(out_s)
    t_s_compile = perf_counter() - t0

    t0 = perf_counter()
    out_s2 = sampled_batched(*batched_args)
    _ = jax.device_get(out_s2)
    t_s_run = perf_counter() - t0

    t0 = perf_counter()
    out_y = symbolic_batched(*batched_args)
    _ = jax.device_get(out_y)
    t_y_compile = perf_counter() - t0

    t0 = perf_counter()
    out_y2 = symbolic_batched(*batched_args)
    _ = jax.device_get(out_y2)
    t_y_run = perf_counter() - t0

    print(f"=== tick vmap batch={batch}, P={P}, samples_per_span={samples_per_span} ===")
    print(f"  sampled : compile {t_s_compile:.4f}s, run {t_s_run:.4f}s  ({t_s_run / batch * 1e3:.3f} ms/env)")
    print(f"  symbolic: compile {t_y_compile:.4f}s, run {t_y_run:.4f}s  ({t_y_run / batch * 1e3:.3f} ms/env)")


def bench_first_hit_vmap_over_envs(
    batch: int, P: int, ticks: int, samples_per_span: int, seed: int, max_block_intervals: int
) -> None:
    d = _make_batched_envs(batch, P, ticks=ticks, seed=seed + 1)

    sps, mb = samples_per_span, max_block_intervals

    # Two distinct callables: ``jax.jit(f) is jax.jit(f)`` for the same object ``f``.
    # A single ``vmap(partial(apply,...))`` reused for both modes would compile once
    # (whichever tick_hit was current) and silently run that program for both labels.

    def one_env_sampled(ox, orad, sp, p0t, p1t, rr, at):
        return gj.first_hit_interval_best_targets_apply_jax(
            ox, orad, sp, p0t, p1t, rr, at, samples_per_span=sps, max_block_intervals=mb
        )

    def one_env_symbolic(ox, orad, sp, p0t, p1t, rr, at):
        return gj.first_hit_interval_best_targets_apply_jax(
            ox, orad, sp, p0t, p1t, rr, at, samples_per_span=sps, max_block_intervals=mb
        )

    args = (
        d["origin_xy"],
        d["origin_radius"],
        d["speed"],
        d["p0_by_tick"],
        d["p1_by_tick"],
        d["r"],
        d["active_by_tick"],
    )

    # ``jit_sampled`` must compile with the real sampled ``tick_hit_intervals_jax``.
    # Do not patch the module to symbolic until after sampled compile + timed runs,
    # or both programs trace symbolic tick_hit and the "sampled" numbers lie.
    gj.tick_hit_intervals_jax = tick_hit_intervals_jax  # type: ignore[assignment]
    jit_sampled = jax.jit(jax.vmap(one_env_sampled))

    t0 = perf_counter()
    out_s = jit_sampled(*args)
    out_s = jax.device_get(out_s)
    t_s_compile = perf_counter() - t0
    _, _, _, ov_s = out_s

    t0 = perf_counter()
    out_s2 = jit_sampled(*args)
    out_s2 = jax.device_get(out_s2)
    t_s_run = perf_counter() - t0

    gj.tick_hit_intervals_jax = tick_hit_symbolic_compat_for_bench  # type: ignore[assignment]
    jit_symbolic = jax.jit(jax.vmap(one_env_symbolic))

    t0 = perf_counter()
    out_y = jit_symbolic(*args)
    out_y = jax.device_get(out_y)
    t_y_compile = perf_counter() - t0
    _, _, _, ov_y = out_y

    t0 = perf_counter()
    out_y2 = jit_symbolic(*args)
    out_y2 = jax.device_get(out_y2)
    t_y_run = perf_counter() - t0

    gj.tick_hit_intervals_jax = tick_hit_intervals_jax  # type: ignore[assignment]

    print(
        f"=== first_hit_interval_best_targets_apply_jax (tick_hit patch test) "
        f"vmap batch={batch}, P={P}, ticks={ticks} ==="
    )
    print(f"  sampled : compile {t_s_compile:.4f}s, run {t_s_run:.4f}s  ({t_s_run / batch * 1e3:.3f} ms/env)")
    print(f"  symbolic: compile {t_y_compile:.4f}s, run {t_y_run:.4f}s  ({t_y_run / batch * 1e3:.3f} ms/env)")
    ov_s_n = int(np.asarray(ov_s, dtype=np.bool_).sum())
    ov_y_n = int(np.asarray(ov_y, dtype=np.bool_).sum())
    print(
        f"  overflow envs (max_block_intervals={max_block_intervals}): "
        f"sampled {ov_s_n}, symbolic {ov_y_n} / {batch}"
    )


def bench_first_hit_stage_breakdown(
    batch: int,
    P: int,
    ticks: int,
    samples_per_span: int,
    seed: int,
    max_block_intervals: int,
    *,
    label: str,
    symbolic: bool,
) -> None:
    """Time precompute-only, sweep-from-precomputed-only, and full ``first_hit`` (separate JIT kernels).

    ``jit_full`` must use ``first_hit_interval_best_targets_apply_jax`` (not ``first_hit_best_targets_jax``)
    under ``jit(vmap(...))``: an inner ``@jit`` blocks vmap batching and can make GPU timings
    look orders of magnitude worse than ``precompute`` + ``sweep`` split kernels.

    Split runtimes need not equal ``full``: two launches vs one fused graph; after the fix above,
    ``full`` should be in the same ballpark as ``pre + sweep``, not wildly slower.
    """

    d = _make_batched_envs(batch, P, ticks=ticks, seed=seed + 3)
    args = (
        d["origin_xy"],
        d["origin_radius"],
        d["speed"],
        d["p0_by_tick"],
        d["p1_by_tick"],
        d["r"],
        d["active_by_tick"],
    )

    if symbolic:
        gj.tick_hit_intervals_jax = tick_hit_symbolic_compat_for_bench  # type: ignore[assignment]
    else:
        gj.tick_hit_intervals_jax = tick_hit_intervals_jax  # type: ignore[assignment]

    try:
        jit_pre = jax.jit(
            jax.vmap(
                lambda ox, orad, sp, p0t, p1t, rr, at: gj._precompute_all_tick_planet_hits_for_best_targets_jax(
                    ox, orad, sp, p0t, p1t, rr, at, samples_per_span
                )
            )
        )
        sweep_vec = jax.vmap(
            partial(
                gj._sweep_best_targets_from_precomputed_hits_jax,
                samples_per_span=samples_per_span,
                max_block_intervals=max_block_intervals,
            )
        )
        jit_sweep = jax.jit(sweep_vec)
        jit_full = jax.jit(
            jax.vmap(
                partial(
                    gj.first_hit_interval_best_targets_apply_jax,
                    samples_per_span=samples_per_span,
                    max_block_intervals=max_block_intervals,
                )
            )
        )

        # Warm pre + materialize for sweep input shape.
        t0 = perf_counter()
        pre1 = jit_pre(*args)
        _ = jax.device_get(pre1)
        t_pre_compile = perf_counter() - t0

        t0 = perf_counter()
        pre2 = jit_pre(*args)
        _ = jax.device_get(pre2)
        t_pre_run = perf_counter() - t0

        lo, hi, va = pre2

        t0 = perf_counter()
        sw1 = jit_sweep(d["origin_xy"], d["origin_radius"], d["speed"], lo, hi, va)
        _ = jax.device_get(sw1)
        t_sweep_compile = perf_counter() - t0

        t0 = perf_counter()
        sw2 = jit_sweep(d["origin_xy"], d["origin_radius"], d["speed"], lo, hi, va)
        _ = jax.device_get(sw2)
        t_sweep_run = perf_counter() - t0

        t0 = perf_counter()
        fu1 = jit_full(*args)
        _ = jax.device_get(fu1)
        t_full_compile = perf_counter() - t0

        t0 = perf_counter()
        fu2 = jit_full(*args)
        _ = jax.device_get(fu2)
        t_full_run = perf_counter() - t0

        ms = batch * 1e-3
        sum_run = t_pre_run + t_sweep_run
        print(f"=== first_hit stage breakdown ({label}) batch={batch}, P={P}, ticks={ticks} ===")
        print(
            f"  precompute only : compile {t_pre_compile:.4f}s, run {t_pre_run:.4f}s  ({t_pre_run / ms:.3f} ms/env)"
        )
        print(
            f"  sweep only       : compile {t_sweep_compile:.4f}s, run {t_sweep_run:.4f}s  ({t_sweep_run / ms:.3f} ms/env)"
        )
        print(
            f"  full fused       : compile {t_full_compile:.4f}s, run {t_full_run:.4f}s  ({t_full_run / ms:.3f} ms/env)"
        )
        print(
            f"  sum(pre+sweep) run {sum_run:.4f}s ({sum_run / ms:.3f} ms/env) vs full {t_full_run:.4f}s "
            f"({t_full_run / ms:.3f} ms/env) — two kernels vs one; gap mostly launch / fusion."
        )
    finally:
        gj.tick_hit_intervals_jax = tick_hit_intervals_jax  # type: ignore[assignment]


def _pct_str(delta: float, base: float) -> str:
    if base <= 0.0:
        return "n/a"
    return f"{100.0 * delta / base:.1f}"


def _make_sweep_vmap_row(
    include_board: bool,
    include_sun: bool,
    samples_per_span: int,
    max_block_intervals: int,
    tag: str,
):
    """Return a uniquely named per-env sweep so ``jax.jit(jax.vmap(...))`` cannot alias across configs."""

    def sweep_row(
        ox: jnp.ndarray,
        orad: jnp.ndarray,
        sp: jnp.ndarray,
        lo: jnp.ndarray,
        hi: jnp.ndarray,
        va: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return gj._sweep_best_targets_from_precomputed_hits_jax(
            ox,
            orad,
            sp,
            lo,
            hi,
            va,
            include_board=include_board,
            include_sun=include_sun,
            samples_per_span=samples_per_span,
            max_block_intervals=max_block_intervals,
        )

    sweep_row.__name__ = f"sweep_row_{tag}"
    sweep_row.__qualname__ = f"sweep_row_{tag}"
    return sweep_row


def bench_sweep_sampled_component_breakdown(
    batch: int,
    P: int,
    ticks: int,
    samples_per_span: int,
    seed: int,
    max_block_intervals: int,
) -> None:
    """Where sampled ``_sweep`` time goes: planet occlusion loop vs board vs sun.

    Uses fixed precomputed ``(lo, hi, valid)`` tensors and only varies
    ``include_board`` / ``include_sun``. Each config gets its own Python callable
    so JIT caches do not collide.
    """

    d = _make_batched_envs(batch, P, ticks=ticks, seed=seed + 5)
    args = (
        d["origin_xy"],
        d["origin_radius"],
        d["speed"],
        d["p0_by_tick"],
        d["p1_by_tick"],
        d["r"],
        d["active_by_tick"],
    )
    ox, orad, sp = d["origin_xy"], d["origin_radius"], d["speed"]

    gj.tick_hit_intervals_jax = tick_hit_intervals_jax  # type: ignore[assignment]
    sps, mb = samples_per_span, max_block_intervals

    jit_pre = jax.jit(
        jax.vmap(
            lambda o, r, s, p0t, p1t, rr, at: gj._precompute_all_tick_planet_hits_for_best_targets_jax(
                o, r, s, p0t, p1t, rr, at, sps
            )
        )
    )
    pre0 = jit_pre(*args)
    _ = jax.device_get(pre0)
    pre1 = jit_pre(*args)
    lo, hi, va = pre1

    configs: list[tuple[str, str, bool, bool]] = [
        ("planets only", "po", False, False),
        ("planets + board", "pb", True, False),
        ("planets + sun (no board)", "ps", False, True),
        ("full (board + sun)", "full", True, True),
    ]

    ms = batch * 1e-3
    results: list[tuple[str, float]] = []

    for label, tag, ib, isun in configs:
        row = _make_sweep_vmap_row(ib, isun, sps, mb, tag)
        jit_sw = jax.jit(jax.vmap(row))
        t0 = perf_counter()
        _ = jax.device_get(jit_sw(ox, orad, sp, lo, hi, va))
        t_compile = perf_counter() - t0
        t0 = perf_counter()
        _ = jax.device_get(jit_sw(ox, orad, sp, lo, hi, va))
        t_run = perf_counter() - t0
        results.append((label, t_run))
        print(
            f"  {label:28s}  compile {t_compile:.4f}s  run {t_run:.4f}s  ({t_run / ms:.3f} ms/env)  "
            f"[board={ib} sun={isun}]"
        )

    t_po = results[0][1]
    t_pb = results[1][1]
    t_ps = results[2][1]
    t_full = results[3][1]
    print("  --- marginal (same precomputed hits; sampled tick_hit) ---")
    print(
        f"  +board vs planets-only     : {(t_pb - t_po) / ms:+.3f} ms/env  "
        f"({_pct_str(t_pb - t_po, t_po)}% of planets-only)"
    )
    print(
        f"  +sun (no board) vs planets : {(t_ps - t_po) / ms:+.3f} ms/env  "
        f"({_pct_str(t_ps - t_po, t_po)}% of planets-only)"
    )
    print(
        f"  +sun given board (full-pb) : {(t_full - t_pb) / ms:+.3f} ms/env  "
        f"({_pct_str(t_full - t_pb, t_pb)}% of planets+board)"
    )


def bench_first_hit_sequential_vs_parallel_planets(
    batch: int,
    P: int,
    ticks: int,
    samples_per_span: int,
    seed: int,
    max_block_intervals: int,
) -> None:
    """``jit(vmap(first_hit_interval_best_targets_apply_jax))``: ordered inner loop vs same-tick ``vmap``."""

    d = _make_batched_envs(batch, P, ticks=ticks, seed=seed + 8)
    args = (
        d["origin_xy"],
        d["origin_radius"],
        d["speed"],
        d["p0_by_tick"],
        d["p1_by_tick"],
        d["r"],
        d["active_by_tick"],
    )
    gj.tick_hit_intervals_jax = tick_hit_intervals_jax  # type: ignore[assignment]
    sps, mb = samples_per_span, max_block_intervals

    def one_env_ordered(ox, orad, sp, p0t, p1t, rr, at):
        return gj.first_hit_interval_best_targets_apply_jax(
            ox,
            orad,
            sp,
            p0t,
            p1t,
            rr,
            at,
            samples_per_span=sps,
            max_block_intervals=mb,
            same_tick_planets_parallel=False,
        )

    def one_env_parallel(ox, orad, sp, p0t, p1t, rr, at):
        return gj.first_hit_interval_best_targets_apply_jax(
            ox,
            orad,
            sp,
            p0t,
            p1t,
            rr,
            at,
            samples_per_span=sps,
            max_block_intervals=mb,
            same_tick_planets_parallel=True,
        )

    jit_ordered = jax.jit(jax.vmap(one_env_ordered))
    jit_parallel = jax.jit(jax.vmap(one_env_parallel))

    t0 = perf_counter()
    out_o1 = jit_ordered(*args)
    out_o1 = jax.device_get(out_o1)
    t_ord_compile = perf_counter() - t0
    t0 = perf_counter()
    out_o2 = jit_ordered(*args)
    out_o2 = jax.device_get(out_o2)
    t_ord_run = perf_counter() - t0

    t0 = perf_counter()
    out_p1 = jit_parallel(*args)
    out_p1 = jax.device_get(out_p1)
    t_par_compile = perf_counter() - t0
    t0 = perf_counter()
    out_p2 = jit_parallel(*args)
    out_p2 = jax.device_get(out_p2)
    t_par_run = perf_counter() - t0

    ms = batch * 1e-3
    _, w_o, _, ov_o = out_o2
    _, w_p, _, ov_p = out_p2
    max_w_diff = float(jnp.max(jnp.abs(w_o - w_p)))
    ov_o_n = int(np.asarray(ov_o, dtype=np.bool_).sum())
    ov_p_n = int(np.asarray(ov_p, dtype=np.bool_).sum())
    ratio = t_ord_run / max(t_par_run, 1e-12)

    print(
        f"=== first_hit sequential vs same-tick-parallel planets (sampled) "
        f"batch={batch}, P={P}, ticks={ticks} ==="
    )
    print(
        f"  sequential (default): compile {t_ord_compile:.4f}s, run {t_ord_run:.4f}s  "
        f"({t_ord_run / ms:.3f} ms/env)"
    )
    print(
        f"  parallel same-tick    : compile {t_par_compile:.4f}s, run {t_par_run:.4f}s  "
        f"({t_par_run / ms:.3f} ms/env)"
    )
    print(f"  run ratio (sequential / parallel): {ratio:.3f}x")
    print(
        f"  output sanity: max |Δ width|={max_w_diff:.6g}  "
        f"overflow envs ordered={ov_o_n} parallel={ov_p_n} / {batch}"
    )


def bench_union_scan_bins_vs_default_first_hit(
    batch: int,
    P: int,
    ticks: int,
    samples_per_span: int,
    seed: int,
    max_block_intervals: int,
    n_block_bins: int,
) -> None:
    """``first_hit_union_scan_bins_apply_jax`` (binned prefix-OR) vs interval first_hit."""

    d = _make_batched_envs(batch, P, ticks=ticks, seed=seed + 9)
    args = (
        d["origin_xy"],
        d["origin_radius"],
        d["speed"],
        d["p0_by_tick"],
        d["p1_by_tick"],
        d["r"],
        d["active_by_tick"],
    )
    gj.tick_hit_intervals_jax = tick_hit_intervals_jax  # type: ignore[assignment]
    sps, mb = samples_per_span, max_block_intervals

    def one_union(ox, orad, sp, p0t, p1t, rr, at):
        return gj.first_hit_union_scan_bins_apply_jax(
            ox,
            orad,
            sp,
            p0t,
            p1t,
            rr,
            at,
            samples_per_span=sps,
            n_block_bins=n_block_bins,
        )

    def one_interval(ox, orad, sp, p0t, p1t, rr, at):
        return gj.first_hit_interval_best_targets_apply_jax(
            ox,
            orad,
            sp,
            p0t,
            p1t,
            rr,
            at,
            samples_per_span=sps,
            max_block_intervals=mb,
        )

    jit_union = jax.jit(jax.vmap(one_union))
    jit_default = jax.jit(jax.vmap(one_interval))

    t0 = perf_counter()
    u1 = jit_union(*args)
    _ = jax.device_get(u1)
    t_u_compile = perf_counter() - t0
    t0 = perf_counter()
    u2 = jit_union(*args)
    _ = jax.device_get(u2)
    t_u_run = perf_counter() - t0

    t0 = perf_counter()
    d1 = jit_default(*args)
    _ = jax.device_get(d1)
    t_d_compile = perf_counter() - t0
    t0 = perf_counter()
    d2 = jit_default(*args)
    _ = jax.device_get(d2)
    t_d_run = perf_counter() - t0

    ms = batch * 1e-3
    ratio = t_d_run / max(t_u_run, 1e-12)

    print(
        f"=== union-scan bins vs interval first_hit (sampled) "
        f"batch={batch}, P={P}, ticks={ticks}, n_block_bins={n_block_bins} ==="
    )
    print(
        "  (union: no board, sun in pre, binned prefix-OR; interval: board+sun, interval sweep)"
    )
    print(
        f"  union-scan bins     : compile {t_u_compile:.4f}s, run {t_u_run:.4f}s  "
        f"({t_u_run / ms:.3f} ms/env)"
    )
    print(
        f"  interval first_hit  : compile {t_d_compile:.4f}s, run {t_d_run:.4f}s  "
        f"({t_d_run / ms:.3f} ms/env)"
    )
    print(f"  run ratio (interval / union-scan): {ratio:.3f}x")


def bench_brute_rays_vs_default_first_hit(
    batch: int,
    P: int,
    ticks: int,
    samples_per_span: int,
    seed: int,
    max_block_intervals: int,
    n_rays: int,
    n_substeps_per_tick: int,
) -> None:
    """``first_hit_brute_rays_baseline_apply_jax`` vs ``first_hit_interval_best_targets_apply_jax``.

    **Note:** the "brute" side here is **only** the per-ray lex scan (``baseline_apply``).
    Training uses ``first_hit_brute_best_targets_from_rays_apply_jax`` (same baseline sweep,
    then per-planet ``argmin`` over hit tick — cheap vs the old widest-run pass). Wall-time
    training still differs from this bench (PyTorch, DLPack, full rollout).
    """

    d = _make_batched_envs(batch, P, ticks=ticks, seed=seed + 10)
    args = (
        d["origin_xy"],
        d["origin_radius"],
        d["speed"],
        d["p0_by_tick"],
        d["p1_by_tick"],
        d["r"],
        d["active_by_tick"],
    )
    gj.tick_hit_intervals_jax = tick_hit_intervals_jax  # type: ignore[assignment]
    sps, mb = samples_per_span, max_block_intervals

    def one_brute(ox, orad, sp, p0t, p1t, rr, at):
        return gj.first_hit_brute_rays_baseline_apply_jax(
            ox,
            orad,
            sp,
            p0t,
            p1t,
            rr,
            at,
            n_rays=n_rays,
            n_substeps_per_tick=n_substeps_per_tick,
            include_sun=True,
        )

    def one_interval(ox, orad, sp, p0t, p1t, rr, at):
        return gj.first_hit_interval_best_targets_apply_jax(
            ox,
            orad,
            sp,
            p0t,
            p1t,
            rr,
            at,
            samples_per_span=sps,
            max_block_intervals=mb,
        )

    jit_brute = jax.jit(jax.vmap(one_brute))
    jit_default = jax.jit(jax.vmap(one_interval))

    t0 = perf_counter()
    b1 = jit_brute(*args)
    _ = jax.device_get(b1)
    t_b_compile = perf_counter() - t0
    t0 = perf_counter()
    b2 = jit_brute(*args)
    _ = jax.device_get(b2)
    t_b_run = perf_counter() - t0

    t0 = perf_counter()
    d1 = jit_default(*args)
    _ = jax.device_get(d1)
    t_d_compile = perf_counter() - t0
    t0 = perf_counter()
    d2 = jit_default(*args)
    _ = jax.device_get(d2)
    t_d_run = perf_counter() - t0

    ms = batch * 1e-3
    lex, hit = b2
    hit_frac = float(jnp.mean(hit.astype(jnp.float32)))
    # ``hit_any`` includes **board OOB** (endpoint leaves the square), not planets only.
    # On 100×100 with a multi-tick horizon, almost every heading eventually exits unless
    # a planet/sun event occurs first — so this fraction is usually near 1, not "planet hit rate".

    print(
        f"=== brute {n_rays} rays (game swept segment/tick) vs interval first_hit "
        f"(sampled) batch={batch}, P={P}, ticks={ticks} ==="
    )
    print(
        "  (brute: jax_orbit_wars fleet segment vs planet segment, sun segment test, board on a1; "
        "interval: tick_hit occlusion sweep + board/sun)"
    )
    print(
        f"  brute rays          : compile {t_b_compile:.4f}s, run {t_b_run:.4f}s  "
        f"({t_b_run / ms:.3f} ms/env)"
    )
    print(
        f"  interval first_hit  : compile {t_d_compile:.4f}s, run {t_d_run:.4f}s  "
        f"({t_d_run / ms:.3f} ms/env)"
    )
    print(f"  run ratio (interval / brute): {t_d_run / max(t_b_run, 1e-12):.3f}x")
    print(
        f"  brute mean P(any terminal event before horizon | ray, env): {hit_frac:.3f} "
        "(planet OR sun segment OR **board OOB**; usually dominated by OOB on finite maps)"
    )


def main() -> None:
    # Object count mimics "planets per env" scale.
    samples_per_span = 17

    for P in (12, 24, 48):
        origin_xy, origin_radius, speed, tick, p0, p1, r, act = _make_random_case(P, seed=1)

        origin_xy_t = jnp.asarray(origin_xy)
        origin_radius_t = jnp.asarray(origin_radius, dtype=jnp.float32)
        speed_t = jnp.asarray(speed, dtype=jnp.float32)
        tick_t = jnp.asarray(tick, dtype=jnp.int32)

        p0_t = jnp.asarray(p0)
        p1_t = jnp.asarray(p1)
        r_t = jnp.asarray(r)
        act_t = jnp.asarray(act)

        sampled_vmap = jax.jit(
            jax.vmap(
                lambda p0i, p1i, ri, ai: tick_hit_intervals_jax(
                    origin_xy_t,
                    origin_radius_t,
                    speed_t,
                    tick_t,
                    p0i,
                    p1i,
                    ri,
                    ai,
                    samples_per_span=samples_per_span,
                )
            )
        )

        symbolic_vmap = jax.jit(
            jax.vmap(
                lambda p0i, p1i, ri, ai: tick_hit_intervals_symbolic_jax_one(
                    origin_xy_t,
                    origin_radius_t,
                    speed_t,
                    tick_t,
                    p0i,
                    p1i,
                    ri,
                    ai,
                )
            )
        )

        # Compile timing.
        t0 = perf_counter()
        lo_s, hi_s, v_s = sampled_vmap(p0_t, p1_t, r_t, act_t)
        _ = jax.device_get((lo_s, hi_s, v_s))
        t_sampled_compile = perf_counter() - t0

        # Steady-state run.
        t0 = perf_counter()
        lo_s2, hi_s2, v_s2 = sampled_vmap(p0_t, p1_t, r_t, act_t)
        _ = jax.device_get((lo_s2, hi_s2, v_s2))
        t_sampled_run = perf_counter() - t0

        t0 = perf_counter()
        lo_y, hi_y, v_y = symbolic_vmap(p0_t, p1_t, r_t, act_t)
        _ = jax.device_get((lo_y, hi_y, v_y))
        t_symbolic_compile = perf_counter() - t0

        t0 = perf_counter()
        lo_y2, hi_y2, v_y2 = symbolic_vmap(p0_t, p1_t, r_t, act_t)
        _ = jax.device_get((lo_y2, hi_y2, v_y2))
        t_symbolic_run = perf_counter() - t0

        print(f"P={P} objects, tick={tick}, samples_per_span={samples_per_span}")
        print(f"  sampled : compile {t_sampled_compile:.4f}s, run {t_sampled_run:.4f}s")
        print(f"  symbolic: compile {t_symbolic_compile:.4f}s, run {t_symbolic_run:.4f}s")

    print("=== Done (single-env tick sweep) ===")


def _bench_first_hit_best_targets(P: int, ticks: int, samples_per_span: int) -> None:
    rng = np.random.default_rng(2)
    origin_xy = rng.uniform(10.0, 90.0, size=(2,)).astype(np.float32)
    origin_radius = float(rng.uniform(1.0, 4.0))
    speed = float(rng.uniform(1.0, 6.0))

    object_p0_by_tick = rng.uniform(0.0, BOARD_SIZE, size=(ticks, P, 2)).astype(np.float32)
    drift = rng.normal(0.0, 3.0, size=(ticks, P, 2)).astype(np.float32)
    object_p1_by_tick = np.clip(object_p0_by_tick + drift, -5.0, BOARD_SIZE + 5.0).astype(np.float32)
    object_radii = rng.uniform(0.8, 4.5, size=(P,)).astype(np.float32)
    object_active_by_tick = (rng.random(size=(ticks, P)) > 0.15).astype(np.bool_)

    origin_xy_t = jnp.asarray(origin_xy)
    origin_radius_t = jnp.asarray(origin_radius, dtype=jnp.float32)
    speed_t = jnp.asarray(speed, dtype=jnp.float32)
    p0_t = jnp.asarray(object_p0_by_tick)
    p1_t = jnp.asarray(object_p1_by_tick)
    r_t = jnp.asarray(object_radii)
    act_t = jnp.asarray(object_active_by_tick)

    # Sampled run
    jit_sampled = jax.jit(
        lambda: gj.first_hit_interval_best_targets_apply_jax(
            origin_xy_t,
            origin_radius_t,
            speed_t,
            p0_t,
            p1_t,
            r_t,
            act_t,
            samples_per_span=samples_per_span,
            max_block_intervals=64,
        )
    )
    t0 = perf_counter()
    _ = jit_sampled()
    _ = jax.device_get(_)
    t_sampled_compile = perf_counter() - t0

    t0 = perf_counter()
    out = jit_sampled()
    out = jax.device_get(out)
    t_sampled_run = perf_counter() - t0
    _, _, _, overflow_s = out
    overflow_s_sum = int(np.asarray(overflow_s).sum())

    gj.tick_hit_intervals_jax = tick_hit_symbolic_compat_for_bench  # type: ignore[assignment]

    jit_symbolic = jax.jit(
        lambda: gj.first_hit_interval_best_targets_apply_jax(
            origin_xy_t,
            origin_radius_t,
            speed_t,
            p0_t,
            p1_t,
            r_t,
            act_t,
            samples_per_span=samples_per_span,
            max_block_intervals=64,
        )
    )
    t0 = perf_counter()
    out2 = jit_symbolic()
    out2 = jax.device_get(out2)
    t_symbolic_compile = perf_counter() - t0

    t0 = perf_counter()
    out3 = jit_symbolic()
    out3 = jax.device_get(out3)
    t_symbolic_run = perf_counter() - t0
    _, _, _, overflow_y = out3
    overflow_y_sum = int(np.asarray(overflow_y).sum())

    # Restore sampled tick function so subsequent tests behave.
    gj.tick_hit_intervals_jax = tick_hit_intervals_jax  # type: ignore[assignment]

    print("=== first_hit_interval_best_targets_apply_jax micro-benchmark ===")
    print(f"P={P}, ticks={ticks}, samples_per_span={samples_per_span}")
    print(f"  sampled : compile {t_sampled_compile:.4f}s, run {t_sampled_run:.4f}s")
    print(f"  symbolic: compile {t_symbolic_compile:.4f}s, run {t_symbolic_run:.4f}s")
    print(f"  overflow: sampled {overflow_s_sum}, symbolic {overflow_y_sum}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=128, help="vmap batch size (envs), default 128")
    p.add_argument(
        "--platform",
        type=str,
        default="cpu",
        choices=("cpu", "cuda"),
        help="JAX platform (set before import; default cpu)",
    )
    p.add_argument("--P", type=int, default=24, help="planets per env for batched bench (default 24)")
    p.add_argument("--ticks", type=int, default=24, help="horizon ticks for first_hit batched bench")
    p.add_argument("--samples-per-span", type=int, default=17, help="sampled tick_hit resolution")
    p.add_argument(
        "--max-block-intervals",
        type=int,
        default=256,
        help="blocker capacity for first_hit batched bench (default 256; rollout uses 32)",
    )
    p.add_argument("--skip-single-env", action="store_true", help="only run batch-128 benches")
    p.add_argument(
        "--sweep-breakdown",
        action="store_true",
        help="sampled sweep only: time planets-only vs +board vs +sun vs full (needs precompute)",
    )
    p.add_argument(
        "--parallel-vs-sequential-sweep",
        action="store_true",
        help="compare same_tick_planets_parallel=False vs True on batched first_hit (sampled tick_hit)",
    )
    p.add_argument(
        "--union-scan-benchmark",
        action="store_true",
        help="binned prefix-OR union-scan variant vs interval first_hit_best_targets",
    )
    p.add_argument(
        "--n-block-bins",
        type=int,
        default=512,
        help="angular bins for --union-scan-benchmark (default 512)",
    )
    p.add_argument(
        "--brute-ray-baseline",
        action="store_true",
        help="benchmark first_hit_brute_rays_baseline_apply_jax vs default apply_jax",
    )
    p.add_argument("--n-brute-rays", type=int, default=2048, help="rays for --brute-ray-baseline")
    p.add_argument(
        "--n-brute-substeps",
        type=int,
        default=8,
        help="ignored (legacy); brute baseline uses one env step segment per tick like jax_orbit_wars",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    # Platform is applied in ``_early_platform_from_argv`` before JAX import.

    if not args.skip_single_env:
        main()
        _bench_first_hit_best_targets(P=12, ticks=4, samples_per_span=args.samples_per_span)

    P = min(args.P, MAX_PLANETS)
    bench_tick_vmap_over_envs(
        batch=args.batch, P=P, samples_per_span=args.samples_per_span, seed=42
    )
    bench_first_hit_vmap_over_envs(
        batch=args.batch,
        P=P,
        ticks=args.ticks,
        samples_per_span=args.samples_per_span,
        seed=42,
        max_block_intervals=args.max_block_intervals,
    )
    bench_first_hit_stage_breakdown(
        batch=args.batch,
        P=P,
        ticks=args.ticks,
        samples_per_span=args.samples_per_span,
        seed=42,
        max_block_intervals=args.max_block_intervals,
        label="sampled tick_hit",
        symbolic=False,
    )
    bench_first_hit_stage_breakdown(
        batch=args.batch,
        P=P,
        ticks=args.ticks,
        samples_per_span=args.samples_per_span,
        seed=42,
        max_block_intervals=args.max_block_intervals,
        label="symbolic tick_hit",
        symbolic=True,
    )
    if args.sweep_breakdown:
        print(f"=== sweep component breakdown (sampled) batch={args.batch}, P={P}, ticks={args.ticks} ===")
        bench_sweep_sampled_component_breakdown(
            batch=args.batch,
            P=P,
            ticks=args.ticks,
            samples_per_span=args.samples_per_span,
            seed=42,
            max_block_intervals=args.max_block_intervals,
        )
    if args.parallel_vs_sequential_sweep:
        bench_first_hit_sequential_vs_parallel_planets(
            batch=args.batch,
            P=P,
            ticks=args.ticks,
            samples_per_span=args.samples_per_span,
            seed=42,
            max_block_intervals=args.max_block_intervals,
        )
    if args.union_scan_benchmark:
        bench_union_scan_bins_vs_default_first_hit(
            batch=args.batch,
            P=P,
            ticks=args.ticks,
            samples_per_span=args.samples_per_span,
            seed=42,
            max_block_intervals=args.max_block_intervals,
            n_block_bins=max(8, args.n_block_bins),
        )
    if args.brute_ray_baseline:
        bench_brute_rays_vs_default_first_hit(
            batch=args.batch,
            P=P,
            ticks=args.ticks,
            samples_per_span=args.samples_per_span,
            seed=42,
            max_block_intervals=args.max_block_intervals,
            n_rays=max(32, args.n_brute_rays),
            n_substeps_per_tick=max(1, args.n_brute_substeps),
        )


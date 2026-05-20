"""Check JAX toggle-based intersection windows against the NumPy reference."""

from __future__ import annotations

import os
import sys
from time import perf_counter

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import jax
import jax.numpy as jnp

from orbit_wars_pt.geometry_jax import (
    intersection_windows_polyline_jax,
    intersection_windows_stationary_jax,
)
from orbit_wars_pt.tangent_geometry_np import (
    intersection_windows,
    make_polyline_motion,
)


def _pack_jax(lo, hi, valid):
    lo_np, hi_np, v_np = jax.device_get((lo, hi, valid))
    return [(float(a), float(b)) for a, b, v in zip(lo_np, hi_np, v_np) if bool(v)]


def _agree(np_w, j_w, tol=1e-5):
    if len(np_w) != len(j_w):
        return False
    for (a0, b0), (a1, b1) in zip(np_w, j_w):
        if abs(a0 - a1) > tol or abs(b0 - b1) > tol:
            return False
    return True


def _stationary_cases(batch=256, seed=0):
    rng = np.random.default_rng(seed)
    circle_center = rng.uniform(0.0, 100.0, size=(batch, 2)).astype(np.float32)
    grow_center = rng.uniform(0.0, 100.0, size=(batch, 2)).astype(np.float32)
    circle_radius = rng.uniform(0.8, 5.0, size=(batch,)).astype(np.float32)
    grow_rate = rng.uniform(0.2, 8.0, size=(batch,)).astype(np.float32)
    launch_offset = rng.uniform(0.0, 8.0, size=(batch,)).astype(np.float32)
    horizon = rng.uniform(2.0, 24.0, size=(batch,)).astype(np.float32)
    return circle_center, circle_radius, grow_center, grow_rate, launch_offset, horizon


def _polyline_cases(batch=256, points_n=25, seed=1):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 2.5, size=(batch, points_n - 1, 2)).astype(np.float32)
    start = rng.uniform(5.0, 95.0, size=(batch, 1, 2)).astype(np.float32)
    points = np.concatenate([start, start + np.cumsum(steps, axis=1)], axis=1)
    points = np.clip(points, -20.0, 120.0).astype(np.float32)
    circle_radius = rng.uniform(0.8, 5.0, size=(batch,)).astype(np.float32)
    grow_center = rng.uniform(0.0, 100.0, size=(batch, 2)).astype(np.float32)
    grow_rate = rng.uniform(0.2, 8.0, size=(batch,)).astype(np.float32)
    launch_offset = rng.uniform(0.0, 8.0, size=(batch,)).astype(np.float32)
    horizon = rng.uniform(2.0, min(24.0, points_n - 1), size=(batch,)).astype(np.float32)
    return points, circle_radius, grow_center, grow_rate, launch_offset, horizon


def main() -> None:
    stat = _stationary_cases()
    stat_fn = jax.jit(jax.vmap(intersection_windows_stationary_jax))
    args = tuple(jnp.asarray(x) for x in stat)
    warm = stat_fn(*args)
    warm[0].block_until_ready()
    t0 = perf_counter()
    stat_out = stat_fn(*args)
    stat_out[0].block_until_ready()
    stat_s = perf_counter() - t0

    stat_ok = 0
    for i in range(stat[0].shape[0]):
        np_w = intersection_windows(
            lambda _t, c=np.asarray(stat[0][i], dtype=np.float64): c,
            stat[2][i],
            float(stat[1][i]),
            float(stat[4][i]),
            float(stat[3][i]),
            float(stat[5][i]),
            stationary_center=stat[0][i],
        )
        j_w = _pack_jax(stat_out[0][i], stat_out[1][i], stat_out[2][i])
        stat_ok += int(_agree(np_w, j_w))

    poly = _polyline_cases()
    poly_fn = jax.jit(jax.vmap(intersection_windows_polyline_jax))
    poly_args = tuple(jnp.asarray(x) for x in poly)
    warm = poly_fn(*poly_args)
    warm[0].block_until_ready()
    t0 = perf_counter()
    poly_out = poly_fn(*poly_args)
    poly_out[0].block_until_ready()
    poly_s = perf_counter() - t0

    poly_ok = 0
    poly_bad = []
    for i in range(poly[0].shape[0]):
        center_at, _velocity_at, _knot_times = make_polyline_motion(poly[0][i])
        np_w = intersection_windows(
            center_at,
            poly[2][i],
            float(poly[1][i]),
            float(poly[4][i]),
            float(poly[3][i]),
            float(poly[5][i]),
            polyline_points=poly[0][i],
        )
        j_w = _pack_jax(poly_out[0][i], poly_out[1][i], poly_out[2][i])
        same = _agree(np_w, j_w)
        poly_ok += int(same)
        if not same and len(poly_bad) < 5:
            poly_bad.append((i, np_w, j_w))

    print(
        f"stationary windows agree: {stat_ok}/{stat[0].shape[0]} "
        f"({100.0 * stat_ok / stat[0].shape[0]:.1f}%), jax batch {stat_s:.4f}s"
    )
    print(
        f"polyline windows agree:   {poly_ok}/{poly[0].shape[0]} "
        f"({100.0 * poly_ok / poly[0].shape[0]:.1f}%), jax batch {poly_s:.4f}s"
    )
    if poly_bad:
        print("mismatches:")
        for row in poly_bad:
            print(" ", row)


if __name__ == "__main__":
    main()

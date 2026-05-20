"""Check isolated JAX tangency-time solvers against the NumPy reference."""

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
    TANGENT_KIND_EXTERNAL,
    tangent_hit_times_polyline_jax,
    tangent_hit_times_stationary_jax,
)
from orbit_wars_pt.tangent_geometry_np import (
    tangent_hit_time_polyline,
    tangent_hit_time_stationary,
)


def _pack_np_hits(hits):
    if hits is None:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    if isinstance(hits, tuple):
        hits = [hits]
    times = np.asarray([h[0] for h in hits], dtype=np.float32)
    kinds = np.asarray([0 if h[1] == "external" else 1 for h in hits], dtype=np.int32)
    return times, kinds


def _pack_jax_hits(times, kinds, valid):
    t = np.asarray(jax.device_get(times))[np.asarray(jax.device_get(valid))]
    k = np.asarray(jax.device_get(kinds))[np.asarray(jax.device_get(valid))]
    return t.astype(np.float32), k.astype(np.int32)


def _agree(np_t, np_k, j_t, j_k, tol=1e-5):
    if np_t.shape != j_t.shape:
        return False, f"count mismatch np={np_t.shape[0]} jax={j_t.shape[0]}"
    if np_t.shape[0] == 0:
        return True, "empty"
    if not np.all(np_k == j_k):
        return False, f"kind mismatch np={np_k.tolist()} jax={j_k.tolist()}"
    if not np.allclose(np_t, j_t, atol=tol, rtol=tol):
        return False, f"time mismatch np={np_t.tolist()} jax={j_t.tolist()}"
    return True, "ok"


def _stationary_cases(batch=512, seed=0):
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
    stat_fn = jax.jit(jax.vmap(tangent_hit_times_stationary_jax))
    args = tuple(jnp.asarray(x) for x in stat)
    warm = stat_fn(*args)
    warm[0].block_until_ready()
    t0 = perf_counter()
    stat_out = stat_fn(*args)
    stat_out[0].block_until_ready()
    stat_s = perf_counter() - t0

    stat_ok = 0
    stat_bad = []
    for i in range(stat[0].shape[0]):
        np_t, np_k = _pack_np_hits(
            tangent_hit_time_stationary(
                stat[0][i],
                float(stat[1][i]),
                stat[2][i],
                float(stat[3][i]),
                float(stat[4][i]),
                float(stat[5][i]),
                return_all=True,
            )
        )
        j_t, j_k = _pack_jax_hits(stat_out[0][i], stat_out[1][i], stat_out[2][i])
        ok, reason = _agree(np_t, np_k, j_t, j_k)
        stat_ok += int(ok)
        if not ok and len(stat_bad) < 5:
            stat_bad.append((i, reason, np_t, np_k, j_t, j_k))

    poly = _polyline_cases()
    poly_fn = jax.jit(jax.vmap(tangent_hit_times_polyline_jax))
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
        np_t, np_k = _pack_np_hits(
            tangent_hit_time_polyline(
                poly[0][i],
                float(poly[1][i]),
                poly[2][i],
                float(poly[3][i]),
                float(poly[4][i]),
                float(poly[5][i]),
                return_all=True,
            )
        )
        j_t, j_k = _pack_jax_hits(poly_out[0][i], poly_out[1][i], poly_out[2][i])
        ok, reason = _agree(np_t, np_k, j_t, j_k)
        poly_ok += int(ok)
        if not ok and len(poly_bad) < 5:
            poly_bad.append((i, reason, np_t, np_k, j_t, j_k))

    print(
        f"stationary agree: {stat_ok}/{stat[0].shape[0]} "
        f"({100.0 * stat_ok / stat[0].shape[0]:.1f}%), jax batch {stat_s:.4f}s"
    )
    if stat_bad:
        print("stationary mismatches:")
        for row in stat_bad:
            print(" ", row[0], row[1])

    print(
        f"polyline agree:   {poly_ok}/{poly[0].shape[0]} "
        f"({100.0 * poly_ok / poly[0].shape[0]:.1f}%), jax batch {poly_s:.4f}s"
    )
    if poly_bad:
        print("polyline mismatches:")
        for row in poly_bad:
            print(" ", row[0], row[1])

    # A tiny sanity stat: earliest-hit kind mix can be useful when we start wiring this in.
    poly_t = np.asarray(jax.device_get(poly_out[0]))
    poly_k = np.asarray(jax.device_get(poly_out[1]))
    poly_v = np.asarray(jax.device_get(poly_out[2]))
    earliest_valid = poly_v[:, 0]
    earliest_external = np.sum(earliest_valid & (poly_k[:, 0] == TANGENT_KIND_EXTERNAL))
    print(
        f"poly earliest valid: {int(np.sum(earliest_valid))}/{poly_v.shape[0]}, "
        f"external first: {int(earliest_external)}"
    )


if __name__ == "__main__":
    main()

"""Check the JAX sextic-root prototype against the NumPy polyline-extrema reference."""

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
    sextic_stationary_root_candidates_jax,
    stationary_angle_sextic_coeffs_jax,
)
from orbit_wars_pt.tangent_geometry_np import (
    _intersection_angle_derivative_linear,
    _refine_stationary_time,
    _stationary_angle_sextic_coeffs,
)


def _polyline_cases(batch=256, points_n=25, seed=2):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 2.5, size=(batch, points_n - 1, 2)).astype(np.float32)
    start = rng.uniform(5.0, 95.0, size=(batch, 1, 2)).astype(np.float32)
    points = np.concatenate([start, start + np.cumsum(steps, axis=1)], axis=1)
    points = np.clip(points, -20.0, 120.0).astype(np.float32)
    circle_radius = rng.uniform(0.8, 5.0, size=(batch,)).astype(np.float32)
    grow_center = rng.uniform(0.0, 100.0, size=(batch, 2)).astype(np.float32)
    grow_rate = rng.uniform(0.2, 8.0, size=(batch,)).astype(np.float32)
    launch_offset = rng.uniform(0.0, 8.0, size=(batch,)).astype(np.float32)
    seg_idx = rng.integers(0, points_n - 1, size=(batch,), dtype=np.int32)
    seg_lo = seg_idx.astype(np.float32) + rng.uniform(0.0, 0.35, size=(batch,)).astype(np.float32)
    seg_hi = np.minimum(seg_idx.astype(np.float32) + 1.0, seg_lo + rng.uniform(0.3, 1.0, size=(batch,)).astype(np.float32))
    return points, circle_radius, grow_center, grow_rate, launch_offset, seg_idx, seg_lo, seg_hi


def _np_candidates(points, circle_radius, grow_center, grow_rate, launch_offset, seg_idx, seg_lo, seg_hi):
    pts = np.asarray(points, dtype=np.float64)
    i = int(seg_idx)
    p0 = pts[i]
    b = pts[i + 1] - p0
    u_off = float(seg_lo) - float(i)
    q0 = p0 - np.asarray(grow_center, dtype=np.float64) + b * u_off
    r0 = float(launch_offset) + float(grow_rate) * float(seg_lo)
    u_len = float(seg_hi - seg_lo)
    poly_asc = _stationary_angle_sextic_coeffs(q0, b, float(circle_radius), float(grow_rate), r0)
    poly_desc = poly_asc[::-1]
    out = []
    roots = np.roots(poly_desc)
    for z in roots:
        if abs(z.imag) > 1e-6:
            continue
        u = float(z.real)
        if u < -1e-10 or u > u_len + 1e-10:
            continue
        u = min(max(u, 0.0), u_len)
        for br in (-1, 1):
            u_refined = _refine_stationary_time(
                u,
                q0,
                b,
                float(circle_radius),
                r0,
                float(grow_rate),
                br,
                0.0,
                u_len,
                deriv_tol=1e-7,
            )
            if u_refined is None:
                continue
            q = q0 + b * u_refined
            d = _intersection_angle_derivative_linear(
                q, b, float(circle_radius), r0, float(grow_rate), u_refined, br
            )
            if not np.isfinite(d) or abs(d) > 1e-7:
                continue
            out.append((float(u_refined), int(br)))
    out.sort(key=lambda x: (x[0], x[1]))
    dedup = []
    for item in out:
        if dedup and abs(item[0] - dedup[-1][0]) <= 1e-6 and item[1] == dedup[-1][1]:
            continue
        dedup.append(item)
    coeff_desc = poly_desc.astype(np.float32)
    return coeff_desc, q0.astype(np.float32), b.astype(np.float32), np.asarray(dedup, dtype=np.float32)


def main() -> None:
    cases = _polyline_cases()
    coeffs = []
    q0s = []
    bs = []
    rp = []
    r0s = []
    vs = []
    ulens = []
    np_rows = []
    for row in zip(*cases):
        coeff_desc, q0, b, np_row = _np_candidates(*row)
        coeffs.append(coeff_desc)
        q0s.append(q0)
        bs.append(b)
        rp.append(np.float32(row[1]))
        r0s.append(np.float32(row[4] + row[3] * row[6]))
        vs.append(np.float32(row[3]))
        ulens.append(np.float32(row[7] - row[6]))
        np_rows.append(np_row)

    coeffs_j = jnp.asarray(np.stack(coeffs, axis=0))
    q0s_j = jnp.asarray(np.stack(q0s, axis=0))
    bs_j = jnp.asarray(np.stack(bs, axis=0))
    rp_j = jnp.asarray(np.asarray(rp, dtype=np.float32))
    r0s_j = jnp.asarray(np.asarray(r0s, dtype=np.float32))
    vs_j = jnp.asarray(np.asarray(vs, dtype=np.float32))
    ulens_j = jnp.asarray(np.asarray(ulens, dtype=np.float32))

    fn = jax.jit(jax.vmap(sextic_stationary_root_candidates_jax))
    warm = fn(coeffs_j, q0s_j, bs_j, rp_j, r0s_j, vs_j, ulens_j)
    warm[0].block_until_ready()
    t0 = perf_counter()
    out = fn(coeffs_j, q0s_j, bs_j, rp_j, r0s_j, vs_j, ulens_j)
    out[0].block_until_ready()
    jax_s = perf_counter() - t0

    coeff_fn = jax.jit(jax.vmap(stationary_angle_sextic_coeffs_jax))
    warm_coeff = coeff_fn(q0s_j, bs_j, rp_j, vs_j, r0s_j)
    warm_coeff.block_until_ready()
    t0 = perf_counter()
    coeff_out = coeff_fn(q0s_j, bs_j, rp_j, vs_j, r0s_j)
    coeff_out.block_until_ready()
    coeff_s = perf_counter() - t0

    j_t = np.asarray(jax.device_get(out[0]))
    j_b = np.asarray(jax.device_get(out[1]))
    j_v = np.asarray(jax.device_get(out[2]))
    coeff_j = np.asarray(jax.device_get(coeff_out))

    coeff_ok = 0
    coeff_bad = []
    coeff_np = np.stack(coeffs, axis=0)
    for i in range(coeff_np.shape[0]):
        same = np.allclose(coeff_np[i], coeff_j[i], atol=1e-2, rtol=1e-5)
        coeff_ok += int(same)
        if not same and len(coeff_bad) < 5:
            coeff_bad.append((i, coeff_np[i].tolist(), coeff_j[i].tolist()))

    ok = 0
    bad = []
    for i, np_row in enumerate(np_rows):
        jt = j_t[i][j_v[i]]
        jb = j_b[i][j_v[i]]
        if np_row.size == 0:
            np_t = np.zeros((0,), dtype=np.float32)
            np_br = np.zeros((0,), dtype=np.int32)
        else:
            np_t = np_row[:, 0].astype(np.float32)
            np_br = np_row[:, 1].astype(np.int32)
        same = (
            jt.shape == np_t.shape
            and np.all(jb.astype(np.int32) == np_br)
            and np.allclose(jt.astype(np.float32), np_t, atol=1e-5, rtol=1e-5)
        )
        ok += int(same)
        if not same and len(bad) < 8:
            bad.append((i, np_t.tolist(), np_br.tolist(), jt.tolist(), jb.tolist()))

    print(
        f"sextic coeff agree: {coeff_ok}/{len(np_rows)} "
        f"({100.0 * coeff_ok / len(np_rows):.1f}%), jax batch {coeff_s:.4f}s"
    )
    if coeff_bad:
        print("coeff mismatches:")
        for row in coeff_bad:
            print(" ", row[0])
    print(f"sextic candidate agree: {ok}/{len(np_rows)} ({100.0 * ok / len(np_rows):.1f}%), jax batch {jax_s:.4f}s")
    if bad:
        print("mismatches:")
        for row in bad:
            print(" ", row)


if __name__ == "__main__":
    main()

"""Consistency and timing check for Python vs JAX angular geometry."""

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

from orbit_wars_pt.geometry import AngleInterval, first_hit_angle_intervals
from orbit_wars_pt.geometry_jax import (
    first_hit_intervals_jax,
    interval_membership,
    probe_angle_grid,
)


TAU = 2.0 * np.pi


def _swept_pair_hit_np(a0, a1, p0, p1, radius: float) -> bool:
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
    *,
    board_size: float = 100.0,
    sun_radius: float = 10.0,
) -> int:
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
                return obj_idx
        if not (0.0 <= a1[0] <= board_size and 0.0 <= a1[1] <= board_size):
            return -2
        if _point_to_segment_distance_np(np.array([50.0, 50.0]), a0, a1) < sun_radius:
            return -1
    return -3


def _angle_in_intervals(angle: float, intervals: list[AngleInterval], eps: float = 1e-9) -> bool:
    a = angle % TAU
    for iv in intervals:
        width = (iv.hi - iv.lo) % TAU
        if width <= eps and iv.hi > iv.lo:
            return True
        lo = iv.lo % TAU
        hi = iv.hi % TAU
        if lo <= hi:
            if lo - eps <= a <= hi + eps:
                return True
        elif a >= lo - eps or a <= hi + eps:
            return True
    return False


def _interval_mask(angles: np.ndarray, intervals: list[AngleInterval]) -> np.ndarray:
    return np.asarray([_angle_in_intervals(float(a), intervals) for a in angles], dtype=bool)


def _make_cases(batch: int, ticks: int, objects: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    origin_xy = rng.uniform(18.0, 82.0, size=(batch, 2)).astype(np.float32)
    origin_radius = rng.uniform(1.0, 4.0, size=(batch,)).astype(np.float32)
    speed = rng.uniform(1.0, 6.0, size=(batch,)).astype(np.float32)

    p0 = rng.uniform(5.0, 95.0, size=(batch, ticks, objects, 2)).astype(np.float32)
    drift = rng.normal(0.0, 2.5, size=(batch, ticks, objects, 2)).astype(np.float32)
    p1 = np.clip(p0 + drift, -5.0, 105.0).astype(np.float32)
    radii = rng.uniform(0.8, 4.5, size=(batch, objects)).astype(np.float32)
    active = rng.random(size=(batch, ticks, objects)) > 0.15
    target_idx = rng.integers(0, objects, size=(batch,), dtype=np.int32)

    # Add one deliberately easy target per case so agreement is not only on
    # empty masks: place the target near the first few reachable radii.
    for b in range(batch):
        t = int(target_idx[b])
        angle = rng.uniform(0.0, TAU)
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        tick = rng.integers(1, max(2, ticks // 2 + 1))
        rho = origin_radius[b] + 0.1 + (tick + 0.5) * speed[b]
        center = origin_xy[b] + rho * direction
        p0[b, tick, t] = center
        p1[b, tick, t] = center + rng.normal(0.0, 0.5, size=(2,))
        radii[b, t] = 2.5
        active[b, tick, t] = True

    return origin_xy, origin_radius, speed, p0, p1, radii, active, target_idx


def _python_masks(origin_xy, origin_radius, speed, p0, p1, radii, active, target_idx, angles):
    masks = []
    for b in range(origin_xy.shape[0]):
        intervals = first_hit_angle_intervals(
            origin_xy[b],
            float(origin_radius[b]),
            float(speed[b]),
            p0[b],
            p1[b],
            radii[b],
            active[b],
            int(target_idx[b]),
            max_depth=9,
        )
        masks.append(_interval_mask(angles, intervals))
    return np.stack(masks, axis=0)


def main() -> None:
    batch = 8
    ticks = 6
    objects = 8
    num_angles = 720

    origin_xy, origin_radius, speed, p0, p1, radii, active, target_idx = _make_cases(
        batch, ticks, objects
    )
    angles_np = np.asarray(jax.device_get(probe_angle_grid(num_angles)))

    jax_fn = jax.jit(
        jax.vmap(
            lambda ox, orad, sp, p0b, p1b, rb, ab, tb: first_hit_intervals_jax(
                ox, orad, sp, p0b, p1b, rb, ab, tb, samples_per_span=33
            )
        )
    )
    args_j = (
        jnp.asarray(origin_xy),
        jnp.asarray(origin_radius),
        jnp.asarray(speed),
        jnp.asarray(p0),
        jnp.asarray(p1),
        jnp.asarray(radii),
        jnp.asarray(active),
        jnp.asarray(target_idx),
    )

    compiled = jax_fn(*args_j)
    compiled[0].block_until_ready()

    t0 = perf_counter()
    j_lo, j_hi, j_valid, j_overflow = jax_fn(*args_j)
    j_lo.block_until_ready()
    jax_s = perf_counter() - t0
    jax_mask = np.asarray(
        jax.device_get(
            jax.vmap(interval_membership, in_axes=(None, 0, 0, 0))(
                jnp.asarray(angles_np), j_lo, j_hi, j_valid
            )
        )
    )

    t0 = perf_counter()
    py_mask = _python_masks(
        origin_xy,
        origin_radius,
        speed,
        p0,
        p1,
        radii,
        active,
        target_idx,
        angles_np,
    )
    py_s = perf_counter() - t0

    agree = py_mask == jax_mask
    mismatches = int((~agree).sum())
    total = int(agree.size)
    print(
        f"consistency: {total - mismatches}/{total} "
        f"({100.0 * (total - mismatches) / total:.3f}%) probe-angle labels agree"
    )
    print(f"python interval+mask: {py_s:.4f}s for batch={batch}")
    print(f"jax interval tensors: {jax_s:.4f}s for batch={batch}")
    if jax_s > 0:
        print(f"speedup: {py_s / jax_s:.2f}x")

    per_case = agree.mean(axis=1)
    print(
        f"per-case agreement: min={per_case.min():.4f} "
        f"mean={per_case.mean():.4f} max={per_case.max():.4f}"
    )
    print(f"overflow cases: {int(np.asarray(jax.device_get(j_overflow)).sum())}/{batch}")

    mismatch_idx = np.argwhere(~agree)
    if mismatch_idx.size:
        py_right = 0
        jax_right = 0
        neither = 0
        print("mismatch audit:")
        for row, (b, a_idx) in enumerate(mismatch_idx[:40]):
            angle = float(angles_np[a_idx])
            exact = _exact_first_hit_np(
                origin_xy[b],
                float(origin_radius[b]),
                float(speed[b]),
                p0[b],
                p1[b],
                radii[b],
                active[b],
                angle,
            )
            target = int(target_idx[b])
            exact_target = exact == target
            py_label = bool(py_mask[b, a_idx])
            jax_label = bool(jax_mask[b, a_idx])
            py_right += py_label == exact_target
            jax_right += jax_label == exact_target
            neither += (py_label != exact_target) and (jax_label != exact_target)
            print(
                f"  #{row:02d} case={b} angle_idx={a_idx} angle={angle:.6f} "
                f"target={target} exact={exact} py={py_label} jax={jax_label}"
            )
        print(f"mismatch verdict: python_right={py_right} jax_right={jax_right} neither={neither}")


if __name__ == "__main__":
    main()

"""Compare CPU interval targets vs JAX ``first_hit_interval_best_targets_apply_jax``."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import jax
import jax.numpy as jnp

from orbit_wars_pt import geometry_jax as gj
from orbit_wars_pt.interval_geometry_np import first_hit_interval_best_targets_np


def _make_case(seed: int = 0, ticks: int = 6, planets: int = 10):
    rng = np.random.default_rng(seed)
    origin_xy = rng.uniform(20.0, 80.0, size=(2,)).astype(np.float32)
    origin_radius = np.float32(rng.uniform(1.5, 4.0))
    speed = np.float32(rng.uniform(1.5, 5.0))
    p0 = rng.uniform(15.0, 85.0, size=(ticks, planets, 2)).astype(np.float32)
    p1 = p0 + rng.uniform(-2.0, 2.0, size=(ticks, planets, 2)).astype(np.float32)
    active = np.ones((ticks, planets), dtype=bool)
    radii = rng.uniform(1.2, 4.0, size=(planets,)).astype(np.float32)
    return origin_xy, origin_radius, speed, p0, p1, radii, active


def main() -> None:
    samples = 9
    origin_xy, origin_radius, speed, p0, p1, radii, active = _make_case()
    order = list(range(int(radii.shape[0])))

    np_angle, np_width, np_valid, np_overflow, np_tick = first_hit_interval_best_targets_np(
        origin_xy,
        float(origin_radius),
        float(speed),
        p0,
        p1,
        radii,
        active,
        object_order=order,
        samples_per_span=samples,
    )

    j_fn = jax.jit(
        lambda ox, orad, sp, p0b, p1b, rb, ab: gj.first_hit_interval_best_targets_apply_jax(
            jnp.asarray(ox),
            jnp.asarray(orad),
            jnp.asarray(sp),
            jnp.asarray(p0b),
            jnp.asarray(p1b),
            jnp.asarray(rb),
            jnp.asarray(ab),
            object_order=order,
            samples_per_span=samples,
        )
    )
    j_angle, j_width, j_valid, j_overflow = j_fn(
        origin_xy, origin_radius, speed, p0, p1, radii, active
    )
    j_angle = np.asarray(jax.device_get(j_angle))
    j_width = np.asarray(jax.device_get(j_width))
    j_valid = np.asarray(jax.device_get(j_valid))

    angle_ok = np.allclose(np_angle, j_angle, atol=2e-3, rtol=0.0) | ~(np_valid & j_valid)
    width_ok = np.allclose(np_width, j_width, atol=2e-3, rtol=0.0) | ~(np_valid & j_valid)
    valid_ok = np_valid == j_valid
    mism = int(np.sum(~(angle_ok & width_ok & valid_ok)))
    print(
        f"interval np vs jax: planets={radii.shape[0]} ticks={p0.shape[0]} "
        f"samples={samples} mismatches={mism}/{radii.shape[0]}"
    )
    if mism:
        idx = np.where(~(angle_ok & width_ok & valid_ok))[0]
        for i in idx[:5]:
            print(
                f"  slot {i}: np ang={np_angle[i]:.4f} w={np_width[i]:.4f} v={np_valid[i]} "
                f"jax ang={j_angle[i]:.4f} w={j_width[i]:.4f} v={j_valid[i]}"
            )
        raise SystemExit(1)
    print("ok")


if __name__ == "__main__":
    main()

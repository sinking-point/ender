"""Compile-friendly real-map benchmark for rays vs analytic tangent pipeline.

This benchmark is shaped to resemble rollout geometry inputs:

- start from batched JAX env state
- forecast future planet/comet paths with the same path-only replay used in rollout
- choose one origin per env from the current state
- benchmark:
  - current ray first-hit kernel
  - fixed-shape analytic pipeline built from
    tangency hits -> toggle windows -> angle extrema

The analytic path uses fixed-capacity tensors over all env/target slots. It
avoids host-side variable-length packing and avoids creating new ``jax.jit``
wrappers inside timed loops.

Caveat: this is still a benchmark harness, not rollout integration. It is meant
to answer "is the compiled kernel shape promising?" rather than "is the whole
policy path finished?"

Example:

    ./.venv/bin/python -m orbit_wars_pt.bench_tangent_pipeline_real --platform cuda --num-envs 256
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
    p.add_argument("--num-envs", type=int, default=256)
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

from jax_orbit_wars import PLANET_OWNER, PLANET_RADIUS, PLANET_SHIPS, PLANET_X, PLANET_Y
from orbit_wars_pt.batched_env import stack_initial_states
from orbit_wars_pt.env_wrapper import OrbitWarsEnvConfig
from orbit_wars_pt.geometry_jax import (
    first_hit_best_targets_apply_jax,
    intersection_windows_polyline_jax,
    intersection_windows_stationary_jax,
    sextic_stationary_root_candidates_batch_jax,
    stationary_angle_sextic_coeffs_jax,
)
from orbit_wars_pt.jax_setup import configure_jax_for_training
from orbit_wars_pt.micro_jax import _fleet_speed_jax, _forecast_planet_paths_one_tick


TAU = 2.0 * jnp.pi
WINDOW_CAP = 6
ROOT_CAP = 12


def _norm_angle(a: jnp.ndarray) -> jnp.ndarray:
    return jnp.mod(a, TAU)


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


def _polyline_window_segment_geometry(
    points: jnp.ndarray,
    lo: jnp.ndarray,
    hi: jnp.ndarray,
    valid: jnp.ndarray,
    grow_center: jnp.ndarray,
    grow_rate: jnp.ndarray,
    launch_offset: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    horizon = points.shape[0] - 1
    cand_cap = 4 + horizon * ROOT_CAP

    def one_window(lo, hi, valid):
        tm = 0.5 * (lo + hi)
        i_mid = jnp.clip(jnp.floor(tm).astype(jnp.int32), 0, points.shape[0] - 2)
        u_mid = tm - i_mid.astype(tm.dtype)
        c_mid = points[i_mid] + u_mid * (points[i_mid + 1] - points[i_mid])
        r_mid = launch_offset + grow_rate * tm
        am_mid, ap_mid, ok_mid = _intersection_angles(c_mid - grow_center, circle_radius, r_mid)
        ref = jnp.where(ok_mid, 0.5 * (am_mid + ap_mid), jnp.arctan2(c_mid[1] - grow_center[1], c_mid[0] - grow_center[0]))

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

        q0_b, b_b, r0_b, u_len_b, use_b = _polyline_window_segment_geometry(
            points, lo, hi, valid, grow_center, grow_rate, launch_offset
        )
        coeff_b = jax.vmap(
            lambda q0, b, r0: stationary_angle_sextic_coeffs_jax(
                q0, b, circle_radius, grow_rate, r0
            )
        )(q0_b, b_b, r0_b)
        t_s, br_s, rv_s = sextic_stationary_root_candidates_batch_jax(
            coeff_b, q0_b, b_b, circle_radius, r0_b, grow_rate, u_len_b
        )

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

    return jax.vmap(one_window)(win_lo, win_hi, win_valid)


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

    return (
        centers.reshape(envs * planets, 2),
        points.reshape(envs * planets, ticks + 1, 2),
        object_radii.reshape(envs * planets),
        origin_rep.reshape(envs * planets, 2),
        speed_rep.reshape(envs * planets),
        launch_rep.reshape(envs * planets),
        horizon_rep.reshape(envs * planets),
        stationary_mask.reshape(envs * planets),
        moving_mask.reshape(envs * planets),
        active_any.reshape(envs * planets),
    )


def _analytic_full_so_far(
    centers: jnp.ndarray,
    points: jnp.ndarray,
    radii: jnp.ndarray,
    origin_xy: jnp.ndarray,
    speed: jnp.ndarray,
    launch_offset: jnp.ndarray,
    horizon: jnp.ndarray,
    stationary_mask: jnp.ndarray,
    moving_mask: jnp.ndarray,
    active_any: jnp.ndarray,
):
    stat_windows = jax.vmap(intersection_windows_stationary_jax)(
        centers, radii, origin_xy, speed, launch_offset, horizon
    )
    move_windows = jax.vmap(intersection_windows_polyline_jax)(
        points, radii, origin_xy, speed, launch_offset, horizon
    )

    stat_w_lo, stat_w_hi, stat_w_valid = stat_windows
    move_w_lo, move_w_hi, move_w_valid = move_windows
    stat_w_valid = stat_w_valid & stationary_mask[:, None]
    move_w_valid = move_w_valid & moving_mask[:, None]

    stat_ext = jax.vmap(_stationary_window_extrema)(
        centers,
        radii,
        origin_xy,
        speed,
        launch_offset,
        stat_w_lo,
        stat_w_hi,
        stat_w_valid,
    )
    move_ext = jax.vmap(_polyline_window_extrema)(
        points,
        radii,
        origin_xy,
        speed,
        launch_offset,
        move_w_lo,
        move_w_hi,
        move_w_valid,
    )

    stat_min, stat_max, stat_valid = stat_ext
    move_min, move_max, move_valid = move_ext
    any_valid = (jnp.any(stat_valid, axis=1) | jnp.any(move_valid, axis=1)) & active_any
    total_windows = jnp.sum(stat_w_valid.astype(jnp.int32)) + jnp.sum(move_w_valid.astype(jnp.int32))

    return {
        "stat_window_count": jnp.sum(stat_w_valid.astype(jnp.int32)),
        "move_window_count": jnp.sum(move_w_valid.astype(jnp.int32)),
        "total_window_count": total_windows,
        "any_valid_target_count": jnp.sum(any_valid.astype(jnp.int32)),
        "stat_min_checksum": jnp.sum(jnp.where(stat_valid, stat_min, 0.0)),
        "move_min_checksum": jnp.sum(jnp.where(move_valid, move_min, 0.0)),
        "stat_max_checksum": jnp.sum(jnp.where(stat_valid, stat_max, 0.0)),
        "move_max_checksum": jnp.sum(jnp.where(move_valid, move_max, 0.0)),
    }


ANALYTIC_FULL_SO_FAR_JIT = jax.jit(_analytic_full_so_far)


def _bench(label: str, fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    compile_run_s = time.perf_counter() - t0

    times = []
    for _ in range(ARGS.repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)

    return {
        "label": label,
        "compile_run_s": compile_run_s,
        "mean_run_s": float(np.mean(times)),
        "min_run_s": float(np.min(times)),
        "out": out,
    }


def main() -> None:
    configure_jax_for_training(prefer_gpu=True, verbose=False)
    print(f"devices {jax.devices()}")

    cfg = OrbitWarsEnvConfig(
        num_agents=2,
        max_fleets=int(ARGS.max_fleets),
        episode_seed=int(ARGS.seed),
    )
    state_b, _ = stack_initial_states(cfg, int(ARGS.num_envs), int(ARGS.seed))

    forecast_row = _bench("forecast", _forecast_batched, state_b, int(ARGS.horizon))
    p0_b, p1_b, active_b = forecast_row["out"]

    origin_idx = _choose_origin_idx(state_b)
    ships = state_b.planets[jnp.arange(int(ARGS.num_envs)), origin_idx, PLANET_SHIPS]
    send = jnp.floor(jnp.asarray(float(ARGS.fraction), dtype=jnp.float32) * ships)
    speed = _fleet_speed_jax(send, float(ARGS.ship_speed))
    origin_xy = state_b.planets[jnp.arange(int(ARGS.num_envs)), origin_idx][:, PLANET_X : PLANET_Y + 1]
    origin_radius = state_b.planets[jnp.arange(int(ARGS.num_envs)), origin_idx, PLANET_RADIUS]
    launch_offset = origin_radius + jnp.asarray(0.1, dtype=jnp.float32)
    object_radii = state_b.planets[:, :, PLANET_RADIUS]
    policy_mask = state_b.planet_active

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
    ray_row = _bench(
        "rays",
        rays_fn,
        origin_xy,
        origin_radius,
        speed,
        p0_b,
        p1_b,
        object_radii,
        active_b,
        policy_mask,
    )

    analytic_inputs = _prepare_targets(
        p0_b,
        p1_b,
        active_b,
        origin_xy,
        speed,
        launch_offset,
        object_radii,
    )
    analytic_row = _bench("analytic_full_so_far", ANALYTIC_FULL_SO_FAR_JIT, *analytic_inputs)

    analytic_np = jax.device_get(analytic_row["out"])
    stationary_mask_np = np.asarray(jax.device_get(analytic_inputs[7]))
    moving_mask_np = np.asarray(jax.device_get(analytic_inputs[8]))

    print(
        f"real analytic-vs-rays benchmark envs={ARGS.num_envs} horizon={ARGS.horizon} "
        f"rays={ARGS.n_rays} repeat={ARGS.repeat}"
    )
    print(
        f"targets stationary={int(stationary_mask_np.sum())} "
        f"moving={int(moving_mask_np.sum())} total={int((stationary_mask_np | moving_mask_np).sum())}"
    )
    print(
        f"analytic windows stationary={int(analytic_np['stat_window_count'])} "
        f"moving={int(analytic_np['move_window_count'])} "
        f"total={int(analytic_np['total_window_count'])}"
    )
    print("name compile+run_s mean_run_s min_run_s per_env_ms")
    for row in (forecast_row, analytic_row, ray_row):
        print(
            f"{row['label']:>20} "
            f"{row['compile_run_s']:>13.4f} "
            f"{row['mean_run_s']:>10.4f} "
            f"{row['min_run_s']:>9.4f} "
            f"{1000.0 * row['mean_run_s'] / float(ARGS.num_envs):>10.3f}"
        )


if __name__ == "__main__":
    main()

"""Ad-hoc parity check: host ``build_observation`` vs JAX ``build_observation_batched_jax``.

Run with::

    python -m orbit_wars_pt.check_observation_jax

Reports max abs diff per field for a fresh 2-agent reset and a state with a
few queued fleets.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import jax
import jax.numpy as jnp

from orbit_wars_pt.batched_env import stack_initial_states
from orbit_wars_pt.env_wrapper import OrbitWarsEnvConfig
from orbit_wars_pt.observation import build_observation, jax_state_to_numpy
from orbit_wars_pt.observation_jax import build_observation_batched_jax


FIELDS = ("entity_type", "owner_idx", "features", "rope_pos", "entity_mask", "planet_mask")


def _slice_jax_obs(obs_jax: Dict[str, jnp.ndarray], env_idx: int) -> Dict[str, np.ndarray]:
    obs_np = jax.device_get(obs_jax)
    return {k: np.asarray(v[env_idx]) for k, v in obs_np.items()}


def _host_obs_dict(state_np, ego: int, ship_speed: float = 6.0) -> Dict[str, np.ndarray]:
    obs = build_observation(state_np, ego, ship_speed=ship_speed)
    # Host builder emits a variable-length sequence; pad to JAX builder's fixed L
    # so the per-field comparison sees aligned shapes.
    from orbit_wars_pt.constants import MAX_PLANETS
    target_L = 1 + MAX_PLANETS
    L = len(obs.entity_type)
    if L < target_L:
        pad = target_L - L
        et = np.pad(obs.entity_type, (0, pad), constant_values=0)
        oi = np.pad(obs.owner_idx, (0, pad), constant_values=0)
        ft = np.pad(obs.features, ((0, pad), (0, 0)), constant_values=0.0)
        rp = np.pad(obs.rope_pos, ((0, pad), (0, 0)), constant_values=0.0)
        em = np.pad(obs.entity_mask, (0, pad), constant_values=False)
        pm = np.pad(obs.planet_mask, (0, pad), constant_values=False)
    elif L == target_L:
        et, oi, ft, rp, em, pm = (
            obs.entity_type, obs.owner_idx, obs.features, obs.rope_pos, obs.entity_mask, obs.planet_mask,
        )
    else:
        raise AssertionError(f"Host obs is longer than JAX obs ({L} > {target_L}). MAX_FLEET_TOKENS too small?")
    return {
        "entity_type": np.asarray(et),
        "owner_idx": np.asarray(oi),
        "features": np.asarray(ft),
        "rope_pos": np.asarray(rp),
        "entity_mask": np.asarray(em),
        "planet_mask": np.asarray(pm),
    }


def _diff_report(host: Dict[str, np.ndarray], jax_: Dict[str, np.ndarray], tag: str) -> bool:
    print(f"\n=== {tag} ===")
    ok = True
    for k in FIELDS:
        h, j = host[k], jax_[k]
        if h.shape != j.shape:
            print(f"  {k}: SHAPE MISMATCH  host={h.shape} jax={j.shape}")
            ok = False
            continue
        if h.dtype.kind == "b" or j.dtype.kind == "b":
            mismatch = int(np.sum(h.astype(bool) != j.astype(bool)))
            print(f"  {k}: bool mismatches = {mismatch} / {h.size}")
            if mismatch > 0:
                ok = False
        elif h.dtype.kind in ("i", "u"):
            mismatch = int(np.sum(h.astype(np.int64) != j.astype(np.int64)))
            print(f"  {k}: int mismatches = {mismatch} / {h.size}")
            if mismatch > 0:
                ok = False
        else:
            diff = np.abs(h.astype(np.float64) - j.astype(np.float64))
            max_diff = float(diff.max()) if diff.size else 0.0
            mean_diff = float(diff.mean()) if diff.size else 0.0
            note = ""
            if k == "rope_pos":
                eta_diff = float(np.abs(h[..., 2].astype(np.float64) - j[..., 2].astype(np.float64)).max())
                xy_diff = float(np.abs(h[..., :2].astype(np.float64) - j[..., :2].astype(np.float64)).max())
                note = f"  (xy max {xy_diff:.6g}, eta max {eta_diff:.6g})"
            print(f"  {k}: max abs diff = {max_diff:.6g}, mean = {mean_diff:.6g}{note}")
            if max_diff > 1e-5:
                ok = False
    print(f"  -> {'PASS' if ok else 'FAIL (see above)'}")
    return ok


def main() -> int:
    cfg = OrbitWarsEnvConfig(num_agents=2, max_fleets=128, episode_seed=0)

    # Fresh reset, no fleets.
    state_b, _ = stack_initial_states(cfg, num_envs=1, seed_base=0)
    state_np = jax_state_to_numpy(jax.tree.map(lambda x: x[0], state_b))

    all_ok = True
    for ego in (0, 1):
        host = _host_obs_dict(state_np, ego)
        obs_jax = build_observation_batched_jax(state_b, ego, 6.0)
        jax_ = _slice_jax_obs(obs_jax, 0)
        ok = _diff_report(host, jax_, f"fresh reset, ego={ego}")
        all_ok = all_ok and ok

    # Inject a couple of synthetic queued arrivals.
    state_one = jax.tree.map(lambda x: x[0], state_b)
    planets_np = np.asarray(state_one.planets)
    pa_np = np.asarray(state_one.planet_active)
    owned_idx = next((i for i in range(planets_np.shape[0]) if pa_np[i] and int(planets_np[i, 1]) == 0), None)
    enemy_idx = next((i for i in range(planets_np.shape[0]) if pa_np[i] and int(planets_np[i, 1]) == 1), None)
    if owned_idx is None or enemy_idx is None:
        print("Could not find self/enemy planets; skipping fleet test")
    else:
        incoming_np = np.asarray(state_one.incoming_fleets).copy()
        for slot, ships in enumerate((5.0, 12.0, 30.0)):
            incoming_np[0, enemy_idx, 10 + slot] = int(ships)
        state_one = state_one._replace(
            incoming_fleets=jnp.asarray(incoming_np),
        )
        state_b2 = jax.tree.map(lambda x: x[None, ...], state_one)
        state_np2 = jax_state_to_numpy(state_one)
        for ego in (0, 1):
            host = _host_obs_dict(state_np2, ego)
            obs_jax = build_observation_batched_jax(state_b2, ego, 6.0)
            jax_ = _slice_jax_obs(obs_jax, 0)
            ok = _diff_report(host, jax_, f"3 queued fleets, ego={ego}")
            all_ok = all_ok and ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

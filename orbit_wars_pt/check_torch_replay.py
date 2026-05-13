"""Consistency check for Torch-only PPO replay against the JAX replay path.

Runs on CPU by default so it is cheap and avoids CUDA allocator noise.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import jax
import jax.numpy as jnp

from jax_orbit_wars import OrbitWarsState, PLANET_X, PLANET_Y

from orbit_wars_pt.batched_env import stack_initial_states
from orbit_wars_pt.constants import MAX_PLANETS
from orbit_wars_pt.env_wrapper import OrbitWarsEnvConfig
from orbit_wars_pt.micro_jax import apply_micro_step_batched, apply_prefix_micro_deltas_batched, compute_pair_geom_and_etas
from orbit_wars_pt.observation_jax import build_observation_batched_jax_per_ego
from orbit_wars_pt.torch_replay import apply_prefix_micro_deltas_torch, build_observation_torch


def _state_to_torch(state: OrbitWarsState, device: torch.device) -> OrbitWarsState:
    return OrbitWarsState(
        **{
            field: torch.as_tensor(np.asarray(jax.device_get(getattr(state, field))), device=device)
            for field in OrbitWarsState._fields
        }
    )


def _torch_to_np_tree(state: OrbitWarsState) -> dict[str, np.ndarray]:
    return {field: getattr(state, field).detach().cpu().numpy() for field in OrbitWarsState._fields}


def _first_valid_pair(pair_geom: np.ndarray, env_idx: int) -> tuple[int, int]:
    flat = np.flatnonzero(pair_geom[env_idx].reshape(-1))
    if flat.size == 0:
        raise AssertionError(f"env {env_idx} has no valid pair")
    k = int(flat[0])
    return k // MAX_PLANETS, k % MAX_PLANETS


def _assert_close(name: str, a, b, *, atol: float = 1e-5) -> None:
    an = np.asarray(a)
    bn = np.asarray(b)
    if an.dtype == np.bool_ or np.issubdtype(an.dtype, np.integer):
        ok = np.array_equal(an, bn)
        diff = 0
    else:
        ok = np.allclose(an, bn, atol=atol, rtol=1e-5)
        diff = float(np.max(np.abs(an - bn))) if an.size else 0.0
    print(f"{name}: shape={an.shape} ok={ok} max_abs={diff}")
    if not ok:
        raise AssertionError(f"{name} mismatch max_abs={diff}")


def main() -> None:
    device = torch.device("cpu")
    cfg = OrbitWarsEnvConfig(num_agents=2, max_fleets=32, episode_seed=0)
    state_b, _ = stack_initial_states(cfg, num_envs=2, seed_base=0)
    pair_geom, _etas = compute_pair_geom_and_etas(state_b, ship_speed=6.0)
    pair_np = np.asarray(jax.device_get(pair_geom))

    pair_flat = np.zeros((2, 2), dtype=np.int32)
    frac_idx = np.zeros((2, 2), dtype=np.int32)
    angle = np.zeros((2, 2), dtype=np.float32)
    fleet_eta = np.zeros((2, 2), dtype=np.float32)
    micro_halt = np.ones((2, 2), dtype=np.bool_)
    slot = np.full((2, 2), -1, dtype=np.int32)
    send = np.zeros((2, 2), dtype=np.float32)

    for env_i in range(2):
        o, d = _first_valid_pair(pair_np, env_i)
        pair_flat[env_i, 0] = o * MAX_PLANETS + d
        frac_idx[env_i, 0] = 4
        planets_i = np.asarray(jax.device_get(state_b.planets[env_i]))
        diff = planets_i[d, PLANET_X : PLANET_Y + 1] - planets_i[o, PLANET_X : PLANET_Y + 1]
        angle[env_i, 0] = float(np.arctan2(diff[1], diff[0]))
        micro_halt[env_i, 0] = False

    # Use JAX's rollout micro-step once to get authoritative send/slot metadata.
    state_after, _oid, angle0, send0, _dispatched, slot0 = apply_micro_step_batched(
        state_b,
        jnp.int32(0),
        jnp.asarray(micro_halt[:, 0]),
        jnp.asarray(pair_flat[:, 0]),
        jnp.asarray(frac_idx[:, 0]),
        jnp.asarray(angle[:, 0]),
        jnp.asarray(fleet_eta[:, 0]),
    )
    del state_after
    send[:, 0] = np.asarray(jax.device_get(send0))
    slot[:, 0] = np.asarray(jax.device_get(slot0))
    angle[:, 0] = np.asarray(jax.device_get(angle0))

    ego = np.asarray([0, 1], dtype=np.int32)
    phase = np.asarray([1, 1], dtype=np.int32)
    apply_mask = np.arange(2, dtype=np.int32)[None, :] < phase[:, None]

    state_jax = apply_prefix_micro_deltas_batched(
        state_b,
        jnp.asarray(ego),
        2,
        jnp.asarray(micro_halt),
        jnp.asarray(send),
        jnp.asarray(slot),
        jnp.asarray(pair_flat),
        jnp.asarray(frac_idx),
        jnp.asarray(angle),
        jnp.asarray(fleet_eta),
        jnp.asarray(apply_mask),
    )
    state_t = apply_prefix_micro_deltas_torch(
        _state_to_torch(state_b, device),
        torch.as_tensor(ego, device=device),
        torch.as_tensor(micro_halt, device=device),
        torch.as_tensor(send, device=device),
        torch.as_tensor(slot, device=device),
        torch.as_tensor(pair_flat, device=device),
        torch.as_tensor(angle, device=device),
        torch.as_tensor(fleet_eta, device=device),
        torch.as_tensor(phase, device=device),
    )

    jax_np = {field: np.asarray(jax.device_get(getattr(state_jax, field))) for field in OrbitWarsState._fields}
    torch_np = _torch_to_np_tree(state_t)
    for field in ("planets", "incoming_fleets"):
        _assert_close(f"state.{field}", torch_np[field], jax_np[field])

    obs_jax = build_observation_batched_jax_per_ego(state_jax, jnp.asarray(ego), 6.0)
    obs_t = build_observation_torch(state_t, torch.as_tensor(ego, device=device), 6.0)
    for key, value_j in obs_jax.items():
        _assert_close(f"obs.{key}", obs_t[key].detach().cpu().numpy(), np.asarray(jax.device_get(value_j)))

    print("Torch replay consistency check passed.")


if __name__ == "__main__":
    main()

"""Batched JAX environment helpers: stacked state, vmap'd step, batched fleet expansion + scores.

Phase 1 of the JAX-exploitation rework: the rollout keeps a single batched
``OrbitWarsState`` on device for the whole episode. ``step`` is vmap'd across
``num_envs`` so each turn issues one fused JAX program rather than ``num_envs``
sequential ones, and host syncs collapse to "once per turn" instead of "once
per env per phase".

Observation building, the intra-turn ``virt`` state, and policy forward still
live on host — those move to JAX in later phases.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import jax
import jax.numpy as jnp
import torch

import jax_orbit_wars as jow
from jax_orbit_wars import OrbitWarsConfig, OrbitWarsState

from orbit_wars_pt.env_wrapper import OrbitWarsEnvConfig
from orbit_wars_pt.scores_jax import _ship_ratio_scores_core, _ship_totals_p01_core


_DEFAULT_STEP_CFG = OrbitWarsConfig()

# vmap leading num_envs axis on (state, actions); broadcast the static config.
_vmapped_step = jax.jit(jax.vmap(jow.step, in_axes=(0, 0, None), out_axes=0))


def reset_env_at_index(
    state_b: OrbitWarsState,
    env_idx: int,
    seed: int,
    cfg: OrbitWarsEnvConfig,
    *,
    fresh_np: Optional[Any] = None,
) -> OrbitWarsState:
    """Replace batch slice ``env_idx`` with a fresh ``reset_from_reference`` game.

    If ``fresh_np`` is set (NumPy tree from ``jax.device_get``), skips in-process
    Kaggle reset and only materializes JAX arrays for scatter.
    """

    if fresh_np is None:
        fresh = jow.reset_from_reference(
            seed,
            int(cfg.num_agents),
            max_fleets=int(cfg.max_fleets),
        )
    else:
        fresh = jax.tree.map(jnp.asarray, fresh_np)
    idx = int(env_idx)

    def scatter(leaf_b: jnp.ndarray, leaf_f: jnp.ndarray) -> jnp.ndarray:
        return leaf_b.at[idx].set(leaf_f)

    return jax.tree.map(scatter, state_b, fresh)


def heal_terminal_env_slices(
    state_b: OrbitWarsState,
    cfg: OrbitWarsEnvConfig,
    episode_turns: List[int],
    seed_cursor: int,
) -> Tuple[OrbitWarsState, int, List[int]]:
    """Resume helper: reset any env slice still terminal (e.g. legacy checkpoints).

    Uses consecutive seeds starting at ``seed_cursor``. Returns updated state,
    number of seeds consumed, and ``episode_turns`` with reset slots cleared.
    """

    num_envs = int(state_b.planets.shape[0])
    done_np = np.asarray(jax.device_get(state_b.done))
    et = list(episode_turns)
    if len(et) != num_envs:
        et = [0] * num_envs
    seeds_used = 0
    for i in range(num_envs):
        if bool(done_np.reshape(-1)[i]):
            state_b = reset_env_at_index(state_b, i, seed_cursor + seeds_used, cfg)
            seeds_used += 1
            et[i] = 0
    return state_b, seeds_used, et


def stack_initial_states(
    cfg_template: OrbitWarsEnvConfig,
    num_envs: int,
    seed_base: int,
    reset_prefetch: Optional[Any] = None,
) -> Tuple[OrbitWarsState, OrbitWarsEnvConfig]:
    """Reset ``num_envs`` envs and stack into a single batched state.

    Returns ``(state_b, cfg)`` where ``state_b`` has a leading ``num_envs`` axis
    on every leaf, and ``cfg`` is a fresh ``OrbitWarsEnvConfig`` that tracks the
    *current* shared ``max_fleets`` (will be mutated on expansion).
    """

    cfg = OrbitWarsEnvConfig(
        num_agents=cfg_template.num_agents,
        max_fleets=cfg_template.max_fleets,
        episode_seed=cfg_template.episode_seed,
    )
    states: List[OrbitWarsState] = []
    for i in range(num_envs):
        if reset_prefetch is not None:
            np_s = reset_prefetch.pop_state(
                int(seed_base + i), int(cfg.num_agents), int(cfg.max_fleets)
            )
            s = jax.tree.map(jnp.asarray, np_s)
        else:
            s = jow.reset_from_reference(seed_base + i, cfg.num_agents, max_fleets=cfg.max_fleets)
        states.append(s)
    state_b = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *states)
    return state_b, cfg


def unstack_state_to_numpy_list(state_b: OrbitWarsState) -> List[OrbitWarsState]:
    """Single ``device_get`` of the entire batched state, then per-env NumPy slicing.

    Returned per-env states have NumPy leaves and are suitable for the existing
    host-side ``build_observation`` / ``virt`` machinery. This is the *only*
    bulk host sync per turn in Phase 1 (everything else is single-scalar).
    """

    state_np = jax.device_get(state_b)
    fields = state_np._fields
    arrs = {f: np.asarray(getattr(state_np, f)) for f in fields}
    num_envs = int(arrs["planets"].shape[0])

    def _slice(env_i: int) -> OrbitWarsState:
        return state_np._replace(**{f: arrs[f][env_i] for f in fields})

    return [_slice(i) for i in range(num_envs)]


def vmap_step_env(state_b: OrbitWarsState, actions_b: jnp.ndarray) -> OrbitWarsState:
    """Single fused vmap'd ``step`` call.

    ``actions_b`` is shaped ``[num_envs, num_agents, max_actions, 3]`` and may
    be either NumPy or JAX (it is converted on the host side in the caller; we
    accept either here).
    """

    actions_jax = jnp.asarray(actions_b, dtype=jnp.float32)
    return _vmapped_step(state_b, actions_jax, _DEFAULT_STEP_CFG)


def expand_fleet_buffers_batched(state: OrbitWarsState, new_max_fleets: int) -> OrbitWarsState:
    """Grow the fleet-buffer dimension for *all* envs to ``new_max_fleets``.

    Existing fleet rows keep the same indices (so ``fleet_active``, fleet ids,
    etc. remain valid). All envs share the buffer dimension because vmap
    requires uniform shapes; expanding when *any* env needs more is the only
    workable policy.
    """

    old_max = int(state.fleets.shape[1])
    if new_max_fleets <= old_max:
        return state
    pad = new_max_fleets - old_max
    num_envs = int(state.fleets.shape[0])
    fleets = jnp.concatenate(
        [state.fleets, jnp.zeros((num_envs, pad, 7), dtype=state.fleets.dtype)],
        axis=1,
    )
    fleet_active = jnp.concatenate(
        [state.fleet_active, jnp.zeros((num_envs, pad), dtype=jnp.bool_)],
        axis=1,
    )
    return state._replace(fleets=fleets, fleet_active=fleet_active)


def shrink_fleet_buffers_batched(state: OrbitWarsState, new_max_fleets: int) -> OrbitWarsState:
    """Truncate the fleet-buffer axis to ``new_max_fleets`` (all envs).

    Caller must ensure no env has an active fleet in a column ``>= new_max_fleets``.
    """

    old_max = int(state.fleets.shape[1])
    if new_max_fleets >= old_max:
        return state
    return state._replace(
        fleets=state.fleets[:, :new_max_fleets, :],
        fleet_active=state.fleet_active[:, :new_max_fleets],
    )


@jax.jit
def max_concurrent_fleets_any_env(state: OrbitWarsState) -> jnp.ndarray:
    """Scalar ``int32``: max over envs of the number of active fleet slots."""

    return jnp.max(jnp.sum(state.fleet_active, axis=1).astype(jnp.int32))


@jax.jit
def ship_ratio_scores_batched(state: OrbitWarsState) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Vmap'd two-player ship-mass ratios; returns ``(r0[N], r1[N])`` on device."""

    r0, r1, _ = jax.vmap(_ship_ratio_scores_core)(
        state.planets, state.planet_active, state.fleets, state.fleet_active
    )
    return r0, r1


@jax.jit
def ship_totals_batched(state: OrbitWarsState) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Vmap'd absolute ship counts for players 0 and 1; returns ``(ships0[N], ships1[N])``."""

    s0, s1 = jax.vmap(_ship_totals_p01_core)(
        state.planets, state.planet_active, state.fleets, state.fleet_active
    )
    return s0, s1


@jax.jit
def inactive_fleet_count_batched(state: OrbitWarsState) -> jnp.ndarray:
    """Per-env count of inactive fleet slots, ``[num_envs]`` int32."""

    return jnp.sum(~state.fleet_active, axis=1).astype(jnp.int32)


def upper_bound_fleet_writes_per_env(actions_np: np.ndarray) -> np.ndarray:
    """Per-env upper bound on fleet writes for ``actions_np[..., 2]`` ships values.

    Mirrors ``_upper_bound_fleet_writes_from_actions`` but emits a per-env
    vector so the expansion check is one batched comparison against
    ``inactive_fleet_count_batched``.
    """

    ships = np.floor(actions_np[..., 2])
    return np.sum(ships > 0, axis=(1, 2)).astype(np.int32)


def update_virt_jax(
    virt_b: OrbitWarsState,
    virts_np: List[OrbitWarsState],
) -> OrbitWarsState:
    """Stack per-env host ``virt`` snapshots back onto a batched JAX state.

    Only the fields ``micro_step_apply`` mutates (``planets``, ``fleets``,
    ``fleet_active``) are pushed; everything else is reused from ``virt_b``,
    which lets us avoid re-uploading the constant comet/initial tables every
    micro-step.
    """

    planets_np = np.stack([np.asarray(v.planets) for v in virts_np], axis=0)
    fleets_np = np.stack([np.asarray(v.fleets) for v in virts_np], axis=0)
    fleet_active_np = np.stack([np.asarray(v.fleet_active) for v in virts_np], axis=0)
    return virt_b._replace(
        planets=jnp.asarray(planets_np),
        fleets=jnp.asarray(fleets_np),
        fleet_active=jnp.asarray(fleet_active_np),
    )


_INT_OBS_FIELDS = ("entity_type", "owner_idx")


def obs_jax_to_torch(obs_jax: Dict[str, jnp.ndarray]) -> Dict[str, torch.Tensor]:
    """Zero-copy handoff (when both runtimes share the device) via the dlpack protocol.

    We ``block_until_ready`` on every output before exposing the arrays to
    PyTorch so the policy forward sees fully-written tensors. JAX and PyTorch
    use independent CUDA streams; without this barrier, PyTorch could read
    while JAX is still writing.

    The two index fields (``entity_type``, ``owner_idx``) are produced as
    ``int32`` on the JAX side (JAX defaults to x32; using ``int64`` triggers
    truncation warnings) and cast to ``torch.long`` here because
    ``nn.Embedding`` requires a ``LongTensor`` index. The cast is on tiny
    tensors of shape ``[B, L]`` so the device-side copy is negligible.
    """

    jax.block_until_ready(obs_jax)
    out: Dict[str, torch.Tensor] = {}
    for k, v in obs_jax.items():
        t = torch.from_dlpack(v)
        if k in _INT_OBS_FIELDS:
            t = t.to(torch.long)
        out[k] = t
    return out

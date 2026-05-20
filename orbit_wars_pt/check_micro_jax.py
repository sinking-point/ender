"""Self-consistency check for Phase 3-5 JAX path.

Runs on CPU; verifies:

* ``compute_pair_geom_and_etas`` returns shape-correct masks/etas and the
  diagonal pair (o == d) is always False.
* ``apply_micro_step_batched`` deducts ships and appends a fleet at the
  first inactive slot; halted envs are unchanged.
* ``init_transition_buffer`` / ``append_to_buffer`` allocate compact per-row
  micro-step fields; canonical state lives in ``turn_state_cache``.
* ``gather_minibatch`` applies a stacked prefix with
  ``apply_prefix_micro_deltas_batched`` and selects per-element from the
  correct player buffer.
* ``replay_logprob_value_entropy_jax`` runs end-to-end on a gathered
  minibatch and returns finite scalars.
"""

from __future__ import annotations

import os
import sys

# Force CPU JAX (avoid sandbox CUDA init issues).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import torch  # noqa: E402

from jax_orbit_wars import PLANET_SHIPS, PLANET_X, PLANET_Y  # noqa: E402

from orbit_wars_pt.batched_env import stack_initial_states  # noqa: E402
from orbit_wars_pt.constants import MAX_PLANETS  # noqa: E402
from orbit_wars_pt.env_wrapper import OrbitWarsEnvConfig  # noqa: E402
from orbit_wars_pt.micro_jax import (  # noqa: E402
    apply_micro_step_batched,
    compute_pair_geom_and_etas,
    must_halt_no_owned_ships_batched,
)
from orbit_wars_pt.model import OrbitWarsPolicy  # noqa: E402
from orbit_wars_pt.observation_jax import build_observation_batched_jax  # noqa: E402
from orbit_wars_pt.ppo_replay import replay_logprob_value_entropy_jax  # noqa: E402
from orbit_wars_pt.transition_buffer import (  # noqa: E402
    append_to_buffer,
    gather_minibatch,
    init_transition_buffer,
    scatter_turn_tags,
)


def _find_valid_pair(pair_geom_valid: np.ndarray, env_idx: int):
    flat = pair_geom_valid[env_idx].flatten()
    idxs = np.where(flat)[0]
    if idxs.size == 0:
        return None
    o = int(idxs[0]) // MAX_PLANETS
    d = int(idxs[0]) % MAX_PLANETS
    return o, d


def main() -> None:
    cfg = OrbitWarsEnvConfig(num_agents=2, max_fleets=32, episode_seed=0)
    state_b, _ = stack_initial_states(cfg, num_envs=4, seed_base=0)

    pair_geom_j, etas_j = compute_pair_geom_and_etas(state_b, ship_speed=6.0)
    pair_geom = np.asarray(jax.device_get(pair_geom_j))
    etas = np.asarray(jax.device_get(etas_j))
    print(
        f"pair_geom_valid shape {pair_geom.shape}; "
        f"per-env counts {[int(pair_geom[i].sum()) for i in range(pair_geom.shape[0])]}"
    )
    assert pair_geom.shape == (4, MAX_PLANETS, MAX_PLANETS)
    for i in range(4):
        diag = np.diagonal(pair_geom[i])
        assert not diag.any(), f"diagonal must be False for env {i}"
    assert etas.shape == (4, MAX_PLANETS, MAX_PLANETS, 5)

    must_halt = np.asarray(jax.device_get(must_halt_no_owned_ships_batched(state_b, jnp.int32(0))))
    print(f"must_halt @ ego=0: {must_halt}")

    pair = _find_valid_pair(pair_geom, env_idx=0)
    assert pair is not None, "expected at least one valid pair on a fresh state"
    o_idx, d_idx = pair

    pair_flat = jnp.zeros((4,), dtype=jnp.int32).at[0].set(o_idx * MAX_PLANETS + d_idx)
    frac_idx = jnp.zeros((4,), dtype=jnp.int32).at[0].set(4)  # FRACTIONS[4] = 1.0
    halt_now = jnp.array([False, True, True, True])
    fleet_eta = jnp.zeros((4,), dtype=jnp.float32)

    pre_planets = np.asarray(jax.device_get(state_b.planets))
    new_state, oid_j, send_j, dispatched_j, _slot_j = apply_micro_step_batched(
        state_b,
        jnp.int32(0),
        halt_now,
        pair_flat,
        frac_idx,
        fleet_eta,
    )
    post_planets = np.asarray(jax.device_get(new_state.planets))
    oid_np, send_np, dispatched_np = jax.device_get((oid_j, send_j, dispatched_j))
    oid_np = np.asarray(oid_np)
    send_np = np.asarray(send_np)
    dispatched_np = np.asarray(dispatched_np)

    pre_ships = float(pre_planets[0, o_idx, PLANET_SHIPS])
    post_ships = float(post_planets[0, o_idx, PLANET_SHIPS])
    delta = pre_ships - post_ships
    print(
        f"env0 launch o={o_idx} d={d_idx} pre_ships={pre_ships:.2f} "
        f"post_ships={post_ships:.2f} delta={delta:.2f} "
        f"send_jax={float(send_np[0]):.2f} dispatched={bool(dispatched_np[0])}"
    )
    assert dispatched_np[0]
    assert abs(delta - float(send_np[0])) < 1e-3
    for i in (1, 2, 3):
        assert np.allclose(pre_planets[i], post_planets[i]), f"env {i} should be unchanged"
        assert not bool(dispatched_np[i])

    incoming_np = np.asarray(jax.device_get(new_state.incoming_fleets))
    ta0 = int(max(np.floor(float(jax.device_get(fleet_eta[0])) - 1.0), 0.0))
    assert incoming_np[0, 0, d_idx, ta0] == int(send_np[0]), "env 0 launch should be scheduled in incoming_fleets"

    # ---- Phase 5: compact buffer + turn_state_cache + gather. ----
    # Single-env slice: two p0 micro rows (dispatch then halt) and one p1 row.
    state_1 = jax.tree.map(lambda x: x[:1], state_b)
    new_state_1, _, send_1, _, slot_1 = apply_micro_step_batched(
        state_1,
        jnp.int32(0),
        jnp.array([False]),
        pair_flat[:1],
        frac_idx[:1],
        fleet_eta[:1],
    )
    pair_flat_1 = pair_flat[:1]
    frac_idx_1 = frac_idx[:1]
    no_valid_pairs_1 = jnp.zeros((1,), dtype=jnp.bool_)
    no_valid_fracs_1 = jnp.zeros((1,), dtype=jnp.bool_)
    must_halt_ns_1 = jnp.zeros((1,), dtype=jnp.bool_)
    tpr_1 = jnp.zeros((1, MAX_PLANETS), dtype=jnp.bool_).at[0, d_idx].set(True)
    tht_1 = jnp.zeros((1, MAX_PLANETS), dtype=jnp.float32).at[0, d_idx].set(float(fleet_eta[0]))

    H_buf = 8
    M = 8
    num_e = 1
    buf0 = init_transition_buffer(num_e, H_buf, M)
    buf1 = init_transition_buffer(num_e, H_buf, M)
    assert buf0.micro_halt_now.shape == (H_buf, num_e, M)
    assert buf0.halt_action.shape == (H_buf, num_e)

    halt_action_r0 = jnp.array([0], dtype=jnp.int32)
    halt_action_r1 = jnp.array([1], dtype=jnp.int32)
    micro_halt_halt = jnp.array([True], dtype=jnp.bool_)
    active_1 = jnp.array([True], dtype=jnp.bool_)
    send_zero = jnp.array([0.0], dtype=jnp.float32)
    eta_zero = jnp.array([0.0], dtype=jnp.float32)
    slot_neg = jnp.array([-1], dtype=jnp.int32)

    buf0 = append_to_buffer(
        buf0,
        halt_now[:1],
        send_1,
        fleet_eta[:1],
        slot_1,
        halt_action_r0,
        pair_flat_1,
        frac_idx_1,
        no_valid_pairs_1,
        no_valid_fracs_1,
        must_halt_ns_1,
        tpr_1,
        tht_1,
        jnp.array([0], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
        active_1,
        M,
    )
    buf0 = append_to_buffer(
        buf0,
        micro_halt_halt,
        send_zero,
        eta_zero,
        slot_neg,
        halt_action_r1,
        pair_flat_1,
        frac_idx_1,
        no_valid_pairs_1,
        no_valid_fracs_1,
        must_halt_ns_1,
        tpr_1,
        tht_1,
        jnp.array([1], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
        active_1,
        M,
    )

    halt_action_p1_r0 = jnp.array([0], dtype=jnp.int32)
    buf1 = append_to_buffer(
        buf1,
        micro_halt_halt,
        send_zero,
        eta_zero,
        slot_neg,
        halt_action_p1_r0,
        pair_flat_1,
        frac_idx_1,
        no_valid_pairs_1,
        no_valid_fracs_1,
        must_halt_ns_1,
        tpr_1,
        tht_1,
        jnp.array([0], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
        active_1,
        M,
    )

    turn_tag_p0 = jnp.full((H_buf, num_e), -1, dtype=jnp.int32)
    turn_tag_p1 = jnp.full((H_buf, num_e), -1, dtype=jnp.int32)
    slot0 = jnp.array([0], dtype=jnp.int32)
    turn_tag_p0 = scatter_turn_tags(turn_tag_p0, jnp.array([0], dtype=jnp.int32), slot0)
    turn_tag_p0 = scatter_turn_tags(turn_tag_p0, jnp.array([1], dtype=jnp.int32), slot0)
    turn_tag_p1 = scatter_turn_tags(turn_tag_p1, jnp.array([0], dtype=jnp.int32), slot0)

    T_cap = 4
    turn_state_cache = jax.tree.map(
        lambda x: jnp.zeros((T_cap,) + x.shape, dtype=x.dtype),
        state_1,
    )
    turn_state_cache = jax.tree.map(
        lambda c, x: c.at[0].set(x), turn_state_cache, state_1
    )

    # (p0, t=1): replay row 0 dispatch -> pre-action matches one-step post state.
    # (p1, t=0): turn-start canonical state, action from buf1.
    mb_player = jnp.array([0, 1], dtype=jnp.int32)
    mb_t = jnp.array([1, 0], dtype=jnp.int32)
    mb_n = jnp.array([0, 0], dtype=jnp.int32)

    state_mb, halt_action_mb, pair_flat_mb, frac_idx_mb, nvp_mb, nvf_mb, _mh_mb, tpr_mb, tht_mb = gather_minibatch(
        buf0,
        buf1,
        mb_player,
        mb_t,
        mb_n,
        turn_state_cache,
        turn_tag_p0,
        turn_tag_p1,
        M,
    )
    assert state_mb.planets.shape == (2, MAX_PLANETS, 7)
    state_mb_planets = np.asarray(jax.device_get(state_mb.planets))
    pre_1 = np.asarray(jax.device_get(state_1.planets[0]))
    post_1 = np.asarray(jax.device_get(new_state_1.planets[0]))
    assert np.allclose(state_mb_planets[0], post_1)
    assert np.allclose(state_mb_planets[1], pre_1)

    halt_action_mb_np = np.asarray(jax.device_get(halt_action_mb))
    assert halt_action_mb_np[0] == 1  # buf0 row 1
    assert halt_action_mb_np[1] == 0  # buf1 row 0

    # ---- Phase 5 replay smoke test on the gathered minibatch. ----
    device = torch.device("cpu")
    policy = OrbitWarsPolicy(d_model=72, n_heads=4, n_layers=2).to(device)
    policy.eval()

    ego_b_j = mb_player  # player_id == ego in our setup

    with torch.inference_mode():
        new_logp, new_v, new_ent = replay_logprob_value_entropy_jax(
            state_b=state_mb,
            halt_action=halt_action_mb,
            pair_flat=pair_flat_mb,
            frac_idx=frac_idx_mb,
            no_valid_pairs=nvp_mb,
            no_valid_fracs=nvf_mb,
            ego_b=ego_b_j,
            policy=policy,
            device=device,
            ship_speed=6.0,
            target_planet_reachable=tpr_mb,
            target_hit_tick=tht_mb,
        )
    print(
        f"replay_jax: logp shape {tuple(new_logp.shape)}, "
        f"value shape {tuple(new_v.shape)}, entropy shape {tuple(new_ent.shape)}"
    )
    assert new_logp.shape == (2,)
    assert new_v.shape == (2,)
    assert new_ent.shape == (2,)
    assert torch.isfinite(new_logp).all()
    assert torch.isfinite(new_v).all()
    assert torch.isfinite(new_ent).all()

    # Smoke-test obs builder still works (rollout path).
    obs_jax = build_observation_batched_jax(state_b, 0, 6.0)
    print(
        "obs_jax (rollout): "
        + ", ".join(f"{k}={tuple(v.shape)}" for k, v in obs_jax.items())
    )

    print("All Phase 3-5 self-consistency checks passed.")


if __name__ == "__main__":
    main()

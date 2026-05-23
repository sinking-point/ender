"""Background consistency check for host-storage rollouts.

When host-storage mode is on, every rollout already lands its per-seat
``TorchTransitionBuffer`` and compressed observation buffer in CPU RAM, plus a
small ``first_reset_event`` sidecar attached to ``RolloutSegment`` that names
``(env_i, seed, write_idx_at_reset_per_seat)`` for the first env that resets
during the segment. This module slices the freshly-reset game's trajectory out
of that host data and submits it to a long-lived background process which:

* spins up an official Kaggle ``orbit_wars`` env with the same ``seed``;
* for each env step (Kaggle ``step``), for each seat:

  * converts the Kaggle observation into the adapter's ``OrbitWarsState`` and
    builds the agent observation tensors via ``_obs_tensors_for_state``;
  * walks only that step's recorded micro rows (``phase_micro_idx`` 0..
    ``max_micro_steps-1``, split when the index wraps), comparing rebuilt obs
    vs rollout obs and applying recorded launches until the seat halts or the
    micro cap is reached;
* aggregates per-seat launches and calls ``env.step(...)`` once per env step.

The background process never imports JAX (verified by the parent process's
worker boot script that poisons ``sys.modules['jax']`` before importing the
adapter). All data flowing parent -> worker is NumPy / built-ins.

Mismatches are saved to ``<out_dir>/iter_<N>_env_<i>_turn_<t>.npz`` plus a
short ``info.json`` per trajectory; a one-line summary is printed too.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import queue
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# On-the-wire trajectory representation. Pickleable, NumPy-only.
# ---------------------------------------------------------------------------


@dataclass
class MicroRow:
    """One stored micro-step row for one seat, copied out of the host segment.

    The row is the policy decision made for this seat at micro index
    ``phase_micro_idx`` within env step ``turn_index`` (Kaggle ``step`` at
    micro time, not a policy-halt segment). ``halt_action == 1`` ends that
    seat's participation for the current env step. When ``halt_action == 0``
    and the safety flags allow it, the row launches a fleet identified by
    ``(origin_slot, target_slot, frac_idx, ships, policy_eta)``.
    """

    turn_index: int
    phase_micro_idx: int
    halt_action: int
    must_halt_no_ships: bool
    no_valid_pairs: bool
    no_valid_fracs: bool
    origin_slot: int
    target_slot: int
    frac_idx: int
    ships: int
    policy_eta: float
    obs: Dict[str, np.ndarray]


@dataclass
class TrajectoryRecord:
    iter_id: int
    env_index: int
    seed: int
    num_agents: int
    ship_speed: float
    n_rays: int
    normalize_obs_to_p0: bool
    max_micro_steps: int = 8
    #: ``rows_per_seat[p]`` is the list of micro rows for seat ``p`` in
    #: chronological order, starting at the freshly-reset game's first row.
    rows_per_seat: List[List[MicroRow]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Host-side extraction: lift the trajectory out of an already-on-host segment.
# ---------------------------------------------------------------------------


def _decode_seat_obs_rows(obs_buf: Any, env_idx: int, row_start: int, row_stop: int) -> List[Dict[str, np.ndarray]]:
    """Decode a contiguous range of compressed observation rows for one env.

    Both compress and decode are idempotent on the rollout side, so we run the
    decoder once (giving the policy-visible obs the rollout actually consumed)
    and return per-row NumPy dicts. The worker will compress+decode the
    adapter's obs through the same code path so the two sides land in the same
    numeric domain.
    """

    import torch

    from orbit_wars_pt.compressed_observation import decode_observation

    if row_stop <= row_start:
        return []
    # ``obs_buf`` fields have shape ``[H, num_envs, ...]``. Slice one env.
    sliced = type(obs_buf)(**{
        name: torch.as_tensor(getattr(obs_buf, name))[row_start:row_stop, env_idx]
        for name in obs_buf._fields
    })
    feature_dim = _infer_feature_dim_from_obs_buf(obs_buf)
    decoded = decode_observation(sliced, feature_dim=feature_dim)
    n = int(row_stop - row_start)
    out: List[Dict[str, np.ndarray]] = []
    for r in range(n):
        out.append({key: np.asarray(v[r].detach().cpu().numpy()) for key, v in decoded.items()})
    return out


def _infer_feature_dim_from_obs_buf(obs_buf: Any) -> int:
    """The compressed buffer doesn't carry feature_dim directly; pick the dim
    that matches the buffer's ``incoming_survivor`` shape (4p has it, 2p
    leaves it at zero but still needs the multi-feature dim if num_agents>2).

    For now, just look at the survivor field's last axis presence: ``compress``
    always emits a non-trivial survivor only when feature_dim is the multi
    variant. As a safe default we use the multi feature dim — ``decode_observation``
    tolerates both as long as ``feature_dim`` matches the source.

    The caller will overwrite this when they know ``num_agents``.
    """

    from orbit_wars_pt.constants import FEATURE_DIM

    return FEATURE_DIM  # caller overrides via _decode_seat_obs_rows_for_num_agents


def _decode_seat_obs_rows_for_num_agents(
    obs_buf: Any,
    env_idx: int,
    row_start: int,
    row_stop: int,
    *,
    num_agents: int,
) -> List[Dict[str, np.ndarray]]:
    """Same as ``_decode_seat_obs_rows`` but with an explicit num_agents arg
    so the decoder picks the right feature dim for 2p vs 4p."""

    import torch

    from orbit_wars_pt.compressed_observation import decode_observation
    from orbit_wars_pt.constants import obs_feature_dim_for_num_agents

    if row_stop <= row_start:
        return []
    sliced = type(obs_buf)(**{
        name: torch.as_tensor(getattr(obs_buf, name))[row_start:row_stop, env_idx]
        for name in obs_buf._fields
    })
    feature_dim = obs_feature_dim_for_num_agents(int(num_agents))
    decoded = decode_observation(sliced, feature_dim=feature_dim)
    n = int(row_stop - row_start)
    out: List[Dict[str, np.ndarray]] = []
    for r in range(n):
        out.append({key: np.asarray(v[r].detach().cpu().numpy()) for key, v in decoded.items()})
    return out


def _env_step_for_phase_row(phase_arr: np.ndarray, row_index: int) -> int:
    """Map a buffer row to the env ``step`` it belongs to (0-based from reset).

    Rollout resets ``micro_k`` on every ``env.step``, so ``phase_micro_idx`` is
    0..max_micro_steps-1 within a step and wraps downward when the env advances.
    """

    env_step = 0
    for r in range(1, int(row_index) + 1):
        if int(phase_arr[r]) <= int(phase_arr[r - 1]):
            env_step += 1
    return env_step


def build_trajectory_from_segment(
    segment: Any,
    *,
    iter_id: int,
    num_agents: int,
    ship_speed: float,
    n_rays: int,
    normalize_obs_to_p0: bool,
    max_micro_steps: int = 8,
) -> Optional[TrajectoryRecord]:
    """Slice the fresh-game trajectory of ``segment.first_reset_event`` out of the host segment.

    Returns ``None`` when the segment has no recorded reset event (e.g. no env
    finished a game during the rollout). The returned ``TrajectoryRecord``
    contains only NumPy data and is safe to send across a multiprocessing queue.
    """

    from orbit_wars_pt.constants import MAX_PLANETS

    if segment.first_reset_event is None:
        return None
    env_i_raw, seed_raw, write_idx_at_reset = segment.first_reset_event
    env_i = int(env_i_raw)
    seed = int(seed_raw)
    n_seats_segment = len(segment.bufs)
    rows_per_seat: List[List[MicroRow]] = []
    for p in range(n_seats_segment):
        buf = segment.bufs[p]
        obs_buf = segment.obs_bufs[p]
        row_start = int(write_idx_at_reset[p])
        row_stop = int(segment.write_idx[p][env_i])
        if row_stop <= row_start:
            rows_per_seat.append([])
            continue
        decoded = _decode_seat_obs_rows_for_num_agents(
            obs_buf,
            env_i,
            row_start,
            row_stop,
            num_agents=int(num_agents),
        )
        # ``buf`` fields are torch tensors on host; index NumPy-style.
        halt_arr = np.asarray(buf.halt_action[row_start:row_stop, env_i].cpu())
        must_halt_arr = np.asarray(buf.must_halt_no_ships[row_start:row_stop, env_i].cpu())
        no_valid_pairs_arr = np.asarray(buf.no_valid_pairs[row_start:row_stop, env_i].cpu())
        no_valid_fracs_arr = np.asarray(buf.no_valid_fracs[row_start:row_stop, env_i].cpu())
        pair_flat_arr = np.asarray(buf.pair_flat[row_start:row_stop, env_i].cpu())
        frac_idx_arr = np.asarray(buf.frac_idx[row_start:row_stop, env_i].cpu())
        send_arr = np.asarray(buf.send[row_start:row_stop, env_i].cpu())
        eta_arr = np.asarray(buf.fleet_eta[row_start:row_stop, env_i].cpu())
        phase_arr = np.asarray(buf.phase_micro_idx[row_start:row_stop, env_i].cpu())
        seat_rows: List[MicroRow] = []
        n = int(row_stop - row_start)
        for r in range(n):
            ms = int(pair_flat_arr[r, 0]) if pair_flat_arr.ndim == 2 else int(pair_flat_arr[r])
            # ``pair_flat``, ``send``, ``frac_idx``, etc., are per-micro within
            # the row (last axis = max_micro_steps). We want the slot indexed
            # by ``phase_micro_idx`` since that is the launch *this row*
            # represents.
            phase_micro = int(phase_arr[r])
            def _scalar(arr: np.ndarray) -> Any:
                if arr.ndim == 2:
                    return arr[r, phase_micro]
                return arr[r]

            pair_flat_scalar = int(_scalar(pair_flat_arr))
            origin_slot = pair_flat_scalar // int(MAX_PLANETS)
            target_slot = pair_flat_scalar % int(MAX_PLANETS)
            ships = int(math.floor(float(_scalar(send_arr))))
            policy_eta = float(_scalar(eta_arr))
            frac_idx_v = int(_scalar(frac_idx_arr))
            halt = int(halt_arr[r])
            row_rec = MicroRow(
                turn_index=int(_env_step_for_phase_row(phase_arr, r)),
                phase_micro_idx=phase_micro,
                halt_action=halt,
                must_halt_no_ships=bool(must_halt_arr[r]),
                no_valid_pairs=bool(no_valid_pairs_arr[r]),
                no_valid_fracs=bool(no_valid_fracs_arr[r]),
                origin_slot=int(origin_slot),
                target_slot=int(target_slot),
                frac_idx=int(frac_idx_v),
                ships=int(ships),
                policy_eta=float(policy_eta),
                obs=decoded[r],
            )
            seat_rows.append(row_rec)
        rows_per_seat.append(seat_rows)

    return TrajectoryRecord(
        iter_id=int(iter_id),
        env_index=env_i,
        seed=seed,
        num_agents=int(num_agents),
        ship_speed=float(ship_speed),
        n_rays=int(n_rays),
        normalize_obs_to_p0=bool(normalize_obs_to_p0),
        max_micro_steps=int(max(1, max_micro_steps)),
        rows_per_seat=rows_per_seat,
    )


# ---------------------------------------------------------------------------
# Worker side: replay through the official Kaggle env and compare.
# ---------------------------------------------------------------------------


_DEFAULT_MISMATCH_DIR_ENV = "ORBIT_WARS_CONSISTENCY_DIR"


def _compress_decode_adapter_obs(obs_tensors: Mapping[str, Any], *, num_agents: int) -> Dict[str, np.ndarray]:
    """Run the rollout's lossy compress+decode on the adapter-built obs.

    We compare in the same numeric domain the policy actually sees during
    training: int16/float16 quantisation applied to the float32 adapter obs.
    """

    import torch

    from orbit_wars_pt.compressed_observation import compress_observation, decode_observation
    from orbit_wars_pt.constants import obs_feature_dim_for_num_agents

    feature_dim = obs_feature_dim_for_num_agents(int(num_agents))
    # The compressor expects a leading batch dim; ``_obs_tensors_for_state``
    # already returns ``[1, ...]`` tensors, so no extra unsqueeze needed.
    obs_batched = {k: v if v.dim() > 0 and v.shape[0] == 1 else v.unsqueeze(0) for k, v in obs_tensors.items()}
    comp = compress_observation(obs_batched)
    decoded = decode_observation(comp, feature_dim=feature_dim)
    return {k: np.asarray(v[0].detach().cpu().numpy()) for k, v in decoded.items()}


def _compare_obs(
    rollout_obs: Mapping[str, np.ndarray],
    adapter_obs: Mapping[str, np.ndarray],
) -> List[Dict[str, Any]]:
    """Compare two compress+decode obs dicts; return per-field mismatch records."""

    mismatches: List[Dict[str, Any]] = []
    keys = ("entity_type", "owner_idx", "features", "rope_pos", "entity_mask", "planet_mask")
    active_rows: Optional[np.ndarray] = None
    try:
        rollout_mask = np.asarray(rollout_obs.get("entity_mask"))
        adapter_mask = np.asarray(adapter_obs.get("entity_mask"))
        if rollout_mask.shape == adapter_mask.shape and rollout_mask.ndim == 1:
            active_rows = rollout_mask.astype(np.bool_) | adapter_mask.astype(np.bool_)
            if active_rows.size > 0:
                # Always keep the CLS token in scope.
                active_rows[0] = True
    except Exception:
        active_rows = None
    for key in keys:
        if key not in rollout_obs or key not in adapter_obs:
            mismatches.append({"field": key, "kind": "missing"})
            continue
        a = np.asarray(rollout_obs[key])
        b = np.asarray(adapter_obs[key])
        if a.shape != b.shape:
            mismatches.append(
                {"field": key, "kind": "shape", "rollout_shape": a.shape, "adapter_shape": b.shape}
            )
            continue
        if active_rows is not None and key not in ("entity_mask", "planet_mask"):
            if a.ndim >= 1 and a.shape[0] == active_rows.shape[0]:
                a = a[active_rows]
                b = b[active_rows]
        if key in ("entity_type", "owner_idx"):
            diff = a.astype(np.int64) != b.astype(np.int64)
            if bool(diff.any()):
                mismatches.append(
                    {
                        "field": key,
                        "kind": "int_inequality",
                        "n_diff": int(diff.sum()),
                        "first_diff": _first_diff_index(diff),
                    }
                )
        elif key in ("entity_mask", "planet_mask"):
            diff = a.astype(np.bool_) != b.astype(np.bool_)
            if bool(diff.any()):
                mismatches.append(
                    {
                        "field": key,
                        "kind": "bool_inequality",
                        "n_diff": int(diff.sum()),
                        "first_diff": _first_diff_index(diff),
                    }
                )
        else:
            # After compress+decode both sides are in the same quantised
            # domain; a tight tolerance is appropriate (the only source of
            # noise is float16 storage on ``xy`` / ``velocity``).
            af = a.astype(np.float64)
            bf = b.astype(np.float64)
            diff_abs = np.abs(af - bf)
            tol = 1e-3 + 1e-3 * np.abs(bf)
            bad = diff_abs > tol
            if bool(bad.any()):
                mismatches.append(
                    {
                        "field": key,
                        "kind": "float_inequality",
                        "n_diff": int(bad.sum()),
                        "max_abs": float(diff_abs.max()),
                        "first_diff": _first_diff_index(bad),
                    }
                )
    return mismatches


def _first_diff_index(diff_mask: np.ndarray) -> Optional[List[int]]:
    flat = np.flatnonzero(diff_mask.reshape(-1))
    if flat.size == 0:
        return None
    return list(map(int, np.unravel_index(int(flat[0]), diff_mask.shape)))


def _resolve_angle_for_target(
    *,
    state: Any,
    origin_slot: int,
    target_slot: int,
    frac_idx: int,
    policy_eta: float,
    ship_speed: float,
    n_rays: int,
    eta_tolerance: float,
) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[float], bool, str, List[Dict[str, Any]]]:
    """Use the adapter's raycast to pick the launch angle for ``target_slot``.

    The Kaggle action must use the angle from the rollout's recorded ``frac_idx``
    (that fraction fixes both the ray fan and ``_fleet_speed`` via planned send).
    We still raycast every fraction into ``per_frac_records`` for dumps, but the
    returned angle / hit metadata always comes from ``frac_idx`` only.

    Returns ``(angle, hit_tick, true_target, true_hit_tick, ok, reason,
    per_frac_records)``.
    """

    from orbit_wars_pt.constants import FRACTIONS, MAX_PLANETS
    from orbit_wars_pt.kaggle_adapter import _raycast_targets_np

    per_frac: List[Dict[str, Any]] = []
    if not (0 <= int(target_slot) < MAX_PLANETS):
        return None, None, None, None, False, "target_slot_out_of_range", per_frac
    if not (0 <= int(frac_idx) < len(FRACTIONS)):
        return None, None, None, None, False, "frac_idx_out_of_range", per_frac

    recorded: Optional[Dict[str, Any]] = None
    for sweep_frac in range(len(FRACTIONS)):
        rec: Dict[str, Any] = {"frac_idx": int(sweep_frac)}
        try:
            out_angle, valid, hit_tick, true_planet, true_hit_tick = _raycast_targets_np(
                state,
                int(origin_slot),
                int(sweep_frac),
                ship_speed=float(ship_speed),
                n_rays=int(n_rays),
            )
        except Exception as exc:
            rec["error"] = repr(exc)
            per_frac.append(rec)
            if int(sweep_frac) == int(frac_idx):
                recorded = rec
            continue
        rec["target_valid"] = bool(valid[int(target_slot)])
        rec["target_hit_tick"] = float(hit_tick[int(target_slot)])
        rec["target_angle"] = float(out_angle[int(target_slot)])
        rec["true_planet"] = int(true_planet[int(target_slot)])
        rec["true_hit_tick"] = float(true_hit_tick[int(target_slot)])
        per_frac.append(rec)
        if int(sweep_frac) == int(frac_idx):
            recorded = rec

    if recorded is None:
        return None, None, None, None, False, "frac_idx_not_raycast", per_frac
    if recorded.get("error") is not None:
        return None, None, None, None, False, f"raycast_error: {recorded['error']}", per_frac
    if not recorded.get("target_valid"):
        return None, None, None, None, False, "no_ray_hits_target", per_frac

    angle = float(recorded["target_angle"])
    tick = float(recorded["target_hit_tick"])
    tp = int(recorded["true_planet"])
    tt = float(recorded["true_hit_tick"])
    err = abs(tick - float(policy_eta))
    if err > float(eta_tolerance):
        return angle, tick, tp, tt, False, f"eta_mismatch err={err:.3f}", per_frac
    return angle, tick, tp, tt, True, "ok", per_frac


def _kaggle_obs_to_agent_state(
    obs: Mapping[str, Any],
    *,
    num_agents: int,
    step_count: int,
) -> Any:
    """Adapter-side: Kaggle observation -> ``OrbitWarsState`` (numpy-backed).

    The official Kaggle env only stamps ``observation.step`` on seat 0, so every
    other seat's obs lacks a step counter. ``observation_to_state``'s default
    fallback is ``0``, which silently corrupts the CLS turn-progress feature
    and any forecast that depends on ``state.step_count`` (planet position
    rollouts, raycast etas, etc.). The worker drives the replay loop and
    knows the canonical turn index for every seat, so we always pass
    ``step_count_override`` rather than trusting per-seat obs.
    """

    from orbit_wars_pt.kaggle_adapter import observation_to_state

    return observation_to_state(
        obs,
        config={"agentCount": int(num_agents)},
        step_count_override=int(step_count),
    )


def _agent_obs_for_virt(
    virt: Any,
    *,
    ego_player: int,
    num_agents: int,
    normalize_obs_to_p0: bool,
) -> Dict[str, np.ndarray]:
    """Build the agent observation for ``virt`` and run the rollout's lossy
    compress+decode so the result is comparable to the host-stored obs."""

    import torch

    from orbit_wars_pt.kaggle_adapter import _obs_tensors_for_state

    tensors = _obs_tensors_for_state(
        virt,
        int(ego_player),
        torch.device("cpu"),
        policy_player_count=int(num_agents),
        normalize_obs_to_p0=bool(normalize_obs_to_p0),
    )
    return _compress_decode_adapter_obs(tensors, num_agents=int(num_agents))


def _check_trajectory(
    traj: TrajectoryRecord,
    *,
    out_dir: Path,
    log_prefix: str,
    eta_tolerance: float,
) -> Dict[str, int]:
    """Replay ``traj`` through the Kaggle env, compare obs at every micro step,
    and dump a single end-of-trajectory diagnostic file iff any mismatch fired.

    The walk accumulates ``_TurnDiag`` records (per-seat Kaggle obs, adapter
    state snapshots, per-row working ``virt``, adapter-built obs, structured
    obs mismatches, and raycast outcomes per attempted launch). Nothing is
    written to disk while the trajectory is clean; if any obs mismatch or
    launch issue fires anywhere in the trajectory, ``_save_trajectory_dump``
    flushes the entire run as a single ``<base>.npz`` + ``<base>.json`` pair.
    """

    from kaggle_environments import make

    from orbit_wars_pt.kaggle_adapter import apply_micro_launch_in_place

    counters = {"turns": 0, "obs_mismatches": 0, "launch_issues": 0, "saved": 0}

    configuration: Dict[str, Any] = {
        "agentCount": int(traj.num_agents),
        "seed": int(traj.seed),
        "actTimeout": 86400.0,
        "runTimeout": 86400.0 * 14.0,
    }
    env = make("orbit_wars", configuration=configuration, debug=False)
    env.reset(int(traj.num_agents))

    cursors = [0] * int(traj.num_agents)
    turns: List[_TurnDiag] = []
    any_mismatch = False

    while True:
        if all(cursors[p] >= len(traj.rows_per_seat[p]) for p in range(traj.num_agents)):
            break

        turn_index = counters["turns"]
        kaggle_step = None
        try:
            kaggle_step = int(env.state[0].observation.get("step", -1))
        except Exception:
            kaggle_step = None

        td = _TurnDiag(turn_index=turn_index, kaggle_step=kaggle_step)
        if env.done:
            td.extra_issues.append({"reason": "kaggle_env_done_early"})
            any_mismatch = True
            counters["launch_issues"] += 1
            print(
                f"{log_prefix} iter={traj.iter_id} env={traj.env_index} "
                f"kaggle_env_done_early at turn {turn_index} (more rollout rows remain)",
                flush=True,
            )
            turns.append(td)
            break

        counters["turns"] += 1

        # Snapshot Kaggle observations and the reconstructed adapter states.
        seat_states: List[Any] = []
        seat_planets: List[np.ndarray] = []
        seat_incoming: List[np.ndarray] = []
        for p in range(traj.num_agents):
            kobs = env.state[p].observation
            td.kaggle_obs_per_seat.append(kobs)
            state = _kaggle_obs_to_agent_state(
                kobs,
                num_agents=traj.num_agents,
                step_count=int(turn_index),
            )
            td.initial_state_per_seat.append(state)
            seat_states.append(state)
            seat_planets.append(np.array(np.asarray(state.planets), copy=True))
            seat_incoming.append(np.array(np.asarray(state.incoming_fleets), copy=True))

        per_seat_actions: List[List[List[float]]] = [[] for _ in range(traj.num_agents)]

        for p in range(traj.num_agents):
            seat_rows = traj.rows_per_seat[p]
            micros_this_env_step = 0
            while cursors[p] < len(seat_rows):
                row = seat_rows[cursors[p]]
                if row.turn_index != turn_index:
                    if row.turn_index < turn_index:
                        td.extra_issues.append(
                            {
                                "seat": p,
                                "reason": "env_step_behind_replay",
                                "row_env_step": int(row.turn_index),
                                "replay_env_step": int(turn_index),
                            }
                        )
                        any_mismatch = True
                        counters["launch_issues"] += 1
                    break
                if micros_this_env_step >= int(traj.max_micro_steps):
                    td.extra_issues.append(
                        {
                            "seat": p,
                            "reason": "too_many_micro_rows_this_env_step",
                            "replay_env_step": int(turn_index),
                            "max_micro_steps": int(traj.max_micro_steps),
                        }
                    )
                    any_mismatch = True
                    counters["launch_issues"] += 1
                    break
                micros_this_env_step += 1

                # Build agent obs from the seat's current working ``virt``.
                virt = seat_states[p]._replace(
                    planets=seat_planets[p],
                    incoming_fleets=seat_incoming[p],
                )
                adapter_obs = _agent_obs_for_virt(
                    virt,
                    ego_player=p,
                    num_agents=traj.num_agents,
                    normalize_obs_to_p0=traj.normalize_obs_to_p0,
                )
                ms = _compare_obs(row.obs, adapter_obs)
                if ms:
                    any_mismatch = True
                    counters["obs_mismatches"] += len(ms)

                rd = _RowDiag(
                    seat=int(p),
                    cursor=int(cursors[p]),
                    turn_index=int(row.turn_index),
                    phase_micro_idx=int(row.phase_micro_idx),
                    halt_action=int(row.halt_action),
                    must_halt_no_ships=bool(row.must_halt_no_ships),
                    no_valid_pairs=bool(row.no_valid_pairs),
                    no_valid_fracs=bool(row.no_valid_fracs),
                    origin_slot=int(row.origin_slot),
                    target_slot=int(row.target_slot),
                    frac_idx=int(row.frac_idx),
                    ships=int(row.ships),
                    policy_eta=float(row.policy_eta),
                    virt_planets=np.array(seat_planets[p], copy=True),
                    virt_incoming=np.array(seat_incoming[p], copy=True),
                    rollout_obs=row.obs,
                    adapter_obs=adapter_obs,
                    obs_mismatches=ms,
                )

                cursors[p] += 1
                if row.halt_action == 1:
                    td.rows.append(rd)
                    break

                if row.ships <= 0:
                    td.rows.append(rd)
                    continue

                angle, tick, true_target, true_tick, ok, reason, per_frac = _resolve_angle_for_target(
                    state=virt,
                    origin_slot=row.origin_slot,
                    target_slot=row.target_slot,
                    frac_idx=row.frac_idx,
                    policy_eta=row.policy_eta,
                    ship_speed=traj.ship_speed,
                    n_rays=traj.n_rays,
                    eta_tolerance=eta_tolerance,
                )
                rd.raycast_per_frac = per_frac
                rd.resolution = {
                    "ok": bool(ok),
                    "reason": reason,
                    "angle": angle,
                    "hit_tick": tick,
                    "true_target": true_target,
                    "true_hit_tick": true_tick,
                }
                if not ok:
                    any_mismatch = True
                    counters["launch_issues"] += 1
                if angle is not None:
                    planet_id = float(seat_planets[p][row.origin_slot, 0])
                    per_seat_actions[p].append([planet_id, float(angle), float(row.ships)])
                    apply_micro_launch_in_place(
                        seat_planets[p],
                        seat_incoming[p],
                        ego_player=p,
                        origin_slot=row.origin_slot,
                        send=row.ships,
                        true_target_slot=true_target if true_target is not None else row.target_slot,
                        true_hit_tick=true_tick if true_tick is not None else row.policy_eta,
                    )

                td.rows.append(rd)

        td.final_actions = per_seat_actions
        try:
            env.step(per_seat_actions)
        except Exception as exc:
            td.env_step_error = repr(exc)
            any_mismatch = True
            counters["launch_issues"] += 1
            print(
                f"{log_prefix} iter={traj.iter_id} env={traj.env_index} turn={turn_index} "
                f"env.step failed: {exc!r}",
                flush=True,
            )
            turns.append(td)
            break

        turns.append(td)

    if any_mismatch:
        npz_path, json_path = _save_trajectory_dump(out_dir, traj, turns, counters=counters)
        counters["saved"] = 1
        print(
            f"{log_prefix} iter={traj.iter_id} env={traj.env_index} "
            f"obs_mismatches={counters['obs_mismatches']} launch_issues={counters['launch_issues']} "
            f"-> {npz_path.name} (+ {json_path.name})",
            flush=True,
        )
    return counters


def _trajectory_basename(traj: TrajectoryRecord) -> str:
    return f"iter_{traj.iter_id:06d}_env_{traj.env_index:04d}"


def _kaggle_obs_to_jsonable(obs: Mapping[str, Any]) -> Any:
    """Best-effort conversion of a Kaggle observation namespace to plain JSON.

    Kaggle returns ``Struct``-like attribute objects with ``.toJSON()`` /
    ``__dict__``; ``json.dumps`` can't serialise them directly. We coerce
    recursively. Falls back to ``repr`` for anything exotic so the dump never
    fails — we'd rather have an imperfect record than no record.
    """

    if obs is None or isinstance(obs, (bool, int, float, str)):
        return obs
    if isinstance(obs, np.ndarray):
        return obs.tolist()
    if isinstance(obs, Mapping):
        return {str(k): _kaggle_obs_to_jsonable(v) for k, v in obs.items()}
    if isinstance(obs, (list, tuple)):
        return [_kaggle_obs_to_jsonable(v) for v in obs]
    if hasattr(obs, "toJSON"):
        try:
            return _kaggle_obs_to_jsonable(obs.toJSON())
        except Exception:
            pass
    if hasattr(obs, "__dict__"):
        try:
            return _kaggle_obs_to_jsonable(dict(obs.__dict__))
        except Exception:
            pass
    try:
        return repr(obs)
    except Exception:
        return None


def _state_to_payload(state: Any, prefix: str) -> Dict[str, np.ndarray]:
    """Snapshot every ``OrbitWarsState`` field into NumPy arrays.

    Includes the canonical planet/fleet/comet tables plus the smaller scalar
    fields (step_count, num_agents, angular_velocity, done flags, …). The
    adapter's ``OrbitWarsState`` is a NamedTuple so iterating ``_fields``
    handles every field forward-compatibly.
    """

    out: Dict[str, np.ndarray] = {}
    for name in state._fields:
        val = getattr(state, name)
        try:
            arr = np.asarray(val)
        except Exception:
            continue
        out[f"{prefix}/{name}"] = arr
    return out


@dataclass
class _RowDiag:
    """Per-row diagnostic snapshot collected as the worker walks one seat's turn.

    Captures the live working ``virt`` state at the start of the row (before
    micro mutation), the adapter-built obs we compared to the rollout's
    stored obs, the structured obs mismatches, and (for non-halt rows that
    attempted a launch) the raycast outputs for every fraction tried plus the
    final resolution decision.
    """

    seat: int
    cursor: int
    turn_index: int
    phase_micro_idx: int
    halt_action: int
    must_halt_no_ships: bool
    no_valid_pairs: bool
    no_valid_fracs: bool
    origin_slot: int
    target_slot: int
    frac_idx: int
    ships: int
    policy_eta: float
    virt_planets: np.ndarray
    virt_incoming: np.ndarray
    rollout_obs: Dict[str, np.ndarray]
    adapter_obs: Dict[str, np.ndarray]
    obs_mismatches: List[Dict[str, Any]]
    raycast_per_frac: List[Dict[str, Any]] = field(default_factory=list)
    resolution: Optional[Dict[str, Any]] = None


@dataclass
class _TurnDiag:
    """Everything the worker observed about one turn during replay."""

    turn_index: int
    kaggle_step: Optional[int]
    kaggle_obs_per_seat: List[Any] = field(default_factory=list)
    initial_state_per_seat: List[Any] = field(default_factory=list)
    rows: List[_RowDiag] = field(default_factory=list)
    final_actions: List[List[List[float]]] = field(default_factory=list)
    extra_issues: List[Dict[str, Any]] = field(default_factory=list)
    env_step_error: Optional[str] = None


def _save_trajectory_dump(
    out_dir: Path,
    traj: TrajectoryRecord,
    turns: List[_TurnDiag],
    *,
    counters: Mapping[str, int],
) -> Tuple[Path, Path]:
    """Single end-of-trajectory dump containing everything diagnostic.

    The full trajectory record (per-seat per-row stored obs and launch
    metadata) is bundled alongside the turn-by-turn replay diagnostics
    (Kaggle obs per seat, adapter-reconstructed pre-step states, per-row
    working ``virt``, adapter-built obs, structured obs/launch mismatches,
    raycast outputs per attempted launch, final actions submitted to
    ``env.step``). Only called when at least one turn produced a mismatch,
    so there is no on-disk noise for clean trajectories.

    Returns ``(npz_path, json_path)``. The ``.npz`` holds all NumPy arrays
    keyed by hierarchical ``traj/...``, ``turn_<t>/...``, ``turn_<t>/row_<r>/...``
    paths; the ``.json`` sidecar holds non-array diagnostics (raw Kaggle obs,
    structured mismatch records, resolution/raycast metadata) which are not
    convenient to encode as NumPy arrays.
    """

    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    base = _trajectory_basename(traj)
    npz_path = out_dir / f"{base}.npz"
    json_path = out_dir / f"{base}.json"

    payload: Dict[str, Any] = {
        "iter_id": np.asarray(traj.iter_id, dtype=np.int64),
        "env_index": np.asarray(traj.env_index, dtype=np.int64),
        "seed": np.asarray(traj.seed, dtype=np.int64),
        "num_agents": np.asarray(traj.num_agents, dtype=np.int64),
        "ship_speed": np.asarray(traj.ship_speed, dtype=np.float64),
        "n_rays": np.asarray(traj.n_rays, dtype=np.int64),
        "normalize_obs_to_p0": np.asarray(traj.normalize_obs_to_p0, dtype=np.bool_),
        "max_micro_steps": np.asarray(int(traj.max_micro_steps), dtype=np.int64),
        "n_seats": np.asarray(len(traj.rows_per_seat), dtype=np.int64),
        "n_turns": np.asarray(len(turns), dtype=np.int64),
        "counters/turns": np.asarray(int(counters.get("turns", 0)), dtype=np.int64),
        "counters/obs_mismatches": np.asarray(int(counters.get("obs_mismatches", 0)), dtype=np.int64),
        "counters/launch_issues": np.asarray(int(counters.get("launch_issues", 0)), dtype=np.int64),
    }

    # Trajectory: per-seat stacked row arrays (compact; one allocation per seat
    # per field rather than per-row pickling).
    obs_keys = ("entity_type", "owner_idx", "features", "rope_pos", "entity_mask", "planet_mask")
    for p, rows in enumerate(traj.rows_per_seat):
        prefix = f"traj/seat{p:02d}"
        n = len(rows)
        payload[f"{prefix}/n_rows"] = np.asarray(n, dtype=np.int64)
        if n == 0:
            continue
        payload[f"{prefix}/turn_index"] = np.asarray([r.turn_index for r in rows], dtype=np.int64)
        payload[f"{prefix}/phase_micro_idx"] = np.asarray([r.phase_micro_idx for r in rows], dtype=np.int64)
        payload[f"{prefix}/halt_action"] = np.asarray([r.halt_action for r in rows], dtype=np.int64)
        payload[f"{prefix}/must_halt_no_ships"] = np.asarray([r.must_halt_no_ships for r in rows], dtype=np.bool_)
        payload[f"{prefix}/no_valid_pairs"] = np.asarray([r.no_valid_pairs for r in rows], dtype=np.bool_)
        payload[f"{prefix}/no_valid_fracs"] = np.asarray([r.no_valid_fracs for r in rows], dtype=np.bool_)
        payload[f"{prefix}/origin_slot"] = np.asarray([r.origin_slot for r in rows], dtype=np.int64)
        payload[f"{prefix}/target_slot"] = np.asarray([r.target_slot for r in rows], dtype=np.int64)
        payload[f"{prefix}/frac_idx"] = np.asarray([r.frac_idx for r in rows], dtype=np.int64)
        payload[f"{prefix}/ships"] = np.asarray([r.ships for r in rows], dtype=np.int64)
        payload[f"{prefix}/policy_eta"] = np.asarray([r.policy_eta for r in rows], dtype=np.float64)
        for key in obs_keys:
            try:
                stacked = np.stack([np.asarray(r.obs[key]) for r in rows], axis=0)
            except Exception:
                continue
            payload[f"{prefix}/obs/{key}"] = stacked

    # Turn-by-turn replay diagnostics (only the non-redundant per-turn data).
    for turn in turns:
        tprefix = f"turn_{turn.turn_index:04d}"
        payload[f"{tprefix}/kaggle_step"] = np.asarray(
            -1 if turn.kaggle_step is None else int(turn.kaggle_step), dtype=np.int64
        )
        payload[f"{tprefix}/n_rows"] = np.asarray(len(turn.rows), dtype=np.int64)
        for p, state in enumerate(turn.initial_state_per_seat):
            payload.update(_state_to_payload(state, f"{tprefix}/adapter_state_seat{p:02d}"))
        for r, rd in enumerate(turn.rows):
            rprefix = f"{tprefix}/row_{r:03d}"
            payload[f"{rprefix}/seat"] = np.asarray(rd.seat, dtype=np.int64)
            payload[f"{rprefix}/cursor"] = np.asarray(rd.cursor, dtype=np.int64)
            payload[f"{rprefix}/virt_planets"] = np.asarray(rd.virt_planets)
            payload[f"{rprefix}/virt_incoming"] = np.asarray(rd.virt_incoming)
            for key, arr in rd.adapter_obs.items():
                payload[f"{rprefix}/adapter_obs/{key}"] = np.asarray(arr)

    np.savez_compressed(npz_path, **payload)

    # JSON sidecar: non-array diagnostics. Includes raw Kaggle obs (which may
    # contain Kaggle-specific structs), structured mismatch records, raycast
    # outputs per attempted launch, and the final actions sent to ``env.step``.
    side: Dict[str, Any] = {
        "iter_id": int(traj.iter_id),
        "env_index": int(traj.env_index),
        "seed": int(traj.seed),
        "num_agents": int(traj.num_agents),
        "ship_speed": float(traj.ship_speed),
        "n_rays": int(traj.n_rays),
        "normalize_obs_to_p0": bool(traj.normalize_obs_to_p0),
        "max_micro_steps": int(traj.max_micro_steps),
        "counters": {k: int(v) for k, v in counters.items()},
        "turns": [
            {
                "turn_index": int(t.turn_index),
                "kaggle_step": None if t.kaggle_step is None else int(t.kaggle_step),
                "kaggle_obs_per_seat": [_kaggle_obs_to_jsonable(o) for o in t.kaggle_obs_per_seat],
                "rows": [
                    {
                        "seat": rd.seat,
                        "cursor": rd.cursor,
                        "turn_index": rd.turn_index,
                        "phase_micro_idx": rd.phase_micro_idx,
                        "halt_action": rd.halt_action,
                        "must_halt_no_ships": rd.must_halt_no_ships,
                        "no_valid_pairs": rd.no_valid_pairs,
                        "no_valid_fracs": rd.no_valid_fracs,
                        "origin_slot": rd.origin_slot,
                        "target_slot": rd.target_slot,
                        "frac_idx": rd.frac_idx,
                        "ships": rd.ships,
                        "policy_eta": rd.policy_eta,
                        "obs_mismatches": rd.obs_mismatches,
                        "raycast_per_frac": rd.raycast_per_frac,
                        "resolution": rd.resolution,
                    }
                    for rd in t.rows
                ],
                "final_actions": t.final_actions,
                "extra_issues": t.extra_issues,
                "env_step_error": t.env_step_error,
            }
            for t in turns
        ],
    }
    try:
        json_path.write_text(json.dumps(side, default=str, indent=2), encoding="utf-8")
    except Exception:
        json_path.write_text(repr(side), encoding="utf-8")

    return npz_path, json_path


def _worker_loop(
    in_q: "mp.Queue[Optional[TrajectoryRecord]]",
    out_dir_str: str,
    log_prefix: str,
    eta_tolerance: float,
) -> None:
    """Long-lived worker. JAX is explicitly poisoned so any accidental import
    breaks loudly instead of pulling XLA into the worker process."""

    import sys
    sys.modules["jax"] = None  # type: ignore[assignment]
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{log_prefix} worker started; out_dir={out_dir}", flush=True)
    while True:
        try:
            item = in_q.get()
        except (EOFError, KeyboardInterrupt):
            break
        if item is None:
            print(f"{log_prefix} worker shutting down", flush=True)
            break
        t0 = time.perf_counter()
        try:
            counters = _check_trajectory(
                item,
                out_dir=out_dir,
                log_prefix=log_prefix,
                eta_tolerance=float(eta_tolerance),
            )
        except Exception:
            print(f"{log_prefix} worker exception:\n{traceback.format_exc()}", flush=True)
            continue
        dt = time.perf_counter() - t0
        print(
            f"{log_prefix} iter={item.iter_id} env={item.env_index} seed={item.seed} "
            f"turns={counters['turns']} obs_mismatches={counters['obs_mismatches']} "
            f"launch_issues={counters['launch_issues']} saved={counters['saved']} "
            f"dt={dt:.2f}s",
            flush=True,
        )


class ConsistencyCheckProcess:
    """Long-lived background process; one instance per training run."""

    def __init__(
        self,
        out_dir: Path,
        *,
        queue_max: int = 4,
        eta_tolerance: float = 1.5,
        log_prefix: str = "[consistency]",
    ) -> None:
        ctx = mp.get_context("spawn")
        self._queue: "mp.Queue[Optional[TrajectoryRecord]]" = ctx.Queue(maxsize=int(queue_max))
        self._process = ctx.Process(
            target=_worker_loop,
            args=(self._queue, str(out_dir), str(log_prefix), float(eta_tolerance)),
            daemon=True,
            name="consistency-check",
        )
        self._log_prefix = str(log_prefix)
        self._process.start()

    def submit(self, traj: TrajectoryRecord) -> bool:
        try:
            self._queue.put_nowait(traj)
            return True
        except queue.Full:
            print(
                f"{self._log_prefix} queue full; dropping iter={traj.iter_id} env={traj.env_index}",
                flush=True,
            )
            return False

    def stop(self, *, timeout_s: float = 5.0) -> None:
        if not self._process.is_alive():
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._process.join(timeout=float(timeout_s))
        if self._process.is_alive():
            self._process.terminate()


def default_output_dir(ckpt_dir: Path) -> Path:
    override = os.environ.get(_DEFAULT_MISMATCH_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path(ckpt_dir).expanduser().parent / "consistency_mismatches"

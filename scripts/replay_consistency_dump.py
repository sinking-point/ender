"""Offline replay of a consistency-check dump file.

Loads the ``traj/...`` payload from a previously-saved
``<consistency_mismatches>/iter_<N>_env_<i>.npz`` (the trajectory part is
self-contained: every recorded micro row + obs + the seed needed to recreate
the official Kaggle env) and re-runs the worker's ``_check_trajectory`` against
it synchronously. Useful for iterating on the consistency checker itself
without waiting for a fresh training-time mismatch.

Usage::

    python -m scripts.replay_consistency_dump <dump.npz> [--out DIR] [--eta-tol 1.5]

The replay writes its own dump pair only if any mismatch fires (same policy as
the live worker). The original dump is never touched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _load_trajectory_from_dump(npz_path: Path) -> "Any":
    """Reconstruct a ``TrajectoryRecord`` from a worker-emitted dump.

    The dump's ``traj/seat<P>/...`` layout mirrors the per-seat row arrays the
    worker received over the queue, so we just stack them back into
    ``MicroRow`` instances. Anything outside ``traj/...`` (turn diagnostics,
    Kaggle obs sidecar) is irrelevant for replay -- the worker rebuilds those
    from scratch when it walks the trajectory through the official env.
    """

    # Import lazily so this script can be invoked without poking JAX globally.
    from orbit_wars_pt.consistency_check import MicroRow, TrajectoryRecord

    z = np.load(npz_path, allow_pickle=False)

    iter_id = int(z["iter_id"])
    env_index = int(z["env_index"])
    seed = int(z["seed"])
    num_agents = int(z["num_agents"])
    ship_speed = float(z["ship_speed"])
    n_rays = int(z["n_rays"])
    normalize_obs_to_p0 = bool(z["normalize_obs_to_p0"])
    max_micro_steps = int(z["max_micro_steps"]) if "max_micro_steps" in z.files else 8
    n_seats = int(z["n_seats"])

    obs_keys = ("entity_type", "owner_idx", "features", "rope_pos", "entity_mask", "planet_mask")

    rows_per_seat: List[List[Any]] = []
    for p in range(n_seats):
        prefix = f"traj/seat{p:02d}"
        n_rows = int(z[f"{prefix}/n_rows"])
        seat_rows: List[Any] = []
        if n_rows == 0:
            rows_per_seat.append(seat_rows)
            continue
        turn_index = z[f"{prefix}/turn_index"]
        phase_micro_idx = z[f"{prefix}/phase_micro_idx"]
        halt_action = z[f"{prefix}/halt_action"]
        must_halt_no_ships = z[f"{prefix}/must_halt_no_ships"]
        no_valid_pairs = z[f"{prefix}/no_valid_pairs"]
        no_valid_fracs = z[f"{prefix}/no_valid_fracs"]
        origin_slot = z[f"{prefix}/origin_slot"]
        target_slot = z[f"{prefix}/target_slot"]
        frac_idx = z[f"{prefix}/frac_idx"]
        ships = z[f"{prefix}/ships"]
        policy_eta = z[f"{prefix}/policy_eta"]
        obs_per_key: Dict[str, np.ndarray] = {}
        for key in obs_keys:
            arr_key = f"{prefix}/obs/{key}"
            if arr_key in z.files:
                obs_per_key[key] = z[arr_key]
        for r in range(n_rows):
            row_obs = {key: np.asarray(arr[r]) for key, arr in obs_per_key.items()}
            seat_rows.append(
                MicroRow(
                    turn_index=int(turn_index[r]),
                    phase_micro_idx=int(phase_micro_idx[r]),
                    halt_action=int(halt_action[r]),
                    must_halt_no_ships=bool(must_halt_no_ships[r]),
                    no_valid_pairs=bool(no_valid_pairs[r]),
                    no_valid_fracs=bool(no_valid_fracs[r]),
                    origin_slot=int(origin_slot[r]),
                    target_slot=int(target_slot[r]),
                    frac_idx=int(frac_idx[r]),
                    ships=int(ships[r]),
                    policy_eta=float(policy_eta[r]),
                    obs=row_obs,
                )
            )
        rows_per_seat.append(seat_rows)

    return TrajectoryRecord(
        iter_id=iter_id,
        env_index=env_index,
        seed=seed,
        num_agents=num_agents,
        ship_speed=ship_speed,
        n_rays=n_rays,
        normalize_obs_to_p0=normalize_obs_to_p0,
        max_micro_steps=max_micro_steps,
        rows_per_seat=rows_per_seat,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=str, help="Path to a consistency dump .npz file.")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for any new dump emitted by the replay (default: alongside the input).",
    )
    parser.add_argument(
        "--eta-tol",
        type=float,
        default=1.5,
        help="Allowed eta error when resolving launch angles, in ticks.",
    )
    args = parser.parse_args()

    # Block any accidental JAX import path inside this process: the worker
    # logic is supposed to be JAX-free and we want the replay to exercise the
    # same surface area.
    sys.modules["jax"] = None  # type: ignore[assignment]

    from orbit_wars_pt.consistency_check import _check_trajectory

    in_path = Path(args.dump).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"dump file not found: {in_path}")
    out_dir = Path(args.out).expanduser().resolve() if args.out else in_path.parent / "replay"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading trajectory from {in_path} ...", flush=True)
    traj = _load_trajectory_from_dump(in_path)
    n_rows = sum(len(rs) for rs in traj.rows_per_seat)
    print(
        f"loaded: iter={traj.iter_id} env={traj.env_index} seed={traj.seed} "
        f"num_agents={traj.num_agents} n_seats={len(traj.rows_per_seat)} total_rows={n_rows}",
        flush=True,
    )
    print(f"replaying through Kaggle env; out_dir={out_dir} eta_tol={args.eta_tol} ...", flush=True)
    counters = _check_trajectory(
        traj,
        out_dir=out_dir,
        log_prefix="[replay]",
        eta_tolerance=float(args.eta_tol),
    )
    print(
        f"done: turns={counters.get('turns', 0)} "
        f"obs_mismatches={counters.get('obs_mismatches', 0)} "
        f"launch_issues={counters.get('launch_issues', 0)} "
        f"saved={counters.get('saved', 0)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

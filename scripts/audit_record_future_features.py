from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from orbit_wars_pt.constants import FUTURE_PLANET_FEATURES_PER_TICK, INCOMING_TA_BINS, MAX_PLANETS
from orbit_wars_pt.kaggle_adapter import (
    _first_hit_targets_np,
    _make_sim_state,
    _obs_tensors_for_state,
    _planned_send,
    _public_obs_from_sim_state,
    _simulate_joint_step_with_kaggle_model,
    observation_to_state,
)
from orbit_wars_pt.model import OrbitWarsPolicy, build_future_planet_features


def _evenly_sample(values: list[int], limit: int) -> list[int]:
    if limit <= 0 or len(values) <= limit:
        return list(values)
    if limit == 1:
        return [values[len(values) // 2]]
    out: list[int] = []
    last = -1
    for i in range(limit):
        idx = round(i * (len(values) - 1) / (limit - 1))
        if idx != last:
            out.append(values[idx])
            last = idx
    return out


def _tick_slice(flat: np.ndarray, tick: int) -> np.ndarray:
    start = tick * FUTURE_PLANET_FEATURES_PER_TICK
    stop = start + FUTURE_PLANET_FEATURES_PER_TICK
    return flat[..., start:stop]


def _owner_garrison_from_future(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    owner = np.argmax(flat[..., :5], axis=-1).astype(np.int64)
    garrison = flat[..., -1].astype(np.float64)
    return owner, garrison


def _owner_garrison_from_obs_batch(batch: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    owner = batch["owner_idx"][0, 1 : 1 + MAX_PLANETS].cpu().numpy().astype(np.int64)
    garrison = batch["features"][0, 1 : 1 + MAX_PLANETS, 1].cpu().numpy().astype(np.float64)
    return owner, garrison


def _future_batches(
    public_obs: dict[str, Any],
    config: dict[str, Any],
    *,
    num_agents: int,
    step_count: int,
    ego_player: int,
    horizon: int,
    first_step_actions: list[list[list[float]]] | None = None,
) -> list[dict[str, torch.Tensor]]:
    sim_state = _make_sim_state(public_obs, num_agents=num_agents, step_count=step_count)
    sim_step = int(step_count)
    out: list[dict[str, torch.Tensor]] = []
    for tick in range(horizon):
        joint_actions = [[] for _ in range(num_agents)] if first_step_actions is None or tick > 0 else first_step_actions
        _simulate_joint_step_with_kaggle_model(sim_state, joint_actions=joint_actions, config=config)
        sim_step += 1
        public_next = _public_obs_from_sim_state(sim_state, step_count=sim_step)
        state_next = observation_to_state(
            public_next,
            config,
            max_fleets=512 + len(public_next.get("fleets", []) or []),
            step_count_override=sim_step,
            num_agents_override=num_agents,
        )
        out.append(
            _obs_tensors_for_state(
                state_next,
                ego_player,
                torch.device("cpu"),
                policy_player_count=num_agents,
                target_abort_enabled=False,
                normalize_obs_to_p0=False,
            )
        )
    return out


def _planet_future_metrics(
    batch0: dict[str, torch.Tensor],
    future_batches: list[dict[str, torch.Tensor]],
) -> dict[str, float]:
    pred = build_future_planet_features(batch0["owner_idx"], batch0["features"])[0].cpu().numpy()
    owner_matches = 0
    owner_total = 0
    garrison_abs = []
    for tick, batch_t in enumerate(future_batches):
        pred_owner, pred_garrison = _owner_garrison_from_future(_tick_slice(pred, tick))
        act_owner, act_garrison = _owner_garrison_from_obs_batch(batch_t)
        active = batch_t["features"][0, 1 : 1 + MAX_PLANETS, 4].cpu().numpy() > 0.5
        owner_matches += int(np.sum((pred_owner == act_owner) & active))
        owner_total += int(np.sum(active))
        garrison_abs.append(np.abs(pred_garrison[active] - act_garrison[active]))
    garrison_all = np.concatenate(garrison_abs) if garrison_abs else np.zeros((0,), dtype=np.float64)
    return {
        "owner_match_rate": float(owner_matches / max(owner_total, 1)),
        "garrison_mae": float(garrison_all.mean()) if garrison_all.size else 0.0,
        "garrison_max_abs": float(garrison_all.max()) if garrison_all.size else 0.0,
    }


def _infer_frac_idx(ships_avail: float, send: int) -> int | None:
    matches = [idx for idx in range(5) if _planned_send(float(ships_avail), idx) == int(send)]
    return matches[0] if matches else None


def _angle_diff(a: float, b: float) -> float:
    d = abs(float(a) - float(b)) % (2.0 * math.pi)
    return min(d, 2.0 * math.pi - d)


def _target_future_metrics(
    public_obs: dict[str, Any],
    config: dict[str, Any],
    *,
    num_agents: int,
    step_count: int,
    ego_player: int,
    action: list[float],
) -> dict[str, Any] | None:
    state = observation_to_state(
        public_obs,
        config,
        max_fleets=512 + len(public_obs.get("fleets", []) or []),
        step_count_override=step_count,
        num_agents_override=num_agents,
    )
    batch0 = _obs_tensors_for_state(
        state,
        ego_player,
        torch.device("cpu"),
        policy_player_count=num_agents,
        target_abort_enabled=False,
        normalize_obs_to_p0=False,
    )
    planets = np.asarray(state.planets)
    origin_pid = int(round(float(action[0])))
    origin_slot_arr = np.where(np.abs(planets[:, 0] - float(origin_pid)) < 0.5)[0]
    if origin_slot_arr.size == 0:
        return None
    origin_slot = int(origin_slot_arr[0])
    ships_avail = float(planets[origin_slot, 5])
    send = int(round(float(action[2])))
    frac_idx = _infer_frac_idx(ships_avail, send)
    if frac_idx is None:
        return None

    ray_angle, ray_valid, ray_hit_tick, true_planet, true_hit_tick = _first_hit_targets_np(
        state,
        origin_slot,
        frac_idx,
        ship_speed=float(config.get("shipSpeed", 6.0)),
        horizon=INCOMING_TA_BINS,
        target_method=str(config.get("target_method", "interval")),
    )
    valid_slots = np.nonzero(ray_valid)[0]
    if valid_slots.size == 0:
        return None
    action_angle = float(action[1]) % (2.0 * math.pi)
    selected_slot = min(valid_slots.tolist(), key=lambda idx: _angle_diff(action_angle, float(ray_angle[idx])))
    angle_error = _angle_diff(action_angle, float(ray_angle[selected_slot]))

    dummy_policy = OrbitWarsPolicy(
        d_model=32,
        n_heads=4,
        n_layers=1,
        feature_dim=int(batch0["features"].shape[-1]),
        future_feature_enabled=True,
    ).cpu().eval()
    with torch.no_grad():
        origin_future, target_future = dummy_policy._target_future_context(  # type: ignore[attr-defined]
            batch0["owner_idx"],
            batch0["features"],
            torch.tensor([origin_slot], dtype=torch.long),
            torch.tensor([float(send)], dtype=torch.float32),
            torch.from_numpy(ray_hit_tick[None, :].astype(np.float32)),
        )
    pred_origin = origin_future[0, selected_slot].cpu().numpy()
    pred_target = target_future[0, selected_slot].cpu().numpy()

    joint_actions = [[action]] + [[] for _ in range(num_agents - 1)]
    future_batches = _future_batches(
        public_obs,
        config,
        num_agents=num_agents,
        step_count=step_count,
        ego_player=ego_player,
        horizon=INCOMING_TA_BINS,
        first_step_actions=joint_actions,
    )

    origin_owner_matches = 0
    origin_abs = []
    target_owner_matches = 0
    target_abs = []
    true_slot = int(true_planet[selected_slot]) if int(true_planet[selected_slot]) >= 0 else -1
    for tick, batch_t in enumerate(future_batches):
        po, pg = _owner_garrison_from_future(_tick_slice(pred_origin, tick))
        to, tg = _owner_garrison_from_future(_tick_slice(pred_target, tick))
        act_owner, act_garrison = _owner_garrison_from_obs_batch(batch_t)
        origin_owner_matches += int(int(po) == int(act_owner[origin_slot]))
        origin_abs.append(abs(float(pg) - float(act_garrison[origin_slot])))
        if 0 <= true_slot < MAX_PLANETS:
            target_owner_matches += int(int(to) == int(act_owner[true_slot]))
            target_abs.append(abs(float(tg) - float(act_garrison[true_slot])))

    return {
        "origin_slot": origin_slot,
        "frac_idx": int(frac_idx),
        "send": int(send),
        "selected_target_slot": int(selected_slot),
        "true_target_slot": int(true_slot),
        "policy_hit_tick": float(ray_hit_tick[selected_slot]),
        "true_hit_tick": float(true_hit_tick[selected_slot]),
        "angle_error": float(angle_error),
        "origin_owner_match_rate": float(origin_owner_matches / INCOMING_TA_BINS),
        "origin_garrison_mae": float(np.mean(origin_abs)) if origin_abs else 0.0,
        "target_owner_match_rate": float(target_owner_matches / INCOMING_TA_BINS) if target_abs else 0.0,
        "target_garrison_mae": float(np.mean(target_abs)) if target_abs else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    parser.add_argument("--player", type=int, default=0)
    parser.add_argument("--planet-samples", type=int, default=6)
    parser.add_argument("--target-samples", type=int, default=6)
    args = parser.parse_args()

    with args.record.open("r", encoding="utf-8") as f:
        record = json.load(f)

    config = dict(record.get("configuration") or {})
    config["target_method"] = str(__import__("os").environ.get("ORBIT_WARS_TARGET_METHOD", "interval"))
    num_agents = int(config.get("agentCount", 2))
    player = int(args.player)
    steps = list(record["steps"])

    active_turns = [
        idx
        for idx, step in enumerate(steps[:-1])
        if step[player].get("status") == "ACTIVE" and "step" in step[player]["observation"]
    ]
    launch_turns = [
        idx
        for idx in active_turns
        if idx > 0
        and steps[idx - 1][player].get("status") == "ACTIVE"
        and "step" in steps[idx - 1][player].get("observation", {})
        and len(step_action := (steps[idx][player].get("action") or [])) == 1
        and len(step_action[0]) >= 3
    ]

    sampled_planet_turns = _evenly_sample(active_turns, int(args.planet_samples))
    sampled_target_turns = _evenly_sample(launch_turns, int(args.target_samples))

    print(f"record={args.record} player={player} active_turns={len(active_turns)} single_launch_turns={len(launch_turns)}")
    print(f"sampled_planet_turns={sampled_planet_turns}")
    print(f"sampled_target_turns={sampled_target_turns}")

    planet_reports: list[dict[str, Any]] = []
    for turn in sampled_planet_turns:
        public_obs = dict(steps[turn][player]["observation"])
        step_count = int(public_obs["step"])
        state = observation_to_state(
            public_obs,
            config,
            max_fleets=512 + len(public_obs.get("fleets", []) or []),
            step_count_override=step_count,
            num_agents_override=num_agents,
        )
        batch0 = _obs_tensors_for_state(
            state,
            player,
            torch.device("cpu"),
            policy_player_count=num_agents,
            target_abort_enabled=False,
            normalize_obs_to_p0=False,
        )
        future_batches = _future_batches(
            public_obs,
            config,
            num_agents=num_agents,
            step_count=step_count,
            ego_player=player,
            horizon=INCOMING_TA_BINS,
        )
        metrics = _planet_future_metrics(batch0, future_batches)
        planet_reports.append({"turn": int(turn), "step": step_count, **metrics})

    target_reports: list[dict[str, Any]] = []
    skipped_target_turns = 0
    for turn in sampled_target_turns:
        public_obs = dict(steps[turn - 1][player]["observation"])
        step_count = int(public_obs["step"])
        action = list((steps[turn][player].get("action") or [])[0])
        metrics = _target_future_metrics(
            public_obs,
            config,
            num_agents=num_agents,
            step_count=step_count,
            ego_player=player,
            action=action,
        )
        if metrics is not None:
            target_reports.append({"turn": int(turn), "obs_step": step_count, "action": action, **metrics})
        else:
            skipped_target_turns += 1

    print("\nPlanet Future Audit")
    for rep in planet_reports:
        print(
            f" turn={rep['turn']:3d} step={rep['step']:3d}"
            f" owner_match={rep['owner_match_rate']:.4f}"
            f" garrison_mae={rep['garrison_mae']:.6f}"
            f" garrison_max={rep['garrison_max_abs']:.6f}"
        )

    print("\nTarget Future Audit")
    print(f" skipped_target_turns={skipped_target_turns}")
    for rep in target_reports:
        print(
            f" turn={rep['turn']:3d} obs_step={rep['obs_step']:3d}"
            f" origin_slot={rep['origin_slot']:2d} frac={rep['frac_idx']} send={rep['send']:3d}"
            f" selected={rep['selected_target_slot']:2d} true={rep['true_target_slot']:2d}"
            f" policy_eta={rep['policy_hit_tick']:.1f} true_eta={rep['true_hit_tick']:.1f}"
            f" angle_err={rep['angle_error']:.3e}"
            f" origin_owner={rep['origin_owner_match_rate']:.4f}"
            f" origin_mae={rep['origin_garrison_mae']:.6f}"
            f" target_owner={rep['target_owner_match_rate']:.4f}"
            f" target_mae={rep['target_garrison_mae']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

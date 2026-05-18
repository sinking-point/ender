#!/usr/bin/env python3
"""Replay submit_refine at selfplay_tangent step 151 (player 0, micro_idx=1).

Reproduces the ``primary interval aim invalid`` warning for slot 28 and prints
diagnostics on why the polished primary misses while edge_same_side succeeds.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from orbit_wars_pt.constants import INCOMING_TA_BINS
from orbit_wars_pt.interval_geometry_np import (
    _candidate_tick_single_planet,
    _closest_angle_in_cells,
    _edge_angle_inside_furthest_on_side,
    _event_first_contact_angle,
    _pick_planet_aim_from_visible_cells,
    _polish_boundary_ground_truth,
    _signed_angle_delta,
    first_hit_tick_single_planet_raycast,
    set_subtract_cells,
    union_angle_intervals,
)
from orbit_wars_pt.kaggle_adapter import (
    _build_interval_micro_geometry,
    _first_hit_targets_np,
    _launch_geometry_from_obs,
    _planned_send,
    observation_to_state,
)
from orbit_wars_pt.orthogonal_geometry_np import cone_to_angle_intervals

RECORD = Path("records/selfplay_tangent.json")
GAME_STEP = 151
EGO_PLAYER = 0
ORIGIN_SLOT = 25
FRAC_IDX = 2
MICRO_IDX = 1
SELECTED_TARGET = 24
SLOT_DEBUG = 28
FIRST_SEND = 1  # micro 0 launch from step 152 record


def _fmt_angle(a: float | None) -> str:
    if a is None:
        return "None"
    return f"{a:.6f} ({math.degrees(a):.2f}°)"


def main() -> None:
    record = json.loads(RECORD.read_text())
    cfg = record["configuration"]
    ship_speed = float(cfg["shipSpeed"])

    obs = record["steps"][GAME_STEP][EGO_PLAYER]["observation"]
    state = observation_to_state(obs, cfg, step_count_override=GAME_STEP)
    launch_geometry = _launch_geometry_from_obs(obs, cfg)

    planets = np.array(state.planets, copy=True)
    planets[ORIGIN_SLOT, 5] -= float(FIRST_SEND)
    virt = state._replace(planets=planets)

    print("=== Context ===")
    print(
        f"step={GAME_STEP} ego={EGO_PLAYER} origin={ORIGIN_SLOT} frac={FRAC_IDX} "
        f"(send={_planned_send(planets[ORIGIN_SLOT, 5] + FIRST_SEND, FRAC_IDX)} after micro0) "
        f"micro={MICRO_IDX} selected_target={SELECTED_TARGET}"
    )
    print(f"origin ships after micro0 booking: {planets[ORIGIN_SLOT, 5]}")

    ray_angle, ray_valid, ray_hit_tick, true_planet, true_hit_tick = _first_hit_targets_np(
        virt,
        ORIGIN_SLOT,
        FRAC_IDX,
        ship_speed=ship_speed,
        horizon=INCOMING_TA_BINS,
        n_rays=360,
        samples_per_span=9,
        target_method="interval",
        game_step=GAME_STEP,
        micro_idx=MICRO_IDX,
        ego_player=EGO_PLAYER,
        launch_geometry=launch_geometry,
        refine_boundaries=True,
        phase="submit_refine",
        selected_target_slot=SELECTED_TARGET,
    )
    print("\n=== submit_refine sweep (selected target) ===")
    print(
        f"slot {SELECTED_TARGET}: valid={bool(ray_valid[SELECTED_TARGET])} "
        f"angle={_fmt_angle(float(ray_angle[SELECTED_TARGET]))} "
        f"tick={ray_hit_tick[SELECTED_TARGET]} true={true_planet[SELECTED_TARGET]}"
    )

    geom = _build_interval_micro_geometry(
        virt,
        ORIGIN_SLOT,
        FRAC_IDX,
        ship_speed=ship_speed,
        horizon=INCOMING_TA_BINS,
        samples_per_span=9,
        target_timing=None,
        launch_geometry=launch_geometry,
    )
    events = geom.events or []
    order_rank = {int(s): i for i, s in enumerate(geom.object_order)}
    sorted_events = sorted(
        events,
        key=lambda e: (
            int(math.floor(e.t)) if math.isfinite(e.t) else 0,
            order_rank.get(e.slot, 10_000) if e.kind == "planet" else 10_001,
        ),
    )

    blocked: list = []
    slot_event = None
    for event in sorted_events:
        hit = cone_to_angle_intervals(event.angle_lo, event.angle_hi)
        if event.kind == "planet" and int(event.slot) == SLOT_DEBUG:
            slot_event = event
            cells = set_subtract_cells(hit, blocked)
            break
        blocked = union_angle_intervals([*blocked, *hit])

    if slot_event is None:
        raise SystemExit(f"no sweep event for slot {SLOT_DEBUG}")

    theta_fc = _event_first_contact_angle(slot_event)
    cells = list(cells)
    target_sig = ("planet", SLOT_DEBUG)

    print(f"\n=== Slot {SLOT_DEBUG} visible cells at first contact ===")
    print(f"first_contact_angle={_fmt_angle(theta_fc)}")
    for lo, hi in cells:
        print(f"  cell [{_fmt_angle(lo)}, {_fmt_angle(hi)}] width={hi - lo:.6f}")

    aim, tick, width = _pick_planet_aim_from_visible_cells(
        cells,
        theta_fc,
        refine_boundaries=True,
        cache=geom.occlusion_cache,
        origin_xy=geom.origin_xy,
        origin_radius=geom.origin_radius,
        speed=geom.speed,
        p0_by_tick=geom.p0_by_tick,
        p1_by_tick=geom.p1_by_tick,
        radii=geom.radii,
        active_by_tick=geom.active_by_tick,
        slot=SLOT_DEBUG,
        object_order=geom.object_order,
        debug_context={
            "phase": "submit_refine",
            "game_step": GAME_STEP,
            "ego_player": EGO_PLAYER,
            "origin_slot": ORIGIN_SLOT,
            "frac_idx": FRAC_IDX,
            "micro_idx": MICRO_IDX,
            "selected_target_slot": SELECTED_TARGET,
        },
    )
    print(f"\n_pick_planet_aim_from_visible_cells -> angle={_fmt_angle(aim)} tick={tick} width={width}")

    theta_primary = _closest_angle_in_cells(
        theta_fc,
        cells,
        refine_boundaries=True,
        target_sig=target_sig,
        cache=geom.occlusion_cache,
        origin_xy=geom.origin_xy,
        origin_radius=geom.origin_radius,
        speed=geom.speed,
        p0_by_tick=geom.p0_by_tick,
        p1_by_tick=geom.p1_by_tick,
        radii=geom.radii,
        active_by_tick=geom.active_by_tick,
        object_order=geom.object_order,
        order_rank=order_rank,
    )
    print(f"\n_closest_angle_in_cells (primary path) -> {_fmt_angle(theta_primary)}")

    lo, hi = cells[0]
    for bound in (lo, hi):
        raw = bound
        outward = -1.0 if abs(raw - lo) <= abs(raw - hi) else 1.0
        polished = _polish_boundary_ground_truth(
            raw,
            owner_cell=(lo, hi),
            target_sig=target_sig,
            outward_dir=outward,
            cache=geom.occlusion_cache,
            origin_xy=geom.origin_xy,
            origin_radius=geom.origin_radius,
            speed=geom.speed,
            p0_by_tick=geom.p0_by_tick,
            p1_by_tick=geom.p1_by_tick,
            radii=geom.radii,
            active_by_tick=geom.active_by_tick,
            object_order=geom.object_order,
            order_rank=order_rank,
        )
        _, tick_b, reason_b = _candidate_tick_single_planet(
            polished,
            origin_xy=geom.origin_xy,
            origin_radius=geom.origin_radius,
            speed=geom.speed,
            p0_by_tick=geom.p0_by_tick,
            p1_by_tick=geom.p1_by_tick,
            radii=geom.radii,
            active_by_tick=geom.active_by_tick,
            slot=SLOT_DEBUG,
        )
        print(
            f"  polish bound={_fmt_angle(raw)} outward={outward:+.0f} -> "
            f"{_fmt_angle(polished)} tick={tick_b} reason={reason_b}"
        )

    edge = _edge_angle_inside_furthest_on_side(
        theta_fc,
        cells,
        side=1,
        refine_boundaries=True,
        target_sig=target_sig,
        cache=geom.occlusion_cache,
        origin_xy=geom.origin_xy,
        origin_radius=geom.origin_radius,
        speed=geom.speed,
        p0_by_tick=geom.p0_by_tick,
        p1_by_tick=geom.p1_by_tick,
        radii=geom.radii,
        active_by_tick=geom.active_by_tick,
        object_order=geom.object_order,
        order_rank=order_rank,
    )
    print(f"\n_edge_same_side -> {_fmt_angle(edge)}")

    for label, theta in (("primary", theta_primary), ("edge_same", edge)):
        if theta is None:
            continue
        tick = first_hit_tick_single_planet_raycast(
            float(theta),
            geom.origin_xy,
            geom.origin_radius,
            geom.speed,
            geom.p0_by_tick[:, SLOT_DEBUG, :],
            geom.p1_by_tick[:, SLOT_DEBUG, :],
            float(geom.radii[SLOT_DEBUG]),
            geom.active_by_tick[:, SLOT_DEBUG],
        )
        print(f"  raycast slot {SLOT_DEBUG} @ {label}: tick={tick}")

    print("\n=== Angle vs first_contact (signed delta) ===")
    for name, theta in (
        ("first_contact", theta_fc),
        ("primary", theta_primary),
        ("edge_same", edge),
        ("recorded_bad_primary", 2.299127498029492),
        ("recorded_ok_edge", 2.7162634336094795),
    ):
        if theta is None:
            continue
        print(f"  {name}: delta_fc={_signed_angle_delta(theta, theta_fc):+.6f}")


if __name__ == "__main__":
    main()

"""Regression tests for tangent launch precision around fleet_id=574.

This case comes from ``records/selfplay_tangent.json``. The recorded fleet
trajectory forecasts to planet slot 12 from step 206. Reconstructing the launch
decision requires the step-205 observation plus an explicit ``step_count``
override, because the player-1 saved observation does not include ``step``.

With the correct launch-time step override, both the whole-board launch-side
raycast and the local target-vs-competitor refinement currently classify the
launch as slot 16 instead. These tests pin that mismatch.

Run::

    python -m unittest orbit_wars_pt.test_tangent_launch_regression -v
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from orbit_wars_pt.constants import INCOMING_TA_BINS
from orbit_wars_pt.interval_geometry_np import _local_first_hit_signature_at_angle
from orbit_wars_pt.kaggle_adapter import (
    _build_interval_micro_geometry,
    _discrete_first_hit_at_angle_np,
    _launch_geometry_from_obs,
    observation_to_state,
)


_RECORD_PATH = Path("records/selfplay_tangent.json")
_FLEET_ID = 574
_LAUNCH_STEP = 205
_LAUNCH_OBS_AGENT = 1
_FORECAST_STEP = 206
_FORECAST_OBS_AGENT = 0
_ORIGIN_SLOT = 23
_FRAC_IDX = 1
_TARGET_SLOT = 16
_COMPETITOR_SLOT = 12


def _load_record() -> dict:
    if not _RECORD_PATH.exists():
        raise unittest.SkipTest(f"missing regression record: {_RECORD_PATH}")
    record = json.loads(_RECORD_PATH.read_text())
    steps = record.get("steps")
    if not isinstance(steps, list) or len(steps) <= max(_LAUNCH_STEP, _FORECAST_STEP):
        raise unittest.SkipTest(
            f"regression record does not contain steps {_LAUNCH_STEP}/{_FORECAST_STEP}: {_RECORD_PATH}"
        )
    return record


def _obs(record: dict, step: int, agent_idx: int) -> dict:
    return record["steps"][step][agent_idx]["observation"]


def _fleet_row_and_index(obs: dict, fleet_id: int) -> tuple[int, list[float]]:
    fleets = obs.get("fleets") or []
    for i, row in enumerate(fleets):
        if int(row[0]) == fleet_id:
            return i, row
    raise AssertionError(f"fleet_id={fleet_id} not found")


def _launch_angle(record: dict) -> float:
    actions = record["steps"][_FORECAST_STEP][_LAUNCH_OBS_AGENT].get("action") or []
    for row in actions:
        if int(row[0]) == _ORIGIN_SLOT and int(row[2]) == 36:
            return float(row[1])
    raise AssertionError("expected launch action for fleet_id=574 not found")


def _format_projected_slot_path(geom, slot: int) -> str:
    lines = [f"projected slot {slot} path:"]
    ticks = int(geom.p0_by_tick.shape[0])
    for tick in range(ticks):
        if not bool(geom.active_by_tick[tick, slot]):
            continue
        p0 = geom.p0_by_tick[tick, slot]
        p1 = geom.p1_by_tick[tick, slot]
        lines.append(
            "  tick="
            f"{tick:02d} "
            f"p0=({p0[0]!r}, {p0[1]!r}) "
            f"p1=({p1[0]!r}, {p1[1]!r})"
        )
    return "\n".join(lines)


class TestTangentLaunchRegression(unittest.TestCase):
    def test_recorded_forecast_for_fleet_574_hits_planet_12(self) -> None:
        record = _load_record()
        obs = _obs(record, _FORECAST_STEP, _FORECAST_OBS_AGENT)
        idx, row = _fleet_row_and_index(obs, _FLEET_ID)

        per_fleet_arrival = np.full((len(obs.get("fleets") or []), 2), -9, dtype=np.int32)
        observation_to_state(
            obs,
            record["configuration"],
            fleet_forecast_arrival=per_fleet_arrival,
        )

        self.assertEqual(int(row[5]), _ORIGIN_SLOT)
        self.assertEqual(per_fleet_arrival[idx].tolist(), [_COMPETITOR_SLOT, 15])

    def test_launch_side_projected_slot_16_matches_record_at_first_future_step(self) -> None:
        record = _load_record()
        obs = _obs(record, _LAUNCH_STEP, _LAUNCH_OBS_AGENT)
        state = observation_to_state(
            obs,
            record["configuration"],
            step_count_override=_LAUNCH_STEP,
        )
        launch_geometry = _launch_geometry_from_obs(obs, record["configuration"])
        geom = _build_interval_micro_geometry(
            state,
            _ORIGIN_SLOT,
            _FRAC_IDX,
            ship_speed=6.0,
            horizon=INCOMING_TA_BINS,
            samples_per_span=9,
            target_timing=None,
            launch_geometry=launch_geometry,
        )

        next_obs = _obs(record, _FORECAST_STEP, _FORECAST_OBS_AGENT)
        row = next_obs["planets"][_TARGET_SLOT]
        predicted = geom.p1_by_tick[0, _TARGET_SLOT]

        self.assertEqual(
            (float(predicted[0]), float(predicted[1])),
            (float(row[2]), float(row[3])),
            "launch-side projected slot 16 position at first future step does not match the record",
        )

    def test_launch_side_whole_board_raycast_for_same_angle_should_hit_planet_12(self) -> None:
        record = _load_record()
        obs = _obs(record, _LAUNCH_STEP, _LAUNCH_OBS_AGENT)
        state = observation_to_state(
            obs,
            record["configuration"],
            step_count_override=_LAUNCH_STEP,
        )
        launch_geometry = _launch_geometry_from_obs(obs, record["configuration"])
        geom = _build_interval_micro_geometry(
            state,
            _ORIGIN_SLOT,
            _FRAC_IDX,
            ship_speed=6.0,
            horizon=INCOMING_TA_BINS,
            samples_per_span=9,
            target_timing=None,
            launch_geometry=launch_geometry,
        )
        angle = _launch_angle(record)

        hit = _discrete_first_hit_at_angle_np(
            angle,
            geom.origin_xy,
            geom.origin_radius,
            geom.speed,
            geom.p0_by_tick,
            geom.p1_by_tick,
            geom.radii,
            geom.active_by_tick,
            state.planet_collision_rank,
            horizon=INCOMING_TA_BINS,
        )
        print(_format_projected_slot_path(geom, _TARGET_SLOT))

        self.assertEqual(
            hit[:2],
            ("planet", _COMPETITOR_SLOT),
            "whole-board launch-side raycast disagrees with the recorded fleet forecast\n"
            + _format_projected_slot_path(geom, _TARGET_SLOT),
        )

    def test_local_two_body_refinement_for_same_angle_should_hit_planet_12(self) -> None:
        record = _load_record()
        obs = _obs(record, _LAUNCH_STEP, _LAUNCH_OBS_AGENT)
        state = observation_to_state(
            obs,
            record["configuration"],
            step_count_override=_LAUNCH_STEP,
        )
        launch_geometry = _launch_geometry_from_obs(obs, record["configuration"])
        geom = _build_interval_micro_geometry(
            state,
            _ORIGIN_SLOT,
            _FRAC_IDX,
            ship_speed=6.0,
            horizon=INCOMING_TA_BINS,
            samples_per_span=9,
            target_timing=None,
            launch_geometry=launch_geometry,
        )
        angle = _launch_angle(record)

        sig = _local_first_hit_signature_at_angle(
            angle,
            target_slot=_TARGET_SLOT,
            competitor_sig=("planet", _COMPETITOR_SLOT),
            origin_xy=geom.origin_xy,
            origin_radius=geom.origin_radius,
            speed=geom.speed,
            p0_by_tick=geom.p0_by_tick,
            p1_by_tick=geom.p1_by_tick,
            radii=geom.radii,
            active_by_tick=geom.active_by_tick,
            object_order=geom.object_order,
        )
        print(_format_projected_slot_path(geom, _TARGET_SLOT))

        self.assertEqual(
            sig,
            ("planet", _COMPETITOR_SLOT),
            "local target-vs-competitor refinement disagrees with the recorded fleet forecast\n"
            + _format_projected_slot_path(geom, _TARGET_SLOT),
        )


if __name__ == "__main__":
    unittest.main()

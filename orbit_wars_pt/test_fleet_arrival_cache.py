from __future__ import annotations

import numpy as np

import orbit_wars_pt.kaggle_adapter as ka


def _base_obs() -> dict[str, object]:
    return {
        "player": 0,
        "step": 10,
        "angular_velocity": 0.0,
        "planets": [
            [0, 0, 20.0, 50.0, 5.0, 60.0, 1.0],
            [1, -1, 80.0, 50.0, 5.0, 40.0, 1.0],
        ],
        "initial_planets": [
            [0, 0, 20.0, 50.0, 5.0, 60.0, 1.0],
            [1, -1, 80.0, 50.0, 5.0, 40.0, 1.0],
        ],
        "fleets": [
            [7, 0, 68.0, 50.0, 0.0, 0.0, 20.0],
        ],
        "comets": [],
    }


def test_fleet_arrival_cache_reuses_next_step_and_invalidates_on_comet_change(monkeypatch) -> None:
    cache = ka.FleetArrivalCache()
    real_swept_pair_hit = ka._swept_pair_hit
    obs0 = _base_obs()
    state0 = ka.observation_to_state(obs0, fleet_arrival_cache=cache)

    target_slot0, eta0 = np.argwhere(np.asarray(state0.incoming_fleets[0]) > 0)[0]
    assert int(target_slot0) == 1
    assert int(eta0) > 0

    speed = ka._fleet_speed(20.0, 6.0)
    obs1 = _base_obs()
    obs1["step"] = 11
    obs1["fleets"] = [[7, 0, 68.0 + speed, 50.0, 0.0, 0.0, 20.0]]

    def fail_if_recomputed(*args, **kwargs):
        raise AssertionError("cached fleet arrival should skip swept-hit recomputation")

    monkeypatch.setattr(ka, "_swept_pair_hit", fail_if_recomputed)
    state1 = ka.observation_to_state(obs1, fleet_arrival_cache=cache)
    incoming1 = np.asarray(state1.incoming_fleets[0])
    assert incoming1[1, int(eta0) - 1] == 20

    calls = {"count": 0}

    def counted_swept_pair_hit(*args, **kwargs):
        calls["count"] += 1
        return real_swept_pair_hit(*args, **kwargs)

    monkeypatch.setattr(ka, "_swept_pair_hit", counted_swept_pair_hit)
    obs2 = _base_obs()
    obs2["step"] = 11
    obs2["fleets"] = [[7, 0, 68.0 + speed, 50.0, 0.0, 0.0, 20.0]]
    obs2["planets"] = list(obs2["planets"]) + [[2, -1, 10.0, 10.0, 4.0, 8.0, 1.0]]
    obs2["comets"] = [
        {
            "planet_ids": [2],
            "paths": [[[10.0, 10.0], [12.0, 10.0]]],
            "path_index": 0,
        }
    ]
    ka.observation_to_state(obs2, fleet_arrival_cache=cache)
    assert calls["count"] > 0


def test_fleet_arrival_cache_reuses_no_hit_within_horizon(monkeypatch) -> None:
    cache = ka.FleetArrivalCache()
    obs0 = _base_obs()
    obs0["fleets"] = [[7, 0, 26.1, 60.0, 0.0, 0.0, 20.0]]
    state0 = ka.observation_to_state(obs0, fleet_arrival_cache=cache)
    assert not np.asarray(state0.incoming_fleets[0]).any()

    speed = ka._fleet_speed(20.0, 6.0)
    obs1 = _base_obs()
    obs1["step"] = 11
    obs1["fleets"] = [[7, 0, 26.1 + speed, 60.0, 0.0, 0.0, 20.0]]

    baseline_calls = {"count": 0}

    def count_baseline(*args, **kwargs):
        baseline_calls["count"] += 1
        return real_swept_pair_hit(*args, **kwargs)

    real_swept_pair_hit = ka._swept_pair_hit
    monkeypatch.setattr(ka, "_swept_pair_hit", count_baseline)
    baseline_state = ka.observation_to_state(obs1, fleet_arrival_cache=None)
    assert not np.asarray(baseline_state.incoming_fleets[0]).any()

    cached_calls = {"count": 0}

    def count_cached(*args, **kwargs):
        cached_calls["count"] += 1
        return real_swept_pair_hit(*args, **kwargs)

    monkeypatch.setattr(ka, "_swept_pair_hit", count_cached)
    state1 = ka.observation_to_state(obs1, fleet_arrival_cache=cache)
    assert not np.asarray(state1.incoming_fleets[0]).any()
    assert cached_calls["count"] < baseline_calls["count"]
    assert cached_calls["count"] <= 2

import unittest

import torch

from orbit_wars_pt.constants import FEATURE_DIM, FUTURE_PLANET_FEATURES_PER_TICK, MAX_PLANETS, NUM_OWNER_SLOTS
from orbit_wars_pt.model import build_future_planet_features, _simulate_future_planet_features


def _tick_slice(flat: torch.Tensor, tick: int) -> torch.Tensor:
    start = tick * FUTURE_PLANET_FEATURES_PER_TICK
    stop = start + FUTURE_PLANET_FEATURES_PER_TICK
    return flat[..., start:stop]


class FutureFeatureTests(unittest.TestCase):
    def test_base_forecast_uses_production_before_arrival(self) -> None:
        owner_idx = torch.zeros((1, 1 + MAX_PLANETS), dtype=torch.long)
        features = torch.zeros((1, 1 + MAX_PLANETS, FEATURE_DIM), dtype=torch.float32)
        owner_idx[:, 1] = 2
        features[:, 1, 0] = torch.log1p(torch.tensor(2.0))
        features[:, 1, 1] = 5.0 / 1000.0
        features[:, 1, 4] = 1.0
        features[:, 1, 8] = 6.0 / 1000.0

        future = build_future_planet_features(owner_idx, features)[0, 0]
        tick0 = _tick_slice(future, 0)
        tick1 = _tick_slice(future, 1)

        self.assertEqual(int(tick0[:NUM_OWNER_SLOTS].argmax().item()), 2)
        self.assertAlmostEqual(float(tick0[-1].item()), 1.0 / 1000.0, places=6)
        self.assertEqual(int(tick1[:NUM_OWNER_SLOTS].argmax().item()), 2)
        self.assertAlmostEqual(float(tick1[-1].item()), 3.0 / 1000.0, places=6)

    def test_launch_merge_captures_neutral_target_at_eta(self) -> None:
        current_owner = torch.zeros((1, 1), dtype=torch.long)
        current_garrison = torch.full((1, 1), 5.0)
        production = torch.zeros((1, 1), dtype=torch.float32)
        arrival_owner = torch.zeros((1, 1, 24), dtype=torch.long)
        arrival_ships = torch.zeros((1, 1, 24), dtype=torch.float32)
        active = torch.ones((1, 1), dtype=torch.bool)

        future = _simulate_future_planet_features(
            current_owner,
            current_garrison,
            production,
            arrival_owner,
            arrival_ships,
            active_mask=active,
            launch_ships=torch.full((1, 1), 7.0),
            launch_eta=torch.ones((1, 1), dtype=torch.long),
            launch_owner=1,
        )[0, 0]

        tick0 = _tick_slice(future, 0)
        tick1 = _tick_slice(future, 1)
        self.assertEqual(int(tick0[:NUM_OWNER_SLOTS].argmax().item()), 0)
        self.assertAlmostEqual(float(tick0[-1].item()), 5.0 / 1000.0, places=6)
        self.assertEqual(int(tick1[:NUM_OWNER_SLOTS].argmax().item()), 1)
        self.assertAlmostEqual(float(tick1[-1].item()), 2.0 / 1000.0, places=6)


if __name__ == "__main__":
    unittest.main()

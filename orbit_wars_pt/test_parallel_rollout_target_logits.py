import unittest

import torch

from orbit_wars_pt.constants import FEATURE_DIM, MAX_PLANETS
from orbit_wars_pt.model import OrbitWarsPolicy
from orbit_wars_pt.parallel_rollout import _target_logits_by_controller


class ParallelRolloutTargetLogitsTests(unittest.TestCase):
    def test_helper_threads_owner_idx_and_features(self) -> None:
        batch = 4
        policy = OrbitWarsPolicy(
            d_model=32,
            n_heads=4,
            n_layers=1,
            feature_dim=FEATURE_DIM,
            population_size=1,
            future_feature_enabled=True,
        ).cpu().eval()

        entity_type = torch.ones((batch, 1 + MAX_PLANETS), dtype=torch.long)
        entity_type[:, 0] = 0
        owner_idx = torch.zeros((batch, 1 + MAX_PLANETS), dtype=torch.long)
        owner_idx[:, 0] = 1
        owner_idx[:, 1] = 1
        owner_idx[:, 2] = 2

        features = torch.zeros((batch, 1 + MAX_PLANETS, FEATURE_DIM), dtype=torch.float32)
        rope_pos = torch.zeros((batch, 1 + MAX_PLANETS, 3), dtype=torch.float32)
        entity_mask = torch.zeros((batch, 1 + MAX_PLANETS), dtype=torch.bool)
        planet_mask = torch.zeros((batch, 1 + MAX_PLANETS), dtype=torch.bool)
        entity_mask[:, 0:3] = True
        planet_mask[:, 1 : 1 + MAX_PLANETS] = True
        features[:, 1:3, 4] = 1.0
        features[:, 1, 0] = torch.log1p(torch.tensor(1.0))
        features[:, 1, 1] = 10.0 / 1000.0
        features[:, 2, 1] = 8.0 / 1000.0
        features[:, 2, 8] = 6.0 / 1000.0

        out = policy.forward_dense_rollout(entity_type, owner_idx, features, rope_pos, entity_mask, planet_mask)
        logits = _target_logits_by_controller(
            policies=[policy],
            planet_hidden=out["planet_hidden"],
            owner_idx=owner_idx,
            features=features,
            origin_idx=torch.zeros((batch,), dtype=torch.long),
            frac_idx=torch.zeros((batch,), dtype=torch.long),
            fleet_size=torch.full((batch,), 4.0),
            target_eta=torch.zeros((batch, MAX_PLANETS), dtype=torch.float32),
            target_ships=features[:, 1 : 1 + MAX_PLANETS, 1] * 1000.0,
            active_population_idx_t=torch.zeros((batch,), dtype=torch.long),
            active_controller_idx_t=torch.zeros((batch,), dtype=torch.long),
        )

        self.assertEqual(tuple(logits.shape), (batch, MAX_PLANETS))
        self.assertTrue(torch.isfinite(logits).all().item())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch

from orbit_wars_pt.batched_env import stack_initial_states
from orbit_wars_pt.constants import obs_feature_dim_for_num_agents
from orbit_wars_pt.env_wrapper import OrbitWarsEnvConfig
from orbit_wars_pt.parallel_rollout import RolloutCarry, collect_parallel_micro_rollouts
from orbit_wars_pt.train_ppo import OrbitWarsPolicy


class TestRolloutCarryShapeRepair(unittest.TestCase):
    def test_stale_player_done_is_rebuilt_from_resized_controller_assignments(self) -> None:
        cfg = OrbitWarsEnvConfig(num_agents=2, max_fleets=32, episode_seed=7)
        feature_dim = obs_feature_dim_for_num_agents(int(cfg.num_agents), target_abort_enabled=False)
        stale_envs = 3
        target_envs = 5
        state_b, _ = stack_initial_states(cfg, num_envs=target_envs, seed_base=11)
        stale_controller_assignments = np.array(
            [
                [0, -1, 0],
                [-1, 0, -1],
            ],
            dtype=np.int32,
        )
        carry = RolloutCarry(
            state_b=state_b,
            cfg=cfg,
            episode_turns=[0] * target_envs,
            player_done=np.zeros((int(cfg.num_agents), stale_envs), dtype=np.bool_),
            population_assignments=np.zeros((int(cfg.num_agents), stale_envs), dtype=np.int32),
            policy_row_for_seat=None,
            controller_assignments=stale_controller_assignments,
            main_player_mask=np.zeros((int(cfg.num_agents), stale_envs), dtype=np.bool_),
            env_mode_by_env=None,
            pending_exploiter_terminal=np.zeros((int(cfg.num_agents), stale_envs), dtype=np.bool_),
        )
        policy = OrbitWarsPolicy(
            d_model=32,
            n_heads=4,
            n_layers=1,
            activation_checkpointing=False,
            feature_dim=feature_dim,
            population_size=1,
            rope_dims=2,
            target_abort_enabled=False,
            disjoint_actor_critic=False,
            halt_init_prob=0.5,
            fraction_init_weights=(1.0, 1.0, 1.0, 1.0, 1.0),
            value_head_count=1,
        ).to(torch.device('cpu'))

        _, _, next_carry, _, _ = collect_parallel_micro_rollouts(
            policy,
            cfg,
            target_envs,
            device=torch.device('cpu'),
            seed_base=17,
            max_micro_steps_per_player=1,
            rollout_micro_horizon=1,
            carry_in=carry,
            max_outer_iters=4,
            first_hit_n_rays=8,
            first_hit_ray_chunk_size=0,
            first_hit_env_chunk_size=0,
        )

        self.assertEqual(next_carry.player_done.shape, (int(cfg.num_agents), target_envs))
        self.assertEqual(next_carry.controller_assignments.shape, (int(cfg.num_agents), target_envs))

    def test_stale_state_batch_is_dropped_when_num_envs_changes(self) -> None:
        cfg = OrbitWarsEnvConfig(num_agents=2, max_fleets=32, episode_seed=7)
        feature_dim = obs_feature_dim_for_num_agents(int(cfg.num_agents), target_abort_enabled=False)
        stale_envs = 3
        target_envs = 5
        stale_state_b, _ = stack_initial_states(cfg, num_envs=stale_envs, seed_base=11)
        carry = RolloutCarry(
            state_b=stale_state_b,
            cfg=cfg,
            episode_turns=[0] * stale_envs,
            player_done=np.zeros((int(cfg.num_agents), stale_envs), dtype=np.bool_),
            population_assignments=np.zeros((int(cfg.num_agents), stale_envs), dtype=np.int32),
            policy_row_for_seat=None,
            controller_assignments=np.zeros((int(cfg.num_agents), stale_envs), dtype=np.int32),
            main_player_mask=np.zeros((int(cfg.num_agents), stale_envs), dtype=np.bool_),
            env_mode_by_env=None,
            pending_exploiter_terminal=np.zeros((int(cfg.num_agents), stale_envs), dtype=np.bool_),
        )
        policy = OrbitWarsPolicy(
            d_model=32,
            n_heads=4,
            n_layers=1,
            activation_checkpointing=False,
            feature_dim=feature_dim,
            population_size=1,
            rope_dims=2,
            target_abort_enabled=False,
            disjoint_actor_critic=False,
            halt_init_prob=0.5,
            fraction_init_weights=(1.0, 1.0, 1.0, 1.0, 1.0),
            value_head_count=1,
        ).to(torch.device('cpu'))

        _, _, next_carry, _, _ = collect_parallel_micro_rollouts(
            policy,
            cfg,
            target_envs,
            device=torch.device('cpu'),
            seed_base=17,
            max_micro_steps_per_player=1,
            rollout_micro_horizon=1,
            carry_in=carry,
            max_outer_iters=4,
            first_hit_n_rays=8,
            first_hit_ray_chunk_size=0,
            first_hit_env_chunk_size=0,
        )

        self.assertEqual(int(next_carry.state_b.planets.shape[0]), target_envs)
        self.assertEqual(len(next_carry.episode_turns), target_envs)


if __name__ == '__main__':
    unittest.main()

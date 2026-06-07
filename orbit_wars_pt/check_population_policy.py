"""Sanity checks for population-tail policy routing.

Run with::

    ./.venv/bin/python -m orbit_wars_pt.check_population_policy

This verifies three key behaviors:

* population members share the trunk but route through independent private
  tails from the final transformer block onward
* omitting ``population_idx`` falls back to member 0, which is the current
  Kaggle adapter behavior
* rollout episode member assignment uses sampling with replacement
"""

from __future__ import annotations

import argparse
import math
import os
import sys

# Force CPU JAX so importing the rollout helper does not probe CUDA.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from orbit_wars_pt.constants import FEATURE_DIM, MAX_PLANETS
from orbit_wars_pt.model import OrbitWarsPolicy
from orbit_wars_pt.parallel_rollout import _sample_population_assignments_for_env


def _build_batch(batch_size: int) -> dict[str, torch.Tensor]:
    seq_len = 1 + MAX_PLANETS
    entity_type = torch.zeros((batch_size, seq_len), dtype=torch.long)
    owner_idx = torch.zeros((batch_size, seq_len), dtype=torch.long)
    features = torch.zeros((batch_size, seq_len, FEATURE_DIM), dtype=torch.float32)
    rope_pos = torch.zeros((batch_size, seq_len, 3), dtype=torch.float32)
    entity_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    planet_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)

    entity_mask[:, 0] = True
    entity_mask[:, 1:5] = True
    planet_mask[:, 1:] = True

    owner_idx[:, 1] = 1
    owner_idx[:, 2] = 2
    owner_idx[:, 3] = 1
    owner_idx[:, 4] = 2

    features[:, 1, 1] = 1.5
    features[:, 2, 1] = 0.8
    features[:, 3, 1] = 2.0
    features[:, 4, 1] = 1.2
    rope_pos[:, 1:5, 0] = torch.tensor([0.1, 0.3, 0.6, 0.8], dtype=torch.float32)
    rope_pos[:, 1:5, 1] = torch.tensor([0.2, 0.7, 0.4, 0.9], dtype=torch.float32)

    return {
        "entity_type": entity_type,
        "owner_idx": owner_idx,
        "features": features,
        "rope_pos": rope_pos,
        "entity_mask": entity_mask,
        "planet_mask": planet_mask,
    }


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def _assert_close(name: str, a: torch.Tensor, b: torch.Tensor, tol: float = 1e-6) -> None:
    diff = _max_diff(a, b)
    print(f"{name}: max_abs_diff={diff:.6g}")
    if diff > tol:
        raise AssertionError(f"{name} mismatch: {diff} > {tol}")


def _assert_not_close(name: str, a: torch.Tensor, b: torch.Tensor, tol: float = 1e-5) -> None:
    diff = _max_diff(a, b)
    print(f"{name}: max_abs_diff={diff:.6g}")
    if diff <= tol:
        raise AssertionError(f"{name} unexpectedly matched: {diff} <= {tol}")


def _copy_tail(dst, src) -> None:
    dst.load_state_dict(src.state_dict())


def _check_default_routes_to_member0(policy: OrbitWarsPolicy, batch: dict[str, torch.Tensor]) -> None:
    pop0 = torch.zeros((batch["entity_type"].shape[0],), dtype=torch.long)
    out_default = policy(**batch)
    out_pop0 = policy(**batch, population_idx=pop0)
    _assert_close("default_vs_member0.value", out_default["value"], out_pop0["value"])
    _assert_close(
        "default_vs_member0.halt_logits",
        out_default["halt_logits"],
        out_pop0["halt_logits"],
    )
    _assert_close(
        "default_vs_member0.origin_frac_logits",
        out_default["origin_frac_logits"],
        out_pop0["origin_frac_logits"],
    )


def _check_identical_tails_match(policy: OrbitWarsPolicy, batch: dict[str, torch.Tensor]) -> None:
    if policy.population_size <= 1:
        return
    for tail in policy.population_tails[1:]:
        _copy_tail(tail, policy.population_tails[0])

    pop0 = torch.zeros((batch["entity_type"].shape[0],), dtype=torch.long)
    pop1 = torch.ones((batch["entity_type"].shape[0],), dtype=torch.long)
    out0 = policy(**batch, population_idx=pop0)
    out1 = policy(**batch, population_idx=pop1)
    _assert_close("identical_tails.value", out0["value"], out1["value"])
    _assert_close("identical_tails.halt_logits", out0["halt_logits"], out1["halt_logits"])


def _check_member_isolation(policy: OrbitWarsPolicy, batch: dict[str, torch.Tensor]) -> None:
    if policy.population_size < 2:
        return
    for tail in policy.population_tails[1:]:
        _copy_tail(tail, policy.population_tails[0])

    with torch.no_grad():
        policy.population_tails[1].value_head.bias.add_(7.0)
        policy.population_tails[1].halt_head.bias.add_(3.0)

    pop_mix = torch.tensor([0, 1, 2, 1, 0, 2], dtype=torch.long)[: batch["entity_type"].shape[0]]
    pop0 = torch.zeros_like(pop_mix)
    out_mix = policy(**batch, population_idx=pop_mix)
    out0 = policy(**batch, population_idx=pop0)

    member1_rows = pop_mix == 1
    other_rows = ~member1_rows
    _assert_not_close(
        "member1_private_tail_changes_member1_values",
        out_mix["value"][member1_rows],
        out0["value"][member1_rows],
    )
    _assert_close(
        "member1_private_tail_preserves_other_values",
        out_mix["value"][other_rows],
        out0["value"][other_rows],
    )

    target0 = policy.target_logits_for_origin_fraction(
        out0["planet_hidden"],
        torch.zeros(pop0.shape[0], dtype=torch.long),
        torch.zeros(pop0.shape[0], dtype=torch.long),
        fleet_size=torch.ones(pop0.shape[0], dtype=torch.float32),
        target_eta=torch.zeros((pop0.shape[0], MAX_PLANETS), dtype=torch.float32),
        target_ships=torch.zeros((pop0.shape[0], MAX_PLANETS), dtype=torch.float32),
        population_idx=pop0,
    )
    target_mix = policy.target_logits_for_origin_fraction(
        out_mix["planet_hidden"],
        torch.zeros(pop_mix.shape[0], dtype=torch.long),
        torch.zeros(pop_mix.shape[0], dtype=torch.long),
        fleet_size=torch.ones(pop_mix.shape[0], dtype=torch.float32),
        target_eta=torch.zeros((pop_mix.shape[0], MAX_PLANETS), dtype=torch.float32),
        target_ships=torch.zeros((pop_mix.shape[0], MAX_PLANETS), dtype=torch.float32),
        population_idx=pop_mix,
    )
    _assert_close(
        "member1_private_tail_preserves_other_target_logits",
        target_mix[other_rows],
        target0[other_rows],
    )


def _check_sampling_with_replacement(seed: int, num_agents: int, population_size: int) -> None:
    assigned = _sample_population_assignments_for_env(seed, num_agents, population_size)

    import numpy as np

    expected = np.random.default_rng(np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)).integers(
        0, population_size, size=(num_agents,), dtype=np.int32
    )
    print(f"sampled_assignments={assigned.tolist()}")
    if not np.array_equal(assigned, expected):
        raise AssertionError("population assignment helper no longer matches replacement sampling")
    if assigned.min() < 0 or assigned.max() >= population_size:
        raise AssertionError("population assignment helper produced out-of-range member ids")


def _check_init_biases() -> None:
    policy = OrbitWarsPolicy(
        d_model=72,
        n_heads=6,
        n_layers=2,
        feature_dim=FEATURE_DIM,
        population_size=1,
        halt_init_prob=0.9,
        fraction_init_weights=(1.0, 1.0, 1.0, 1.0, 15.0),
    )
    halt_bias = policy.halt_head.bias.detach().cpu()
    expected_halt = math.log(0.9 / 0.1)
    if abs(float(halt_bias[0].item())) > 1e-6 or abs(float(halt_bias[1].item()) - expected_halt) > 1e-6:
        raise AssertionError("halt init bias did not match requested prior")
    expected_frac = torch.log(torch.tensor([1.0, 1.0, 1.0, 1.0, 15.0], dtype=policy.origin_frac_head.bias.dtype))
    if not torch.allclose(policy.origin_frac_head.bias.detach().cpu(), expected_frac.cpu(), atol=1e-6, rtol=0.0):
        raise AssertionError("origin_frac_head init bias did not match requested ratio")
    frac_head_biases = torch.tensor(
        [float(head.bias.detach().cpu().item()) for head in policy.frac_heads],
        dtype=expected_frac.dtype,
    )
    if not torch.allclose(frac_head_biases, expected_frac.cpu(), atol=1e-6, rtol=0.0):
        raise AssertionError("legacy frac_heads init bias did not match requested ratio")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--population-size", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=6)
    args = p.parse_args()

    if args.population_size < 2:
        raise SystemExit("--population-size must be >= 2 for this checker")

    torch.manual_seed(args.seed)
    batch = _build_batch(args.batch_size)
    policy = OrbitWarsPolicy(
        d_model=72,
        n_heads=6,
        n_layers=4,
        feature_dim=FEATURE_DIM,
        population_size=args.population_size,
    )
    policy.eval()

    print("checking default Kaggle-style routing")
    _check_default_routes_to_member0(policy, batch)

    print("checking cloned tails produce identical outputs")
    _check_identical_tails_match(policy, batch)

    print("checking private tail edits affect only routed members")
    _check_member_isolation(policy, batch)

    print("checking rollout member assignment sampling")
    _check_sampling_with_replacement(args.seed, num_agents=4, population_size=args.population_size)

    print("checking optional head init biases")
    _check_init_biases()

    print("population policy checks: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

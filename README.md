# Orbit Wars

PyTorch training and inference code for a Kaggle Orbit Wars agent, backed by a JAX environment for high-throughput self-play.

This repository contains:

- PPO training for 2-player and 4-player Orbit Wars
- Local replay and evaluation tools for Kaggle-style episodes
- Packaging utilities for building Kaggle submission bundles
- Tournament and debugging scripts for comparing packaged agents

The approach is described in more detail in [writeup.md](/home/billy/orbit-wars/writeup.md).

## Repository Layout

- `orbit_wars_pt/`: main package for policy code, training, inference, tests, and utilities
- `orbit_wars_pt/cli/run_train_ppo.py`: lightweight `python -m` entry point for PPO training
- `orbit_wars_pt/package_kaggle_submission.py`: builds a submission bundle or `.tar.gz`
- `orbit_wars_pt/play_kaggle_selfplay.py`: runs local Kaggle self-play and saves a record
- `orbit_wars_pt/play_kaggle_vs_checkpoint.py`: lets a human or GUI play against a checkpoint
- `scripts/run_packaged_tournament.py`: runs repeated matches between packaged bundles
- `docs/kaggle-docker-runner.md`: replay harness using Kaggle's public Docker image

## Requirements

Python 3.10+ is recommended.

Install GPU-enabled frameworks first, matching your machine:

- PyTorch: follow https://pytorch.org/get-started/locally/
- JAX: follow https://jax.readthedocs.io/en/latest/installation.html

Then install the remaining Python dependencies:

```bash
pip install -r requirements-train.txt
```

`requirements-train.txt` intentionally does not pin a CUDA wheel for `torch` or `jax`, so you can choose the right build for your driver and hardware.

## Training

The main training entry point is:

```bash
python -m orbit_wars_pt.cli.run_train_ppo --help
```

Minimal examples:

```bash
python -m orbit_wars_pt.cli.run_train_ppo \
  --experiment ppo-2p-baseline \
  --num-agents 2
```

```bash
python -m orbit_wars_pt.cli.run_train_ppo \
  --experiment ppo-4p-baseline \
  --num-agents 4 \
  --num-envs 384
```

Artifacts are written under `experiments/<experiment>/`, with TensorBoard logs grouped under `experiments/tensorboard/`.

To view all runs:

```bash
tensorboard --logdir experiments/tensorboard
```

## Local Evaluation

Run one local Kaggle self-play episode and save the record:

```bash
python -m orbit_wars_pt.play_kaggle_selfplay \
  --checkpoint experiments/ppo-2p-baseline/checkpoints/iter_00000020.pt \
  --num-agents 2 \
  --out records/selfplay.json
```

Play against a checkpoint yourself:

```bash
python -m orbit_wars_pt.play_kaggle_vs_checkpoint \
  --checkpoint experiments/ppo-2p-baseline/checkpoints/iter_00000020.pt \
  --human-player 0
```

Add `--gui` to use the Tkinter board UI instead of terminal input.

## Kaggle Submission Bundles

Build a packaged submission from 4-player and 2-player checkpoints:

```bash
python -m orbit_wars_pt.package_kaggle_submission \
  --checkpoint-4p experiments/ppo-4p-baseline/checkpoints/iter_00000020.pt \
  --checkpoint-2p experiments/ppo-2p-baseline/checkpoints/iter_00000020.pt \
  --out dist/orbit-wars-submission.tar.gz
```

The generated bundle contains a root `main.py`, a slim inference copy of `orbit_wars_pt`, and packaged checkpoint files ready for Kaggle submission or local replay.

## Packaged-Agent Tournaments

Run an incremental tournament over packaged agents:

```bash
python scripts/run_packaged_tournament.py \
  --agent baseline=dist/orbit-wars-submission.tar.gz \
  --agent challenger=dist/challenger.tar.gz \
  --out-dir tournament/2p \
  --num-players 2
```

This writes match records, cached extracted bundles, and rankings into the chosen output directory.

## Kaggle Docker Replay

For a closer approximation of Kaggle's runtime, see [docs/kaggle-docker-runner.md](/home/billy/orbit-wars/docs/kaggle-docker-runner.md).

## Tests

The repo includes targeted test modules under `orbit_wars_pt/`, for example:

```bash
pytest orbit_wars_pt/test_terminal_rewards.py
```

Or run the full test set discovered in that package:

```bash
pytest orbit_wars_pt
```

## Notes

- Training uses PyTorch for models and optimization, with JAX for the environment backend and rollout-heavy geometry work.
- Several scripts are specialized debugging and benchmarking tools for search, geometry, replay consistency, and Kaggle validation issues.
- `bundle_orbit_wars.py` can create a portable zip of the project for remote machines.

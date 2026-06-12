#!/usr/bin/env python3
"""Lightweight python -m launcher for PPO training."""

from __future__ import annotations

def main() -> None:
    from orbit_wars_pt.train_ppo import parse_args, train

    train(parse_args())


if __name__ == "__main__":
    main()

from __future__ import annotations

import unittest

import numpy as np

from orbit_wars_pt.kaggle_adapter import (
    MAX_PLANETS,
    _search_opponent_should_halt_on_neutral_underlaunch,
)


class TestSearchOpponentNeutralGuard(unittest.TestCase):
    def test_halts_opponent_neutral_underlaunch_with_no_incoming(self) -> None:
        planets = np.zeros((MAX_PLANETS, 7), dtype=np.float32)
        incoming = np.zeros((2, MAX_PLANETS, 8), dtype=np.float32)
        planets[5, 1] = -1.0
        planets[5, 5] = 17.0

        self.assertTrue(
            _search_opponent_should_halt_on_neutral_underlaunch(
                planets,
                incoming,
                player=1,
                search_root_player=0,
                send=17,
                true_target_slot=5,
            )
        )

    def test_allows_root_player_to_keep_same_launch(self) -> None:
        planets = np.zeros((MAX_PLANETS, 7), dtype=np.float32)
        incoming = np.zeros((2, MAX_PLANETS, 8), dtype=np.float32)
        planets[5, 1] = -1.0
        planets[5, 5] = 17.0

        self.assertFalse(
            _search_opponent_should_halt_on_neutral_underlaunch(
                planets,
                incoming,
                player=0,
                search_root_player=0,
                send=17,
                true_target_slot=5,
            )
        )

    def test_allows_when_existing_fleet_is_already_heading_to_target(self) -> None:
        planets = np.zeros((MAX_PLANETS, 7), dtype=np.float32)
        incoming = np.zeros((2, MAX_PLANETS, 8), dtype=np.float32)
        planets[5, 1] = -1.0
        planets[5, 5] = 17.0
        incoming[0, 5, 3] = 4.0

        self.assertFalse(
            _search_opponent_should_halt_on_neutral_underlaunch(
                planets,
                incoming,
                player=1,
                search_root_player=0,
                send=17,
                true_target_slot=5,
            )
        )

    def test_allows_when_launch_is_large_enough_to_capture_alone(self) -> None:
        planets = np.zeros((MAX_PLANETS, 7), dtype=np.float32)
        incoming = np.zeros((2, MAX_PLANETS, 8), dtype=np.float32)
        planets[5, 1] = -1.0
        planets[5, 5] = 17.0

        self.assertFalse(
            _search_opponent_should_halt_on_neutral_underlaunch(
                planets,
                incoming,
                player=1,
                search_root_player=0,
                send=18,
                true_target_slot=5,
            )
        )


if __name__ == "__main__":
    unittest.main()

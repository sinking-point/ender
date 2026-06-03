from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from orbit_wars_pt.kaggle_adapter import _greedy_from_env, _normalize_greedy


class TestGreedyNormalization(unittest.TestCase):
    def test_bool_greedy_applies_to_all_four_seats(self) -> None:
        self.assertEqual(_normalize_greedy(True), {0: True, 1: True, 2: True, 3: True})
        self.assertEqual(_normalize_greedy(False), {0: False, 1: False, 2: False, 3: False})

    def test_mapping_keeps_per_seat_overrides(self) -> None:
        self.assertEqual(_normalize_greedy({2: True}), {0: False, 1: False, 2: True, 3: False})

    def test_env_per_seat_values_override_global_fallback(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ORBIT_WARS_GREEDY": "1",
                "ORBIT_WARS_GREEDY_P2": "0",
            },
            clear=False,
        ):
            self.assertEqual(_greedy_from_env(), {0: True, 1: True, 2: False, 3: True})


if __name__ == "__main__":
    unittest.main()

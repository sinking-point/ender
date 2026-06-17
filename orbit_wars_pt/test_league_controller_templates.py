from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from orbit_wars_pt.league_templates import league_controller_templates


class TestLeagueControllerTemplates(unittest.TestCase):
    def test_four_player_league_uses_p0_and_p3_for_main_policy(self) -> None:
        controller, main_mask, require_all_dead = league_controller_templates(4)

        np.testing.assert_array_equal(
            controller,
            np.asarray([[0], [1], [1], [0]], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            main_mask,
            np.asarray([[True], [False], [False], [True]], dtype=np.bool_),
        )
        self.assertTrue(require_all_dead)

    def test_two_player_league_template_is_unchanged(self) -> None:
        controller, main_mask, require_all_dead = league_controller_templates(2)

        np.testing.assert_array_equal(
            controller,
            np.asarray([[0], [1]], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            main_mask,
            np.asarray([[True], [False]], dtype=np.bool_),
        )
        self.assertFalse(require_all_dead)


if __name__ == "__main__":
    unittest.main()

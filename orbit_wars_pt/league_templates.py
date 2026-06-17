from __future__ import annotations

import numpy as np


def league_controller_templates(num_agents: int) -> tuple[np.ndarray, np.ndarray, bool]:
    agents = int(num_agents)
    if agents == 4:
        controller = np.asarray([[0], [1], [1], [0]], dtype=np.int32)
        main_mask = np.asarray([[True], [False], [False], [True]], dtype=np.bool_)
        return controller, main_mask, True
    if agents == 2:
        controller = np.asarray([[0], [1]], dtype=np.int32)
        main_mask = np.asarray([[True], [False]], dtype=np.bool_)
        return controller, main_mask, False
    raise ValueError(f"league templates only support 2 or 4 agents, got {agents}")

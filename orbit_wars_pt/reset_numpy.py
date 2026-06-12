"""NumPy-backed reset builders for lightweight prefetch workers.

These helpers mirror the reset structure of ``jax_orbit_wars.reset_from_reference``
without importing JAX, so background reset workers can stay lightweight and avoid
XLA/PJRT threadpool startup.
"""

from __future__ import annotations

import random
from typing import NamedTuple

import numpy as np

MAX_BASE_PLANETS = 40
MAX_COMETS = 20
MAX_PLANETS = MAX_BASE_PLANETS + MAX_COMETS
MAX_COMET_GROUPS = 5
MAX_COMET_PATH = 40
INCOMING_TA_BINS = 24
DEFAULT_MAX_FLEETS = 4096
FLEET_ROW_WIDTH = 9
COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)
COMET_RADIUS = 1.0

PLANET_ID = 0
PLANET_OWNER = 1
PLANET_X = 2
PLANET_Y = 3
PLANET_RADIUS = 4
PLANET_SHIPS = 5
PLANET_PRODUCTION = 6

EXPLOITER_MODE_SELFPLAY_2P = 0
EXPLOITER_MODE_SELFPLAY_4P = 1
EXPLOITER_MODE_VS_2P = 2
EXPLOITER_MODE_VS_4P = 3


class NumpyOrbitWarsState(NamedTuple):
    planets: np.ndarray
    planet_active: np.ndarray
    initial_planets: np.ndarray
    initial_active: np.ndarray
    origin_frac_blocked: np.ndarray
    fleets: np.ndarray
    fleet_active: np.ndarray
    incoming_fleets: np.ndarray
    incoming_fake_correction: np.ndarray
    comet_paths: np.ndarray
    comet_path_lengths: np.ndarray
    comet_ships: np.ndarray
    comet_group_active: np.ndarray
    comet_path_index: np.ndarray
    comet_planet_ids: np.ndarray
    comet_slots: np.ndarray
    next_fleet_id: np.ndarray
    angular_velocity: np.ndarray
    step_count: np.ndarray
    num_agents: np.ndarray
    rewards: np.ndarray
    done: np.ndarray
    overflow: np.ndarray


def unified_exploiter_active_seat_count(mode_code: int) -> int:
    mode = int(mode_code)
    if mode in (EXPLOITER_MODE_SELFPLAY_2P, EXPLOITER_MODE_VS_2P):
        return 2
    if mode in (EXPLOITER_MODE_SELFPLAY_4P, EXPLOITER_MODE_VS_4P):
        return 4
    raise ValueError(f"unknown unified exploiter mode {mode}")


def _empty_state_arrays(num_agents: int, max_fleets: int) -> NumpyOrbitWarsState:
    return NumpyOrbitWarsState(
        planets=np.zeros((MAX_PLANETS, 7), dtype=np.float32),
        planet_active=np.zeros((MAX_PLANETS,), dtype=np.bool_),
        initial_planets=np.zeros((MAX_PLANETS, 7), dtype=np.float32),
        initial_active=np.zeros((MAX_PLANETS,), dtype=np.bool_),
        origin_frac_blocked=np.zeros((MAX_PLANETS, 5), dtype=np.bool_),
        fleets=np.zeros((max_fleets, FLEET_ROW_WIDTH), dtype=np.float32),
        fleet_active=np.zeros((max_fleets,), dtype=np.bool_),
        incoming_fleets=np.zeros((num_agents, MAX_PLANETS, INCOMING_TA_BINS), dtype=np.uint16),
        incoming_fake_correction=np.zeros((num_agents, MAX_PLANETS, INCOMING_TA_BINS), dtype=np.uint16),
        comet_paths=np.zeros((MAX_COMET_GROUPS, 4, MAX_COMET_PATH, 2), dtype=np.float32),
        comet_path_lengths=np.zeros((MAX_COMET_GROUPS, 4), dtype=np.int32),
        comet_ships=np.zeros((MAX_COMET_GROUPS,), dtype=np.float32),
        comet_group_active=np.zeros((MAX_COMET_GROUPS,), dtype=np.bool_),
        comet_path_index=np.full((MAX_COMET_GROUPS,), -1, dtype=np.int32),
        comet_planet_ids=np.full((MAX_COMET_GROUPS, 4), -1, dtype=np.int32),
        comet_slots=np.full((MAX_COMET_GROUPS, 4), -1, dtype=np.int32),
        next_fleet_id=np.asarray(0, dtype=np.int32),
        angular_velocity=np.asarray(0.0, dtype=np.float32),
        step_count=np.asarray(0, dtype=np.int32),
        num_agents=np.asarray(num_agents, dtype=np.int32),
        rewards=np.zeros((4,), dtype=np.float32),
        done=np.asarray(False, dtype=np.bool_),
        overflow=np.asarray(False, dtype=np.bool_),
    )


def reset_from_reference_numpy(seed: int, num_agents: int = 2, *, max_fleets: int = DEFAULT_MAX_FLEETS) -> NumpyOrbitWarsState:
    from kaggle_environments.envs.orbit_wars.orbit_wars import generate_comet_paths, generate_planets

    init_rng = random.Random(seed)
    angular_velocity = init_rng.uniform(0.025, 0.05)
    planets_list = generate_planets(init_rng)
    initial_planets_list = [p.copy() for p in planets_list]

    num_groups = len(planets_list) // 4
    if num_groups > 0:
        home_group = init_rng.randint(0, num_groups - 1)
        base = home_group * 4
        if num_agents == 2:
            planets_list[base][PLANET_OWNER] = 0
            planets_list[base][PLANET_SHIPS] = 10
            planets_list[base + 3][PLANET_OWNER] = 1
            planets_list[base + 3][PLANET_SHIPS] = 10
        elif num_agents == 4:
            for player in range(4):
                planets_list[base + player][PLANET_OWNER] = player
                planets_list[base + player][PLANET_SHIPS] = 10
        else:
            raise ValueError('Orbit Wars supports 2 or 4 agents.')

    state = _empty_state_arrays(int(num_agents), int(max_fleets))
    comet_paths = state.comet_paths.copy()
    comet_path_lengths = state.comet_path_lengths.copy()
    comet_ships = state.comet_ships.copy()
    comet_planet_ids_for_generation: list[int] = []
    scratch_initial = [p.copy() for p in initial_planets_list]

    for group_idx, spawn_step in enumerate(COMET_SPAWN_STEPS):
        comet_rng = random.Random(f'orbit_wars-comet-{seed}-{spawn_step}')
        paths = generate_comet_paths(
            scratch_initial,
            angular_velocity,
            spawn_step,
            comet_planet_ids_for_generation,
            4.0,
            rng=comet_rng,
        )
        if not paths:
            continue
        for comet_idx, path in enumerate(paths):
            length = min(len(path), MAX_COMET_PATH)
            comet_path_lengths[group_idx, comet_idx] = length
            comet_paths[group_idx, comet_idx, :length] = np.asarray(path[:length], dtype=np.float32)
        ships = min(
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
        )
        comet_ships[group_idx] = float(ships)

    base_count = len(planets_list)
    if base_count > MAX_BASE_PLANETS:
        raise ValueError(f'Reference generated {base_count} planets, max is {MAX_BASE_PLANETS}.')

    planets = state.planets.copy()
    initial_planets = state.initial_planets.copy()
    planet_active = state.planet_active.copy()
    initial_active = state.initial_active.copy()
    planets[:base_count] = np.asarray(planets_list, dtype=np.float32)
    initial_planets[:base_count] = np.asarray(initial_planets_list, dtype=np.float32)
    planet_active[:base_count] = True
    initial_active[:base_count] = True

    return state._replace(
        planets=planets,
        planet_active=planet_active,
        initial_planets=initial_planets,
        initial_active=initial_active,
        comet_paths=comet_paths,
        comet_path_lengths=comet_path_lengths,
        comet_ships=comet_ships,
        angular_velocity=np.asarray(angular_velocity, dtype=np.float32),
    )


def pad_2p_reset_to_4p_numpy(base_state: NumpyOrbitWarsState) -> NumpyOrbitWarsState:
    incoming = np.asarray(base_state.incoming_fleets)
    incoming_fake = np.asarray(base_state.incoming_fake_correction)
    incoming4 = np.zeros((4,) + incoming.shape[1:], dtype=incoming.dtype)
    incoming_fake4 = np.zeros((4,) + incoming_fake.shape[1:], dtype=incoming_fake.dtype)
    incoming4[:2] = incoming
    incoming_fake4[:2] = incoming_fake
    return base_state._replace(
        incoming_fleets=incoming4,
        incoming_fake_correction=incoming_fake4,
        num_agents=np.asarray(4, dtype=np.int32),
        rewards=np.zeros((4,), dtype=np.float32),
        done=np.asarray(False, dtype=np.bool_),
        overflow=np.asarray(False, dtype=np.bool_),
    )


def build_unified_exploiter_state_variant_numpy(seed: int, *, active_seat_count: int, max_fleets: int) -> NumpyOrbitWarsState:
    if int(active_seat_count) == 2:
        base = reset_from_reference_numpy(int(seed), 2, max_fleets=int(max_fleets))
        return pad_2p_reset_to_4p_numpy(base)
    if int(active_seat_count) == 4:
        return reset_from_reference_numpy(int(seed), 4, max_fleets=int(max_fleets))
    raise ValueError(f'unsupported active seat count {int(active_seat_count)} for unified exploiter mode')

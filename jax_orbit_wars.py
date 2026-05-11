"""JAX port of Kaggle's Orbit Wars environment.

The official Kaggle environment stores planets, fleets, and comets in Python
lists. This module uses fixed-size JAX arrays so `step` can be `jit` compiled
and `vmap`'d for RL training. `reset_from_reference` intentionally calls the
official package to reproduce map generation and comet schedules exactly for a
given seed; the turn update is implemented in JAX.
"""

from __future__ import annotations

import math
import random
from typing import NamedTuple

import jax
import jax.numpy as jnp


BOARD_SIZE = 100.0
CENTER = BOARD_SIZE / 2.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
COMET_RADIUS = 1.0
COMET_PRODUCTION = 1.0
COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)

MAX_BASE_PLANETS = 40
MAX_COMETS = 20
MAX_PLANETS = MAX_BASE_PLANETS + MAX_COMETS
MAX_COMET_GROUPS = 5
MAX_COMET_PATH = 40
DEFAULT_MAX_FLEETS = 4096
DEFAULT_MAX_ACTIONS = 64

PLANET_ID = 0
PLANET_OWNER = 1
PLANET_X = 2
PLANET_Y = 3
PLANET_RADIUS = 4
PLANET_SHIPS = 5
PLANET_PRODUCTION = 6

FLEET_ID = 0
FLEET_OWNER = 1
FLEET_X = 2
FLEET_Y = 3
FLEET_ANGLE = 4
FLEET_FROM_PLANET = 5
FLEET_SHIPS = 6


class OrbitWarsConfig(NamedTuple):
    episode_steps: jnp.ndarray = jnp.asarray(500, dtype=jnp.int32)
    ship_speed: jnp.ndarray = jnp.asarray(6.0, dtype=jnp.float32)
    sun_radius: jnp.ndarray = jnp.asarray(SUN_RADIUS, dtype=jnp.float32)
    board_size: jnp.ndarray = jnp.asarray(BOARD_SIZE, dtype=jnp.float32)


class OrbitWarsState(NamedTuple):
    planets: jnp.ndarray  # [MAX_PLANETS, 7], float32
    planet_active: jnp.ndarray  # [MAX_PLANETS], bool
    initial_planets: jnp.ndarray  # [MAX_PLANETS, 7], float32
    initial_active: jnp.ndarray  # [MAX_PLANETS], bool
    fleets: jnp.ndarray  # [max_fleets, 7], float32
    fleet_active: jnp.ndarray  # [max_fleets], bool
    comet_paths: jnp.ndarray  # [5, 4, 40, 2], float32
    comet_path_lengths: jnp.ndarray  # [5, 4], int32
    comet_ships: jnp.ndarray  # [5], float32
    comet_group_active: jnp.ndarray  # [5], bool
    comet_path_index: jnp.ndarray  # [5], int32
    comet_planet_ids: jnp.ndarray  # [5, 4], int32
    comet_slots: jnp.ndarray  # [5, 4], int32
    next_fleet_id: jnp.ndarray
    angular_velocity: jnp.ndarray
    step_count: jnp.ndarray
    num_agents: jnp.ndarray
    rewards: jnp.ndarray  # [4], float32
    done: jnp.ndarray
    overflow: jnp.ndarray


def empty_actions(num_agents: int = 2, max_actions: int = DEFAULT_MAX_ACTIONS) -> jnp.ndarray:
    """Returns a no-op action tensor shaped [num_agents, max_actions, 3]."""

    return jnp.zeros((num_agents, max_actions, 3), dtype=jnp.float32)


def reset_from_reference(
    seed: int,
    num_agents: int = 2,
    *,
    max_fleets: int = DEFAULT_MAX_FLEETS,
) -> OrbitWarsState:
    """Builds an initial JAX state using the official Orbit Wars generator.

    This keeps the fiddly Python RNG/map-generation behavior identical to
    Kaggle for a seed while leaving the hot `step` path pure JAX.
    """

    from kaggle_environments.envs.orbit_wars.orbit_wars import (
        generate_comet_paths,
        generate_planets,
    )

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
            raise ValueError("Orbit Wars supports 2 or 4 agents.")

    comet_paths = jnp.zeros((MAX_COMET_GROUPS, 4, MAX_COMET_PATH, 2), dtype=jnp.float32)
    comet_path_lengths = jnp.zeros((MAX_COMET_GROUPS, 4), dtype=jnp.int32)
    comet_ships = jnp.zeros((MAX_COMET_GROUPS,), dtype=jnp.float32)
    comet_planet_ids_for_generation: list[int] = []
    scratch_initial = [p.copy() for p in initial_planets_list]

    for group_idx, spawn_step in enumerate(COMET_SPAWN_STEPS):
        comet_rng = random.Random(f"orbit_wars-comet-{seed}-{spawn_step}")
        paths = generate_comet_paths(
            scratch_initial,
            angular_velocity,
            spawn_step,
            comet_planet_ids_for_generation,
            4.0,
            rng=comet_rng,
        )
        if paths:
            for comet_idx, path in enumerate(paths):
                length = min(len(path), MAX_COMET_PATH)
                comet_path_lengths = comet_path_lengths.at[group_idx, comet_idx].set(length)
                comet_paths = comet_paths.at[group_idx, comet_idx, :length].set(
                    jnp.asarray(path[:length], dtype=jnp.float32)
                )
            ships = min(
                comet_rng.randint(1, 99),
                comet_rng.randint(1, 99),
                comet_rng.randint(1, 99),
                comet_rng.randint(1, 99),
            )
            comet_ships = comet_ships.at[group_idx].set(float(ships))

    planets = jnp.zeros((MAX_PLANETS, 7), dtype=jnp.float32)
    initial_planets = jnp.zeros((MAX_PLANETS, 7), dtype=jnp.float32)
    planet_active = jnp.zeros((MAX_PLANETS,), dtype=bool)
    initial_active = jnp.zeros((MAX_PLANETS,), dtype=bool)

    base_count = len(planets_list)
    if base_count > MAX_BASE_PLANETS:
        raise ValueError(f"Reference generated {base_count} planets, max is {MAX_BASE_PLANETS}.")

    planets = planets.at[:base_count].set(jnp.asarray(planets_list, dtype=jnp.float32))
    initial_planets = initial_planets.at[:base_count].set(
        jnp.asarray(initial_planets_list, dtype=jnp.float32)
    )
    planet_active = planet_active.at[:base_count].set(True)
    initial_active = initial_active.at[:base_count].set(True)

    return OrbitWarsState(
        planets=planets,
        planet_active=planet_active,
        initial_planets=initial_planets,
        initial_active=initial_active,
        fleets=jnp.zeros((max_fleets, 7), dtype=jnp.float32),
        fleet_active=jnp.zeros((max_fleets,), dtype=bool),
        comet_paths=comet_paths,
        comet_path_lengths=comet_path_lengths,
        comet_ships=comet_ships,
        comet_group_active=jnp.zeros((MAX_COMET_GROUPS,), dtype=bool),
        comet_path_index=jnp.full((MAX_COMET_GROUPS,), -1, dtype=jnp.int32),
        comet_planet_ids=jnp.full((MAX_COMET_GROUPS, 4), -1, dtype=jnp.int32),
        comet_slots=jnp.full((MAX_COMET_GROUPS, 4), -1, dtype=jnp.int32),
        next_fleet_id=jnp.asarray(0, dtype=jnp.int32),
        angular_velocity=jnp.asarray(angular_velocity, dtype=jnp.float32),
        step_count=jnp.asarray(0, dtype=jnp.int32),
        num_agents=jnp.asarray(num_agents, dtype=jnp.int32),
        rewards=jnp.zeros((4,), dtype=jnp.float32),
        done=jnp.asarray(False),
        overflow=jnp.asarray(False),
    )


def _point_to_segment_distance(point, start, end):
    delta = end - start
    l2 = jnp.sum(delta * delta)
    t = jnp.where(l2 == 0.0, 0.0, jnp.sum((point - start) * delta) / l2)
    t = jnp.clip(t, 0.0, 1.0)
    projection = start + t * delta
    return jnp.linalg.norm(point - projection)


def _swept_pair_hit(a0, a1, p0, p1, radius):
    d0 = a0 - p0
    dv = (a1 - a0) - (p1 - p0)
    qa = jnp.sum(dv * dv)
    qb = 2.0 * jnp.sum(d0 * dv)
    qc = jnp.sum(d0 * d0) - radius * radius
    disc = qb * qb - 4.0 * qa * qc
    static_hit = qc <= 0.0
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    t1 = (-qb - sqrt_disc) / (2.0 * qa)
    t2 = (-qb + sqrt_disc) / (2.0 * qa)
    moving_hit = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
    return jnp.where(qa < 1e-12, static_hit, moving_hit)


def _first_true(mask):
    any_true = jnp.any(mask)
    idx = jnp.argmax(mask.astype(jnp.int32))
    return any_true, idx


def _expire_comets(state: OrbitWarsState) -> OrbitWarsState:
    lengths = state.comet_path_lengths
    expired_groups = state.comet_group_active & jnp.all(
        state.comet_path_index[:, None] >= lengths, axis=1
    )
    expired_slots = jnp.zeros_like(state.planet_active)

    def mark_group(group_idx, acc):
        slots = state.comet_slots[group_idx]
        valid_slots = (slots >= 0) & expired_groups[group_idx]
        return acc.at[jnp.maximum(slots, 0)].set(valid_slots | acc[jnp.maximum(slots, 0)])

    expired_slots = jax.lax.fori_loop(0, MAX_COMET_GROUPS, mark_group, expired_slots)
    keep_planets = state.planet_active & ~expired_slots
    keep_initial = state.initial_active & ~expired_slots

    return state._replace(
        planet_active=keep_planets,
        initial_active=keep_initial,
        comet_group_active=state.comet_group_active & ~expired_groups,
        comet_planet_ids=jnp.where(expired_groups[:, None], -1, state.comet_planet_ids),
        comet_slots=jnp.where(expired_groups[:, None], -1, state.comet_slots),
    )


def _spawn_comets(state: OrbitWarsState) -> OrbitWarsState:
    spawn_steps = jnp.asarray(COMET_SPAWN_STEPS, dtype=jnp.int32)
    spawn_matches = (state.step_count + 1) == spawn_steps
    has_spawn, group_idx = _first_true(spawn_matches)
    has_paths = jnp.all(state.comet_path_lengths[group_idx] > 0)
    should_spawn = has_spawn & has_paths

    max_pid = jnp.max(jnp.where(state.planet_active, state.planets[:, PLANET_ID], -1.0)).astype(
        jnp.int32
    )
    next_pid = max_pid + 1

    def add_one(carry, comet_idx):
        planets, initial_planets, planet_active, initial_active, comet_ids, comet_slots, overflow = carry
        inactive = ~planet_active
        has_slot, slot = _first_true(inactive)
        pid = next_pid + comet_idx
        planet = jnp.asarray(
            [0.0, -1.0, -99.0, -99.0, COMET_RADIUS, 0.0, COMET_PRODUCTION],
            dtype=jnp.float32,
        )
        planet = planet.at[PLANET_ID].set(pid.astype(jnp.float32))
        planet = planet.at[PLANET_SHIPS].set(state.comet_ships[group_idx])
        write = should_spawn & has_slot
        safe_slot = jnp.where(has_slot, slot, jnp.asarray(0, dtype=jnp.int32))
        planets = planets.at[safe_slot].set(jnp.where(write, planet, planets[safe_slot]))
        initial_planets = initial_planets.at[safe_slot].set(
            jnp.where(write, planet, initial_planets[safe_slot])
        )
        planet_active = planet_active.at[safe_slot].set(write | planet_active[safe_slot])
        initial_active = initial_active.at[safe_slot].set(write | initial_active[safe_slot])
        comet_ids = comet_ids.at[group_idx, comet_idx].set(
            jnp.where(write, pid, comet_ids[group_idx, comet_idx]).astype(jnp.int32)
        )
        comet_slots = comet_slots.at[group_idx, comet_idx].set(
            jnp.where(write, safe_slot, comet_slots[group_idx, comet_idx]).astype(jnp.int32)
        )
        overflow = overflow | (should_spawn & ~has_slot)
        return (
            planets,
            initial_planets,
            planet_active,
            initial_active,
            comet_ids,
            comet_slots,
            overflow,
        ), None

    carry = (
        state.planets,
        state.initial_planets,
        state.planet_active,
        state.initial_active,
        state.comet_planet_ids,
        state.comet_slots,
        state.overflow,
    )
    carry, _ = jax.lax.scan(add_one, carry, jnp.arange(4, dtype=jnp.int32))
    planets, initial_planets, planet_active, initial_active, comet_ids, comet_slots, overflow = carry

    return state._replace(
        planets=planets,
        initial_planets=initial_planets,
        planet_active=planet_active,
        initial_active=initial_active,
        comet_planet_ids=comet_ids,
        comet_slots=comet_slots,
        comet_path_index=state.comet_path_index.at[group_idx].set(
            jnp.where(should_spawn, -1, state.comet_path_index[group_idx])
        ),
        comet_group_active=state.comet_group_active.at[group_idx].set(
            should_spawn | state.comet_group_active[group_idx]
        ),
        overflow=overflow,
    )


def _launch_fleets(state: OrbitWarsState, actions: jnp.ndarray) -> OrbitWarsState:
    actions = jnp.asarray(actions, dtype=jnp.float32)

    def launch_one(carry, flat_idx):
        planets, fleets, fleet_active, next_fleet_id, overflow = carry
        max_actions = actions.shape[1]
        player = flat_idx // max_actions
        action_idx = flat_idx % max_actions
        from_id, angle, ships_raw = actions[player, action_idx]
        ships = jnp.floor(ships_raw)
        planet_id_match = planets[:, PLANET_ID] == from_id
        valid_planet_mask = state.planet_active & planet_id_match
        has_planet, planet_idx = _first_true(valid_planet_mask)
        planet = planets[planet_idx]
        valid = (
            (player < state.num_agents)
            & has_planet
            & (planet[PLANET_OWNER] == player.astype(jnp.float32))
            & (planet[PLANET_SHIPS] >= ships)
            & (ships > 0)
        )
        has_fleet_slot, fleet_idx = _first_true(~fleet_active)
        write = valid & has_fleet_slot
        start_x = planet[PLANET_X] + jnp.cos(angle) * (planet[PLANET_RADIUS] + 0.1)
        start_y = planet[PLANET_Y] + jnp.sin(angle) * (planet[PLANET_RADIUS] + 0.1)
        fleet = jnp.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
        fleet = fleet.at[FLEET_ID].set(next_fleet_id.astype(jnp.float32))
        fleet = fleet.at[FLEET_OWNER].set(player.astype(jnp.float32))
        fleet = fleet.at[FLEET_X].set(start_x)
        fleet = fleet.at[FLEET_Y].set(start_y)
        fleet = fleet.at[FLEET_ANGLE].set(angle)
        fleet = fleet.at[FLEET_FROM_PLANET].set(from_id)
        fleet = fleet.at[FLEET_SHIPS].set(ships)

        safe_fleet_idx = jnp.where(has_fleet_slot, fleet_idx, 0)
        planets = planets.at[planet_idx, PLANET_SHIPS].add(jnp.where(valid, -ships, 0.0))
        fleets = fleets.at[safe_fleet_idx].set(jnp.where(write, fleet, fleets[safe_fleet_idx]))
        fleet_active = fleet_active.at[safe_fleet_idx].set(write | fleet_active[safe_fleet_idx])
        next_fleet_id = next_fleet_id + write.astype(jnp.int32)
        overflow = overflow | (valid & ~has_fleet_slot)
        return (planets, fleets, fleet_active, next_fleet_id, overflow), None

    total_actions = actions.shape[0] * actions.shape[1]
    carry = (
        state.planets,
        state.fleets,
        state.fleet_active,
        state.next_fleet_id,
        state.overflow,
    )
    carry, _ = jax.lax.scan(launch_one, carry, jnp.arange(total_actions, dtype=jnp.int32))
    planets, fleets, fleet_active, next_fleet_id, overflow = carry
    return state._replace(
        planets=planets,
        fleets=fleets,
        fleet_active=fleet_active,
        next_fleet_id=next_fleet_id,
        overflow=overflow,
    )


def _planet_paths(state: OrbitWarsState):
    old_pos = state.planets[:, PLANET_X : PLANET_Y + 1]
    init_pos = state.initial_planets[:, PLANET_X : PLANET_Y + 1]
    delta = init_pos - CENTER
    orbital_r = jnp.linalg.norm(delta, axis=1)
    initial_angle = jnp.arctan2(delta[:, 1], delta[:, 0])
    rotating = (
        state.planet_active
        & state.initial_active
        & (orbital_r + state.planets[:, PLANET_RADIUS] < ROTATION_RADIUS_LIMIT)
    )
    current_angle = initial_angle + state.angular_velocity * state.step_count.astype(jnp.float32)
    rotated = jnp.stack(
        [
            CENTER + orbital_r * jnp.cos(current_angle),
            CENTER + orbital_r * jnp.sin(current_angle),
        ],
        axis=1,
    )
    new_pos = jnp.where(rotating[:, None], rotated, old_pos)
    check_collision = state.planet_active.copy()

    next_path_index = state.comet_path_index + state.comet_group_active.astype(jnp.int32)
    expired_after_move = jnp.zeros_like(state.planet_active)

    def group_body(carry, group_idx):
        new_pos, check_collision, expired_after_move = carry
        idx = next_path_index[group_idx]
        slots = state.comet_slots[group_idx]
        active_group = state.comet_group_active[group_idx]

        def comet_body(inner_carry, comet_idx):
            new_pos, check_collision, expired_after_move = inner_carry
            slot = slots[comet_idx]
            safe_slot = jnp.maximum(slot, 0)
            active = active_group & (slot >= 0)
            length = state.comet_path_lengths[group_idx, comet_idx]
            expired = active & (idx >= length)
            in_path = active & (idx < length)
            path_pos = state.comet_paths[group_idx, comet_idx, jnp.maximum(idx, 0)]
            first_placement = state.planets[safe_slot, PLANET_X] < 0.0
            new_pos = new_pos.at[safe_slot].set(jnp.where(in_path, path_pos, new_pos[safe_slot]))
            check = active & (~first_placement | expired)
            check_collision = check_collision.at[safe_slot].set(check)
            expired_after_move = expired_after_move.at[safe_slot].set(
                expired | expired_after_move[safe_slot]
            )
            return (new_pos, check_collision, expired_after_move), None

        carry, _ = jax.lax.scan(
            comet_body,
            (new_pos, check_collision, expired_after_move),
            jnp.arange(4, dtype=jnp.int32),
        )
        return carry, None

    (new_pos, check_collision, expired_after_move), _ = jax.lax.scan(
        group_body,
        (new_pos, check_collision, expired_after_move),
        jnp.arange(MAX_COMET_GROUPS, dtype=jnp.int32),
    )
    return old_pos, new_pos, check_collision, next_path_index, expired_after_move


def _move_fleets_and_collect_combats(
    state: OrbitWarsState,
    old_planet_pos: jnp.ndarray,
    new_planet_pos: jnp.ndarray,
    planet_collision_enabled: jnp.ndarray,
    config: OrbitWarsConfig,
):
    combat_ships = jnp.zeros((MAX_PLANETS, 4), dtype=jnp.float32)
    fleets = state.fleets
    fleet_active = state.fleet_active

    def move_one(carry, fleet_idx):
        fleets, fleet_active, combat_ships = carry
        fleet = fleets[fleet_idx]
        active = fleet_active[fleet_idx]
        angle = fleet[FLEET_ANGLE]
        ships = fleet[FLEET_SHIPS]
        speed = 1.0 + (config.ship_speed - 1.0) * (
            jnp.log(ships) / jnp.log(jnp.asarray(1000.0, dtype=jnp.float32))
        ) ** 1.5
        speed = jnp.minimum(speed, config.ship_speed)
        old_pos = fleet[FLEET_X : FLEET_Y + 1]
        new_pos = old_pos + jnp.asarray([jnp.cos(angle) * speed, jnp.sin(angle) * speed])

        hit_mask = jax.vmap(_swept_pair_hit, in_axes=(None, None, 0, 0, 0))(
            old_pos,
            new_pos,
            old_planet_pos,
            new_planet_pos,
            state.planets[:, PLANET_RADIUS],
        )
        hit_mask = hit_mask & state.planet_active & planet_collision_enabled & active
        hit_planet, planet_idx = _first_true(hit_mask)
        in_bounds = (
            (new_pos[0] >= 0.0)
            & (new_pos[0] <= config.board_size)
            & (new_pos[1] >= 0.0)
            & (new_pos[1] <= config.board_size)
        )
        sun_hit = _point_to_segment_distance(jnp.asarray([CENTER, CENTER]), old_pos, new_pos) < config.sun_radius
        remove = active & (hit_planet | (~in_bounds) | sun_hit)
        owner = fleet[FLEET_OWNER].astype(jnp.int32)
        combat_ships = combat_ships.at[planet_idx, owner].add(
            jnp.where(hit_planet, ships, 0.0)
        )
        fleet = fleet.at[FLEET_X : FLEET_Y + 1].set(jnp.where(active, new_pos, old_pos))
        fleets = fleets.at[fleet_idx].set(fleet)
        fleet_active = fleet_active.at[fleet_idx].set(active & ~remove)
        return (fleets, fleet_active, combat_ships), None

    carry = (fleets, fleet_active, combat_ships)
    carry, _ = jax.lax.scan(move_one, carry, jnp.arange(fleets.shape[0], dtype=jnp.int32))
    return carry


def _resolve_combats(state: OrbitWarsState, combat_ships: jnp.ndarray) -> OrbitWarsState:
    def resolve_one(planets, planet_idx):
        planet = planets[planet_idx]
        ships_by_player = combat_ships[planet_idx]
        order = jnp.argsort(-ships_by_player)
        top_player = order[0]
        second_player = order[1]
        top_ships = ships_by_player[top_player]
        second_ships = ships_by_player[second_player]
        survivor_ships = jnp.where(top_ships == second_ships, 0.0, top_ships - second_ships)
        survivor_owner = top_player.astype(jnp.float32)
        has_survivor = survivor_ships > 0.0
        same_owner = planet[PLANET_OWNER] == survivor_owner
        new_ships = jnp.where(
            same_owner,
            planet[PLANET_SHIPS] + survivor_ships,
            planet[PLANET_SHIPS] - survivor_ships,
        )
        captured = has_survivor & (~same_owner) & (new_ships < 0.0)
        planet = planet.at[PLANET_OWNER].set(
            jnp.where(captured, survivor_owner, planet[PLANET_OWNER])
        )
        planet = planet.at[PLANET_SHIPS].set(
            jnp.where(
                has_survivor,
                jnp.where(captured, -new_ships, new_ships),
                planet[PLANET_SHIPS],
            )
        )
        write = state.planet_active[planet_idx] & (top_ships > 0.0)
        planets = planets.at[planet_idx].set(jnp.where(write, planet, planets[planet_idx]))
        return planets, None

    planets, _ = jax.lax.scan(resolve_one, state.planets, jnp.arange(MAX_PLANETS, dtype=jnp.int32))
    return state._replace(planets=planets)


def _score_and_done(state: OrbitWarsState, config: OrbitWarsConfig) -> OrbitWarsState:
    planet_owners = state.planets[:, PLANET_OWNER].astype(jnp.int32)
    fleet_owners = state.fleets[:, FLEET_OWNER].astype(jnp.int32)
    safe_planet_owners = jnp.maximum(planet_owners, 0)
    safe_fleet_owners = jnp.maximum(fleet_owners, 0)
    planet_values = jnp.where(
        state.planet_active & (planet_owners >= 0), state.planets[:, PLANET_SHIPS], 0.0
    )
    fleet_values = jnp.where(state.fleet_active, state.fleets[:, FLEET_SHIPS], 0.0)
    planet_scores = jnp.zeros((4,), dtype=jnp.float32).at[safe_planet_owners].add(
        planet_values
    )
    fleet_scores = jnp.zeros((4,), dtype=jnp.float32).at[safe_fleet_owners].add(
        fleet_values
    )
    scores = planet_scores + fleet_scores
    alive_planet_values = (state.planet_active & (planet_owners >= 0)).astype(jnp.int32)
    alive_fleet_values = state.fleet_active.astype(jnp.int32)
    alive_from_planets = (
        jnp.zeros((4,), dtype=jnp.int32).at[safe_planet_owners].max(alive_planet_values)
        > 0
    )
    alive_from_fleets = (
        jnp.zeros((4,), dtype=jnp.int32).at[safe_fleet_owners].max(alive_fleet_values) > 0
    )
    alive = alive_from_planets | alive_from_fleets
    player_mask = jnp.arange(4, dtype=jnp.int32) < state.num_agents
    alive_count = jnp.sum(alive & player_mask)
    terminated = (state.step_count >= config.episode_steps - 2) | (alive_count <= 1)
    max_score = jnp.max(jnp.where(player_mask, scores, -jnp.inf))
    rewards = jnp.where((scores == max_score) & (max_score > 0.0), 1.0, -1.0)
    rewards = jnp.where(terminated, rewards, state.rewards)
    return state._replace(done=terminated, rewards=rewards)


def step(
    state: OrbitWarsState,
    actions: jnp.ndarray,
    config: OrbitWarsConfig = OrbitWarsConfig(),
) -> OrbitWarsState:
    """Runs one Orbit Wars turn.

    `actions` is shaped `[num_agents, max_actions, 3]` with rows
    `[from_planet_id, direction_angle, num_ships]`. Zero rows are no-ops.
    """

    def do_step(s):
        s = _expire_comets(s)
        s = _spawn_comets(s)
        s = _launch_fleets(s, actions)

        owned = s.planet_active & (s.planets[:, PLANET_OWNER] != -1.0)
        s = s._replace(
            planets=s.planets.at[:, PLANET_SHIPS].add(
                jnp.where(owned, s.planets[:, PLANET_PRODUCTION], 0.0)
            )
        )

        old_pos, new_pos, collision_enabled, next_path_index, expired_after_move = _planet_paths(s)
        fleets, fleet_active, combat_ships = _move_fleets_and_collect_combats(
            s, old_pos, new_pos, collision_enabled, config
        )
        planets = s.planets.at[:, PLANET_X : PLANET_Y + 1].set(new_pos)
        planet_active = s.planet_active & ~expired_after_move
        initial_active = s.initial_active & ~expired_after_move

        expired_group = jnp.zeros((MAX_COMET_GROUPS,), dtype=bool)

        def expire_group(group_idx, acc):
            slots = s.comet_slots[group_idx]
            expired = jnp.any(expired_after_move[jnp.maximum(slots, 0)] & (slots >= 0))
            return acc.at[group_idx].set(expired)

        expired_group = jax.lax.fori_loop(0, MAX_COMET_GROUPS, expire_group, expired_group)

        s = s._replace(
            planets=planets,
            planet_active=planet_active,
            initial_active=initial_active,
            fleets=fleets,
            fleet_active=fleet_active,
            comet_path_index=next_path_index,
            comet_group_active=s.comet_group_active & ~expired_group,
            comet_planet_ids=jnp.where(expired_group[:, None], -1, s.comet_planet_ids),
            comet_slots=jnp.where(expired_group[:, None], -1, s.comet_slots),
        )
        s = _resolve_combats(s, combat_ships)
        s = _score_and_done(s, config)
        return s._replace(step_count=s.step_count + 1)

    return jax.lax.cond(state.done, lambda s: s, do_step, state)


def expand_fleet_buffers(state: OrbitWarsState, new_max_fleets: int) -> OrbitWarsState:
    """Copies fleet rows into a larger `[new_max_fleets, 7]` buffer.

    Call when `_launch_fleets` cannot place a fleet because `fleet_active` is
    full (`overflow=True`). Preserves existing fleet slots and flags.
    """

    old_max = int(state.fleets.shape[0])
    if new_max_fleets <= old_max:
        return state
    fleets = jnp.zeros((new_max_fleets, 7), dtype=jnp.float32)
    fleet_active = jnp.zeros((new_max_fleets,), dtype=bool)
    fleets = fleets.at[:old_max].set(state.fleets)
    fleet_active = fleet_active.at[:old_max].set(state.fleet_active)
    return state._replace(fleets=fleets, fleet_active=fleet_active)


jit_step = jax.jit(step)

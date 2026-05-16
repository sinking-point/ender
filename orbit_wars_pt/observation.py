"""NumPy observation tensors for the policy (ego framed as player 0)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import jax
import numpy as np

from jax_orbit_wars import FLEET_ETA, FLEET_OWNER, FLEET_SHIPS, FLEET_TARGET_PLANET, OrbitWarsConfig

from orbit_wars_pt.constants import (
    BOARD_SIZE,
    CENTER,
    ENTITY_CLS,
    ENTITY_COMET,
    FEATURE_DIM_MULTI,
    INCOMING_SURVIVOR_FLAT,
    INCOMING_TA_BINS,
    ENTITY_PLANET,
    MAX_FLEET_TOKENS,
    MAX_PLANETS,
    NUM_OWNER_SLOTS,
    obs_feature_dim_for_num_agents,
)
from orbit_wars_pt.geometry import planet_pred_velocity


def _cls_turn_fraction(step_count: int) -> float:
    """Fraction of the way to the env step limit (for value bootstrapping / timeouts)."""

    ep = int(np.asarray(jax.device_get(OrbitWarsConfig().episode_steps)))
    denom = max(ep - 2, 1)
    return float(np.clip(float(step_count) / float(denom), 0.0, 1.0))


def _remap_owner(owner: float, ego: int, num_agents: int) -> int:
    o = int(owner)
    if o < 0:
        return 0  # neutral bucket
    if o == ego:
        return 1  # self
    # 2p: single enemy bucket; 4p-ready: distinct opponent ids 2..num_agents
    if num_agents <= 2:
        return 2
    # compress other players to slots 2..num_agents (reserve 0 neutral 1 self)
    if o < ego:
        return 2 + o
    return 2 + (o - 1)


def _incoming_interfleet_np(
    incoming: np.ndarray,
    ego: int,
    num_agents: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Interfleet largest-vs-2nd reduction; returns (signed_net[P,T], survivor_slot[P,T] int)."""

    a = incoming.shape[0]
    padded = np.pad(incoming.astype(np.float32), ((0, 4 - a), (0, 0), (0, 0)))
    ships = np.transpose(padded, (1, 2, 0))
    order = np.argsort(-ships, axis=-1)
    top_s = np.take_along_axis(ships, order[..., :1], axis=-1)[..., 0]
    second_s = np.take_along_axis(ships, order[..., 1:2], axis=-1)[..., 0]
    top_p = order[..., 0].astype(np.int32)
    survivor = np.where(top_s == second_s, 0.0, top_s - second_s)
    ego_j = int(ego)
    signed = np.where(survivor <= 0.0, 0.0, np.where(top_p == ego_j, survivor, -survivor))
    is_self = top_p == ego_j
    slot_le2 = np.where(survivor <= 0.0, 0, np.where(is_self, 1, 2))
    slot_gt2 = np.where(
        survivor <= 0.0,
        0,
        np.where(is_self, 1, np.where(top_p < ego_j, 2 + top_p, 2 + (top_p - 1))),
    )
    survivor_slot = np.where(num_agents <= 2, slot_le2, slot_gt2)
    survivor_slot = np.clip(survivor_slot, 0, NUM_OWNER_SLOTS - 1).astype(np.int32)
    return signed / 1000.0, survivor_slot


@dataclass
class ObservationBatch:
    entity_type: np.ndarray  # [L] int64
    owner_idx: np.ndarray  # [L] int64
    features: np.ndarray  # [L, F]
    rope_pos: np.ndarray  # [L, 3] float32 (x, y, time) scaled ~ O(1)
    entity_mask: np.ndarray  # [L] bool — True where token is real
    planet_mask: np.ndarray  # [L] bool — planet slots (incl. inactive padding)
    cls_index: int
    planet_positions: np.ndarray  # [P, 2] for masking / updates
    planet_ids: np.ndarray  # [P] float planet id per slot
    planet_idx_by_id: Dict[float, int]
    ego_player: int
    num_planets: int  # physical slots P == MAX_PLANETS


def build_observation(
    state: Any,
    ego_player: int,
    *,
    max_fleet_tokens: int = MAX_FLEET_TOKENS,
    ship_speed: float = 6.0,
) -> ObservationBatch:
    """Builds a padded entity sequence `[CLS] + planets + fleets` from a JAX `OrbitWarsState` or numpy views."""

    planets = np.asarray(state.planets)
    planet_active = np.asarray(state.planet_active)
    initial_planets = np.asarray(state.initial_planets)
    initial_active = np.asarray(state.initial_active)
    incoming_fleets = np.asarray(state.incoming_fleets)
    angular_velocity = float(np.asarray(state.angular_velocity))
    step_count = int(np.asarray(state.step_count))
    num_agents = int(np.asarray(state.num_agents))

    comet_ids = np.asarray(state.comet_planet_ids)
    comet_set = set(float(x) for x in comet_ids.flatten() if int(x) >= 0)

    cls_index = 0
    p_slots = MAX_PLANETS
    planet_ids = planets[:p_slots, 0].astype(np.float64)
    planet_xy = planets[:p_slots, 2:4].astype(np.float64)
    planet_r = planets[:p_slots, 4].astype(np.float64)
    planet_owner = planets[:p_slots, 1].astype(np.float64)
    planet_ships = planets[:p_slots, 5].astype(np.float64)
    planet_prod = planets[:p_slots, 6].astype(np.float64)
    init_xy = initial_planets[:p_slots, 2:4].astype(np.float64)

    planet_idx_by_id = {}
    for i in range(p_slots):
        if planet_active[i]:
            planet_idx_by_id[float(planet_ids[i])] = i

    cls_turn = _cls_turn_fraction(step_count)
    incoming = incoming_fleets.astype(np.float32)
    fdim = obs_feature_dim_for_num_agents(num_agents)
    incoming_net, survivor_slot = _incoming_interfleet_np(incoming, int(ego_player), num_agents)

    cls_feat = np.zeros((fdim,), dtype=np.float32)
    cls_feat[6] = np.float32(cls_turn)

    types_list: List[int] = [ENTITY_CLS]
    owner_list: List[int] = [1]
    feat_list: List[np.ndarray] = [cls_feat]
    rope_list: List[np.ndarray] = [np.array([CENTER / BOARD_SIZE, CENTER / BOARD_SIZE, 0.0], dtype=np.float32)]
    mask_list: List[bool] = [True]
    planet_mask_list: List[bool] = [False]

    # Planet tokens occupy fixed indices 1 .. P_slots
    for i in range(p_slots):
        active = bool(planet_active[i])
        pid = float(planet_ids[i])
        is_comet = pid in comet_set
        etype = ENTITY_COMET if is_comet else ENTITY_PLANET
        ox = _remap_owner(planet_owner[i], ego_player, num_agents)

        vx, vy = planet_pred_velocity(
            init_xy[i],
            planet_xy[i],
            float(planet_r[i]),
            angular_velocity,
            step_count,
            bool(initial_active[i]),
            active,
        )

        if is_comet and active:
            group_row = np.where(comet_ids == int(pid))
            if group_row[0].size > 0:
                g, k = int(group_row[0][0]), int(group_row[1][0])
                paths = np.asarray(state.comet_paths[g, k])
                lens = int(np.asarray(state.comet_path_lengths[g, k]))
                idx = int(np.asarray(state.comet_path_index[g]))
                if lens > 1 and 0 <= idx < lens - 1:
                    p0 = paths[idx]
                    p1 = paths[idx + 1]
                    vx = float(p1[0] - p0[0])
                    vy = float(p1[1] - p0[1])

        feat = np.zeros((fdim,), dtype=np.float32)
        feat[0] = np.float32(np.log1p(max(planet_prod[i], 0.0)))
        feat[1] = np.float32(planet_ships[i] / 1000.0)
        feat[2] = np.float32(vx / 5.0)
        feat[3] = np.float32(vy / 5.0)
        feat[4] = np.float32(active)
        feat[5] = np.float32(planet_r[i] / 10.0)
        feat[8 : 8 + INCOMING_TA_BINS] = incoming_net[i].astype(np.float32)
        if fdim == FEATURE_DIM_MULTI and num_agents > 2:
            oh = np.eye(NUM_OWNER_SLOTS, dtype=np.float32)[survivor_slot[i]]
            feat[
                8 + INCOMING_TA_BINS : 8 + INCOMING_TA_BINS + INCOMING_SURVIVOR_FLAT
            ] = oh.reshape(INCOMING_SURVIVOR_FLAT)

        xy = planet_xy[i] if active else np.zeros((2,), dtype=np.float64)
        rope = np.array([float(xy[0]) / BOARD_SIZE, float(xy[1]) / BOARD_SIZE, 0.0], dtype=np.float32)

        types_list.append(etype)
        owner_list.append(min(ox, NUM_OWNER_SLOTS - 1))
        feat_list.append(feat)
        rope_list.append(rope)
        mask_list.append(active)
        planet_mask_list.append(True)

    L = len(types_list)
    entity_type = np.asarray(types_list, dtype=np.int64)
    owner_idx = np.asarray(owner_list, dtype=np.int64)
    features = np.stack(feat_list, axis=0).astype(np.float32)
    rope_pos = np.stack(rope_list, axis=0).astype(np.float32)
    entity_mask = np.asarray(mask_list, dtype=np.bool_)
    planet_mask = np.asarray(planet_mask_list, dtype=np.bool_)

    return ObservationBatch(
        entity_type=entity_type,
        owner_idx=owner_idx,
        features=features,
        rope_pos=rope_pos,
        entity_mask=entity_mask,
        planet_mask=planet_mask,
        cls_index=cls_index,
        planet_positions=planet_xy.astype(np.float32),
        planet_ids=planet_ids.astype(np.float32),
        planet_idx_by_id=planet_idx_by_id,
        ego_player=ego_player,
        num_planets=p_slots,
    )


def jax_state_to_numpy(state: Any) -> Any:
    """Convert JAX `OrbitWarsState` leaves to NumPy (CPU)."""

    def _to_np(x):
        return np.asarray(jax.device_get(x))

    if hasattr(state, "_replace"):
        return state._replace(
            planets=_to_np(state.planets),
            planet_active=_to_np(state.planet_active),
            initial_planets=_to_np(state.initial_planets),
            initial_active=_to_np(state.initial_active),
            fleets=_to_np(state.fleets),
            fleet_active=_to_np(state.fleet_active),
            incoming_fleets=_to_np(state.incoming_fleets),
            comet_paths=_to_np(state.comet_paths),
            comet_path_lengths=_to_np(state.comet_path_lengths),
            comet_ships=_to_np(state.comet_ships),
            comet_group_active=_to_np(state.comet_group_active),
            comet_path_index=_to_np(state.comet_path_index),
            comet_planet_ids=_to_np(state.comet_planet_ids),
            comet_slots=_to_np(state.comet_slots),
            next_fleet_id=_to_np(state.next_fleet_id),
            angular_velocity=_to_np(state.angular_velocity),
            step_count=_to_np(state.step_count),
            num_agents=_to_np(state.num_agents),
            rewards=_to_np(state.rewards),
            done=_to_np(state.done),
            overflow=_to_np(state.overflow),
        )
    return state


def total_ship_ratio_scores(state_np: Any) -> Tuple[float, float, float]:
    """Returns (ratio_ego0, ratio_ego1, total_ships) for 2 agents (ego ids 0 and 1)."""

    planets = np.asarray(state_np.planets)
    planet_active = np.asarray(state_np.planet_active)
    incoming_fleets = np.asarray(state_np.incoming_fleets)

    scores = np.zeros((4,), dtype=np.float64)
    total = 1e-6
    for i in range(planets.shape[0]):
        if not planet_active[i]:
            continue
        owner = int(planets[i, 1])
        if owner < 0:
            continue
        sh = float(planets[i, 5])
        scores[owner] += sh
        total += sh
    fleet_scores = incoming_fleets.astype(np.float64).sum(axis=(1, 2))
    scores[: fleet_scores.shape[0]] += fleet_scores
    total += float(fleet_scores.sum())

    r0 = float(scores[0] / total)
    r1 = float(scores[1] / total)
    return r0, r1, float(total)

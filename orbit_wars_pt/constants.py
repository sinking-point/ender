"""Shared constants aligned with `jax_orbit_wars`."""

BOARD_SIZE = 100.0
CENTER = BOARD_SIZE / 2.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0

MAX_PLANETS = 60  # MAX_BASE_PLANETS + MAX_COMETS in jax

# Tangent geometry: peak moving bodies (orbiting + comets) and overlap windows per horizon.
MAX_MOVING_TARGETS = 32
MAX_POLYLINE_TANGENCY_HITS = 4
MAX_POLYLINE_INTERSECTION_WINDOWS = 2

# Incoming-fleet countdown bins stored directly on each planet token.
# Dispatch geometry only allows hits inside this horizon.
INCOMING_TA_BINS = 24

# Owner ids after ego-remap (neutral, ego=0, ally unused in 2p, enemies) — also one-hot width for survivor plane.
NUM_OWNER_SLOTS = 5
FUTURE_PLANET_FEATURES_PER_TICK = NUM_OWNER_SLOTS + 1

BASE_FEATURE_DIM = 8
FEATURE_DIM = BASE_FEATURE_DIM + INCOMING_TA_BINS
# After signed interfleet mass (``INCOMING_TA_BINS``), each TA bin uses a one-hot over ``NUM_OWNER_SLOTS``.
INCOMING_SURVIVOR_FLAT = INCOMING_TA_BINS * NUM_OWNER_SLOTS
FEATURE_DIM_MULTI = BASE_FEATURE_DIM + INCOMING_TA_BINS + INCOMING_SURVIVOR_FLAT
FUTURE_PLANET_FEATURE_DIM = INCOMING_TA_BINS * FUTURE_PLANET_FEATURES_PER_TICK

# Legacy cap retained for compatibility with older host/debug helpers.
MAX_FLEET_TOKENS = 128

ENTITY_CLS = 0
ENTITY_PLANET = 1
ENTITY_COMET = 2
ENTITY_FLEET = 3

NUM_ENTITY_TYPES = 4

# Five dispatch fractions: 20% .. 100%
FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
NUM_FRACTIONS = len(FRACTIONS)
BLOCKED_FRAC_FEATURES = NUM_FRACTIONS
FEATURE_DIM_ABORT = FEATURE_DIM + BLOCKED_FRAC_FEATURES
FEATURE_DIM_MULTI_ABORT = FEATURE_DIM_MULTI + BLOCKED_FRAC_FEATURES

# Owner ids after ego-remap (semantic aliases; indices must match ``NUM_OWNER_SLOTS``).
OWNER_NEUTRAL = 0
OWNER_SELF = 1
OWNER_ENEMY_1 = 2
OWNER_ENEMY_2 = 3
OWNER_ENEMY_3 = 4


def obs_feature_dim_for_num_agents(num_agents: int, target_abort_enabled: bool = False) -> int:
    """Policy/observation feature width: 2p keeps the original 32-dim layout."""

    if int(num_agents) > 2:
        return FEATURE_DIM_MULTI_ABORT if bool(target_abort_enabled) else FEATURE_DIM_MULTI
    return FEATURE_DIM_ABORT if bool(target_abort_enabled) else FEATURE_DIM

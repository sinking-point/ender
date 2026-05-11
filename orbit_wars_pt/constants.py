"""Shared constants aligned with `jax_orbit_wars`."""

BOARD_SIZE = 100.0
CENTER = BOARD_SIZE / 2.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0

MAX_PLANETS = 60  # MAX_BASE_PLANETS + MAX_COMETS in jax

# Policy / tensor caps (truncate beyond this for speed)
MAX_FLEET_TOKENS = 128

ENTITY_CLS = 0
ENTITY_PLANET = 1
ENTITY_COMET = 2
ENTITY_FLEET = 3

NUM_ENTITY_TYPES = 4

# Five dispatch fractions: 20% .. 100%
FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
NUM_FRACTIONS = len(FRACTIONS)

# Owner ids after ego-remap (neutral, ego=0, ally unused in 2p, enemies)
OWNER_NEUTRAL = 0
OWNER_SELF = 1
OWNER_ENEMY_1 = 2
OWNER_ENEMY_2 = 3
OWNER_ENEMY_3 = 4
NUM_OWNER_SLOTS = 5

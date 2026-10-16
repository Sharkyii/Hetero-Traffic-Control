# ── Simulation ────────────────────────────────────────────────────────────────
FPS            = 60
FPS_SIM        = 4
PLAYBACK_SPEED = 1.0

START_FRAME = 111
END_FRAME   = 290

# ── Light ─────────────────────────────────────────────────────────────────────
LIGHT_TURN_RED_FRAME = 60      # adjust after you see the data
STOP_BUFFER_METERS   = 2.0     # metres before stop line

# ── Forces ────────────────────────────────────────────────────────────────────
REACTION_TIME          = 0.005
REPULSION_STRENGTH     = 200.0
REPULSION_RANGE        = 0.3
WALL_REPULSION_STRENGTH = 400.0
WALL_REPULSION_RANGE   = 1.5
CONTACT_STIFFNESS      = 60000.0
CONTACT_FRICTION       = 120000.0
INTERACTION_RANGE      = 10.0
REPULSION_FORCE_MAG_BASE          = 100.0
REPULSION_DISTANCE_THRESHOLD_PX   = 50

FORCES_ENABLED = {
    "drive":     True,
    "repulsive": True,
    "obstacle":  True,
    "stop":      False,
}

# ── Car defaults ──────────────────────────────────────────────────────────────
DESIRED_SPEED   = 15
MAX_SPEED       = 10.0
DEFAULT_CAR_W   = 20
DEFAULT_CAR_H   = 10
CAR_LENGTH      = 25
CAR_WIDTH       = 12

STATIC_MOVEMENT_THRESHOLD = 2.0
STATIC_SPEED_THRESHOLD    = 0.5

# ── Vehicle classes ───────────────────────────────────────────────────────────
VEHICLE_SIZE_SMALL = 300
VEHICLE_SIZE_LARGE = 600

VEHICLE_PARAMS = {
    'small':  {'desired_speed': DESIRED_SPEED, 'reaction_time': REACTION_TIME,
               'repulsion_strength': 5619.5, 'repulsion_range': 0.57},
    'medium': {'desired_speed': DESIRED_SPEED, 'reaction_time': REACTION_TIME,
               'repulsion_strength': 3806.4, 'repulsion_range': 0.46},
    'large':  {'desired_speed': DESIRED_SPEED, 'reaction_time': REACTION_TIME,
               'repulsion_strength': 3570.9, 'repulsion_range': 0.14},
}

# ── Display ───────────────────────────────────────────────────────────────────
WIDTH  = 1600
HEIGHT = 900
SHOW_FORCE_ARROWS = False
# Unused -- nothing in the codebase reads COLLISION_STOP_FRAMES; the
# car.stopped_frames field it was presumably meant to drive is only ever
# initialized to 0 and decremented, never set positive. Left commented
# rather than removed in case the collision-cooldown feature gets wired
# up later.
# COLLISION_STOP_FRAMES = 2

# ── Boundary ──────────────────────────────────────────────────────────────────
EXIT_BUFFER_PX           = 20
BOUNDARY_PUSH_STRENGTH_PX = 5.0
EPS = 1e-9

# ── Stopping behavior ─────────────────────────────────────────────────────────
BRAKING_ZONE_METERS  = 30.56
STOP_LINE_REPULSION  = 2260.60
FORCE_ARROW_SCALE    = 0.02

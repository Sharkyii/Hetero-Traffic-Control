# =========================================================
# CONSTANTS - Fixed values (don't change these)
# =========================================================
# These are technical constants that should remain fixed
# for the simulation to work properly
# =========================================================

# ---------------------------------------------------------
# DISPLAY CONSTANTS
# ---------------------------------------------------------
WIDTH = 1200             # Window width in pixels
HEIGHT = 600             # Window height in pixels

# ---------------------------------------------------------
# UNIT CONVERSION
# ---------------------------------------------------------
PIXELS_PER_METER = 6.75  # How many pixels = 1 meter in physics
METERS_PER_PIXEL = 1.0 / PIXELS_PER_METER

# ---------------------------------------------------------
# PHYSICS CONSTANTS
# ---------------------------------------------------------
AGENT_RADIUS = 0.3       # Car radius for physics (meters)
AGENT_MASS = 80.0        # Car mass (kg)

# Small number to prevent division by zero
EPS = 1e-9

# Road geometry
ROAD_MARGIN_M = 2.0       # Margin for road boundaries in meters

# ---------------------------------------------------------
# LANE 7 DATA CONSTANTS
# ---------------------------------------------------------
# These come from analyzing the CSV data
LANE7_MIN_X = 1500
LANE7_MAX_X = 2700
LANE7_MIN_Y = 650
LANE7_MAX_Y = 800

# Coordinate transformations
LANE_X_OFFSET = -1500
LANE_Y_OFFSET = -725
LANE7_SCREEN_Y = HEIGHT // 2

# CSV data scaling
CSV_PIXELS_PER_METER = 20.0
LANE7_PIXELS_PER_METER = 20.0

# ---------------------------------------------------------
# FORCE VISUALIZATION COLORS
# ---------------------------------------------------------
COLORS = {
    "force_drive": (0, 255, 0),        # GREEN
    "force_repulsive": (255, 50, 50),  # RED  
    "force_obstacle": (255, 165, 0),   # ORANGE
    "force_total": (255, 0, 255),      # MAGENTA
    "car_body": (50, 150, 255),
    "hud_text": (255, 255, 255),
}
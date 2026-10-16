"""
Simulation Configuration and Physical Constants.

Defines spatial calibration, temporal discretization, vehicle dimension profiles,
and Indian Roads Congress (IRC 106-1990) Passenger Car Unit (PCU) equivalents.
"""

import math

# Temporal Discretization
FPS = 4
DT = 1.0 / FPS  # 0.25 seconds per discrete time step

# Spatial Calibration (Empirically measured from drone altitude and road markings)
PIXELS_PER_METER = 11.0
METERS_PER_PIXEL = 1.0 / PIXELS_PER_METER

# Physical Vehicle Classification Dimensions and IRC 106-1990 PCU Equivalents
VEHICLE_CLASSES = {
    "small": {
        "description": "Two-wheelers (motorcycles, scooters, bicycles)",
        "area_max_px": 300.0,
        "pcu": 0.5,
        "default_width_m": 0.8,
        "default_length_m": 1.8,
        "desired_speed": 0.66,      # m/s (~2.4 km/h in dense queue)
        "time_headway": 1.20,       # seconds
        "max_accel": 0.70,          # m/s^2
        "standstill_distance": 0.05,# meters
    },
    "medium": {
        "description": "Passenger cars, auto-rickshaws, small vans",
        "area_min_px": 300.0,
        "area_max_px": 600.0,
        "pcu": 1.0,
        "default_width_m": 1.8,
        "default_length_m": 4.2,
        "desired_speed": 0.59,      # m/s (~2.1 km/h)
        "time_headway": 0.92,       # seconds
        "max_accel": 0.65,          # m/s^2
        "standstill_distance": 0.05,# meters
    },
    "large": {
        "description": "Buses, commercial trucks, heavy utility vehicles",
        "area_min_px": 600.0,
        "pcu": 2.5,
        "default_width_m": 2.5,
        "default_length_m": 8.5,
        "desired_speed": 0.88,      # m/s (~3.2 km/h)
        "time_headway": 1.05,       # seconds
        "max_accel": 0.64,          # m/s^2
        "standstill_distance": 0.05,# meters
    }
}

# Signal Control Bounds
DEFAULT_MIN_GREEN_S = 15.0
DEFAULT_MAX_GREEN_S = 45.0
DEFAULT_CLEARANCE_TIMER_S = 6.0  # Fallback IRC clearance interval if box clearance is disabled
DYNAMIC_CLEARANCE_BUFFER_S = 1.0 # Minimum safety buffer once box polygon is vacant

# Default Approach Configuration
ENTRY_ROADS = [2, 4, 6, 8]
EXIT_ROADS = [1, 3, 5, 7]

# Kinematic and Force Engine Configuration
STOP_BUFFER_METERS = 2.0
REACTION_TIME = 0.005

FORCES_ENABLED = {
    "drive": True,
    "repulsive": True,
    "obstacle": True,
    "stop": False,
}

VEHICLE_PARAMS = {
    'small': {
        'desired_speed': 15.0,
        'reaction_time': REACTION_TIME,
        'repulsion_strength': 5619.5,
        'repulsion_range': 0.57,
    },
    'medium': {
        'desired_speed': 15.0,
        'reaction_time': REACTION_TIME,
        'repulsion_strength': 3806.4,
        'repulsion_range': 0.46,
    },
    'large': {
        'desired_speed': 15.0,
        'reaction_time': REACTION_TIME,
        'repulsion_strength': 3570.9,
        'repulsion_range': 0.14,
    },
}

# Display Dimensions
WIDTH = 1600
HEIGHT = 900


"""
Simulation Package: Discrete-event microscopic traffic simulator and road network geometry.
"""

from simulation.config import (
    FPS, DT, PIXELS_PER_METER, METERS_PER_PIXEL,
    VEHICLE_CLASSES, ENTRY_ROADS, EXIT_ROADS,
    DEFAULT_MIN_GREEN_S, DEFAULT_MAX_GREEN_S
)
from simulation.data_loader import TrajectoryDataLoader
from simulation.road_geometry import (
    is_point_in_polygon, calculate_road_direction, make_car,
    clear_forces, sim_step, start_transit, step_transit_cars
)

__all__ = [
    "FPS",
    "DT",
    "PIXELS_PER_METER",
    "METERS_PER_PIXEL",
    "VEHICLE_CLASSES",
    "ENTRY_ROADS",
    "EXIT_ROADS",
    "DEFAULT_MIN_GREEN_S",
    "DEFAULT_MAX_GREEN_S",
    "TrajectoryDataLoader",
    "is_point_in_polygon",
    "calculate_road_direction",
    "make_car",
    "clear_forces",
    "sim_step",
    "start_transit",
    "step_transit_cars",
]

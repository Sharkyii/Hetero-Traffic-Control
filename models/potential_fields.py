"""
2D Potential Fields and Lateral Guidance Forces.

Implements solenoidal driving forces pulling vehicles along lane guidelines
and exponential boundary wall repulsion preventing road departures.
"""

import math
from typing import List, Tuple
import numpy as np
from simulation.config import METERS_PER_PIXEL

# Physical Constants for Force Integration
AGENT_MASS = 1000.0                # Nominal vehicle mass (kg)
REACTION_TIME = 0.5                # Relaxation time constant tau (s)
CONTACT_STIFFNESS = 1.2e5          # Physical collision restitution stiffness
CONTACT_FRICTION = 2.4e5           # Sliding friction coefficient
WALL_REPULSION_STRENGTH = 1094.5   # Lateral wall repulsion amplitude
WALL_REPULSION_RANGE = 0.91        # Decay distance (meters)
INTERACTION_RANGE = 20.0           # Maximum potential field cutoff distance (meters)
EPS = 1e-6


def normalize(v: np.ndarray) -> np.ndarray:
    """Returns normalized unit vector or zero vector if magnitude < EPS."""
    norm = float(np.linalg.norm(v))
    return v / norm if norm > EPS else np.zeros(2)


def driving_force(car) -> np.ndarray:
    """
    Computes steering force pulling the vehicle heading toward target_dir.
    Velocity magnitude is decoupled from orientation to prevent rotational resonance.
    """
    tau = float(getattr(car, "reaction_time", REACTION_TIME))
    v0_ms = float(car.v_desired) * METERS_PER_PIXEL
    speed = float(np.linalg.norm(car.vel))
    drive_scale = float(getattr(car, "drive_force_scale", 1.0))
    target_dir = np.asarray(car.target_dir, dtype=float)
    return drive_scale * (v0_ms - speed) * target_dir / max(tau, 1e-3)


def repulsive_force(car1, car2) -> np.ndarray:
    """
    Computes smooth exponential social-force repulsion between two vehicles.
    """
    diff = car1.pos - car2.pos
    distance = float(np.linalg.norm(diff))

    if distance < EPS or distance > INTERACTION_RANGE:
        return np.zeros(2)

    direction = diff / distance
    combined_radius = float(car1.radius + car2.radius)
    overlap = combined_radius - distance

    strength = float(getattr(car1, "repulsion_strength", 1500.0))
    rng = float(getattr(car1, "repulsion_range", 1.5))
    repulsion_magnitude = strength * math.exp(overlap / max(rng, 1e-3))

    contact = 0.0
    if overlap > 0.0:
        contact = CONTACT_STIFFNESS * overlap
        rel_vel = car2.vel - car1.vel
        tangent = np.array([-direction[1], direction[0]])
        sliding_speed = float(np.dot(rel_vel, tangent))
        contact += CONTACT_FRICTION * overlap * sliding_speed

    raw_force = repulsion_magnitude + contact
    max_force = float(getattr(car1, "max_repulsion_force", 500.0))
    raw_force = min(raw_force, max_force)

    return (raw_force / AGENT_MASS) * direction


def obstacle_force(car, boundaries: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Computes repulsive force from static road curb boundaries.
    """
    total_accel = np.zeros(2)

    for wall_start, wall_end in boundaries:
        wall_vec = wall_end - wall_start
        wall_len_sq = float(np.dot(wall_vec, wall_vec))

        if wall_len_sq < EPS:
            closest_pt = wall_start
        else:
            t = np.clip(np.dot(car.pos - wall_start, wall_vec) / wall_len_sq, 0.0, 1.0)
            closest_pt = wall_start + t * wall_vec

        to_car = car.pos - closest_pt
        dist = float(np.linalg.norm(to_car))

        if dist < EPS:
            continue

        direction = to_car / dist
        overlap = car.radius - dist

        repulsion = WALL_REPULSION_STRENGTH * math.exp(overlap / WALL_REPULSION_RANGE)
        contact = 0.0
        if overlap > 0.0:
            contact = CONTACT_STIFFNESS * overlap
            wall_tangent = normalize(wall_vec)
            v_along = float(np.dot(car.vel, wall_tangent))
            contact += CONTACT_FRICTION * overlap * v_along

        force_accel = ((repulsion + contact) / AGENT_MASS) * direction
        total_accel += force_accel

    return total_accel


def compute_all_forces(car, all_cars: list, boundaries: list, forces_enabled: dict,
                       red_light: bool = False, car_proj: float = 0.0, stop_line: np.ndarray = None,
                       stop_line_proj: float = 0.0, road_direction: np.ndarray = None,
                       current_v_desired: float = None):
    """
    Integrates driving, vehicle-to-vehicle repulsion, and boundary obstacle forces.
    """
    car.F_drive = np.zeros(2)
    car.F_repulsive = np.zeros(2)
    car.F_obstacle = np.zeros(2)
    car.F_stop = np.zeros(2)

    if forces_enabled.get("drive", True):
        car.F_drive = driving_force(car)

    if forces_enabled.get("repulsive", True):
        for other in all_cars:
            if other.id != car.id:
                car.F_repulsive += repulsive_force(car, other)

    if forces_enabled.get("obstacle", True):
        car.F_obstacle = obstacle_force(car, boundaries)

    car.F_total = car.F_drive + car.F_repulsive + car.F_obstacle + car.F_stop

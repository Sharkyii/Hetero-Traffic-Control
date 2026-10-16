import numpy as np
from settings import *
from constants import *


def normalize(v):
    """Return unit vector, or zero if too small"""
    n = np.linalg.norm(v)
    return v / n if n > EPS else np.zeros(2)


def contact_function(overlap):
    """Returns overlap if positive, zero otherwise"""
    return max(overlap, 0.0)


# ---------------------------------------------------------
# DRIVING FORCE - Cars try to reach their desired speed
# ---------------------------------------------------------
DRIVE_FORCE_SCALE = 1.0  # multiplier to strengthen driving force relative to repulsion

def driving_force(car):
    """
    Steering force: pulls the car's heading toward target_dir.
    v_desired converted to m/s to match car.vel units.
    IDM owns the speed magnitude — this only contributes direction to F_total.

    Uses only the *magnitude* of car.vel (speed), not its direction, when
    forming the error term -- deliberately, not an oversight. sim_step's
    rotation re-derives car.vel's direction from car.theta every tick
    (car.vel = spd*[cos(theta), sin(theta)]) so velocity stays consistent
    with the car's actual facing. Feeding that same theta-coupled vel back
    into driving_force's direction closes a feedback loop: theta sets vel's
    angle, vel's angle swings F_drive's angle, F_drive's angle drives the
    next theta rotation. Confirmed by tracing car 657 (od4/road_4): once
    theta drifted off target_dir by even a little, subtracting a
    comparably-sized but differently-angled vel from v0*target_dir swung
    F_drive's angle by 50-100+ degrees tick to tick, and the fixed 20°/tick
    turn-rate cap overshot back and forth forever -- a genuine limit cycle,
    reproduced identically across a REACTION_TIME sweep from 0.005 to 60
    (a 10,000x gain range) since it's a direction-coupling bug, not a gain
    problem. Using speed (a scalar, no direction) instead of the raw vel
    vector removes the coupling: F_drive's angle is now always target_dir's
    own angle (or its reverse, if wildly overspeed), so steering can't
    resonate off the rotation it just performed.
    """
    tau = getattr(car, 'reaction_time', REACTION_TIME)
    v0_ms = car.v_desired * METERS_PER_PIXEL  # convert to m/s
    speed = np.linalg.norm(car.vel)
    return DRIVE_FORCE_SCALE * (v0_ms - speed) * car.target_dir / tau


# ---------------------------------------------------------
# REPULSIVE FORCE - Cars avoid each other
# ---------------------------------------------------------
def repulsive_force(car1, car2):
    """Repulsion acceleration (m/s²) between two cars.

    Per-pair force is capped at MAX_REPULSION_FORCE to prevent the
    ~7.7 M N spikes that arise when real-video cars spawn in bbox-
    overlapping positions. Without the cap those spikes exceed IDM's
    0.8 m/s² drive acceleration by six orders of magnitude, permanently
    freezing the front queue.
    """
    diff = car1.pos - car2.pos
    distance = np.linalg.norm(diff)

    if distance < EPS or distance > INTERACTION_RANGE:
        return np.zeros(2)

    direction = diff / distance
    combined_radius = car1.radius + car2.radius
    overlap = combined_radius - distance

    strength = getattr(car1, 'repulsion_strength', REPULSION_STRENGTH)
    rng      = getattr(car1, 'repulsion_range',    REPULSION_RANGE)
    repulsion_magnitude = strength * np.exp(overlap / rng)

    contact = 0
    if overlap > 0:
        contact = CONTACT_STIFFNESS * overlap
        relative_velocity = car2.vel - car1.vel
        tangent = np.array([-direction[1], direction[0]])
        sliding_speed = np.dot(relative_velocity, tangent)
        contact += CONTACT_FRICTION * overlap * sliding_speed

    raw_force = repulsion_magnitude + contact
    # Cap to prevent overlapping spawn pairs from dominating — real-video
    # bounding boxes often overlap at the moment of first detection.
    MAX_REPULSION_FORCE = getattr(car1, 'max_repulsion_force', 500.0)  # N
    raw_force = min(raw_force, MAX_REPULSION_FORCE)

    # divide by mass to get acceleration
    return (raw_force / AGENT_MASS) * direction


# ---------------------------------------------------------
# OBSTACLE FORCE - Cars avoid walls/boundaries
# ---------------------------------------------------------
def obstacle_force(car, boundaries):
    """
    Force to avoid hitting walls/boundaries only.
    Red-light stopping is handled by IDM (idm1.py).
    """
    total_force = np.zeros(2)

    for wall_start, wall_end in boundaries:
        wall_vector = wall_end - wall_start
        wall_length_sq = np.dot(wall_vector, wall_vector)

        if wall_length_sq < EPS:
            closest_point = wall_start
        else:
            t = np.clip(np.dot(car.pos - wall_start, wall_vector) / wall_length_sq, 0, 1)
            closest_point = wall_start + t * wall_vector

        to_car = car.pos - closest_point
        distance = np.linalg.norm(to_car)

        if distance < EPS:
            continue

        direction = to_car / distance
        overlap = car.radius - distance

        repulsion = WALL_REPULSION_STRENGTH * np.exp(overlap / WALL_REPULSION_RANGE)

        contact = 0
        if overlap > 0:
            contact = CONTACT_STIFFNESS * overlap
            wall_tangent = normalize(wall_vector)
            velocity_along_wall = np.dot(car.vel, wall_tangent)
            contact += CONTACT_FRICTION * overlap * velocity_along_wall

        # divide by mass to get acceleration
        force = ((repulsion + contact) / AGENT_MASS) * direction
        total_force += force

    return total_force


# ---------------------------------------------------------
# COMPUTE ALL FORCES
# ---------------------------------------------------------
def compute_all_forces(car, all_cars, boundaries, forces_enabled,
                       red_light=False, car_proj=0, stop_line=None,
                       stop_line_proj=0, road_direction=None, current_v_desired=None):
    """
    Calculate all forces acting on one car.
    """
    car.F_drive = np.zeros(2)
    car.F_repulsive = np.zeros(2)
    car.F_obstacle = np.zeros(2)
    car.F_stop = np.zeros(2)
    car.F_total = np.zeros(2)

    if road_direction is None:
        road_direction = np.array([1.0, 0.0])

    # Driving force
    if forces_enabled.get("drive", True):
        car.F_drive = driving_force(car)

    # Repulsive force from other cars
    if forces_enabled.get("repulsive", True):
        for other_car in all_cars:
            if other_car.id != car.id:
                car.F_repulsive += repulsive_force(car, other_car)

    # Obstacle force from walls/boundaries
    if forces_enabled.get("obstacle", True):
        car.F_obstacle = obstacle_force(car, boundaries)

    car.F_total = car.F_drive + car.F_repulsive + car.F_obstacle + car.F_stop

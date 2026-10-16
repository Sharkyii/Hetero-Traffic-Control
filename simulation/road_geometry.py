"""
simulation/road_geometry.py — Intersection Road Geometry, Turning Trajectories, and Solenoidal Physics.

Theoretical & Algorithmic Formulation:
  1. Ray-Casting Polygon Containment:
     Evaluates interior approach corridor and junction conflict zone occupancy.
  2. Cubic Bezier Turning Attractor Trajectories:
     Constructs C^1 smooth curvature curves between entry and destination stop lines.
  3. Solenoidal Yielding & Dynamic Priority Resolution:
     Resolves right-of-way between conflicting paths using entry timestamp ordering
     and cached priority latches to prevent mid-transit deadlocks.
  4. Non-Penetration Collision Mechanics:
     Implements swept-circle collision clipping and iterative positional impulse separation.
"""

import math
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from types import SimpleNamespace

from simulation.config import (
    METERS_PER_PIXEL, PIXELS_PER_METER, DT, FPS,
    VEHICLE_PARAMS, STOP_BUFFER_METERS,
)
from models.idm import (
    compute_idm_acceleration, find_crossing_threat, _idm_acceleration,
    IDM_A, IDM_B, IDM_T, IDM_S0,
)
from models.potential_fields import compute_all_forces

AGENT_MASS = 1.0

# Steering / Path Following Parameters
LOOKAHEAD_DISTANCE_M = 3.5    # metres ahead on Bezier to steer toward
CURVE_RESTORING_GAIN = 40.0   # elastic pull (N/m) back toward curve centerline
BASE_GUIDANCE_MAG = 150.0     # forward guidance force magnitude (N)

# Solenoidal Yield Parameters
YIELD_RADIUS = 3.5            # metres - crossing-conflict detection radius
MIN_TRANSIT_SPEED = 0.3       # m/s - ribbon crawl floor
DEFLECT_GAIN = 0.45           # lateral steering nudge magnitude


def is_point_in_polygon(point: Tuple[float, float], polygon_points: List[Tuple[float, float]]) -> bool:
    """Ray-casting algorithm determining whether a 2D point lies within a polygon."""
    x, y = point
    n = len(polygon_points)
    inside = False
    if n < 3:
        return False
    p1x, p1y = polygon_points[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon_points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def calculate_road_direction(polygon_points: List[Tuple[float, float]], red_line: Optional[List[Tuple[float, float]]] = None) -> np.ndarray:
    """Computes normalized road centerline direction vector pointing toward the stop line."""
    if red_line and len(red_line) == 2:
        pts = np.array(polygon_points, dtype=float)
        centroid = pts.mean(axis=0)
        rl_mid = np.mean(np.array(red_line, dtype=float), axis=0)
        d = rl_mid - centroid
        n = float(np.linalg.norm(d))
        if n > 1e-6:
            return d / n
    if len(polygon_points) != 4:
        return np.array([1.0, 0.0])
    pts = np.array(polygon_points)
    e12 = pts[2] - pts[1]
    e30 = pts[0] - pts[3]
    v1 = e12 / (np.linalg.norm(e12) + 1e-6)
    v2 = -e30 / (np.linalg.norm(e30) + 1e-6)
    d = (v1 + v2) / 2.0
    return d / (np.linalg.norm(d) + 1e-6)


def _segment_intersect(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
    """Tests 2D line segment intersection between [p1, p2] and [p3, p4]."""
    d1 = p2 - p1
    d2 = p4 - p3
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-10:
        return False, None
    diff = p3 - p1
    t = (diff[0] * d2[1] - diff[1] * d2[0]) / cross
    u = (diff[0] * d1[1] - diff[1] * d1[0]) / cross
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return True, p1 + t * d1
    return False, None


def make_car(tid: int, sx: float, sy: float, iv: Tuple[float, float], v_desired: float,
             length_px: float, width_px: float, offset_x: float, offset_y: float,
             road_direction: np.ndarray, vp: Optional[dict] = None) -> Any:
    """Instantiates a vehicle agent with physical and kinematic properties."""
    if vp is None:
        vp = {}
    lm = length_px * METERS_PER_PIXEL
    wm = width_px * METERS_PER_PIXEL
    vel = np.array(iv, dtype=float)

    car = SimpleNamespace(
        track_id=tid,
        id=tid,
        pos=np.array([(sx + offset_x) * METERS_PER_PIXEL, (sy + offset_y) * METERS_PER_PIXEL]),
        vel=vel,
        target_dir=np.array(road_direction, dtype=float),
        theta=math.atan2(road_direction[1], road_direction[0]),
        length_px=float(length_px),
        width_px=float(width_px),
        length_m=lm,
        width_m=wm,
        radius=0.5 * lm,
        v_desired=float(v_desired),
        active=True,
        state='entry',
        current_road=2,
        dest_road=None,
        in_transit=False,
        stopped_frames=0,
        transit_entry_frame=0,
        _priority_vs={},
        custom_params={},
        category=vp.get("category", "medium"),
        pcu=vp.get("pcu", 1.0),
        repulsion_strength=vp.get("repulsion_strength", 1500.0),
        repulsion_range=vp.get("repulsion_range", 1.5),
        max_repulsion_force=vp.get("max_repulsion_force", 500.0),
        mass=AGENT_MASS,
        F_drive=np.zeros(2),
        F_repulsive=np.zeros(2),
        F_obstacle=np.zeros(2),
        F_stop=np.zeros(2),
        F_total=np.zeros(2),
        F_steer=np.zeros(2),
    )
    return car


def clear_forces(car):
    """Zeros all accumulative force vectors on the vehicle."""
    car.F_drive = np.zeros(2)
    car.F_repulsive = np.zeros(2)
    car.F_obstacle = np.zeros(2)
    car.F_stop = np.zeros(2)
    car.F_total = np.zeros(2)
    car.F_steer = np.zeros(2)


def _clamp_to_boundary(car, boundaries: List[Any]):
    """Restricts vehicle positions to remain within road boundaries."""
    bm = 0.5 * METERS_PER_PIXEL
    for ws, we in boundaries:
        wv = we - ws
        wls = float(np.dot(wv, wv))
        if wls < 1e-12:
            continue
        t = float(np.clip(np.dot(car.pos - ws, wv) / wls, 0.0, 1.0))
        closest = ws + t * wv
        to_car = car.pos - closest
        dist = float(np.linalg.norm(to_car))
        min_dist = car.radius + bm
        push_dir = (to_car / dist) if dist > 1e-9 else np.array([-wv[1], wv[0]]) / (np.linalg.norm(wv) + 1e-12)
        if dist < min_dist * 2.0:
            f_in = float(np.dot(car.F_total, -push_dir))
            if f_in > 0:
                car.F_total += f_in * push_dir
        if dist < min_dist:
            car.pos = closest + push_dir * min_dist
            v_in = float(np.dot(car.vel, -push_dir))
            if v_in > 0:
                car.vel += v_in * push_dir


def _strip_collision_forces(cars: list):
    """Zeros repulsive forces for overlapping vehicles to prevent explosive displacement."""
    for i, a in enumerate(cars):
        if not a.active or getattr(a, "stopped_frames", 0) > 0:
            continue
        for b in cars[i + 1:]:
            if not b.active or getattr(b, "stopped_frames", 0) > 0:
                continue
            diff = a.pos - b.pos
            dist = float(np.linalg.norm(diff))
            min_dist = a.radius + b.radius
            if dist < 1e-9 or dist > min_dist * 2.0:
                continue
            push_a = diff / dist
            f = float(np.dot(a.F_total, -push_a))
            if f > 0:
                a.F_total += f * push_a
            f = float(np.dot(b.F_total, push_a))
            if f > 0:
                b.F_total -= f * push_a


def _resolve_collision_positions(cars: list):
    """Applies iterative positional impulse separation to enforce physical non-penetration."""
    for i, a in enumerate(cars):
        if not a.active or getattr(a, "stopped_frames", 0) > 0:
            continue
        for b in cars[i + 1:]:
            if not b.active or getattr(b, "stopped_frames", 0) > 0:
                continue
            diff = a.pos - b.pos
            dist = float(np.linalg.norm(diff))
            min_dist = a.radius + b.radius
            if dist >= min_dist:
                continue
            push_a = diff / dist if dist > 1e-9 else np.array([1.0, 0.0])
            overlap = min_dist - dist
            a.pos += push_a * (overlap * 0.5)
            b.pos -= push_a * (overlap * 0.5)

            # Damp forward speed along heading rather than adding sideways velocity kicks
            va = float(np.dot(a.vel, -push_a))
            if va > 0:
                spd_a = max(0.0, float(np.linalg.norm(a.vel)) - va)
                a.vel = np.array([spd_a * math.cos(a.theta), spd_a * math.sin(a.theta)])
            vb = float(np.dot(b.vel, push_a))
            if vb > 0:
                spd_b = max(0.0, float(np.linalg.norm(b.vel)) - vb)
                b.vel = np.array([spd_b * math.cos(b.theta), spd_b * math.sin(b.theta)])


def _clip_movement_for_collision(pos0: np.ndarray, delta: np.ndarray, radius: float, others: list) -> float:
    """Swept-circle collision clipping so vehicle update stops short of another car."""
    t_max = 1.0
    a = float(np.dot(delta, delta))
    if a < 1e-12:
        return 1.0
    for other_pos, other_radius in others:
        R = radius + other_radius
        f = pos0 - other_pos
        c = float(np.dot(f, f)) - R * R
        if c <= 0:
            continue
        b = 2.0 * float(np.dot(f, delta))
        disc = b * b - 4 * a * c
        if disc < 0:
            continue
        t_hit = (-b - np.sqrt(disc)) / (2 * a)
        if 0.0 <= t_hit < t_max:
            t_max = t_hit
    return max(t_max, 0.0)


def sim_step(car, dt: float = 0.25):
    """Advances vehicle kinematic state by dt seconds."""
    spd = float(np.linalg.norm(car.vel))
    if spd > 1e-4:
        f_steer = getattr(car, "F_steer", car.F_total)
        f_mag = float(np.linalg.norm(f_steer))
        desired_angle = math.atan2(f_steer[1], f_steer[0]) if f_mag > 1e-3 else math.atan2(car.target_dir[1], car.target_dir[0])
        w = math.atan2(math.sin(desired_angle - car.theta), math.cos(desired_angle - car.theta))
        w = float(np.clip(w, -math.pi / 9, math.pi / 9))
        car.theta += w

    delta = np.array([spd * math.cos(car.theta), spd * math.sin(car.theta)]) * dt
    car.pos += delta


def _get_ordered_stopline(rc: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Return (stop_left, stop_right) consistently ordered left-to-right."""
    if "stop_left" in rc and "stop_right" in rc:
        return rc["stop_left"], rc["stop_right"]
    d = rc["direction"]
    v_right = np.array([-d[1], d[0]])
    pA, pB = rc["stop_p1"], rc["stop_p2"]
    if np.dot(pB - pA, v_right) > 0:
        return pA, pB
    return pB, pA


def start_transit(car, destination_road: int, entrance_point_m: np.ndarray,
                  cfg: dict, s_override: Optional[float] = None, sim_frame: int = 0):
    """
    Called once when a car clears its origin stop line on green.
    Constructs a smooth 40-point cubic Bezier attractor curve across the junction.
    """
    p0 = car.pos.copy()
    entry_dir = cfg[car.current_road]["direction"]
    exit_dir = cfg[destination_road]["direction"]

    if s_override is not None:
        s = float(np.clip(s_override, 0.05, 0.95))
    else:
        e_left, e_right = _get_ordered_stopline(cfg[car.current_road])
        entry_vec = e_right - e_left
        entry_w2 = float(np.dot(entry_vec, entry_vec))
        s = (float(np.clip(np.dot(p0 - e_left, entry_vec) / entry_w2, 0.05, 0.95))
             if entry_w2 > 1e-4 else 0.5)

    d_left, d_right = _get_ordered_stopline(cfg[destination_road])
    target_entrance = d_left + s * (d_right - d_left)

    p3 = target_entrance + exit_dir * 5.0
    dist = float(np.linalg.norm(p3 - p0))
    d = dist * 0.5
    p1 = p0 + entry_dir * d
    p2 = p3 - exit_dir * d

    path = []
    for t in np.linspace(0.0, 1.0, 40):
        pt = ((1.0 - t)**3 * p0 +
              3.0 * (1.0 - t)**2 * t * p1 +
              3.0 * (1.0 - t) * t**2 * p2 +
              t**3 * p3)
        path.append(pt)

    car.transit_path = np.array(path)
    car.transit_path_idx = 0
    car.transit_s = s

    target = car.transit_path[min(1, len(car.transit_path) - 1)] - car.pos
    n = float(np.linalg.norm(target))
    car.target_dir = (target / n) if n > 1e-6 else entry_dir.copy()

    car.transit_entry_frame = sim_frame
    car._priority_vs = {}

    car.state = 'transit'
    car.in_transit = True
    car.origin_road = car.current_road
    car.dest_road = destination_road
    car.current_road = None


def has_priority(car, other) -> bool:
    """
    Evaluates right-of-way between two vehicles traversing the conflict area.
    Earlier entry into transit holds priority. Priority is cached to prevent mid-transit flipping.
    """
    if not hasattr(car, "_priority_vs") or car._priority_vs is None:
        car._priority_vs = {}
    if id(other) in car._priority_vs:
        return car._priority_vs[id(other)]

    frame_car = getattr(car, "transit_entry_frame", 0)
    frame_other = getattr(other, "transit_entry_frame", 0)

    if frame_car != frame_other:
        result = frame_car < frame_other
    else:
        prog_car = getattr(car, "transit_path_idx", 0) / max(1, len(getattr(car, "transit_path", [1])))
        prog_other = getattr(other, "transit_path_idx", 0) / max(1, len(getattr(other, "transit_path", [1])))
        if abs(prog_car - prog_other) > 0.05:
            result = prog_car > prog_other
        elif hasattr(car, 'transit_path') and hasattr(other, 'transit_path'):
            dist_car = float(np.linalg.norm(car.transit_path[-1] - car.pos))
            dist_other = float(np.linalg.norm(other.transit_path[-1] - other.pos))
            if abs(dist_car - dist_other) > 0.5:
                result = dist_car < dist_other
            else:
                result = getattr(car, 'track_id', id(car)) < getattr(other, 'track_id', id(other))
        else:
            result = getattr(car, "track_id", id(car)) < getattr(other, "track_id", id(other))

    car._priority_vs[id(other)] = result
    return result


def step_transit(transit_cars: list, middle_boundaries: list, forces_enabled: dict,
                 dt: Optional[float] = None, all_cars: Optional[list] = None,
                 ghost_leaders: Optional[list] = None):
    """One physics tick for all cars currently crossing the intersection."""
    if dt is None:
        dt = DT
    boundaries = middle_boundaries or []

    for car in transit_cars:
        if not hasattr(car, 'transit_path') or len(car.transit_path) < 2:
            continue

        # 1. Steering: Bezier lookahead + cross-track correction
        dists = np.linalg.norm(car.transit_path - car.pos, axis=1)
        closest_idx = int(np.argmin(dists))
        car.transit_path_idx = closest_idx
        closest_pt = car.transit_path[closest_idx]
        cross_track_vec = closest_pt - car.pos

        target_idx = closest_idx
        cum_dist = 0.0
        for i in range(closest_idx, len(car.transit_path) - 1):
            seg_len = float(np.linalg.norm(car.transit_path[i + 1] - car.transit_path[i]))
            cum_dist += seg_len
            target_idx = i + 1
            if cum_dist >= LOOKAHEAD_DISTANCE_M:
                break

        target_pt = car.transit_path[target_idx]
        tangent = target_pt - car.pos
        t_mag = float(np.linalg.norm(tangent))
        tangent_unit = (tangent / t_mag) if t_mag > 1e-6 else car.target_dir.copy()

        desired_heading = tangent_unit + 0.25 * cross_track_vec
        h_mag = float(np.linalg.norm(desired_heading))
        if h_mag > 1e-6:
            car.target_dir = desired_heading / h_mag

        # 2. Longitudinal Speed Regulation
        v = float(np.linalg.norm(car.vel))
        v0 = float(getattr(car, 'v_desired', 15)) * METERS_PER_PIXEL

        free_road = (v / (v0 + 1e-9)) ** 4
        a_idm = IDM_A * (1.0 - free_road)
        new_speed = float(np.clip(v + a_idm * dt, 0.0, v0))

        yield_ceiling = v0
        all_peers = list(transit_cars)
        if ghost_leaders:
            all_peers.extend(ghost_leaders)

        for other in all_peers:
            if other is car:
                continue
            to_other = other.pos - car.pos
            dist = float(np.linalg.norm(to_other))
            if dist < 1e-6:
                continue

            same_dest = (getattr(other, 'dest_road', None) == getattr(car, 'dest_road', None))

            # Case A: Same destination -> ribbon IDM car-following
            if same_dest:
                forward_proj = float(np.dot(to_other, car.target_dir))
                if forward_proj > 0:
                    lat_dist = float(np.linalg.norm(to_other - forward_proj * car.target_dir))
                    s_diff = abs(getattr(car, 'transit_s', 0.5) - getattr(other, 'transit_s', 0.5))
                    if lat_dist < 1.6 and s_diff < 0.25:
                        g = dist - (car.radius + other.radius)
                        closing = v - float(np.linalg.norm(other.vel))
                        a_follow = _idm_acceleration(v, v0, g, closing)
                        new_speed = float(np.clip(v + a_follow * dt, 0.0, v0))
                        if g > 0.8:
                            new_speed = max(new_speed, min(1.2, v0))
                        elif g > 0.4:
                            crawl = min(1.2, v0) * ((g - 0.4) / 0.4)
                            new_speed = max(new_speed, max(crawl, 0.4))
                        elif g > 0.0:
                            new_speed = max(new_speed, min(0.4, v0))
                        else:
                            new_speed = max(new_speed, MIN_TRANSIT_SPEED)

            # Case B: Different destination -> Solenoidal Yield
            else:
                if dist < YIELD_RADIUS and not has_priority(car, other):
                    g_cross = dist - (car.radius + other.radius)
                    proximity_factor = max(0.0, g_cross / YIELD_RADIUS)
                    v_yield = max(v0 * proximity_factor, 0.0)
                    yield_ceiling = min(yield_ceiling, v_yield)
                    new_speed = min(new_speed, v_yield)

                    repulse_dir = car.pos - other.pos
                    r_norm = float(np.linalg.norm(repulse_dir))
                    if r_norm > 1e-6:
                        repulse_dir /= r_norm
                        perp = np.array([-car.target_dir[1], car.target_dir[0]])
                        deflect_sign = float(np.dot(repulse_dir, perp))
                        cross_track_dist = float(np.linalg.norm(cross_track_vec))
                        effective_gain = DEFLECT_GAIN * max(0.2, 1.0 - cross_track_dist / 2.0)
                        nudge = deflect_sign * effective_gain * perp
                        new_dir = car.target_dir + nudge
                        n_mag = float(np.linalg.norm(new_dir))
                        if n_mag > 1e-6:
                            cand_dir = new_dir / n_mag
                            angle_from_tangent = math.atan2(cand_dir[1] * tangent_unit[0] - cand_dir[0] * tangent_unit[1],
                                                            cand_dir[0] * tangent_unit[0] + cand_dir[1] * tangent_unit[1])
                            if abs(angle_from_tangent) > math.radians(30):
                                base_ang = math.atan2(tangent_unit[1], tangent_unit[0])
                                clamped_ang = base_ang + np.clip(angle_from_tangent, -math.radians(30), math.radians(30))
                                cand_dir = np.array([math.cos(clamped_ang), math.sin(clamped_ang)])
                            car.target_dir = cand_dir

        new_speed = min(new_speed, yield_ceiling)
        heading_angle = car.theta if v > 1e-4 else math.atan2(car.target_dir[1], car.target_dir[0])
        car.vel = np.array([new_speed * math.cos(heading_angle), new_speed * math.sin(heading_angle)])

        # 3. Forces: Social repulsion + elastic curve attraction + forward drive
        orig_rng = getattr(car, 'repulsion_range', 0.3)
        car.repulsion_range = max(orig_rng, 1.2)

        clear_forces(car)
        compute_all_forces(car, [c for c in transit_cars if c is not car],
                           boundaries, forces_enabled,
                           red_light=False, road_direction=car.target_dir)

        car.repulsion_range = orig_rng

        F_restore = cross_track_vec * CURVE_RESTORING_GAIN
        F_forward = car.target_dir * BASE_GUIDANCE_MAG
        car.F_total += F_restore + F_forward
        car.F_steer = car.F_total.copy()

    # 4. Strip collision forces & integrate
    _strip_collision_forces(transit_cars)

    move_snapshot = [(c, c.pos.copy(), c.radius) for c in transit_cars]
    for car in transit_cars:
        others = [(pos, r) for c2, pos, r in move_snapshot if c2 is not car]
        spd = float(np.linalg.norm(car.vel))
        if spd > 1e-4:
            desired_angle = math.atan2(car.target_dir[1], car.target_dir[0])
            err = math.atan2(math.sin(desired_angle - car.theta), math.cos(desired_angle - car.theta))
            w = float(np.clip(err * 0.25, -math.radians(5.0), math.radians(5.0)))
            car.theta += w

        delta = np.array([spd * math.cos(car.theta), spd * math.sin(car.theta)]) * dt
        t = _clip_movement_for_collision(car.pos, delta, car.radius, others)
        car.pos += delta * t
        car.vel = np.array([spd * math.cos(car.theta), spd * math.sin(car.theta)])
        if boundaries:
            _clamp_to_boundary(car, boundaries)

    _resolve_collision_positions(transit_cars)


def check_arrival(car, dest_polygon_m: List[Tuple[float, float]], offset: Optional[Tuple[float, float]] = None) -> bool:
    """Verifies whether transiting vehicle has entered destination exit polygon in metre-space."""
    return is_point_in_polygon(tuple(car.pos), dest_polygon_m)
step_transit_cars = step_transit

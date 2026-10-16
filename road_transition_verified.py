"""
road_transition_verified.py — Bug-Fixed Verified Solenoidal Junction Physics Engine

Fixes Applied:
  - Bug 3: Remove MIN_TRANSIT_SPEED floor in Case B yield so cars can come to a full stop if needed.
  - Bug 4 & Bug 14: Store `transit_entry_frame = sim_frame` in start_transit; replace has_priority
    with entry-frame comparison and per-pair latch cache so priority never flips mid-transit.
  - Bug 5: Scale DEFLECT_GAIN inversely with cross-track distance and cap target_dir deviation
    from Bezier tangent to 30 degrees to prevent lateral deflection oscillations.
  - Bug 6: Support ghost_leaders in Case A ribbon following so trailing cars don't ram newly arrived exit cars.
  - Bug 10: Track yield_ceiling across all Case B peers and apply it after peer loop so Case A ribbon floors
    cannot cancel Case B cross-traffic yield braking.
  - Bug 12: Check arrival directly against metre-space destination polygon (dest_polygon_m) avoiding
    pixel-conversion offset mismatches.
"""
import math
import numpy as np

from constants import PIXELS_PER_METER, METERS_PER_PIXEL, AGENT_MASS
from evaluate import is_point_in_polygon, clear_forces, _clamp_to_boundary, \
    _strip_collision_forces, _resolve_collision_positions, sim_step, _clip_movement_for_collision
from Forces import compute_all_forces
from Forces.idm1_verified import find_crossing_threat, _idm_acceleration, IDM_A, IDM_B, IDM_T, IDM_S0
from settings import FPS_SIM, MAX_SPEED

# --- Steering / path following ---
LOOKAHEAD_DISTANCE_M  = 3.5    # metres ahead on the Bezier to steer toward
CURVE_RESTORING_GAIN  = 40.0   # elastic pull (N/m) back toward curve centreline
BASE_GUIDANCE_MAG     = 150.0  # forward guidance force magnitude (N)

# --- Solenoidal yield parameters ---
YIELD_RADIUS      = 3.5   # m — crossing-conflict detection radius
MIN_TRANSIT_SPEED = 0.3   # m/s — ribbon crawl floor only
DEFLECT_GAIN      = 0.45  # lateral steering nudge magnitude (fraction of perp unit vec)


def _get_ordered_stopline(rc):
    """Return (stop_left, stop_right) consistently ordered left-to-right."""
    if "stop_left" in rc and "stop_right" in rc:
        return rc["stop_left"], rc["stop_right"]
    d = rc["direction"]
    v_right = np.array([-d[1], d[0]])
    pA, pB = rc["stop_p1"], rc["stop_p2"]
    if np.dot(pB - pA, v_right) > 0:
        return pA, pB
    return pB, pA


def start_transit(car, destination_road, entrance_point_m, cfg, s_override=None, sim_frame=0):
    """Called once when a car clears its origin stop line on green.
    Constructs a smooth Bezier attractor curve from the car's current position
    to the destination road's entrance stop line.

    Bug 14 Fix:
    Stores car.transit_entry_frame = sim_frame and car._priority_vs = {} fresh for each transit.
    """
    p0 = car.pos.copy()
    entry_dir = cfg[car.current_road]["direction"]
    exit_dir  = cfg[destination_road]["direction"]

    # Lateral fraction s along the destination road's stop line
    if s_override is not None:
        s = float(np.clip(s_override, 0.05, 0.95))
    else:
        e_left, e_right = _get_ordered_stopline(cfg[car.current_road])
        entry_vec = e_right - e_left
        entry_w2  = float(np.dot(entry_vec, entry_vec))
        s = (float(np.clip(np.dot(p0 - e_left, entry_vec) / entry_w2, 0.05, 0.95))
             if entry_w2 > 1e-4 else 0.5)

    d_left, d_right  = _get_ordered_stopline(cfg[destination_road])
    target_entrance  = d_left + s * (d_right - d_left)

    # P3: 5 m into the exit road so the car arrives cleanly aligned
    p3   = target_entrance + exit_dir * 5.0
    dist = float(np.linalg.norm(p3 - p0))
    d    = dist * 0.5
    p1   = p0 + entry_dir * d   # control point near entry
    p2   = p3 - exit_dir  * d   # control point near exit

    # 40-point cubic Bezier
    path = []
    for t in np.linspace(0.0, 1.0, 40):
        pt = ((1.0 - t)**3 * p0 +
              3.0 * (1.0 - t)**2 * t * p1 +
              3.0 * (1.0 - t) * t**2 * p2 +
              t**3 * p3)
        path.append(pt)

    car.transit_path     = np.array(path)
    car.transit_path_idx = 0
    car.transit_s        = s        # stored for same-ribbon detection

    target = car.transit_path[min(1, len(car.transit_path) - 1)] - car.pos
    n = float(np.linalg.norm(target))
    car.target_dir = (target / n) if n > 1e-6 else entry_dir.copy()

    # Bug 14 Fix: Record transit entry frame and reset priority latch cache
    car.transit_entry_frame = sim_frame
    car._priority_vs = {}

    car.state        = 'transit'
    car.origin_road  = car.current_road
    car.dest_road    = destination_road
    car.current_road = None


def has_priority(car, other):
    """Right-of-way: True if `car` has priority over `other`.

    Bug 4 Fix:
    1. Latch Cache: Once computed for a pair, priority never flips mid-transit.
    2. Primary Rule: Earlier entry into transit has priority (transit_entry_frame).
    3. Same-Frame Tie-Breaker: Normalized path progress, then track_id.
    """
    cache = getattr(car, '_priority_vs', None)
    if cache is not None and id(other) in cache:
        return cache[id(other)]

    frame_car   = getattr(car,   'transit_entry_frame', 0)
    frame_other = getattr(other, 'transit_entry_frame', 0)

    if frame_car != frame_other:
        result = frame_car < frame_other
    else:
        # Same frame tie-breaker: path-progress ratio
        prog_car   = getattr(car,   'transit_path_idx', 0) / max(1, len(getattr(car,   'transit_path', [1])))
        prog_other = getattr(other, 'transit_path_idx', 0) / max(1, len(getattr(other, 'transit_path', [1])))

        if abs(prog_car - prog_other) > 0.05:
            result = prog_car > prog_other
        elif hasattr(car, 'transit_path') and hasattr(other, 'transit_path'):
            dist_car   = float(np.linalg.norm(car.transit_path[-1]   - car.pos))
            dist_other = float(np.linalg.norm(other.transit_path[-1] - other.pos))
            if abs(dist_car - dist_other) > 0.5:
                result = dist_car < dist_other
            else:
                result = getattr(car, 'track_id', id(car)) < getattr(other, 'track_id', id(other))
        else:
            result = getattr(car, 'track_id', id(car)) < getattr(other, 'track_id', id(other))

    if not hasattr(car, '_priority_vs'):
        car._priority_vs = {}
    car._priority_vs[id(other)] = result
    return result


def step_transit(transit_cars, middle_boundaries, forces_enabled, dt=None, all_cars=None, ghost_leaders=None):
    """One physics tick for all cars currently crossing the intersection.

    Fixes integrated:
      - Bug 3: Allows new_speed to reach 0.0 in Case B yielding (no forced 0.3 m/s floor).
      - Bug 5: Damped lateral deflection gain and 30-degree tangent deviation clamp.
      - Bug 6: Incorporates ghost_leaders for Case A ribbon following.
      - Bug 10: Enforces yield_ceiling across all Case B peers.
    """
    if dt is None:
        dt = 1.0 / FPS_SIM
    boundaries = middle_boundaries or []

    for car in transit_cars:
        if not hasattr(car, 'transit_path') or len(car.transit_path) < 2:
            continue

        # ------------------------------------------------------------------ #
        # 1. STEERING: Bezier lookahead + cross-track correction              #
        # ------------------------------------------------------------------ #
        dists       = np.linalg.norm(car.transit_path - car.pos, axis=1)
        closest_idx = int(np.argmin(dists))
        car.transit_path_idx = closest_idx
        closest_pt    = car.transit_path[closest_idx]
        cross_track_vec = closest_pt - car.pos   # points from car toward curve

        # Walk forward along the curve to find the lookahead point
        target_idx = closest_idx
        cum_dist   = 0.0
        for i in range(closest_idx, len(car.transit_path) - 1):
            seg_len = float(np.linalg.norm(car.transit_path[i + 1] - car.transit_path[i]))
            cum_dist += seg_len
            target_idx = i + 1
            if cum_dist >= LOOKAHEAD_DISTANCE_M:
                break

        target_pt    = car.transit_path[target_idx]
        tangent      = target_pt - car.pos
        t_mag        = float(np.linalg.norm(tangent))
        tangent_unit = (tangent / t_mag) if t_mag > 1e-6 else car.target_dir.copy()

        # Blend lookahead tangent with a gentle cross-track pull to stay on curve
        desired_heading = tangent_unit + 0.25 * cross_track_vec
        h_mag = float(np.linalg.norm(desired_heading))
        if h_mag > 1e-6:
            car.target_dir = desired_heading / h_mag

        # ------------------------------------------------------------------ #
        # 2. LONGITUDINAL SPEED REGULATION                                    #
        # ------------------------------------------------------------------ #
        v  = float(np.linalg.norm(car.vel))
        v0 = float(getattr(car, 'v_desired', 15)) * METERS_PER_PIXEL

        # Start from free-road acceleration
        free_road = (v / (v0 + 1e-9)) ** 4
        a_idm     = IDM_A * (1.0 - free_road)
        new_speed = float(np.clip(v + a_idm * dt, 0.0, v0))

        # Bug 10: yield_ceiling accumulates strictest Case B constraint
        yield_ceiling = v0

        # Bug 6: Combine transit peers with ghost_leaders
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

            # ---- Case A: SAME destination → ribbon IDM car-following ---- #
            if same_dest:
                forward_proj = float(np.dot(to_other, car.target_dir))
                if forward_proj > 0:
                    lat_dist = float(np.linalg.norm(to_other - forward_proj * car.target_dir))
                    s_diff   = abs(getattr(car, 'transit_s', 0.5) - getattr(other, 'transit_s', 0.5))
                    if lat_dist < 1.6 and s_diff < 0.25:   # same ribbon
                        g       = dist - (car.radius + other.radius)
                        closing = v - float(np.linalg.norm(other.vel))
                        a_follow = _idm_acceleration(v, v0, g, closing)
                        new_speed = float(np.clip(v + a_follow * dt, 0.0, v0))
                        # Ribbon speed floor: crawl don't freeze
                        if g > 0.8:
                            new_speed = max(new_speed, min(1.2, v0))
                        elif g > 0.4:
                            crawl = min(1.2, v0) * ((g - 0.4) / 0.4)
                            new_speed = max(new_speed, max(crawl, 0.4))
                        elif g > 0.0:
                            new_speed = max(new_speed, min(0.4, v0))
                        else:
                            new_speed = max(new_speed, MIN_TRANSIT_SPEED)

            # ---- Case B: DIFFERENT destination → SOLENOIDAL YIELD ---- #
            else:
                if dist < YIELD_RADIUS and not has_priority(car, other):
                    g_cross = dist - (car.radius + other.radius)

                    # 2a. Bug 3 Fix: soft speed reduction can decelerate to 0.0
                    proximity_factor = max(0.0, g_cross / YIELD_RADIUS)
                    v_yield   = max(v0 * proximity_factor, 0.0)
                    yield_ceiling = min(yield_ceiling, v_yield)  # Bug 10
                    new_speed = min(new_speed, v_yield)

                    # 2b. Bug 5 Fix: Damped lateral deflection capped to 30 degrees
                    repulse_dir = car.pos - other.pos
                    r_norm      = float(np.linalg.norm(repulse_dir))
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
                            # Cap angle from tangent_unit to 30 degrees
                            angle_from_tangent = math.atan2(cand_dir[1] * tangent_unit[0] - cand_dir[0] * tangent_unit[1],
                                                            cand_dir[0] * tangent_unit[0] + cand_dir[1] * tangent_unit[1])
                            if abs(angle_from_tangent) > math.radians(30):
                                base_ang = math.atan2(tangent_unit[1], tangent_unit[0])
                                clamped_ang = base_ang + np.clip(angle_from_tangent, -math.radians(30), math.radians(30))
                                cand_dir = np.array([math.cos(clamped_ang), math.sin(clamped_ang)])
                            car.target_dir = cand_dir

        # Bug 10 Fix: Apply yield_ceiling across all Case B peers
        new_speed = min(new_speed, yield_ceiling)

        heading_angle = (car.theta if v > 1e-4
                         else math.atan2(car.target_dir[1], car.target_dir[0]))
        car.vel = np.array([new_speed * math.cos(heading_angle),
                            new_speed * math.sin(heading_angle)])

        # ------------------------------------------------------------------ #
        # 3. FORCES: Social repulsion + elastic curve attraction + drive      #
        # ------------------------------------------------------------------ #
        orig_rng = getattr(car, 'repulsion_range', 0.3)
        car.repulsion_range = max(orig_rng, 1.2)   # wider bubble in junction

        clear_forces(car)
        compute_all_forces(car, [c for c in transit_cars if c is not car],
                           boundaries, forces_enabled,
                           red_light=False, road_direction=car.target_dir)

        car.repulsion_range = orig_rng

        # Elastic pull back toward curve centre + forward drive
        F_restore  = cross_track_vec * CURVE_RESTORING_GAIN
        F_forward  = car.target_dir * BASE_GUIDANCE_MAG
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
            err = math.atan2(math.sin(desired_angle - car.theta),
                             math.cos(desired_angle - car.theta))
            # Smooth steering along the Bézier curve, max 5 degrees per tick to prevent sharp rotations
            w = float(np.clip(err * 0.25, -math.radians(5.0), math.radians(5.0)))
            car.theta += w

        delta = np.array([spd * math.cos(car.theta), spd * math.sin(car.theta)]) * dt
        t = _clip_movement_for_collision(car.pos, delta, car.radius, others)
        car.pos += delta * t
        car.vel = np.array([spd * math.cos(car.theta), spd * math.sin(car.theta)])
        if boundaries:
            _clamp_to_boundary(car, boundaries)

    _resolve_collision_positions(transit_cars)


def check_arrival(car, dest_polygon_m, offset=None):
    """True once a transiting car's position has entered its destination road polygon.
    Bug 12 Fix: Accepts dest_polygon_m in metre space directly, eliminating pixel-conversion offset errors."""
    return is_point_in_polygon(tuple(car.pos), dest_polygon_m)

import numpy as np
import math
from constants import METERS_PER_PIXEL

# ── IDM parameters ────────────────────────────────────────────────────────────
IDM_S0    = 0.3    # minimum jam gap           (m)
IDM_T     = 0.6    # safe time headway         (s)
IDM_A     = 0.8    # max acceleration          (m/s²)
IDM_B     = 2.5    # comfortable deceleration  (m/s²)
IDM_DELTA = 4      # acceleration exponent
LATERAL_THRESHOLD = 1.5   # metres — same-lane filter (wider to catch real leaders)
CROSSING_HORIZON_S = 5.0  # ignore a crossing threat further out than this --
                          # IDM's own gap-based deceleration already tapers
                          # to ~0 for a distant one anyway, this just skips
                          # the bookkeeping for one that's not relevant yet
# ─────────────────────────────────────────────────────────────────────────────


def heading(car) -> np.ndarray:
    speed = np.linalg.norm(car.vel)
    if speed > 1e-6:
        return car.vel / speed
    return np.asarray(car.target_dir, dtype=float)


def front_proj(car, road_dir: np.ndarray) -> float:
    alpha = math.atan2(car.width_px, car.length_px)
    half_diag = math.sqrt(car.length_px ** 2 + car.width_px ** 2) / 2.0 * METERS_PER_PIXEL
    theta = math.atan2(heading(car)[1], heading(car)[0])
    pos = car.pos
    a = pos + half_diag * np.array([math.cos(theta + alpha), math.sin(theta + alpha)])
    b = pos + half_diag * np.array([math.cos(theta - alpha), math.sin(theta - alpha)])
    return max(float(np.dot(a, road_dir)), float(np.dot(b, road_dir)))


def back_proj(car, road_dir: np.ndarray) -> float:
    alpha = math.atan2(car.width_px, car.length_px)
    half_diag = math.sqrt(car.length_px ** 2 + car.width_px ** 2) / 2.0 * METERS_PER_PIXEL
    theta = math.atan2(heading(car)[1], heading(car)[0])
    pos = car.pos
    c = pos + half_diag * np.array([math.cos(theta + alpha + math.pi), math.sin(theta + alpha + math.pi)])
    d = pos + half_diag * np.array([math.cos(theta - alpha + math.pi), math.sin(theta - alpha + math.pi)])
    return min(float(np.dot(c, road_dir)), float(np.dot(d, road_dir)))


def gap(follower, leader, road_dir: np.ndarray) -> float:
    """Bumper-to-bumper gap between follower and leader along road_dir."""
    return back_proj(leader, road_dir) - front_proj(follower, road_dir)


def find_leader(car, active: list, road_dir: np.ndarray):
    """
    Return (leader, bumper-to-bumper gap, dv) for the closest car ahead
    in the same lane. Returns (None, inf, 0.0) if road is clear.
    """
    ego_proj = front_proj(car, road_dir)
    perp = np.array([-road_dir[1], road_dir[0]])
    best_gap = float("inf")
    best_car = None

    for other in active:
        if other is car:
            continue

        # same-lane filter
        lateral = abs(float(np.dot(other.pos - car.pos, perp)))
        if lateral > LATERAL_THRESHOLD:
            continue

        if front_proj(other, road_dir) <= ego_proj:
            continue

        g = gap(car, other, road_dir)
        if 0 < g < best_gap:
            best_gap = g
            best_car = other

    if best_car is None:
        return None, float("inf"), 0.0

    dv = float(np.linalg.norm(car.vel)) - float(np.linalg.norm(best_car.vel))
    return best_car, best_gap, dv


def find_crossing_threat(car, pool):
    """
    (gap, dv) against the most urgent car on a genuine collision course with
    `car` -- unlike find_leader, not restricted to "ahead, in the same
    lane": any other car whose current straight-line trajectory (constant
    velocity assumed) would bring its collision circle within touching
    distance of car's own, at any future point, counts. This is what
    catches crossing/merging traffic that find_leader's lane projection
    misses entirely -- a car converging from the side never satisfies
    find_leader's lateral cutoff, so IDM never brakes for it, and the only
    thing that used to stop the two of them was a hard geometric clip at
    the very last instant (see sim_step's swept-collision clip). This
    computes the same "distance-to-first-contact" relative-motion quadratic
    that clip already uses per-tick, just with BOTH cars' velocities (not
    one frozen), and turns it into a gap/closing-speed pair shaped exactly
    like find_leader's, so it can feed the same IDM braking curve and
    produce a smooth, graduated slowdown instead of a last-instant stop.

    Returns (inf, 0.0) if nothing on `pool` is actually on a collision
    course (paths don't come close enough, already diverging, or already
    overlapping -- that last case isn't this function's job, the swept
    clip + _resolve_collision_positions handle it).
    """
    p0 = car.pos
    v0 = car.vel
    r0 = float(getattr(car, 'radius', 0.0))
    best_gap = float('inf')
    best_dv = 0.0

    for other in pool:
        if other is car:
            continue
        dp = other.pos - p0
        rel_v = other.vel - v0
        R = r0 + float(getattr(other, 'radius', 0.0))

        dist = float(np.linalg.norm(dp))
        c = dist * dist - R * R
        if c <= 0:
            continue  # already touching/overlapping -- not this function's job

        a = float(np.dot(rel_v, rel_v))
        if a < 1e-9:
            continue  # no relative motion -- this gap isn't closing on its own

        b = 2.0 * float(np.dot(dp, rel_v))
        disc = b * b - 4.0 * a * c
        if disc < 0:
            continue  # paths never come within R of each other

        t_hit = (-b - math.sqrt(disc)) / (2.0 * a)
        if t_hit < 0.0 or t_hit > CROSSING_HORIZON_S:
            continue  # already past closest approach, or too far out to act on yet

        closing = -float(np.dot(dp, rel_v)) / dist  # rate the surface gap is shrinking, m/s
        if closing <= 0:
            continue  # not actually closing right now

        g = dist - R
        if g < best_gap:
            best_gap = g
            best_dv = closing

    return best_gap, best_dv


# ── IDM core ──────────────────────────────────────────────────────────────────

def _idm_acceleration(v: float, v0: float, g: float, dv: float) -> float:
    """a = a_max * [ 1 - (v/v0)^delta - (s*/gap)^2 ]"""
    g      = max(g, 1e-3)
    s_star = max(IDM_S0 + v * IDM_T + v * dv / (2.0 * math.sqrt(IDM_A * IDM_B) + 1e-9), IDM_S0)
    free_road   = (v / (v0 + 1e-9)) ** IDM_DELTA
    interaction = (s_star / g) ** 2
    return IDM_A * (1.0 - free_road - interaction)


# ── public API ────────────────────────────────────────────────────────────────

def apply_idm(cars: list, road_direction: np.ndarray,
              stop_line_proj: float, red_light: bool,
              dt: float = 1.0, crossing_pool: list = None):
    """
    IDM car-following + red-light stopping, longitudinal only.
    Velocity is written back along road_direction.

    crossing_pool: optional list of every other active car worth checking
    for a genuine (not necessarily same-lane) collision course -- see
    find_crossing_threat. Omit to skip that check entirely (old behavior:
    only same-lane leaders and the red-light stop line brake a car).
    """
    active = [c for c in cars if c.active and c.stopped_frames == 0
              and not getattr(c, 'stopped_at_line', False)]

    for car in active:
        v  = float(np.linalg.norm(car.vel))
        v0 = float(getattr(car, 'v_desired', 15)) * METERS_PER_PIXEL

        # 1. real leader
        _, gap_real, dv_real = find_leader(car, active, road_direction)

        # 2. virtual stop-line car when red — use gap() with a dummy car
        gap_stop = float("inf")
        dv_stop  = v
        if red_light:
            # build a zero-size virtual car sitting exactly at the stop line
            stop_car = type('StopCar', (), {
                'pos':       road_direction * stop_line_proj,
                'vel':       np.zeros(2),
                'target_dir': road_direction,
                'length_px': 0.0,
                'width_px':  0.0,
            })()
            stop_car.theta = math.atan2(road_direction[1], road_direction[0])
            g_s = gap(car, stop_car, road_direction)
            if g_s > 0:
                gap_stop = g_s

        # 3. any car on a genuine collision course, same-lane or not --
        # catches crossing/merging traffic find_leader's lane projection
        # misses entirely (see find_crossing_threat).
        gap_cross, dv_cross = float("inf"), 0.0
        if crossing_pool:
            gap_cross, dv_cross = find_crossing_threat(car, crossing_pool)

        # 4. most constraining of the three
        g, dv = gap_real, dv_real
        if gap_stop < g:
            g, dv = gap_stop, dv_stop
        if gap_cross < g:
            g, dv = gap_cross, dv_cross

        # 5. hard stop at minimum gap
        if g <= IDM_S0:
            car.vel = np.zeros(2)
            continue

        # 6. IDM acceleration
        a = _idm_acceleration(v, v0, g, dv)

        # 7. Euler speed update — clamp to [0, v0]
        new_speed = float(np.clip(v + a * dt, 0.0, v0))

        # 8. write speed back along road_direction (longitudinal only)
        if new_speed < 1e-4:
            car.vel = np.zeros(2)
        else:
            car.vel = road_direction * new_speed

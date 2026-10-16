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
CROSSING_HORIZON_S = 5.0  # ignore a crossing threat further out than this
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
    `car`.

    Bug 9 Fix:
    If two cars are already touching/overlapping (c <= 0), instead of skipping
    with `continue` which returned gap=inf (causing IDM to accelerate), return
    gap=0.0 and positive closing speed, commanding immediate maximum braking.
    """
    p0 = car.pos
    v0 = car.vel
    r0 = float(getattr(car, 'radius', 0.0))
    best_gap = float('inf')
    best_dv = 0.0

    fwd_dir = getattr(car, 'target_dir', None)
    if fwd_dir is None or np.linalg.norm(fwd_dir) < 0.1:
        fwd_dir = v0
    fwd_norm = fwd_dir / (np.linalg.norm(fwd_dir) + 1e-9) if np.linalg.norm(fwd_dir) > 0.01 else np.array([1.0, 0.0])

    for other in pool:
        if other is car:
            continue
        dp = other.pos - p0
        rel_v = other.vel - v0
        R = r0 + float(getattr(other, 'radius', 0.0))

        dist = float(np.linalg.norm(dp))
        c = dist * dist - R * R

        # Ignore vehicles entirely behind us along our forward heading
        fwd_proj = float(np.dot(dp, fwd_norm))
        if fwd_proj < -0.1:
            continue

        closing = -float(np.dot(dp, rel_v)) / (dist + 1e-9)

        if c <= 0:
            # Bug 9 Fix: already touching/overlapping in front or converging — brake immediately
            if closing > 0:
                return 0.0, closing
            continue

        a = float(np.dot(rel_v, rel_v))
        if a < 1e-9:
            continue

        b = 2.0 * float(np.dot(dp, rel_v))
        disc = b * b - 4.0 * a * c
        if disc < 0:
            continue

        t_hit = (-b - math.sqrt(disc)) / (2.0 * a)
        if t_hit < 0.0 or t_hit > CROSSING_HORIZON_S:
            continue

        closing = -float(np.dot(dp, rel_v)) / dist
        if closing <= 0:
            continue

        g = dist - R
        if g < best_gap:
            best_gap = g
            best_dv = closing

    return best_gap, best_dv


# ── IDM core ──────────────────────────────────────────────────────────────────

def _idm_acceleration(v: float, v0: float, g: float, dv: float,
                      a_max: float = None, b_max: float = None,
                      t_headway: float = None, s0: float = None) -> float:
    """a = a_max * [ 1 - (v/v0)^delta - (s*/gap)^2 ]"""
    a_m = IDM_A if a_max is None else float(a_max)
    b_m = IDM_B if b_max is None else float(b_max)
    t_h = IDM_T if t_headway is None else float(t_headway)
    s_0 = IDM_S0 if s0 is None else float(s0)

    g      = max(g, 1e-3)
    s_star = max(s_0 + v * t_h + v * dv / (2.0 * math.sqrt(a_m * b_m) + 1e-9), s_0)
    free_road   = (v / (v0 + 1e-9)) ** IDM_DELTA
    interaction = (s_star / g) ** 2
    return a_m * (1.0 - free_road - interaction)


# ── public API ────────────────────────────────────────────────────────────────

def apply_idm(cars: list, road_direction: np.ndarray,
              stop_line_proj: float, red_light: bool,
              dt: float = 1.0, crossing_pool: list = None):
    """
    IDM car-following + red-light stopping, longitudinal only.
    Velocity is written back along road_direction.
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

        # 3. any car on a genuine collision course, same-lane or not
        gap_cross, dv_cross = float("inf"), 0.0
        if crossing_pool:
            gap_cross, dv_cross = find_crossing_threat(car, crossing_pool)

        # 4. most constraining of the three
        g, dv = gap_real, dv_real
        if gap_stop < g:
            g, dv = gap_stop, dv_stop
        if gap_cross < g:
            g, dv = gap_cross, dv_cross

        # Check for per-vehicle personalized parameters
        cp = getattr(car, 'custom_params', None)
        a_m = cp.get('IDM_A', IDM_A) if cp else IDM_A
        b_m = cp.get('IDM_B', IDM_B) if cp else IDM_B
        t_h = cp.get('IDM_T', IDM_T) if cp else IDM_T
        s_0 = cp.get('IDM_S0', IDM_S0) if cp else IDM_S0

        # 5. hard stop at minimum gap
        if g <= s_0:
            car.vel = np.zeros(2)
            continue

        # 6. IDM acceleration
        a = _idm_acceleration(v, v0, g, dv, a_max=a_m, b_max=b_m, t_headway=t_h, s0=s_0)

        # 7. Euler speed update — clamp to [0, v0]
        new_speed = float(np.clip(v + a * dt, 0.0, v0))

        # 8. write speed back along road_direction (longitudinal only)
        if new_speed < 1e-4:
            car.vel = np.zeros(2)
        else:
            car.vel = road_direction * new_speed

"""
evaluate.py - Offline trajectory error metrics for new_sim.
Metrics: ADE, FDE, RMSE, MissRate

Usage:
    python evaluate.py
    python evaluate.py --miss-threshold 30
"""
import argparse
import csv
import json
import os
import sys
import numpy as np
import pygame
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from settings import *
from phase_schedule import is_green
from constants import *
from data_loader import DataLoader
from Forces import compute_all_forces, apply_idm


def is_point_in_polygon(point, polygon_points):
    x, y = point
    n = len(polygon_points)
    inside = False
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


def calculate_road_direction(polygon_points, red_line=None):
    """Unit vector from the polygon's center toward the stop line. See the
    matching function in sim.py for why this is derived from red_line
    rather than polygon point order."""
    if red_line and len(red_line) == 2:
        pts = np.array(polygon_points, dtype=float)
        centroid = pts.mean(axis=0)
        rl_mid = np.mean(np.array(red_line, dtype=float), axis=0)
        d = rl_mid - centroid
        n = np.linalg.norm(d)
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


def _segment_intersect(p1, p2, p3, p4):
    d1 = p2 - p1
    d2 = p4 - p3
    cross = d1[0]*d2[1] - d1[1]*d2[0]
    if abs(cross) < 1e-10:
        return False, None
    diff = p3 - p1
    t = (diff[0]*d2[1] - diff[1]*d2[0]) / cross
    u = (diff[0]*d1[1] - diff[1]*d1[0]) / cross
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return True, p1 + t * d1
    return False, None


def make_car(tid, sx, sy, iv, v_desired, length_px, width_px,
             offset_x, offset_y, road_direction, vp=None):
    if vp is None:
        vp = {}
    lm = length_px * METERS_PER_PIXEL
    wm = width_px  * METERS_PER_PIXEL
    vel = np.array(iv, dtype=float)
    # per-car color, same formula as sim.py's make_car -- was flat (0,0,0)
    # here since evaluate.py itself never draws anything, but junction_sim.py
    # imports this exact make_car and does draw with it.
    hue = (tid * 73) % 360
    c = pygame.Color(0)
    c.hsva = (hue, 80, 100, 100)
    return SimpleNamespace(
        id=tid,
        pos=np.array([sx * METERS_PER_PIXEL, sy * METERS_PER_PIXEL]),
        vel=vel,
        # always spawn parallel to the road, not the raw recorded velocity's
        # direction -- see sim.py's make_car for the full explanation (a
        # noisy single-frame velocity reading used to get masked by
        # sim_step's old uncapped resync; with that removed, it would
        # otherwise stay visibly wrong for several seconds after spawn).
        theta=np.arctan2(road_direction[1], road_direction[0]),
        v_desired=v_desired,
        target_dir=road_direction.copy(),
        radius=(lm + wm) / 4.0,
        length_px=length_px, width_px=width_px,
        color=(c.r, c.g, c.b),
        F_drive=np.zeros(2), F_repulsive=np.zeros(2),
        F_obstacle=np.zeros(2), F_stop=np.zeros(2), F_total=np.zeros(2),
        F_steer=np.zeros(2),
        trajectory=[],
        offset_x=offset_x, offset_y=offset_y,
        active=True, stopped_at_line=False,
        reaction_time=vp.get('reaction_time', REACTION_TIME),
        repulsion_strength=vp.get('repulsion_strength', REPULSION_STRENGTH),
        repulsion_range=vp.get('repulsion_range', REPULSION_RANGE),
        stopped_frames=0,
        mass=AGENT_MASS,
    )


def clear_forces(car):
    car.F_drive = car.F_repulsive = car.F_obstacle = car.F_stop = car.F_total = car.F_steer = np.zeros(2)


def _clamp_to_boundary(car, boundaries):
    BM = 0.5 * METERS_PER_PIXEL
    for ws, we in boundaries:
        wv = we - ws
        wls = float(np.dot(wv, wv))
        if wls < 1e-12:
            continue
        t = float(np.clip(np.dot(car.pos - ws, wv) / wls, 0.0, 1.0))
        closest = ws + t * wv
        to_car = car.pos - closest
        dist = float(np.linalg.norm(to_car))
        min_dist = car.radius + BM
        push_dir = (to_car / dist) if dist > 1e-9 else \
                   np.array([-wv[1], wv[0]]) / (np.linalg.norm(wv) + 1e-12)
        if dist < min_dist * 2.0:
            f_in = np.dot(car.F_total, -push_dir)
            if f_in > 0:
                car.F_total += f_in * push_dir
        if dist < min_dist:
            car.pos = closest + push_dir * min_dist
            v_in = np.dot(car.vel, -push_dir)
            if v_in > 0:
                car.vel += v_in * push_dir


def _strip_collision_forces(cars):
    for i, a in enumerate(cars):
        if not a.active or a.stopped_frames > 0:
            continue
        for b in cars[i + 1:]:
            if not b.active or b.stopped_frames > 0:
                continue
            diff = a.pos - b.pos
            dist = float(np.linalg.norm(diff))
            min_dist = a.radius + b.radius
            if dist < 1e-9 or dist > min_dist * 2.0:
                continue
            push_a = diff / dist
            f = np.dot(a.F_total, -push_a)
            if f > 0:
                a.F_total += f * push_a
            f = np.dot(b.F_total, push_a)
            if f > 0:
                b.F_total -= f * push_a


def _resolve_collision_positions(cars):
    for i, a in enumerate(cars):
        if not a.active or a.stopped_frames > 0:
            continue
        for b in cars[i + 1:]:
            if not b.active or b.stopped_frames > 0:
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
            va = np.dot(a.vel, -push_a)
            if va > 0:
                spd_a = max(0.0, float(np.linalg.norm(a.vel)) - va)
                a.vel = np.array([spd_a * np.cos(a.theta), spd_a * np.sin(a.theta)])
            vb = np.dot(b.vel, push_a)
            if vb > 0:
                spd_b = max(0.0, float(np.linalg.norm(b.vel)) - vb)
                b.vel = np.array([spd_b * np.cos(b.theta), spd_b * np.sin(b.theta)])


def _clip_movement_for_collision(pos0, delta, radius, others):
    """See sim.py's version for the full explanation -- swept-circle check
    so a car's position update stops short of another car it would
    otherwise move into/through this tick, instead of moving the full
    distance and only untangling the overlap afterward."""
    t_max = 1.0
    a = float(np.dot(delta, delta))
    if a < 1e-12:
        return 1.0
    for other_pos, other_radius in others:
        R = radius + other_radius
        f = pos0 - other_pos
        c = float(np.dot(f, f)) - R * R
        if c <= 0:
            continue  # already overlapping at the start of this move
        b = 2.0 * float(np.dot(f, delta))
        disc = b * b - 4 * a * c
        if disc < 0:
            continue
        t_hit = (-b - np.sqrt(disc)) / (2 * a)
        if 0.0 <= t_hit < t_max:
            t_max = t_hit
    return max(t_max, 0.0)


def sim_step(car, others=None):
    spd = float(np.linalg.norm(car.vel))
    # theta changes ONLY via the capped rotation below, never a direct
    # resync to car.vel's direction -- that resync (removed) snapped a
    # resuming stop-and-go car's heading straight to road_direction in one
    # uncapped tick, however large the gap was; see sim.py's sim_step for
    # the full trace that caught it.
    if spd > 1e-4:
        # steer off F_steer, not the (possibly wall/collision-clamp-modified)
        # car.F_total -- see sim.py's sim_step for the full explanation and
        # the trace that caught a car spinning in place, pinned against a
        # wall, chasing the clamp's wall-avoidance correction instead of
        # steering toward where it actually wanted to go.
        f_steer = getattr(car, "F_steer", car.F_total)
        f_mag = np.linalg.norm(f_steer)
        desired_angle = np.arctan2(f_steer[1], f_steer[0]) if f_mag > 1e-3 \
                        else np.arctan2(car.target_dir[1], car.target_dir[0])
        w = np.arctan2(np.sin(desired_angle - car.theta),
                       np.cos(desired_angle - car.theta))
        w = float(np.clip(w, -np.pi / 9, np.pi / 9))  # up to 20°/tick
        car.theta += w
    # else: stopped -- hold heading instead of rotating unboundedly toward
    # residual F_total with nothing to resync it back (see sim.py's sim_step)
    dt = 1.0 / FPS_SIM
    delta = np.array([spd * np.cos(car.theta), spd * np.sin(car.theta)]) * dt
    if others:
        t = _clip_movement_for_collision(car.pos, delta, car.radius, others)
        car.pos += delta * t
    else:
        car.pos += delta
    car.vel = np.array([spd * np.cos(car.theta), spd * np.sin(car.theta)])


def compute_metrics(sim_traj, gt_traj):
    gt_lk = {p['frame_num'] - gt_traj[0]['frame_num']: (p['x'], p['y']) for p in gt_traj}
    s0 = sim_traj[0]['frame_num']
    errors = []
    last_err = None
    for sp in sim_traj:
        pt = gt_lk.get(sp['frame_num'] - s0)
        if pt is None:
            continue
        err = np.hypot(sp['x'] - pt[0], sp['y'] - pt[1])
        errors.append(err)
        last_err = err
    if not errors:
        return None
    return {
        'ADE':      float(np.mean(errors)),
        'FDE':      float(last_err),
        'RMSE':     float(np.sqrt(np.mean(np.array(errors) ** 2))),
        'n_frames': len(errors),
    }


def run_simulation(csv_path, polygon_path, start_frame=None, end_frame=None, road_num=None):
    with open(polygon_path) as f:
        pdata = json.load(f)
    polygon_points = pdata["polygon_points"]
    loader = DataLoader(csv_path)

    start_frame = START_FRAME if start_frame is None else start_frame
    end_frame = END_FRAME if end_frame is None else end_frame
    # Follows the same cyclic signal (phase_schedule.is_green) that
    # junction_sim.py uses for this road inside the full 8-way intersection
    # -- green for one 170-frame slot per 680-frame cycle, red otherwise,
    # repeating for the whole run. See sim.py's run loop for the same fix
    # and the full explanation of the frame_start-reuse bug this replaces.
    road_num = 2 if road_num is None else road_num

    cars, spawned_ids = [], set()
    sim_frame = start_frame
    dt_sim = 1.0 / FPS_SIM
    red_light = False

    poly_np = np.array(polygon_points)
    sim_offset_x = -poly_np[:, 0].min()
    sim_offset_y = -poly_np[:, 1].min()
    road_dir = calculate_road_direction(polygon_points, pdata.get("red_line"))

    shifted = [(p[0]+sim_offset_x, p[1]+sim_offset_y) for p in polygon_points]
    boundaries = [
        (np.array(shifted[i]) * METERS_PER_PIXEL,
         np.array(shifted[(i+1) % len(shifted)]) * METERS_PER_PIXEL)
        for i in range(len(shifted))
    ]

    road_projs = [p[0]*road_dir[0] + p[1]*road_dir[1] for p in polygon_points]
    raw_max_proj = max(road_projs)
    shifted_projs = [(p[0]+sim_offset_x)*road_dir[0] + (p[1]+sim_offset_y)*road_dir[1]
                     for p in polygon_points]
    stop_line_proj_m = (max(shifted_projs) - STOP_BUFFER_METERS * PIXELS_PER_METER) * METERS_PER_PIXEL

    mi = int(np.argmax(road_projs))
    n = len(polygon_points)
    oi = (mi-1)%n if road_projs[(mi-1)%n] > road_projs[(mi+1)%n] else (mi+1)%n
    stop_p1 = (np.array(polygon_points[mi], float) + [sim_offset_x, sim_offset_y]) * METERS_PER_PIXEL
    stop_p2 = (np.array(polygon_points[oi], float) + [sim_offset_x, sim_offset_y]) * METERS_PER_PIXEL

    finished = {}

    # rotation-stability tracking -- counts how many times each car's
    # rendered heading reverses direction tick to tick (the "shivering"
    # signature: a real oscillation is many sign-flips in a row, not one
    # legitimate turn). Used by tune_params.py to penalize parameter
    # combinations that cause steering instability, which plain ADE/FDE
    # doesn't detect at all -- a car can oscillate wildly around the
    # correct mean position and still score well on trajectory match.
    _prev_angle = {}
    _prev_sign = {}
    flip_counts = {}

    def _rendered_angle(car):
        speed = np.hypot(car.vel[0], car.vel[1])
        return np.degrees(np.arctan2(-car.vel[1], car.vel[0])) if speed > 1e-4 else np.degrees(-car.theta)

    def _track_rotation(car):
        ang = _rendered_angle(car)
        prev = _prev_angle.get(car.id)
        if prev is not None:
            d = (ang - prev + 180) % 360 - 180
            if abs(d) > 1.0:
                sign = 1 if d > 0 else -1
                if _prev_sign.get(car.id, 0) not in (0, sign):
                    flip_counts[car.id] = flip_counts.get(car.id, 0) + 1
                _prev_sign[car.id] = sign
        _prev_angle[car.id] = ang

    for _ in range(end_frame - start_frame):
        for cdata in loader.get_new_cars_for_frame(sim_frame):
            tid = cdata['track_id']
            if tid in spawned_ids:
                continue
            if hasattr(loader, 'static_ids') and tid in loader.static_ids:
                continue
            if not is_point_in_polygon((cdata['x'], cdata['y']), polygon_points):
                continue
            spawned_ids.add(tid)
            sx = cdata['x'] + sim_offset_x
            sy = cdata['y'] + sim_offset_y
            area = cdata.get('w', DEFAULT_CAR_W) * cdata.get('h', DEFAULT_CAR_H)
            vc = 'small' if area < VEHICLE_SIZE_SMALL else ('large' if area >= VEHICLE_SIZE_LARGE else 'medium')
            vp = VEHICLE_PARAMS[vc]
            vx = vy = 0.0
            traj = loader.cars_by_id.get(tid, [])
            for i, p in enumerate(traj):
                if p['frame_num'] == sim_frame and i < len(traj) - 1:
                    nx = traj[i + 1]
                    vx = (nx['x'] - cdata['x']) * FPS_SIM
                    vy = (nx['y'] - cdata['y']) * FPS_SIM
                    break
            # w/h are the axis-aligned image-space bounding box, not
            # length/width in vehicle-oriented terms -- see sim.py's spawn
            # code for the full explanation (a fixed w=length/h=width
            # mapping is backwards for most vehicles on a vertically-running
            # road). A vehicle's length is its longer bounding dimension.
            raw_w = cdata.get('w', CAR_LENGTH)
            raw_h = cdata.get('h', CAR_WIDTH)
            cars.append(make_car(tid, sx, sy,
                                 [vx * METERS_PER_PIXEL, vy * METERS_PER_PIXEL],
                                 vp['desired_speed'],
                                 max(raw_w, raw_h), min(raw_w, raw_h),
                                 sim_offset_x, sim_offset_y, road_dir, vp))

        sim_frame += 1
        red_light = not is_green(road_num, sim_frame)
        if not red_light:
            # phase cycled back to green: release anyone this road's
            # earlier red phase had frozen at the stop line -- stopped_at_line
            # is otherwise a one-way flag that never clears itself.
            for car in cars:
                car.stopped_at_line = False

        for car in cars:
            if not car.active:
                continue
            px = car.pos * PIXELS_PER_METER
            orig = (px[0] - sim_offset_x, px[1] - sim_offset_y)
            car_proj = orig[0]*road_dir[0] + orig[1]*road_dir[1]
            car.trajectory.append({'frame_num': sim_frame, 'x': orig[0], 'y': orig[1]})
            clear_forces(car)
            compute_all_forces(car, [c for c in cars if c is not car],
                               boundaries, FORCES_ENABLED,
                               red_light=red_light, car_proj=car_proj,
                               stop_line=(stop_p1, stop_p2),
                               road_direction=road_dir)
            car.F_steer = car.F_total.copy()  # snapshot before the boundary clamp below can rewrite F_total's angle

        for car in cars:
            # stopped_frames is never set above 0 anywhere in the codebase
            # (COLLISION_STOP_FRAMES in settings.py is unused too) -- this
            # decrement is a no-op. Left commented rather than removed in
            # case the collision-cooldown feature gets wired up later.
            # if car.stopped_frames > 0:
            #     car.stopped_frames = max(0, car.stopped_frames - 1)
            #     continue
            _clamp_to_boundary(car, boundaries)
        _strip_collision_forces(cars)

        # crossing_pool=cars: see sim.py's main loop for why -- brakes a car
        # smoothly for anyone on a collision course, not just a same-lane
        # leader, instead of leaning on the swept-position clip alone.
        apply_idm(cars, road_direction=road_dir,
                  stop_line_proj=stop_line_proj_m,
                  red_light=red_light, dt=dt_sim,
                  crossing_pool=cars)

        # frozen pre-move snapshot -- see sim.py's main loop for why (order-
        # independent swept-collision checks, not whatever position a car
        # already ended up at if processed earlier in this same loop)
        move_snapshot = [(c, c.pos.copy(), c.radius) for c in cars if c.active]
        for car in cars:
            if not car.active or car.stopped_frames > 0:
                continue
            if getattr(car, 'stopped_at_line', False):
                continue
            prev = car.pos.copy()
            others = [(pos, r) for c2, pos, r in move_snapshot if c2 is not car]
            sim_step(car, others=others)
            _track_rotation(car)
            if red_light and not car.stopped_at_line:
                hit, pt = _segment_intersect(prev, car.pos, stop_p1, stop_p2)
                if hit:
                    car.pos = pt
                    car.vel = np.zeros(2)
                    car.F_total = np.zeros(2)
                    car.stopped_at_line = True
            _clamp_to_boundary(car, boundaries)
            orig = (car.pos[0]*PIXELS_PER_METER - sim_offset_x,
                    car.pos[1]*PIXELS_PER_METER - sim_offset_y)
            if orig[0]*road_dir[0] + orig[1]*road_dir[1] > raw_max_proj + 50:
                car.active = False
            elif not is_point_in_polygon(orig, polygon_points):
                car.active = False

        _resolve_collision_positions(cars)
        for car in cars:
            if not car.active and car.id not in finished:
                finished[car.id] = car.trajectory[:]
        cars = [c for c in cars if c.active]

    for car in cars:
        if car.id not in finished:
            finished[car.id] = car.trajectory[:]

    return finished, loader, flip_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',            default='selected_area_data.csv')
    parser.add_argument('--polygon',        default='polygon_points.json')
    parser.add_argument('--miss-threshold', type=float, default=50.0)
    parser.add_argument('--output',         default='results.csv')
    parser.add_argument('--profile',        default=None,
                         help="path to a params_profiles/*.json to apply before running "
                              "(leave unset to use the current hardcoded settings)")
    args = parser.parse_args()

    if args.profile:
        import param_profile
        param_profile.apply_from_file(args.profile)
        print(f"Applied parameter profile: {args.profile}")

    with open(args.polygon) as f:
        _pdata = json.load(f)
    # roads produced by add_polygon.py carry their own frame window.
    start_frame = _pdata.get("frame_start")
    end_frame = _pdata.get("frame_end")
    road_num = _pdata.get("road", 2)

    print("Running simulation headlessly...")
    sim_trajs, loader, flip_counts = run_simulation(args.csv, args.polygon,
                                        start_frame=start_frame, end_frame=end_frame,
                                        road_num=road_num)
    print(f"Done. {len(sim_trajs)} cars tracked.\n")
    if flip_counts:
        avg_flips = sum(flip_counts.values()) / len(sim_trajs)
        worst = sorted(flip_counts.items(), key=lambda kv: -kv[1])[:5]
        print(f"Rotation stability: {avg_flips:.2f} avg direction-flips/car "
              f"(worst: {', '.join(f'car {c}={n}' for c, n in worst)})\n")

    rows = []
    all_ade, all_fde, all_rmse = [], [], []

    for tid, sim_traj in sim_trajs.items():
        gt = loader.cars_by_id.get(tid)
        if not gt or not sim_traj:
            continue
        m = compute_metrics(sim_traj, gt)
        if m is None:
            continue
        rows.append({'car_id': tid, **m})
        all_ade.append(m['ADE'])
        all_fde.append(m['FDE'])
        all_rmse.append(m['RMSE'])

    if not rows:
        print("No matched trajectories found.")
        return

    miss_rate = sum(1 for r in rows if r['FDE'] > args.miss_threshold) / len(rows) * 100

    print(f"{'Car':>6}  {'ADE(px)':>10}  {'FDE(px)':>10}  {'RMSE(px)':>10}  {'ADE(m)':>8}  {'FDE(m)':>8}  {'Frames':>6}")
    print("-" * 74)
    for r in sorted(rows, key=lambda x: x['car_id']):
        print(f"{r['car_id']:>6}  {r['ADE']:>10.2f}  {r['FDE']:>10.2f}  {r['RMSE']:>10.2f}"
              f"  {r['ADE']*METERS_PER_PIXEL:>8.3f}  {r['FDE']*METERS_PER_PIXEL:>8.3f}  {r['n_frames']:>6}")

    print("\n" + "=" * 74)
    print(f"  Cars evaluated : {len(rows)}")
    print(f"  ADE            : {np.mean(all_ade):.2f} px  /  {np.mean(all_ade)*METERS_PER_PIXEL:.3f} m")
    print(f"  FDE            : {np.mean(all_fde):.2f} px  /  {np.mean(all_fde)*METERS_PER_PIXEL:.3f} m")
    print(f"  RMSE           : {np.mean(all_rmse):.2f} px  /  {np.mean(all_rmse)*METERS_PER_PIXEL:.3f} m")
    print(f"  Miss Rate      : {miss_rate:.1f}%  (FDE > {args.miss_threshold:.0f} px)")
    print("=" * 74)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['car_id', 'ADE', 'FDE', 'RMSE', 'n_frames'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
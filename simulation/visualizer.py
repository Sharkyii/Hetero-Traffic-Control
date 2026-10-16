"""
junction_sim_hetero_qc.py — Heterogeneous Queue-Clearing Max-Pressure & Online Vehicle Learner Simulation

Core Architectural Features:
  1. Heterogeneous Queue-Clearing Max-Pressure Controller (H-QC-MPC):
     - Exhaustive Service: Holds green until all waiting and approaching vehicles have crossed.
     - Event-Driven Dynamic Box Clearance: Eliminates the fixed 6.0s all-red timer. Next green
       turns on the instant the junction box is physically clear of crossing vehicles.
     - PCU-weighted pressure for mixed Indian urban traffic (IRC standard: 0.4 small, 1.0 med, 2.8 large).
     - Anti-starvation preemption governor.
  2. Online Self-Optimizing Heterogeneous Vehicle Learner:
     - Contextual prior seeding conditioned on vehicle geometry (L, W) and entry velocity.
     - Dual-EKF online parameter tracking adjusting individual IDM parameters (T_i, a_max_i, s0_i).
  3. 100% Pure CPU Execution with real-time Pygame telemetry HUD.

Usage:
  python junction_sim_hetero_qc.py --source od4
  SDL_VIDEODRIVER=dummy python junction_sim_hetero_qc.py --source od4 --max-frames 200
"""

import argparse
import json
import os
import sys
import math
import time as _time
import numpy as np

# Ensure root directory is on import path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# Auto-switch to virtualenv if available
_venv_python = os.path.abspath(os.path.join(ROOT_DIR, "..", "venv", "bin", "python"))
if os.path.exists(_venv_python) and sys.executable != _venv_python:
    try:
        import pygame
    except ImportError:
        os.execv(_venv_python, [_venv_python] + sys.argv)

import pygame

from simulation.config import (
    PIXELS_PER_METER, METERS_PER_PIXEL, DT, FPS,
    VEHICLE_PARAMS, FORCES_ENABLED, STOP_BUFFER_METERS,
    ENTRY_ROADS, EXIT_ROADS, WIDTH, HEIGHT,
)
from simulation.data_loader import DataLoader
from models.potential_fields import compute_all_forces
from models.idm import apply_idm, find_crossing_threat, CROSSING_HORIZON_S
from models.vehicle_classifier import classify_vehicle
from controllers.max_pressure import (
    HeteroQueueClearingController,
    ENTRY_ROADS, ALL_EXITS, DEFAULT_PCU_WEIGHTS
)
from models.vehicle_learner import VehicleOnlineLearner
from calibration import param_profile
from simulation.road_geometry import (
    is_point_in_polygon, calculate_road_direction, _segment_intersect,
    make_car, clear_forces, _clamp_to_boundary, _strip_collision_forces,
    _resolve_collision_positions, sim_step, _clip_movement_for_collision,
)
import simulation.road_geometry as RT


ROAD_COLOR_GREEN = (0, 200, 0)
ROAD_COLOR_RED = (200, 0, 0)
MIDDLE_COLOR = (140, 140, 140)
ARROW_COLOR = (255, 255, 255)


def load_roads():
    json_path = os.path.join(ROOT_DIR, "data", "metadata", "intersection_roads.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(ROOT_DIR, "picked_roads.json")
    data = json.load(open(json_path))
    roads = {r["road"]: r for r in data if r["road"] != "middle"}
    middle = next((r for r in data if r["road"] == "middle"), None)
    return roads, middle



def which_road(x, y, roads, allowed=None):
    for num, r in roads.items():
        if allowed is not None and num not in allowed:
            continue
        if is_point_in_polygon((x, y), r["polygon"]):
            return num
    return None


def compute_real_exits(csv_path, source, turning_movements_path, cfg, offset):
    import csv
    if not os.path.exists(turning_movements_path) or not os.path.exists(csv_path):
        return {}
    cache_path = os.path.join(os.path.dirname(csv_path), f".cache_real_exits_{source}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                raw = json.load(f)
                return {int(k): (v[0], v[1]) for k, v in raw.items()}
        except Exception:
            pass
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    from collections import defaultdict
    tracks = defaultdict(list)
    for r in rows:
        tracks[int(r["track_id"])].append(r)
    with open(turning_movements_path) as f:
        turning_data = json.load(f)
    movements = {m["track_id"]: m for m in turning_data.get("movements", [])}
    real_exits = {}
    for tid, m_info in movements.items():
        dest_road = m_info.get("dest_road")
        if dest_road is None or dest_road not in cfg or tid not in tracks:
            continue
        rc = cfg[dest_road]
        stop_left = rc["stop_left"]
        lat_vec = rc["lateral_vec"]
        lat_w2 = float(np.dot(lat_vec, lat_vec))
        if lat_w2 < 1e-4:
            continue

        pts_m = np.array([[(float(r["x"]) + offset[0]) * METERS_PER_PIXEL,
                           (float(r["y"]) + offset[1]) * METERS_PER_PIXEL] for r in tracks[tid]])
        dists = []
        for pt in pts_m:
            t = np.clip(np.dot(pt - stop_left, lat_vec) / lat_w2, 0.0, 1.0)
            proj = stop_left + t * lat_vec
            dists.append(np.linalg.norm(pt - proj))
        min_idx = np.argmin(dists)
        if dists[min_idx] < 3.0:
            s_exit = float(np.dot(pts_m[min_idx] - stop_left, lat_vec) / lat_w2)
            real_exits[tid] = (dest_road, float(np.clip(s_exit, 0.05, 0.95)))
        else:
            real_exits[tid] = (dest_road, None)
    try:
        with open(cache_path, "w") as f:
            json.dump({str(k): list(v) for k, v in real_exits.items()}, f)
    except Exception:
        pass
    return real_exits


def get_natural_resting_yaw(track_id: int) -> float:
    h = (int(track_id) * 2654435761) & 0xFFFFFFFF
    sign = 1.0 if (h & 1) else -1.0
    deg = 3.0 + float((h >> 1) % 41) * 0.1
    return math.radians(sign * deg)


def build_road_config(roads, offset):
    cfg = {}
    for num, r in roads.items():
        poly = r["polygon"]
        red_line = r["redLine"]
        direction = calculate_road_direction(poly, red_line)

        if num in [1, 3, 5, 7]:
            direction = -direction

        shifted = [(p[0] + offset[0], p[1] + offset[1]) for p in poly]
        all_edges = [
            (np.array(shifted[i]) * METERS_PER_PIXEL,
             np.array(shifted[(i + 1) % len(shifted)]) * METERS_PER_PIXEL)
            for i in range(len(shifted))
        ]
        boundaries = []
        for p1, p2 in all_edges:
            edge_vec = p2 - p1
            n = np.linalg.norm(edge_vec)
            if n < 1e-9:
                continue
            alignment = abs(float(np.dot(edge_vec / n, direction)))
            if alignment > 0.5:
                boundaries.append((p1, p2))

        road_projs = [p[0] * direction[0] + p[1] * direction[1] for p in poly]
        raw_max_proj = max(road_projs)

        rl_shifted = [(p[0] + offset[0], p[1] + offset[1]) for p in red_line]
        stop_p1 = np.array(rl_shifted[0]) * METERS_PER_PIXEL
        stop_p2 = np.array(rl_shifted[1]) * METERS_PER_PIXEL
        rl_mid_shifted = np.mean(rl_shifted, axis=0)
        stop_line_proj_m = (rl_mid_shifted[0] * direction[0] + rl_mid_shifted[1] * direction[1]) * METERS_PER_PIXEL

        v_right = np.array([-direction[1], direction[0]])
        if np.dot(stop_p2 - stop_p1, v_right) > 0:
            stop_left, stop_right = stop_p1, stop_p2
        else:
            stop_left, stop_right = stop_p2, stop_p1
        lateral_vec = stop_right - stop_left

        cfg[num] = {
            "direction": direction,
            "boundaries": boundaries,
            "polygon": poly,
            "polygon_m": [np.array(p) * METERS_PER_PIXEL for p in shifted],
            "red_line": red_line,
            "raw_max_proj": raw_max_proj,
            "stop_p1": stop_p1,
            "stop_p2": stop_p2,
            "stop_left": stop_left,
            "stop_right": stop_right,
            "lateral_vec": lateral_vec,
            "lane_width_m": float(np.linalg.norm(lateral_vec)),
            "stop_line_proj_m": stop_line_proj_m,
        }
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Heterogeneous Queue-Clearing Max-Pressure & Online Learner Sim.")
    parser.add_argument("--source", default="od4", help="Data source folder name (e.g. od4)")
    parser.add_argument("--profile", default="params_profiles/calibrated_hetero.json", help="Calibrated parameter profile")
    parser.add_argument("--max-frames", type=int, default=0, help="Exit after N frames (0 = run indefinitely)")
    parser.add_argument("--fps", type=int, default=30, help="Display FPS target")
    args = parser.parse_args()

    # Load calibrated parameters
    prof_path = os.path.join(ROOT_DIR, args.profile) if args.profile else os.path.join(ROOT_DIR, "calibration", "profiles", "universal_profile.json")
    if os.path.exists(prof_path):
        print(f"[Init] Loading calibrated parameters from {prof_path}...", flush=True)
        param_profile.apply_from_file(prof_path)

    roads, middle = load_roads()
    csv_path = os.path.join(ROOT_DIR, "data", "trajectories", f"{args.source}.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(ROOT_DIR, "segments_new_versions", args.source, "full_data.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(ROOT_DIR, "segments", args.source, "full_data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find trajectory CSV dataset for source '{args.source}'.")

    loader = DataLoader(csv_path)

    all_pts = [p for r in roads.values() for p in r["polygon"]]
    if middle:
        all_pts += middle["polygon"]
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    offset = np.array([-min(xs) + 40, -min(ys) + 40])

    turning_path = os.path.join(ROOT_DIR, "data", "metadata", "turning_movements.json")
    if not os.path.exists(turning_path):
        turning_path = os.path.join(ROOT_DIR, "turning_movements.json")
    cfg = build_road_config(roads, offset)
    real_exits = compute_real_exits(csv_path, args.source, turning_path, cfg, offset)


    # Initialize Controllers
    signal_controller = HeteroQueueClearingController(
        entry_roads=ENTRY_ROADS,
        exit_roads=ALL_EXITS,
        min_hold_frames=8,
        max_starvation_frames=240,
        starvation_pcu_threshold=15.0,
        min_clearance_frames=2,
    )
    vehicle_learner = VehicleOnlineLearner(learning_rate=1.0)

    # Pure CPU Pygame Initialization
    pygame.init()
    view_w, view_h = 1000, 750
    screen = pygame.display.set_mode((view_w, view_h), pygame.DOUBLEBUF)
    pygame.display.set_caption("Heterogeneous Queue-Clearing MP & Online Learner (Pure CPU)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)
    font_bold = pygame.font.SysFont("monospace", 15, bold=True)

    dt_sim = 1.0 / 4.0  # 4 FPS simulation tick (0.25s)
    sim_frame = 100
    cars = []
    paused = False
    running = True

    print("\n=======================================================", flush=True)
    print(" H-QC-MPC & Online Vehicle Learner Simulation Started", flush=True)
    print(" Mode: Pure CPU (Exhaustive Queue Clearing + Dynamic Clearance)", flush=True)
    print("=======================================================\n", flush=True)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                paused = not paused

        if not paused:
            # 1. Spawn vehicles from drone dataset
            new_records = loader.get_cars_for_frame(sim_frame)
            for r in new_records:
                tid = int(r["track_id"])
                if any(c.track_id == tid for c in cars):
                    continue
                orig_x, orig_y = float(r["x"]), float(r["y"])
                px_x = orig_x + offset[0]
                px_y = orig_y + offset[1]
                road_num = which_road(orig_x, orig_y, roads, allowed=ENTRY_ROADS)
                if road_num is None:
                    continue

                raw_w = float(r.get("w", r.get("width", 20)))
                raw_h = float(r.get("h", r.get("height", 40)))
                v_class = classify_vehicle(raw_w, raw_h)
                vp = VEHICLE_PARAMS[v_class]
                rc = cfg[road_num]

                raw_spd = float(r.get("robust_speed", r.get("speed", 0.0)))
                init_spd = float(np.clip(raw_spd * METERS_PER_PIXEL, 0.0, vp['desired_speed']))

                car = make_car(tid, px_x, px_y, rc["direction"] * init_spd,
                               vp['desired_speed'], max(raw_w, raw_h), min(raw_w, raw_h),
                               offset[0], offset[1], rc["direction"], vp)
                car.current_road = road_num
                car.entry_road = road_num
                car.state = 'entry'
                car.dest_road = None
                car.vehicle_class = v_class
                car.track_id = tid
                car.resting_yaw_offset = get_natural_resting_yaw(tid)

                # Register with Online Learner to assign personalized parameters
                custom_params = vehicle_learner.register_vehicle(
                    veh_id=tid,
                    veh_class=v_class,
                    length_px=max(raw_w, raw_h),
                    width_px=min(raw_w, raw_h),
                    v_entry=float(np.linalg.norm(car.vel)) if hasattr(car, 'vel') else 0.0,
                    meters_per_pixel=METERS_PER_PIXEL
                )
                car.custom_params = custom_params
                cars.append(car)

            sim_frame += 1

            # 2. Identify In-Junction Crossing Vehicles (Box Occupancy)
            transit_cars = [c for c in cars if c.active and c.state == 'transit']
            box_occupied = len(transit_cars) > 0

            # 3. Heterogeneous Queue and Approach Aggregation
            queue_hetero = {rn: {'small': 0, 'medium': 0, 'large': 0} for rn in ENTRY_ROADS}
            approach_hetero = {rn: {'small': 0, 'medium': 0, 'large': 0} for rn in ENTRY_ROADS}
            exit_hetero = {rn: {'small': 0, 'medium': 0, 'large': 0} for rn in ALL_EXITS}

            for car in cars:
                if not car.active:
                    continue
                v_cls = getattr(car, 'vehicle_class', 'medium')
                if car.state == 'entry' and car.current_road in ENTRY_ROADS:
                    rc = cfg[car.current_road]
                    d_to_stop = rc["stop_line_proj_m"] - float(np.dot(car.pos, rc["direction"]))
                    if 0.0 <= d_to_stop <= 45.0:
                        spd = float(np.linalg.norm(car.vel))
                        if spd < 0.8 or getattr(car, 'stopped_at_line', False):
                            queue_hetero[car.current_road][v_cls] += 1
                        elif d_to_stop <= 14.0:
                            approach_hetero[car.current_road][v_cls] += 1
                elif car.current_road in ALL_EXITS:
                    exit_hetero[car.current_road][v_cls] += 1

            # 4. Step H-QC-MPC Controller
            signal_controller.step(
                queue_data=queue_hetero,
                approach_data=approach_hetero,
                exit_data=exit_hetero,
                box_occupied=box_occupied
            )

            # 5. Entry Road Microscopic Forces & Physics
            by_road = {}
            for car in cars:
                if car.active and car.state == 'entry':
                    by_road.setdefault(car.current_road, []).append(car)

            for road_num, road_cars in by_road.items():
                road_is_green = signal_controller.is_green(road_num)
                if road_is_green:
                    for car in road_cars:
                        car.stopped_at_line = False

                for car in road_cars:
                    px = car.pos * PIXELS_PER_METER
                    orig = (px[0] - offset[0], px[1] - offset[1])
                    rc = cfg[car.current_road]
                    car_proj = orig[0] * rc["direction"][0] + orig[1] * rc["direction"][1]
                    car.trajectory.append({'frame_num': sim_frame, 'x': orig[0], 'y': orig[1]})
                    clear_forces(car)
                    compute_all_forces(car, [c for c in road_cars if c is not car],
                                       rc["boundaries"], FORCES_ENABLED,
                                       red_light=(not road_is_green), car_proj=car_proj,
                                       stop_line=(rc["stop_p1"], rc["stop_p2"]),
                                       road_direction=rc["direction"])
                    road_dir = rc["direction"] / (np.linalg.norm(rc["direction"]) + 1e-9)
                    road_lat = np.array([-road_dir[1], road_dir[0]])
                    f_lat = np.dot(car.F_total, road_lat) * road_lat
                    f_fwd = max(100.0, np.dot(car.F_drive, road_dir)) * road_dir
                    car.F_steer = f_fwd + f_lat

            # 6. IDM Car Following (with personalized custom_params per vehicle)
            crossing_pool_cars = [c for c in cars if c.active and c.state in ('entry', 'transit')]
            for road_num, road_cars in by_road.items():
                rc = cfg[road_num]
                red_light = not signal_controller.is_green(road_num)
                apply_idm(road_cars, road_direction=rc["direction"],
                          stop_line_proj=rc["stop_line_proj_m"], red_light=red_light, dt=dt_sim,
                          crossing_pool=crossing_pool_cars)

            # 7. Kinematic Update & Stop-Line Advance
            move_snapshot_by_road = {rn: [(c, c.pos.copy(), c.radius) for c in rc_cars]
                                     for rn, rc_cars in by_road.items()}

            for car in cars:
                if not car.active or car.state != 'entry':
                    continue
                car.stopped_at_line = False
                rc = cfg[car.current_road]
                spd = float(np.linalg.norm(car.vel))
                base_road_angle = math.atan2(rc["direction"][1], rc["direction"][0])

                if car.current_road in ENTRY_ROADS:
                    if spd > 1e-4:
                        desired_angle = math.atan2(car.F_steer[1], car.F_steer[0])
                        ang_diff = math.atan2(math.sin(desired_angle - base_road_angle),
                                              math.cos(desired_angle - base_road_angle))
                        ang_diff = np.clip(ang_diff, -math.radians(5.0), math.radians(5.0))
                        desired_angle = base_road_angle + ang_diff
                        err = math.atan2(math.sin(desired_angle - car.theta),
                                         math.cos(desired_angle - car.theta))
                        if abs(err) > math.radians(0.5):
                            max_turn = float(np.clip(spd * math.radians(1.0) + math.radians(0.3),
                                                     math.radians(0.2), math.radians(1.0)))
                            w = float(np.clip(err * 0.18, -max_turn, max_turn))
                            car.theta += w

                    theta_dev = math.atan2(math.sin(car.theta - base_road_angle),
                                           math.cos(car.theta - base_road_angle))
                    if abs(theta_dev) > math.radians(5.0):
                        car.theta = base_road_angle + np.clip(theta_dev, -math.radians(5.0), math.radians(5.0))

                delta = np.array([spd * math.cos(car.theta), spd * math.sin(car.theta)]) * dt_sim
                others = [(pos, r) for c2, pos, r in move_snapshot_by_road.get(car.current_road, []) if c2 is not car]
                t = _clip_movement_for_collision(car.pos, delta, car.radius, others)
                car.pos += delta * t
                car.vel = np.array([spd * math.cos(car.theta), spd * math.sin(car.theta)])

                orig = (car.pos[0] * PIXELS_PER_METER - offset[0],
                        car.pos[1] * PIXELS_PER_METER - offset[1])
                proj = orig[0] * rc["direction"][0] + orig[1] * rc["direction"][1]
                left_road = proj > rc["raw_max_proj"] + 50 or not is_point_in_polygon(orig, rc["polygon"])

                if car.current_road not in ENTRY_ROADS:
                    if left_road:
                        car.active = False
                        vehicle_learner.deregister_vehicle(car.track_id)
                    continue

                red_light = not signal_controller.is_green(car.current_road)
                car_proj_m = float(np.dot(car.pos, rc["direction"]))
                stop_m = rc["stop_line_proj_m"]

                if red_light:
                    if car_proj_m >= stop_m - 0.5 or left_road:
                        overshoot = car_proj_m - (stop_m - 0.5)
                        if overshoot > 0:
                            car.pos -= rc["direction"] * overshoot
                        car.vel = np.zeros(2)
                        car.F_total = np.zeros(2)
                        car.stopped_at_line = True
                        continue

                _clamp_to_boundary(car, rc["boundaries"])

                if not left_road:
                    continue

                # Transition to Transit
                e_left, e_right = rc["stop_left"], rc["stop_right"]
                e_vec = rc["lateral_vec"]
                e_w2 = float(np.dot(e_vec, e_vec))
                s_raw = float(np.clip(np.dot(car.pos - e_left, e_vec) / e_w2, 0.05, 0.95)) if e_w2 > 1e-4 else 0.5

                ROUTES = {2: {"straight": 5, "left": 3, "right": 7},
                          4: {"straight": 7, "left": 5, "right": 1},
                          6: {"straight": 1, "left": 7, "right": 3},
                          8: {"straight": 3, "left": 1, "right": 5}}
                r_opts = ROUTES.get(car.current_road, {"straight": 5, "left": 3, "right": 7})

                dest_info = real_exits.get(car.track_id)
                if dest_info is not None:
                    dest, real_s = dest_info
                    s_exit = real_s if real_s is not None else s_raw
                else:
                    dest = r_opts["straight"]
                    s_exit = s_raw

                if dest is not None and dest in cfg:
                    entrance_point_m = (cfg[dest]["stop_p1"] + cfg[dest]["stop_p2"]) / 2.0
                    RT.start_transit(car, dest, entrance_point_m, cfg, s_override=s_exit, sim_frame=sim_frame)
                else:
                    car.active = False
                    vehicle_learner.deregister_vehicle(car.track_id)

            # 8. Transit Simulation & Arrival
            middle_boundaries = []
            transit_cars = [c for c in cars if c.active and c.state == 'transit']
            just_arrived = []
            for car in transit_cars:
                if RT.check_arrival(car, cfg[car.dest_road]["polygon_m"]):
                    dest_rc = cfg[car.dest_road]
                    car.current_road = car.dest_road
                    car.state = 'entry'
                    car.target_dir = dest_rc["direction"].copy()
                    just_arrived.append(car)

            active_transit = [c for c in cars if c.active and c.state == 'transit']
            RT.step_transit(active_transit, middle_boundaries, FORCES_ENABLED, all_cars=cars, ghost_leaders=just_arrived)
            cars = [c for c in cars if c.active]

        # 9. Rendering & HUD
        screen.fill((30, 30, 30))
        # Draw roads
        for num, r in roads.items():
            pts = [(int(p[0] + offset[0]), int(p[1] + offset[1])) for p in r["polygon"]]
            is_grn = signal_controller.is_green(num)
            color = (30, 90, 30) if is_grn else (90, 30, 30) if num in ENTRY_ROADS else (50, 50, 50)
            pygame.draw.polygon(screen, color, pts)
            pygame.draw.polygon(screen, (100, 100, 100), pts, 2)

        # Draw middle box
        if middle:
            m_pts = [(int(p[0] + offset[0]), int(p[1] + offset[1])) for p in middle["polygon"]]
            box_col = (120, 90, 30) if box_occupied else (60, 60, 60)
            pygame.draw.polygon(screen, box_col, m_pts, 2)

        # Draw cars
        for car in cars:
            if not car.active:
                continue
            cx = int(car.pos[0] * PIXELS_PER_METER)
            cy = int(car.pos[1] * PIXELS_PER_METER)
            col = (60, 200, 255) if car.vehicle_class == 'small' else (255, 200, 50) if car.vehicle_class == 'medium' else (255, 80, 80)
            pygame.draw.circle(screen, col, (cx, cy), max(3, int(car.radius * PIXELS_PER_METER)))

        # Telemetry HUD
        hud_bg = pygame.Surface((440, 150))
        hud_bg.set_alpha(220)
        hud_bg.fill((10, 10, 10))
        screen.blit(hud_bg, (10, 10))

        status_txt = font_bold.render(f"STATUS: {signal_controller.phase_status}", True, (255, 255, 0))
        box_txt = font.render(f"BOX OCCUPANCY: {'OCCUPIED (' + str(len(transit_cars)) + ' cars)' if box_occupied else 'CLEAR'}", True,
                              (255, 100, 100) if box_occupied else (100, 255, 100))
        saved_txt = font.render(f"SAVED ALL-RED: {signal_controller.saved_clearance_frames/4:.1f}s vs 6.0s fixed timer", True, (100, 255, 255))
        switches_txt = font.render(f"PHASE SWITCHES: {signal_controller.total_phase_switches} | PREEMPTIONS: {signal_controller.preemption_events}", True, (200, 200, 200))

        v_stats = vehicle_learner.get_summary_stats()
        learner_txt = font.render(f"ONLINE LEARNERS ({v_stats['active_learners']}): T={v_stats['mean_T']:.2f}s, a={v_stats['mean_A']:.2f}m/s2", True, (150, 255, 150))

        screen.blit(status_txt, (18, 16))
        screen.blit(box_txt, (18, 42))
        screen.blit(saved_txt, (18, 68))
        screen.blit(switches_txt, (18, 94))
        screen.blit(learner_txt, (18, 120))

        pygame.display.flip()
        if os.environ.get("SDL_VIDEODRIVER") != "dummy" and args.fps > 0:
            clock.tick(args.fps)

        if (sim_frame - 100) % 25 == 0:
            print(f"[Frame {sim_frame - 100:3d}] {signal_controller.phase_status} | Box: {'OCCUPIED' if box_occupied else 'CLEAR'} | Active Cars: {len(cars)} | Saved Clearance: {signal_controller.saved_clearance_frames / 4.0:.1f}s", flush=True)

        if args.max_frames > 0 and sim_frame >= 100 + args.max_frames:
            break

    print(f"\n[Completed] Simulated {sim_frame - 100} frames.")
    print(f"Total Phase Switches: {signal_controller.total_phase_switches}")
    print(f"Total All-Red Time Saved vs 6.0s Timer: {signal_controller.saved_clearance_frames / 4.0:.2f} seconds")
    pygame.quit()


if __name__ == "__main__":
    main()

"""
evaluate_dual_new_models.py — High-Performance CPU Evaluator for the Two New Traffic Intelligence Models

Evaluates:
  Model 1: H-QC-MPC (Heterogeneous Queue-Clearing Max-Pressure Controller)
           - Dynamic polygon box clearance (eliminates static 6s all-red delay)
           - Exhaustive queue service (holds green until queued platoon completely clears)
           - PCU-weighted pressure routing
           - Anti-starvation safety preemption

  Model 2: VehicleOnlineLearner (H-QC-MPC + Online Self-Optimizing Dual-EKF Vehicle Learner)
           - Online Bayesian estimation of microscopic IDM driving parameters (T_i, a_max^(i), s0_i)
           - Rapid adaptation to heterogeneous driving styles (aggressive vs cautious)
           - Real-time parameter trajectory logging

Saves detailed logs and metrics into:
  - logs/model1_h_qc_mpc/metrics.json
  - logs/model1_h_qc_mpc/simulation_log.json
  - logs/model2_vehicle_online_learner/metrics.json
  - logs/model2_vehicle_online_learner/simulation_log.json
  - logs/model2_vehicle_online_learner/parameter_trajectories.json
  - logs/comparison_summary.json

Usage:
  python evaluate_dual_new_models.py --source flight_dji003_21_8 --frames 400
"""

import os
import sys
import math
import json
import argparse
import numpy as np
from collections import defaultdict

# Pure CPU Execution
os.environ["SDL_VIDEODRIVER"] = "dummy"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import pygame
pygame.init()

from simulation.config import (
    PIXELS_PER_METER, METERS_PER_PIXEL, DT, FPS,
    VEHICLE_PARAMS, FORCES_ENABLED, STOP_BUFFER_METERS,
    ENTRY_ROADS, EXIT_ROADS, WIDTH, HEIGHT,
)
from simulation.data_loader import DataLoader
from models.potential_fields import compute_all_forces
from models.idm import apply_idm
from models.vehicle_classifier import classify_vehicle
import simulation.road_geometry as RT
from calibration import param_profile

from controllers.max_pressure import (
    HeteroQueueClearingController,
    ENTRY_ROADS, ALL_EXITS, DEFAULT_PCU_WEIGHTS
)
from models.vehicle_learner import VehicleOnlineLearner
from simulation.road_geometry import (
    is_point_in_polygon, calculate_road_direction, make_car, clear_forces,
    _clamp_to_boundary, _strip_collision_forces, _resolve_collision_positions,
    _clip_movement_for_collision, sim_step,
)


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


def resolve_dataset_paths(source_name, root="data/trajectories"):
    """Finds CSV dataset for the chosen flight or fallback."""
    candidate_clean = os.path.join(ROOT_DIR, "data", "trajectories", f"{source_name}.csv")
    if os.path.exists(candidate_clean):
        return candidate_clean
    candidate_new = os.path.join(ROOT_DIR, "segments_new_versions", source_name, "full_data.csv")
    if os.path.exists(candidate_new):
        return candidate_new
    candidate_old = os.path.join(ROOT_DIR, "segments", source_name, "full_data.csv")
    if os.path.exists(candidate_old):
        return candidate_old
    raise FileNotFoundError(f"Cannot find trajectory CSV dataset for source '{source_name}'.")



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
    tracks = defaultdict(list)
    for r in rows:
        tracks[int(r["track_id"])].append(r)

    tm = json.load(open(turning_movements_path)).get(source, {}).get("entry_exit_by_track", {})
    real_exits = {}
    for tid_str, (entry, exit_) in tm.items():
        tid = int(tid_str)
        if exit_ is None or tid not in tracks:
            continue
        dest_road = exit_
        if dest_road not in cfg:
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


def simulate_model(model_name: str, source: str = "flight_dji003_21_8",
                   max_frames: int = 400, profile_path: str = None):
    """
    Simulates either:
      - 'model1_h_qc_mpc': Queue-clearing max pressure alone
      - 'model2_vehicle_online_learner': Queue-clearing max pressure + Online Vehicle Learner
    """
    # 1. Apply calibrated parameter profile
    if profile_path and os.path.exists(profile_path):
        param_profile.apply_from_file(profile_path)
    elif os.path.exists(os.path.join(ROOT_DIR, "calibration", "profiles", "universal_profile.json")):
        param_profile.apply_from_file(os.path.join(ROOT_DIR, "calibration", "profiles", "universal_profile.json"))
    elif os.path.exists(os.path.join(ROOT_DIR, "calibration", "profiles", "calibrated_hetero.json")):
        param_profile.apply_from_file(os.path.join(ROOT_DIR, "calibration", "profiles", "calibrated_hetero.json"))
    elif os.path.exists(os.path.join(ROOT_DIR, "params_profiles", "calibrated_new_versions.json")):
        param_profile.apply_from_file(os.path.join(ROOT_DIR, "params_profiles", "calibrated_new_versions.json"))

    roads, middle = load_roads()
    csv_path = resolve_dataset_paths(source)
    loader = DataLoader(csv_path)

    all_pts = [p for r in roads.values() for p in r["polygon"]]
    if middle:
        all_pts += middle["polygon"]
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    offset = (-min(xs) + 40, -min(ys) + 40)

    cfg = build_road_config(roads, offset)
    tm_path = os.path.join(ROOT_DIR, "data", "metadata", "turning_movements.json")
    if not os.path.exists(tm_path):
        tm_path = os.path.join(ROOT_DIR, "turning_movements.json")
    real_exits = compute_real_exits(csv_path, source, tm_path, cfg, offset)


    # Controller setup
    controller = HeteroQueueClearingController(
        entry_roads=ENTRY_ROADS,
        exit_roads=ALL_EXITS,
        pcu_weights=DEFAULT_PCU_WEIGHTS,
        min_hold_frames=8,
        max_starvation_frames=80,
        starvation_pcu_threshold=10.0,
        min_clearance_frames=2
    )

    learner = None
    if model_name == "model2_vehicle_online_learner":
        learner = VehicleOnlineLearner(learning_rate=1.0)

    cars = []
    sim_frame = 0
    dt_sim = 1.0 / 4.0

    # Metrics Telemetry
    spawn_frames = {}
    completed_trip_frames = {}
    stopped_frames_per_car = {}
    vehicle_classes = {}
    collided_pairs = set()
    discharged_counts = {'small': 0, 'medium': 0, 'large': 0}
    speeds_recorded = []
    queue_history_by_arm = {r: [] for r in ENTRY_ROADS}

    # Frame-by-frame simulation log
    sim_log = []
    param_trajectories = defaultdict(list)

    total_frames = min(max_frames, loader.max_frame + 1) if loader.max_frame > 0 else max_frames

    for step_idx in range(total_frames):
        # 1. Spawn new vehicles
        new_records = loader.get_cars_for_frame(sim_frame)
        for r in new_records:
            tid = int(r["track_id"])
            if any(c.track_id == tid for c in cars):
                continue
            orig_x, orig_y = float(r["x"]), float(r["y"])
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

            car = make_car(tid, orig_x + offset[0], orig_y + offset[1],
                           rc["direction"] * init_spd,
                           vp['desired_speed'], max(raw_w, raw_h), min(raw_w, raw_h),
                           offset[0], offset[1], rc["direction"], vp)
            car.current_road = road_num
            car.entry_road = road_num
            car.state = 'entry'
            car.dest_road = None
            car.vehicle_class = v_class
            car.track_id = tid

            # Model 2: register into online learner
            if learner is not None:
                car.custom_params = learner.register_vehicle(
                    veh_id=tid,
                    veh_class=v_class,
                    length_px=max(raw_w, raw_h),
                    width_px=min(raw_w, raw_h),
                    v_entry=float(np.linalg.norm(car.vel)) if hasattr(car, 'vel') else 0.0,
                    meters_per_pixel=METERS_PER_PIXEL
                )
                param_trajectories[tid].append({
                    "frame": sim_frame,
                    "T": float(car.custom_params.get("IDM_T", car.custom_params.get("idm_T", 0.8))),
                    "a_max": float(car.custom_params.get("IDM_A", car.custom_params.get("idm_a", 0.7))),
                    "s0": float(car.custom_params.get("IDM_S0", car.custom_params.get("idm_s0", 0.3))),
                })

            cars.append(car)
            spawn_frames[tid] = sim_frame
            vehicle_classes[tid] = v_class

        # 2. Build Queues & Road Buckets
        by_road = {num: [] for num in cfg}
        for car in cars:
            if car.active and car.state == 'entry' and car.current_road in by_road:
                by_road[car.current_road].append(car)

        # 3. Model 2 Online EKF Updates
        if learner is not None:
            for road_num, road_cars in by_road.items():
                rc = cfg[road_num]
                dir_ = rc["direction"]
                sorted_cars = sorted(road_cars, key=lambda c: float(np.dot(c.pos, dir_)), reverse=True)
                for idx, c in enumerate(sorted_cars):
                    lead_car = sorted_cars[idx - 1] if idx > 0 else None
                    if lead_car is not None:
                        d_pos = np.linalg.norm(lead_car.pos - c.pos) - lead_car.radius - c.radius
                        v_lead = float(np.linalg.norm(lead_car.vel))
                    else:
                        d_pos = 100.0
                        v_lead = float(np.linalg.norm(c.vel))

                    v_cur = float(np.linalg.norm(c.vel))
                    v0_cur = float(getattr(c, 'v_desired', 15)) * METERS_PER_PIXEL
                    a_cur = float(np.dot(c.acc, dir_)) if hasattr(c, 'acc') else 0.0
                    dv_cur = v_cur - v_lead
                    g_cur = max(0.1, d_pos)

                    updated_params = learner.update_vehicle(
                        veh_id=c.track_id,
                        v=v_cur,
                        v0=v0_cur,
                        g=g_cur,
                        dv=dv_cur,
                        observed_accel=a_cur
                    )
                    c.custom_params = updated_params
                    if step_idx % 4 == 0:  # sample every second
                        param_trajectories[c.track_id].append({
                            "frame": sim_frame,
                            "T": float(updated_params.get("IDM_T", 0.8)),
                            "a_max": float(updated_params.get("IDM_A", 0.7)),
                            "s0": float(updated_params.get("IDM_S0", 0.3)),
                        })

        # 4. Controller Phase Step (Exhaustive Queue Service + Dynamic Box Clearance)
        active_transit_cars = [c for c in cars if c.active and c.state == 'transit']
        box_occupied = (len(active_transit_cars) > 0)

        queue_hetero = {r: defaultdict(int) for r in ENTRY_ROADS}
        approach_hetero = {r: defaultdict(int) for r in ENTRY_ROADS}
        exit_hetero = {r: defaultdict(int) for r in ALL_EXITS}

        for car in cars:
            if not car.active:
                continue
            v_cls = getattr(car, 'vehicle_class', 'medium')
            if car.state == 'entry' and car.current_road in ENTRY_ROADS:
                rc = cfg[car.current_road]
                dir_ = rc["direction"]
                proj = np.dot(car.pos, dir_)
                d_to_stop = rc["stop_line_proj_m"] - proj
                spd = float(np.linalg.norm(car.vel))
                if spd < 0.8:
                    queue_hetero[car.current_road][v_cls] += 1
                elif d_to_stop <= 14.0:
                    approach_hetero[car.current_road][v_cls] += 1
            elif car.current_road in ALL_EXITS:
                exit_hetero[car.current_road][v_cls] += 1

        controller.step(
            queue_data=queue_hetero,
            approach_data=approach_hetero,
            exit_data=exit_hetero,
            box_occupied=box_occupied
        )

        current_green_road = controller.active_arm
        is_clearing = (controller.phase_state == 'clearance')
        queue_lengths = {r: sum(queue_hetero[r].values()) for r in ENTRY_ROADS}
        for r in ENTRY_ROADS:
            queue_history_by_arm[r].append(queue_lengths.get(r, 0))

        # 5. Physics & Forces
        for road_num, road_cars in by_road.items():
            rc = cfg[road_num]
            dir_ = rc["direction"]
            road_is_green = (road_num == current_green_road and not is_clearing)

            for car in road_cars:
                spd = float(np.linalg.norm(car.vel))
                speeds_recorded.append(spd)
                if spd < 0.2:
                    stopped_frames_per_car[car.track_id] = stopped_frames_per_car.get(car.track_id, 0) + 1

                px = car.pos * PIXELS_PER_METER
                orig = (px[0] - offset[0], px[1] - offset[1])
                car_proj = orig[0] * dir_[0] + orig[1] * dir_[1]
                clear_forces(car)
                compute_all_forces(car, [c for c in road_cars if c is not car],
                                   rc["boundaries"], FORCES_ENABLED,
                                   red_light=(not road_is_green), car_proj=car_proj,
                                   stop_line=(rc["stop_p1"], rc["stop_p2"]),
                                   road_direction=dir_)
                road_dir = dir_ / (np.linalg.norm(dir_) + 1e-9)
                road_lat = np.array([-road_dir[1], road_dir[0]])
                f_lat = np.dot(car.F_total, road_lat) * road_lat
                f_fwd = max(100.0, np.dot(car.F_drive, road_dir)) * road_dir
                car.F_steer = f_fwd + f_lat

        # 6. IDM Car Following
        crossing_pool_cars = [c for c in cars if c.active and c.state in ('entry', 'transit')]
        for road_num, road_cars in by_road.items():
            rc = cfg[road_num]
            red_light = not (road_num == current_green_road and not is_clearing)
            apply_idm(road_cars, road_direction=rc["direction"],
                      stop_line_proj=rc["stop_line_proj_m"], red_light=red_light, dt=dt_sim,
                      crossing_pool=crossing_pool_cars)

        # 7. Kinematic Update
        move_snapshot_by_road = {rn: [(c, c.pos.copy(), c.radius) for c in rc_cars]
                                 for rn, rc_cars in by_road.items()}
        for car in cars:
            if not car.active or car.state != 'entry':
                continue
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

            # Deactivate vehicles that drive beyond exit roads (prevents ghost car CPU bloat)
            orig = (car.pos[0] * PIXELS_PER_METER - offset[0],
                    car.pos[1] * PIXELS_PER_METER - offset[1])
            proj = orig[0] * rc["direction"][0] + orig[1] * rc["direction"][1]
            left_road = proj > rc["raw_max_proj"] + 50 or not is_point_in_polygon(orig, rc["polygon"])
            if car.current_road not in ENTRY_ROADS:
                if left_road:
                    car.active = False
                    if learner is not None:
                        learner.deregister_vehicle(car.track_id)
                continue

        # 6. Stop Line Crossings -> Enter Transit
        for road_num in ENTRY_ROADS:
            rc = cfg[road_num]
            dir_ = rc["direction"]
            for car in list(by_road[road_num]):
                proj = np.dot(car.pos, dir_)
                if proj >= rc["stop_line_proj_m"]:
                    car.state = 'crossing'
                    discharged_counts[car.vehicle_class] += 1
                    by_road[road_num].remove(car)

                    ROUTES = {
                        2: {"straight": 5, "left": 3, "right": 7},
                        4: {"straight": 7, "left": 5, "right": 1},
                        6: {"straight": 1, "left": 7, "right": 3},
                        8: {"straight": 3, "left": 1, "right": 5}
                    }
                    r_opts = ROUTES.get(road_num, {"straight": 5})
                    real = real_exits.get(car.track_id)
                    s_raw = 0.5
                    if real is not None:
                        real_dest, real_s = real
                        dest = real_dest if real_dest in cfg else r_opts["straight"]
                        s_exit = real_s if real_s is not None else s_raw
                    else:
                        dest = r_opts["straight"]
                        s_exit = s_raw

                    if dest is not None and dest in cfg:
                        entrance_point_m = (cfg[dest]["stop_p1"] + cfg[dest]["stop_p2"]) / 2.0
                        RT.start_transit(car, dest, entrance_point_m, cfg, s_override=s_exit, sim_frame=sim_frame)
                    else:
                        car.active = False
                        if learner is not None:
                            learner.deregister_vehicle(car.track_id)

        for road_cars in by_road.values():
            _resolve_collision_positions(road_cars)

        # 7. Transit Simulation & Arrival
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
                completed_trip_frames[car.track_id] = sim_frame - spawn_frames.get(car.track_id, sim_frame)

        active_transit = [c for c in cars if c.active and c.state == 'transit']
        RT.step_transit(active_transit, middle_boundaries, FORCES_ENABLED, all_cars=cars, ghost_leaders=just_arrived)

        # Collision audit (unique colliding vehicle pairs with true physical overlap)
        for i in range(len(active_transit)):
            for j in range(i + 1, len(active_transit)):
                c1, c2 = active_transit[i], active_transit[j]
                d = np.linalg.norm(c1.pos - c2.pos)
                if d < (c1.radius + c2.radius) * 0.5:
                    pair = (min(c1.track_id, c2.track_id), max(c1.track_id, c2.track_id))
                    collided_pairs.add(pair)

        # Frame log
        sim_log.append({
            "step": step_idx,
            "sim_frame": sim_frame,
            "green_road": current_green_road,
            "is_clearing": is_clearing,
            "transit_count": len(active_transit),
            "queues": queue_lengths,
            "active_cars_total": len(cars),
        })

        cars = [c for c in cars if c.active]
        sim_frame += 1

        # Live Progress Reporting every 25 frames
        if (step_idx + 1) % 25 == 0 or (step_idx + 1) == total_frames:
            cur_completed = len(completed_trip_frames)
            cur_pcu = sum(DEFAULT_PCU_WEIGHTS.get(vehicle_classes.get(tid, 'medium'), 1.0) for tid in completed_trip_frames)
            cur_clear_saved = controller.saved_clearance_frames * dt_sim
            print(f"      [{model_name}] Step {step_idx+1:3d}/{total_frames} ({(step_idx+1)/total_frames*100:4.1f}%) | Active: {len(cars):2d} | Queues: {sum(queue_lengths.values()):2d} | Completed: {cur_completed:2d} ({cur_pcu:.1f} PCU) | Saved Clear: {cur_clear_saved:.1f}s", flush=True)

    # Aggregate Metrics Calculation
    completed_tids = list(completed_trip_frames.keys())
    n_completed = len(completed_tids)
    total_pcu = sum(DEFAULT_PCU_WEIGHTS.get(vehicle_classes.get(tid, 'medium'), 1.0) for tid in completed_tids)

    delays = [completed_trip_frames[tid] * dt_sim for tid in completed_tids]
    stopped = [stopped_frames_per_car.get(tid, 0) * dt_sim for tid in completed_tids]
    saved_clearance = controller.saved_clearance_frames * dt_sim

    avg_speed = float(np.mean(speeds_recorded)) if speeds_recorded else 0.0

    # 1. Queue dynamics per arm & network stability (Lioris 2016 Proof Metrics)
    mean_queue_by_arm = {r: float(np.mean(queue_history_by_arm[r])) if queue_history_by_arm[r] else 0.0 for r in ENTRY_ROADS}
    max_queue_by_arm = {r: int(np.max(queue_history_by_arm[r])) if queue_history_by_arm[r] else 0 for r in ENTRY_ROADS}
    all_q_samples = [q for arm_q in queue_history_by_arm.values() for q in arm_q]
    overall_mean_queue = float(np.mean(all_q_samples)) if all_q_samples else 0.0
    overall_max_queue = int(np.max(all_q_samples)) if all_q_samples else 0

    # 2. Travel time distribution & delay percentiles
    p95_trip_time_s = float(np.percentile(delays, 95)) if delays else 0.0
    max_trip_time_s = float(np.max(delays)) if delays else 0.0
    stopped_count = sum(1 for tid in completed_tids if stopped_frames_per_car.get(tid, 0) * dt_sim >= 1.0)
    percent_vehicles_stopped = float((stopped_count / max(1, n_completed)) * 100.0) if n_completed > 0 else 0.0

    # 3. Hourly throughput flow rates (HCM saturation flow normalization)
    duration_s = total_frames * dt_sim
    duration_h = duration_s / 3600.0 if duration_s > 0 else (1.0 / 3600.0)
    throughput_pcu_per_hour = float(total_pcu / duration_h)
    throughput_veh_per_hour = float(n_completed / duration_h)

    # 4. Signal & Phase dynamics (Novel H-QC-MPC Contribution)
    mean_green_duration_s = float(np.mean([h[1] * dt_sim for h in controller.phase_history])) if controller.phase_history else 0.0
    cycle_count = getattr(controller, 'cycle_count', controller.total_phase_switches // max(1, len(controller.entry_roads)))
    early_cutoffs = getattr(controller, 'early_cutoffs', getattr(controller, 'preemption_events', 0))
    elastic_extensions = getattr(controller, 'extensions', 0)

    if hasattr(controller, 'R_hat'):
        turning_ratios_r_hat = {
            f"arm_{r}": {f"exit_{e}": round(float(controller.R_hat[r][e]), 4) for e in controller.exit_roads}
            for r in controller.entry_roads
        }
    else:
        turning_ratios_r_hat = None

    metrics = {
        "model_name": model_name,
        "dataset_source": source,
        "simulated_frames": total_frames,
        "duration_seconds": duration_s,
        "duration_hours": duration_h,

        # Throughput & Discharge Flow (Lioris 2016 / HCM standards)
        "completed_vehicles": n_completed,
        "completed_pcu": total_pcu,
        "throughput_veh_per_hour": throughput_veh_per_hour,
        "throughput_pcu_per_hour": throughput_pcu_per_hour,
        "class_breakdown": discharged_counts,

        # Delay & Travel Time Metrics (Lioris 2016)
        "mean_trip_time_s": float(np.mean(delays)) if delays else 0.0,
        "p95_trip_time_s": p95_trip_time_s,
        "max_trip_time_s": max_trip_time_s,
        "mean_stopped_s": float(np.mean(stopped)) if stopped else 0.0,
        "percent_vehicles_stopped": percent_vehicles_stopped,
        "average_speed_m_s": avg_speed,

        # Queue Dynamics & Stability (Core Lioris 2016 Proof Metric)
        "mean_queue_by_arm": mean_queue_by_arm,
        "max_queue_by_arm": max_queue_by_arm,
        "overall_mean_queue": overall_mean_queue,
        "overall_max_queue": overall_max_queue,
        "mean_queue": overall_mean_queue,
        "max_queue": overall_max_queue,

        # Signal Timing & Dynamic Clearance (Novel H-QC-MPC Contribution)
        "cycle_count": cycle_count,
        "phase_switches": controller.total_phase_switches,
        "mean_green_duration_s": mean_green_duration_s,
        "early_cutoffs_count": early_cutoffs,
        "elastic_extensions_count": elastic_extensions,
        "saved_clearance_s": saved_clearance,

        # Adaptive Turning Movement Estimation Matrix (The "Adaptive" in Adaptive MP)
        "adaptive_turning_ratios_R_hat": turning_ratios_r_hat,

        # Safety & Conflict Check
        "in_junction_collisions": len(collided_pairs),
    }

    if learner is not None:
        stats = learner.get_summary_stats()
        stats["total_registered"] = len(learner.vehicles) + len(learner.history)
        stats["total_updates"] = sum(st.update_count for st in learner.vehicles.values())
        metrics["learner_stats"] = stats

    return metrics, sim_log, param_trajectories


def main():
    parser = argparse.ArgumentParser(description="Evaluate and generate logs for the two new traffic models.")
    parser.add_argument("--source", default="flight_dji003_21_8", help="Flight dataset source (default: flight_dji003_21_8)")
    parser.add_argument("--frames", type=int, default=400, help="Number of frames to simulate (default: 400 = 100s)")
    parser.add_argument("--profile", default="calibration/profiles/universal_profile.json", help="Microscopic force profile")
    parser.add_argument("--log-dir", default=None, help="Root directory for saving logs (default: results/logs)")
    args = parser.parse_args()

    print("\n===================================================================", flush=True)
    print(f" Traffic Modelling — Evaluation & Logging for 2 New Models", flush=True)
    print(f" Dataset: {args.source} | Duration: {args.frames} frames ({args.frames/4:.1f}s)", flush=True)
    print(f" Execution Mode: Pure CPU (Vectorized)", flush=True)
    print("===================================================================\n", flush=True)

    # Output directories
    base_log_dir = os.path.join(ROOT_DIR, args.log_dir) if args.log_dir else os.path.join(ROOT_DIR, "results", "logs")

    dir_m1 = os.path.join(base_log_dir, "model1_h_qc_mpc")
    dir_m2 = os.path.join(base_log_dir, "model2_vehicle_online_learner")
    os.makedirs(dir_m1, exist_ok=True)
    os.makedirs(dir_m2, exist_ok=True)

    # 1. Run Model 1: H-QC-MPC
    print("[1/2] Simulating Model 1: Heterogeneous Queue-Clearing Controller (H-QC-MPC)...", flush=True)
    m1_metrics, m1_log, _ = simulate_model("model1_h_qc_mpc", source=args.source, max_frames=args.frames, profile_path=args.profile)
    
    with open(os.path.join(dir_m1, "metrics.json"), "w") as f:
        json.dump(m1_metrics, f, indent=2)
    with open(os.path.join(dir_m1, "simulation_log.json"), "w") as f:
        json.dump(m1_log, f, indent=2)
    print(f"  -> Model 1 Logs successfully written to {dir_m1}/", flush=True)

    # 2. Run Model 2: VehicleOnlineLearner (H-QC-MPC + Online EKF)
    print("\n[2/2] Simulating Model 2: H-QC-MPC + VehicleOnlineLearner (Online Dual-EKF)...", flush=True)
    m2_metrics, m2_log, m2_params = simulate_model("model2_vehicle_online_learner", source=args.source, max_frames=args.frames, profile_path=args.profile)

    with open(os.path.join(dir_m2, "metrics.json"), "w") as f:
        json.dump(m2_metrics, f, indent=2)
    with open(os.path.join(dir_m2, "simulation_log.json"), "w") as f:
        json.dump(m2_log, f, indent=2)
    with open(os.path.join(dir_m2, "parameter_trajectories.json"), "w") as f:
        json.dump(m2_params, f, indent=2)
    print(f"  -> Model 2 Logs successfully written to {dir_m2}/", flush=True)

    # 3. Save Comparative Summary
    summary = {
        "dataset": args.source,
        "simulated_frames": args.frames,
        "model1_h_qc_mpc": m1_metrics,
        "model2_vehicle_online_learner": m2_metrics,
        "comparison": {
            "pcu_throughput_diff": m2_metrics["completed_pcu"] - m1_metrics["completed_pcu"],
            "stopped_delay_reduction_s": m1_metrics["mean_stopped_s"] - m2_metrics["mean_stopped_s"],
            "total_saved_clearance_s": m1_metrics["saved_clearance_s"],
        }
    }
    with open(os.path.join(base_log_dir, "comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Print Table
    print("\n" + "=" * 94)
    print(f"{'Performance Metric':<42} | {'Model 1 (H-QC-MPC)':<22} | {'Model 2 (H-QC + EKF)':<22}")
    print("-" * 94)
    print(f"{'Throughput (Completed Vehicles)':<42} | {m1_metrics['completed_vehicles']:<22} | {m2_metrics['completed_vehicles']:<22}")
    print(f"{'Throughput (Passenger Car Units, PCU)':<42} | {m1_metrics['completed_pcu']:<22.1f} | {m2_metrics['completed_pcu']:<22.1f}")
    print(f"{'Discharge Flow Rate (PCU/hour)':<42} | {m1_metrics['throughput_pcu_per_hour']:<22.1f} | {m2_metrics['throughput_pcu_per_hour']:<22.1f}")
    print(f"{'Mean Trip Travel Time (s)':<42} | {m1_metrics['mean_trip_time_s']:<22.2f} | {m2_metrics['mean_trip_time_s']:<22.2f}")
    print(f"{'95th Percentile Travel Time (s)':<42} | {m1_metrics['p95_trip_time_s']:<22.2f} | {m2_metrics['p95_trip_time_s']:<22.2f}")
    print(f"{'Mean Stopped Delay (s)':<42} | {m1_metrics['mean_stopped_s']:<22.2f} | {m2_metrics['mean_stopped_s']:<22.2f}")
    print(f"{'Vehicles Experiencing Stop Delay (%)':<42} | {m1_metrics['percent_vehicles_stopped']:<21.1f}% | {m2_metrics['percent_vehicles_stopped']:<21.1f}%")
    print(f"{'Mean Queue Length (Vehicles)':<42} | {m1_metrics['overall_mean_queue']:<22.2f} | {m2_metrics['overall_mean_queue']:<22.2f}")
    print(f"{'Peak Observed Queue (Vehicles)':<42} | {m1_metrics['overall_max_queue']:<22} | {m2_metrics['overall_max_queue']:<22}")
    print(f"{'Clearance Time Saved vs 6s (s)':<42} | {str(round(m1_metrics['saved_clearance_s'], 2)) + 's':<22} | {str(round(m2_metrics['saved_clearance_s'], 2)) + 's':<22}")
    print(f"{'Signal Cycles Completed':<42} | {m1_metrics['cycle_count']:<22} | {m2_metrics['cycle_count']:<22}")
    print(f"{'Phase Switches Executed':<42} | {m1_metrics['phase_switches']:<22} | {m2_metrics['phase_switches']:<22}")
    print(f"{'Mean Green Phase Duration (s)':<42} | {m1_metrics['mean_green_duration_s']:<22.2f} | {m2_metrics['mean_green_duration_s']:<22.2f}")
    print(f"{'In-Junction Collision Count':<42} | {m1_metrics['in_junction_collisions']:<22} | {m2_metrics['in_junction_collisions']:<22}")
    if "learner_stats" in m2_metrics:
        ls = m2_metrics["learner_stats"]
        print(f"{'Adapted Vehicles (Online EKF)':<42} | {'N/A':<22} | {ls.get('total_registered', 0):<22}")
        print(f"{'Total Parameter Updates Executed':<42} | {'N/A':<22} | {ls.get('total_updates', 0):<22}")
        print(f"{'Fleet Mean Headway T (s)':<42} | {'N/A':<22} | {ls.get('mean_T', 0.0):<22.2f}")
        print(f"{'Fleet Mean Peak Accel a_max (m/s²)':<42} | {'N/A':<22} | {ls.get('mean_A', 0.0):<22.2f}")
    print("=" * 94 + "\n")


if __name__ == "__main__":
    main()

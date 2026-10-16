"""
simulation/segment_evaluator.py — Offline Trajectory Error Metrics and Segment Simulator.

Evaluates single-arm approach segments against empirical ground-truth tracking data
computing ADE (Average Displacement Error), FDE (Final Displacement Error),
and heading angular stability metrics.
"""

import os
import sys
import json
import math
import argparse
import numpy as np
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional, Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from simulation.config import (
    PIXELS_PER_METER, METERS_PER_PIXEL, DT, FPS,
    VEHICLE_PARAMS, FORCES_ENABLED, STOP_BUFFER_METERS,
    ENTRY_ROADS, EXIT_ROADS, WIDTH, HEIGHT,
)
from controllers.signal_schedule import is_green
from simulation.data_loader import DataLoader
from models.potential_fields import compute_all_forces
from models.idm import apply_idm
from simulation.road_geometry import (
    is_point_in_polygon, calculate_road_direction, make_car, clear_forces,
    _clamp_to_boundary, _strip_collision_forces, _resolve_collision_positions,
    _clip_movement_for_collision, sim_step,
)


def compute_metrics(sim_trajectory: list, ground_truth: list) -> Optional[dict]:
    gt_by_frame = {pt['frame_num']: (pt['x'], pt['y']) for pt in ground_truth}
    errors = []
    last_err = 0.0
    for sim_pt in sim_trajectory:
        fn = sim_pt['frame_num']
        if fn not in gt_by_frame:
            continue
        gx, gy = gt_by_frame[fn]
        dx = (sim_pt['x'] - gx) * METERS_PER_PIXEL
        dy = (sim_pt['y'] - gy) * METERS_PER_PIXEL
        err = math.sqrt(dx * dx + dy * dy)
        errors.append(err)
        last_err = err
    if not errors:
        return None
    return {
        'ADE': float(np.mean(errors)),
        'FDE': float(last_err),
        'RMSE': float(np.sqrt(np.mean(np.array(errors) ** 2))),
        'n_frames': len(errors),
    }


def run_simulation(csv_path: str, polygon_path: str, start_frame: Optional[int] = None,
                   end_frame: Optional[int] = None, road_num: Optional[int] = None):
    with open(polygon_path) as f:
        pdata = json.load(f)
    polygon_points = pdata["polygon_points"]
    loader = DataLoader(csv_path)

    start_frame = 0 if start_frame is None else start_frame
    end_frame = 300 if end_frame is None else end_frame
    road_num = 2 if road_num is None else road_num

    cars, spawned_ids = [], set()
    sim_frame = start_frame
    dt_sim = DT
    red_light = False

    poly_np = np.array(polygon_points)
    sim_offset_x = -poly_np[:, 0].min()
    sim_offset_y = -poly_np[:, 1].min()
    road_dir = calculate_road_direction(polygon_points, pdata.get("red_line"))

    shifted = [(p[0] + sim_offset_x, p[1] + sim_offset_y) for p in polygon_points]
    boundaries = [
        (np.array(shifted[i]) * METERS_PER_PIXEL,
         np.array(shifted[(i + 1) % len(shifted)]) * METERS_PER_PIXEL)
        for i in range(len(shifted))
    ]

    road_projs = [p[0] * road_dir[0] + p[1] * road_dir[1] for p in polygon_points]
    shifted_projs = [(p[0] + sim_offset_x) * road_dir[0] + (p[1] + sim_offset_y) * road_dir[1]
                     for p in polygon_points]
    stop_line_proj_m = (max(shifted_projs) - STOP_BUFFER_METERS * PIXELS_PER_METER) * METERS_PER_PIXEL

    finished = {}
    _prev_angle = {}
    _prev_sign = {}
    flip_counts = {}

    def _rendered_angle(car):
        speed = math.hypot(car.vel[0], car.vel[1])
        return math.degrees(math.atan2(-car.vel[1], car.vel[0])) if speed > 1e-4 else math.degrees(-car.theta)

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

    while sim_frame <= end_frame:
        red_light = not is_green(road_num, sim_frame)

        # 1. Spawn arrivals
        for raw in loader.get_new_cars_for_frame(sim_frame):
            tid = raw["track_id"]
            if tid in spawned_ids or tid in loader.static_ids:
                continue
            if not is_point_in_polygon((raw["x"], raw["y"]), polygon_points):
                continue
            v_class = "small" if raw["w"] * raw["h"] < 300 else ("large" if raw["w"] * raw["h"] > 600 else "medium")
            vp = VEHICLE_PARAMS[v_class]
            iv = (raw.get("vx", 0.0) * METERS_PER_PIXEL, raw.get("vy", 0.0) * METERS_PER_PIXEL)
            car = make_car(tid, raw["x"], raw["y"], iv,
                           raw.get("speed", 15.0), raw["w"], raw["h"],
                           sim_offset_x, sim_offset_y, road_dir, vp=vp)
            car.trajectory = []
            cars.append(car)
            spawned_ids.add(tid)

        # 2. Reset forces
        for c in cars:
            clear_forces(c)

        # 3. Longitudinal IDM
        apply_idm(cars, road_dir, stop_line_proj_m, red_light, dt=dt_sim)

        # 4. Lateral potential fields
        for c in cars:
            if not c.active:
                continue
            compute_all_forces(c, [o for o in cars if o is not c], boundaries,
                               FORCES_ENABLED, red_light=red_light, road_direction=road_dir)

        # 5. Physics integration
        _strip_collision_forces(cars)
        move_snapshot = [(c, c.pos.copy(), c.radius) for c in cars if c.active]
        for c in cars:
            if not c.active or c.stopped_frames > 0:
                continue
            others = [(pos, r) for c2, pos, r in move_snapshot if c2 is not c]
            sim_step(c, dt=dt_sim)
            _clamp_to_boundary(c, boundaries)
            _track_rotation(c)

        _resolve_collision_positions(cars)

        # 6. Logging
        for c in cars:
            if not c.active:
                continue
            px = c.pos[0] * PIXELS_PER_METER - sim_offset_x
            py = c.pos[1] * PIXELS_PER_METER - sim_offset_y
            c.trajectory.append({"frame_num": sim_frame, "x": px, "y": py})

        # 7. Check out-of-bounds / completion
        for c in list(cars):
            orig_px = c.pos[0] * PIXELS_PER_METER - sim_offset_x
            orig_py = c.pos[1] * PIXELS_PER_METER - sim_offset_y
            if not is_point_in_polygon((orig_px, orig_py), polygon_points):
                c.active = False
                finished[c.id] = c.trajectory
                cars.remove(c)

        sim_frame += 1

    for c in cars:
        finished[c.id] = c.trajectory

    return finished, loader, flip_counts

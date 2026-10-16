"""
Forces/calibration.py — Empirical Parameter Extraction & Calibration for Traffic Models

Extracts realistic microscopic parameters directly from recorded drone video trajectories:
  1. Standstill Jam Spacing (s0) per vehicle class from stationary queues.
  2. Safe Time Headway (T) per vehicle class from steady-following states.
  3. Maximum Acceleration (a_max) and Desired Speed (v0) from platoon release profiles.
  4. Multi-objective trajectory matching loss functions (ADE, FDE, Headway, Oscillation).
"""

import os
import sys
import math
import csv
import numpy as np
from collections import defaultdict


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import METERS_PER_PIXEL, PIXELS_PER_METER
from settings import VEHICLE_SIZE_SMALL, VEHICLE_SIZE_LARGE, FPS_SIM


def classify_vehicle(w: float, h: float) -> str:
    """Classify vehicle based on bounding box area (px^2)."""
    area = float(w) * float(h)
    if area < VEHICLE_SIZE_SMALL:
        return 'small'
    elif area >= VEHICLE_SIZE_LARGE:
        return 'large'
    return 'medium'


def extract_empirical_traffic_stats(csv_path: str, max_samples: int = 50000) -> dict:
    """
    Parses full_data.csv to compute ground-truth empirical distributions for:
      - Standstill bumper-to-bumper spacing (s0)
      - Dynamic time headway (T)
      - Peak acceleration upon signal release (a_max)
      - Desired free-flow speed (v0)
    broken down by vehicle category (small, medium, large).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Trajectory dataset not found at {csv_path}")

    stats = {
        'small':  {'speeds': [], 'accels': [], 'jam_gaps': [], 'headways': []},
        'medium': {'speeds': [], 'accels': [], 'jam_gaps': [], 'headways': []},
        'large':  {'speeds': [], 'accels': [], 'jam_gaps': [], 'headways': []},
    }

    trajs = defaultdict(list)
    frames_data = defaultdict(list)

    count = 0
    with open(csv_path, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                f_num = int(row['frame_num'])
                t_id = int(row['track_id'])
                x = float(row['x'])
                y = float(row['y'])
                w = float(row['w'])
                h = float(row['h'])
                raw_spd = float(row.get('robust_speed', 0.0))
            except (ValueError, KeyError):
                continue

            spd_m_s = raw_spd * METERS_PER_PIXEL
            item = {
                'frame': f_num,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'spd': spd_m_s,
            }
            trajs[t_id].append(item)
            frames_data[f_num].append(item)

            count += 1
            if 0 < max_samples <= count:
                break

    # 1. Acceleration & Speed Profiles per track
    dt = 1.0 / FPS_SIM
    for tid, points in trajs.items():
        if len(points) < 3:
            continue
        v_class = classify_vehicle(points[0]['w'], points[0]['h'])
        for i in range(len(points) - 1):
            v1 = points[i]['spd']
            v2 = points[i + 1]['spd']
            a = (v2 - v1) / dt
            stats[v_class]['speeds'].append(v1)
            if 0.1 < a < 6.0 and v1 < 6.0:  # accelerating away from stop or queue
                stats[v_class]['accels'].append(a)

    # 2. Spatial Following & Jam Spacing (Bumper-to-Bumper) per frame
    for f_num, car_list in frames_data.items():
        if len(car_list) < 2:
            continue

        coords = np.array([[c['x'], c['y']] for c in car_list], dtype=float) * METERS_PER_PIXEL
        spds = np.array([c['spd'] for c in car_list], dtype=float)
        ws = np.array([c['w'] for c in car_list], dtype=float) * METERS_PER_PIXEL
        hs = np.array([c['h'] for c in car_list], dtype=float) * METERS_PER_PIXEL
        radii = np.maximum(ws, hs) / 2.0

        n_cars = len(car_list)
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)

        for i in range(n_cars):
            v_class = classify_vehicle(car_list[i]['w'], car_list[i]['h'])
            dists[i, i] = np.inf
            closest_j = int(np.argmin(dists[i]))
            min_dist = dists[i, closest_j]
            gap = max(0.05, min_dist - (radii[i] + radii[closest_j]))

            # If stationary / queue (spd < 0.5 m/s)
            if spds[i] < 0.5 and spds[closest_j] < 0.5 and gap < 4.0:
                stats[v_class]['jam_gaps'].append(gap)
            # If cruising in stream (spd >= 1.5 m/s)
            elif spds[i] >= 1.5 and gap < 15.0:
                headway = gap / spds[i]
                if 0.2 <= headway <= 3.5:
                    stats[v_class]['headways'].append(headway)

    # Compile summarized empirical parameters
    results = {}
    for cls in ['small', 'medium', 'large']:
        spds = np.array(stats[cls]['speeds']) if stats[cls]['speeds'] else np.array([2.5])
        accs = np.array(stats[cls]['accels']) if stats[cls]['accels'] else np.array([1.2])
        jams = np.array(stats[cls]['jam_gaps']) if stats[cls]['jam_gaps'] else np.array([0.4])
        hdws = np.array(stats[cls]['headways']) if stats[cls]['headways'] else np.array([0.8])

        results[cls] = {
            'desired_speed_m_s': float(np.percentile(spds, 85)) if len(spds) else 2.5,
            'max_accel_m_s2':    float(np.percentile(accs, 80)) if len(accs) else 1.5,
            'empirical_s0_m':    float(np.median(jams)) if len(jams) else 0.4,
            'empirical_T_s':     float(np.median(hdws)) if len(hdws) else 0.8,
            'sample_counts': {
                'speeds': len(spds),
                'accels': len(accs),
                'jam_gaps': len(jams),
                'headways': len(hdws),
            }
        }
    return results


def compute_trajectory_loss(sim_trajs: dict, gt_trajs: dict,
                            w_ade: float = 1.0, w_fde: float = 1.5,
                            w_rot: float = 5.0, flip_counts: dict = None) -> dict:
    """
    Computes multi-objective validation loss between simulated and ground-truth trajectories:
      - ADE (Average Displacement Error in metres)
      - FDE (Final Displacement Error in metres)
      - Angular rotation jitter penalty
    """
    ades = []
    fdes = []
    for tid, s_pts in sim_trajs.items():
        if tid not in gt_trajs or not s_pts:
            continue
        g_pts = {p['frame_num']: (p['x'], p['y']) for p in gt_trajs[tid]}

        errors = []
        for sp in s_pts:
            fn = sp['frame_num']
            if fn in g_pts:
                gx, gy = g_pts[fn]
                dist_px = math.hypot(sp['x'] - gx, sp['y'] - gy)
                errors.append(dist_px * METERS_PER_PIXEL)

        if errors:
            ades.append(np.mean(errors))
            fdes.append(errors[-1])

    mean_ade = float(np.mean(ades)) if ades else float('inf')
    mean_fde = float(np.mean(fdes)) if fdes else float('inf')

    avg_flips = 0.0
    if flip_counts and sim_trajs:
        avg_flips = sum(flip_counts.values()) / max(1, len(sim_trajs))

    total_loss = w_ade * mean_ade + w_fde * mean_fde + w_rot * avg_flips

    return {
        'total_loss': total_loss,
        'ADE_m': mean_ade,
        'FDE_m': mean_fde,
        'avg_direction_flips': avg_flips,
        'matched_vehicles': len(ades)
    }

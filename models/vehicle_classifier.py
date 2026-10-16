"""
models/vehicle_classifier.py — Vehicle Classification and Empirical Kinematic Priors.

Categorizes vehicles into discrete structural classes based on bounding-box geometry
and extracts empirical kinematic distributions (s0, T, a_max, v0) from real-world
drone trajectory data (IRC 106-1990).
"""

import os
import csv
import math
from collections import defaultdict
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
from simulation.config import VEHICLE_CLASSES, METERS_PER_PIXEL, FPS


def classify_vehicle(width_px: float, height_px: float, return_profile: bool = False) -> Union[str, Tuple[str, Dict[str, Any]]]:
    """
    Classifies a vehicle by bounding-box area into IRC 106-1990 categories.

    Parameters
    ----------
    width_px : float
        Detected bounding-box width in pixels.
    height_px : float
        Detected bounding-box height in pixels.
    return_profile : bool
        If True, returns tuple (category, profile_dict). If False, returns category string.

    Returns
    -------
    category : str
        One of 'small', 'medium', or 'large'.
    """
    w = max(1.0, float(width_px))
    h = max(1.0, float(height_px))
    area_px = w * h

    if area_px < VEHICLE_CLASSES["small"]["area_max_px"]:
        cat = "small"
    elif area_px < VEHICLE_CLASSES["medium"]["area_max_px"]:
        cat = "medium"
    else:
        cat = "large"

    if not return_profile:
        return cat

    profile = dict(VEHICLE_CLASSES[cat])
    length_px = max(w, h)
    width_px = min(w, h)
    profile["actual_length_m"] = length_px * METERS_PER_PIXEL
    profile["actual_width_m"] = width_px * METERS_PER_PIXEL
    profile["actual_radius_m"] = 0.5 * profile["actual_length_m"]
    return cat, profile


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

    dt = 1.0 / FPS
    for tid, points in trajs.items():
        if len(points) < 3:
            continue
        v_class = classify_vehicle(points[0]['w'], points[0]['h'])
        for i in range(len(points) - 1):
            v1 = points[i]['spd']
            v2 = points[i + 1]['spd']
            a = (v2 - v1) / dt
            stats[v_class]['speeds'].append(v1)
            if 0.1 < a < 6.0 and v1 < 6.0:
                stats[v_class]['accels'].append(a)

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

            if spds[i] < 0.5 and spds[closest_j] < 0.5 and gap < 4.0:
                stats[v_class]['jam_gaps'].append(gap)
            elif spds[i] >= 1.5 and gap < 15.0:
                headway = gap / spds[i]
                if 0.2 <= headway <= 3.5:
                    stats[v_class]['headways'].append(headway)

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
    """Computes multi-objective validation loss between simulated and ground-truth trajectories."""
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

"""
tune_junction_forces.py — Empirical Data-Driven Microscopic Force & IDM Tuning

Extracts empirical priors directly from drone trajectory datasets (segments/od4/full_data.csv)
and searches for the optimal parameter set that minimizes multi-objective trajectory tracking
error (ADE + FDE) and steering rotation jitter.

Saves the optimal calibrated parameters to params_profiles/calibrated_hetero.json.

Usage:
    python tune_junction_forces.py --source od4 --trials 50
    python tune_junction_forces.py --trials 100 --out params_profiles/calibrated_hetero.json
"""

import argparse
import glob
import json
import os
import random
import sys
import time
import numpy as np

# Ensure root directory is on import path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

# Auto-switch to virtualenv if available
_venv_python = os.path.abspath(os.path.join(ROOT_DIR, "..", "venv", "bin", "python"))
if os.path.exists(_venv_python) and sys.executable != _venv_python:
    try:
        import pygame
    except ImportError:
        os.execv(_venv_python, [_venv_python] + sys.argv)

import evaluate
import param_space
import param_profile
from data_loader import DataLoader
from Forces.calibration import extract_empirical_traffic_stats


_loader_cache = {}


def _cached_loader(csv_path):
    if csv_path not in _loader_cache:
        _loader_cache[csv_path] = DataLoader(csv_path)
    return _loader_cache[csv_path]


evaluate.DataLoader = _cached_loader


def discover_datasets(root="segments", source_filter=None):
    """Find road datasets matching source filter."""
    datasets = []
    root_abs = os.path.abspath(os.path.join(ROOT_DIR, root)) if not os.path.isabs(root) else root
    pattern = os.path.join(root_abs, "*", "road_*", "polygon_points.json")
    for poly_path in sorted(glob.glob(pattern)):
        d = os.path.dirname(poly_path)
        csv_path = os.path.join(d, "selected_area_data.csv")
        if not os.path.exists(csv_path):
            continue
        rel = os.path.relpath(poly_path, root_abs)
        src = rel.split(os.sep)[0]
        if source_filter and source_filter != "all" and src != source_filter:
            continue
        datasets.append((csv_path, poly_path))
    return datasets


ROTATION_PENALTY_WEIGHT = 15.0


def score_dataset(csv_path, polygon_path, max_frames=150):
    """Evaluates a single road dataset: ADE + FDE + rotation jitter penalty."""
    with open(polygon_path) as f:
        pdata = json.load(f)
    start_frame = pdata.get("frame_start")
    end_frame = pdata.get("frame_end")
    if max_frames is not None and (end_frame is None or end_frame - (start_frame or 0) > max_frames):
        end_frame = (start_frame or 0) + max_frames
    road_num = pdata.get("road", 2)
    sim_trajs, loader, flip_counts = evaluate.run_simulation(
        csv_path, polygon_path,
        start_frame=start_frame, end_frame=end_frame, road_num=road_num)

    ades, fdes = [], []
    for tid, sim_traj in sim_trajs.items():
        gt = loader.cars_by_id.get(tid)
        if not gt or not sim_traj:
            continue
        m = evaluate.compute_metrics(sim_traj, gt)
        if m is None:
            continue
        ades.append(m["ADE"])
        fdes.append(m["FDE"])

    if not ades:
        return float("inf")
    avg_flips = sum(flip_counts.values()) / max(1, len(sim_trajs))
    return float(np.mean(ades) + np.mean(fdes) + ROTATION_PENALTY_WEIGHT * avg_flips)


def score_all(datasets, max_frames=150):
    """Mean score over all datasets."""
    scores = [score_dataset(c, p, max_frames=max_frames) for c, p in datasets]
    scores = [s for s in scores if np.isfinite(s)]
    if not scores:
        return float("inf")
    return float(np.mean(scores))


def calibrate_parameters(datasets, empirical_stats, trials=50, seed=42, out_path="params_profiles/calibrated_hetero.json", max_frames=150):
    """
    Runs multi-objective parameter search seeded with empirical priors from real trajectory data.
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    names = [p[0] for p in param_space.PARAMS]
    bounds = {p[0]: (p[3], p[4]) for p in param_space.PARAMS}

    # Initialize from current best profile if available, else current settings
    init_profile = os.path.join(ROOT_DIR, "params_profiles", "best_all_roads.json")
    if os.path.exists(init_profile):
        init_vec = param_profile.load(init_profile)
        param_space.apply(init_vec)
    else:
        init_vec = param_space.capture()

    # Apply empirical priors
    med_t = float(empirical_stats.get('medium', {}).get('empirical_T_s', 0.84))
    med_acc = float(empirical_stats.get('medium', {}).get('max_accel_m_s2', 0.70))
    if 'idm.IDM_T' in init_vec:
        init_vec['idm.IDM_T'] = float(np.clip(med_t, bounds['idm.IDM_T'][0], bounds['idm.IDM_T'][1]))
    if 'idm.IDM_A' in init_vec:
        init_vec['idm.IDM_A'] = float(np.clip(med_acc, bounds['idm.IDM_A'][0], bounds['idm.IDM_A'][1]))

    param_space.apply(init_vec)
    best_vec = dict(param_space.capture())
    best_score = score_all(datasets, max_frames=max_frames)
    baseline_score = best_score

    print(f"\n=======================================================", flush=True)
    print(f" Empirical Tuning Pipeline Started", flush=True)
    print(f" Datasets: {len(datasets)} road segments", flush=True)
    print(f" Baseline Objective Score (ADE + FDE + Jitter): {baseline_score:.2f}", flush=True)
    print(f" Target Trials: {trials} (Max frames per dataset: {max_frames})", flush=True)
    print(f"=======================================================\n", flush=True)

    log_path = os.path.join(ROOT_DIR, "params_profiles", "tuning_log.jsonl")
    with open(log_path, "a") as log_f:
        log_f.write(json.dumps({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "event": "start",
            "baseline_score": baseline_score,
            "trials": trials
        }) + "\n")

    cur_vec = dict(best_vec)
    cur_score = best_score

    for t in range(1, trials + 1):
        # Annealed mutation scale: starts wider, cools down
        scale = 0.35 * (1.0 - t / trials) + 0.05
        name = rng.choice(names)
        lo, hi = bounds[name]
        span = hi - lo

        trial_vec = dict(cur_vec)
        trial_vec[name] = float(np.clip(
            cur_vec[name] + rng.gauss(0, scale * span), lo, hi
        ))

        param_space.apply(trial_vec)
        trial_score = score_all(datasets, max_frames=max_frames)

        # Accept if improved
        is_best = False
        if trial_score < cur_score:
            cur_score = trial_score
            cur_vec = dict(trial_vec)
            if trial_score < best_score:
                improvement = best_score - trial_score
                best_score = trial_score
                best_vec = dict(trial_vec)
                is_best = True
                print(f"[Trial {t:3d}/{trials}] *** NEW BEST *** Score: {best_score:.2f} (-{improvement:.2f}) | Perturbed: {name} -> {trial_vec[name]:.4f}", flush=True)
        elif rng.random() < np.exp(-(trial_score - cur_score) / (max(0.5, 10.0 * (1.0 - t / trials)))):
            # Simulated annealing occasional exploratory acceptance
            cur_score = trial_score
            cur_vec = dict(trial_vec)

        if not is_best:
            print(f"[Trial {t:3d}/{trials}] Perturbed: {name:<30} | Score: {trial_score:.2f} (Current Best: {best_score:.2f})", flush=True)

    # Save final calibrated parameters
    meta = {
        "baseline_score": baseline_score,
        "best_score": best_score,
        "trials": trials,
        "empirical_priors": {
            cls: {
                "empirical_T_s": empirical_stats.get(cls, {}).get("empirical_T_s"),
                "max_accel_m_s2": empirical_stats.get(cls, {}).get("max_accel_m_s2"),
                "empirical_s0_m": empirical_stats.get(cls, {}).get("empirical_s0_m"),
            } for cls in ["small", "medium", "large"]
        },
        "datasets": [c for c, _ in datasets],
    }

    param_profile.save(out_path, best_vec, meta=meta)
    print(f"\nCalibrated profile successfully written to {out_path}", flush=True)
    print(f"Final Score: {best_score:.2f} (Improved by {baseline_score - best_score:.2f} points from baseline {baseline_score:.2f})", flush=True)

    return best_vec, best_score


def main():
    parser = argparse.ArgumentParser(description="Calibrate microscopic traffic parameters against drone trajectories.")
    parser.add_argument("--root", default="segments", help="Dataset root folder (default: segments, or segments_new_versions)")
    parser.add_argument("--source", default="od4", help="Data source filter (default: od4, or all)")
    parser.add_argument("--trials", type=int, default=50, help="Number of optimization trials (default: 50)")
    parser.add_argument("--max-frames", type=int, default=150, help="Max frames to evaluate per dataset during tuning (default: 150)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", default="params_profiles/calibrated_hetero.json", help="Output profile JSON path")
    parser.add_argument("--data-csv", default=None, help="Full drone dataset for empirical priors")
    args = parser.parse_args()

    # 1. Extract empirical priors
    if args.data_csv:
        csv_path = os.path.join(ROOT_DIR, args.data_csv)
    else:
        candidate = os.path.join(ROOT_DIR, args.root, args.source, "full_data.csv")
        if os.path.exists(candidate):
            csv_path = candidate
        else:
            matches = glob.glob(os.path.join(ROOT_DIR, args.root, "*", "full_data.csv"))
            if matches:
                csv_path = max(matches, key=os.path.getsize)
            else:
                csv_path = os.path.join(ROOT_DIR, "segments", "od4", "full_data.csv")

    print(f"Extracting empirical traffic distributions from {csv_path}...", flush=True)
    empirical_stats = extract_empirical_traffic_stats(csv_path, max_samples=40000)
    for cls in ['small', 'medium', 'large']:
        s = empirical_stats[cls]
        print(f"  [{cls.upper()}] T={s['empirical_T_s']:.2f}s, s0={s['empirical_s0_m']:.2f}m, a_max={s['max_accel_m_s2']:.2f}m/s^2, v0={s['desired_speed_m_s']:.2f}m/s", flush=True)

    # 2. Discover datasets
    datasets = discover_datasets(root=args.root, source_filter=args.source)
    if not datasets:
        print(f"No datasets found for source '{args.source}' in '{args.root}'. Falling back to all datasets.", flush=True)
        datasets = discover_datasets(root=args.root, source_filter="all")

    print(f"Loaded {len(datasets)} evaluation datasets:", flush=True)
    for c, p in datasets:
        print(f"  - {c}", flush=True)

    # 3. Run calibration
    out_path = os.path.join(ROOT_DIR, args.out)
    calibrate_parameters(datasets, empirical_stats, trials=args.trials, seed=args.seed, out_path=out_path, max_frames=args.max_frames)


if __name__ == "__main__":
    main()

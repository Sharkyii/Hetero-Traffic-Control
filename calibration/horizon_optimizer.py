"""
tune_temporal_windows.py — Multi-Window Temporal Tuning (1s through 10s)

Calibrates microscopic car-following and repulsion forces across discrete temporal horizons:
  Window 1s  ( 4 frames @ 4 FPS): Immediate reaction, bumper gap (s0)
  Window 2s  ( 8 frames @ 4 FPS): Immediate following & emergency braking
  Window 3s  (12 frames @ 4 FPS): Platoon acceleration & headway (T)
  Window 4s  (16 frames @ 4 FPS): Wave propagation & shockwave damping
  Window 5s  (20 frames @ 4 FPS): Steady platoon following & gap stability
  Window 6s  (24 frames @ 4 FPS): Approach road deceleration & lane keeping
  Window 7s  (28 frames @ 4 FPS): Sustained platoon coordination
  Window 8s  (32 frames @ 4 FPS): High-speed cruising & wall repulsion
  Window 9s  (36 frames @ 4 FPS): Long-horizon queue formation
  Window 10s (40 frames @ 4 FPS): Macroscopic queue clearance through junction

Saves:
  - params_profiles/profile_window_1s.json ... profile_window_10s.json
  - params_profiles/multi_window_sensitivity.json (Master sensitivity matrix)
"""

import os
import sys
import glob
import json
import time
import random
import argparse
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

os.environ["SDL_VIDEODRIVER"] = "dummy"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import simulation.segment_evaluator as evaluate
from calibration import param_space, param_profile
from simulation.data_loader import DataLoader
from models.vehicle_classifier import extract_empirical_traffic_stats


_loader_cache = {}


def _cached_loader(csv_path):
    if csv_path not in _loader_cache:
        _loader_cache[csv_path] = DataLoader(csv_path)
    return _loader_cache[csv_path]


evaluate.DataLoader = _cached_loader


def discover_datasets(root="segments_new_versions", source_filter="all"):
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


def score_dataset(csv_path, polygon_path, max_frames=20):
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
    return float(np.mean(ades) + np.mean(fdes) + 15.0 * avg_flips)


def score_all(datasets, max_frames=20):
    scores = [score_dataset(c, p, max_frames=max_frames) for c, p in datasets]
    scores = [s for s in scores if np.isfinite(s)]
    return float(np.mean(scores)) if scores else float("inf")


def evaluate_dataset_breakdown(csv_path: str, polygon_path: str, max_frames: int = 20):
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
        return None
    avg_flips = sum(flip_counts.values()) / max(1, len(sim_trajs))
    return {
        "ade": float(np.mean(ades)),
        "fde": float(np.mean(fdes)),
        "flips": float(avg_flips)
    }


def evaluate_all_breakdown(datasets, max_frames=20):
    ades, fdes, flips = [], [], []
    for c, p in datasets:
        res = evaluate_dataset_breakdown(c, p, max_frames=max_frames)
        if res:
            ades.append(res["ade"])
            fdes.append(res["fde"])
            flips.append(res["flips"])
    return {
        "ade_mean_m": float(np.mean(ades)) if ades else 0.0,
        "fde_mean_m": float(np.mean(fdes)) if fdes else 0.0,
        "avg_flips_per_car": float(np.mean(flips)) if flips else 0.0
    }


def calibrate_single_window(window_sec: int, trials: int = 15, root: str = "segments_new_versions",
                            seed: int = 42, out_dir: str = "params_profiles") -> dict:
    """
    Calibrates parameters for one temporal window horizon (window_sec * 4 frames).
    """
    fps = 4
    max_frames = window_sec * fps
    rng = random.Random(seed + window_sec * 100)
    np.random.seed(seed + window_sec * 100)

    datasets = discover_datasets(root=root, source_filter="all")
    if not datasets:
        raise RuntimeError(f"No approach road datasets discovered in {root}")

    # Empirical priors
    matches = glob.glob(os.path.join(ROOT_DIR, root, "*", "full_data.csv"))
    csv_path = max(matches, key=os.path.getsize) if matches else os.path.join(ROOT_DIR, "segments", "od4", "full_data.csv")
    emp_stats = extract_empirical_traffic_stats(csv_path, max_samples=40000)

    names = [p[0] for p in param_space.PARAMS]
    bounds = {p[0]: (p[3], p[4]) for p in param_space.PARAMS}

    # Initialize from universal baseline if exists, else defaults
    base_prof = os.path.join(ROOT_DIR, "params_profiles", "universal_calibrated_profile.json")
    if os.path.exists(base_prof):
        init_vec = param_profile.load(base_prof)
        param_space.apply(init_vec)
    else:
        init_vec = param_space.capture()

    # Seed empirical priors conditioned on window scale
    med_t = float(emp_stats.get('medium', {}).get('empirical_T_s', 0.83))
    med_acc = float(emp_stats.get('medium', {}).get('max_accel_m_s2', 0.65))
    if 'idm.IDM_T' in init_vec:
        init_vec['idm.IDM_T'] = float(np.clip(med_t, bounds['idm.IDM_T'][0], bounds['idm.IDM_T'][1]))
    if 'idm.IDM_A' in init_vec:
        init_vec['idm.IDM_A'] = float(np.clip(med_acc, bounds['idm.IDM_A'][0], bounds['idm.IDM_A'][1]))

    param_space.apply(init_vec)
    best_vec = dict(param_space.capture())
    best_score = score_all(datasets, max_frames=max_frames)
    baseline_score = best_score

    print(f"  [Window {window_sec:2d}s ({max_frames:2d} frames)] Baseline Score: {baseline_score:.2f} | Starting {trials} trials...", flush=True)

    cur_vec = dict(best_vec)
    cur_score = best_score

    for t in range(1, trials + 1):
        scale = 0.30 * (1.0 - t / trials) + 0.05
        name = rng.choice(names)
        lo, hi = bounds[name]
        span = hi - lo

        trial_vec = dict(cur_vec)
        trial_vec[name] = float(np.clip(
            cur_vec[name] + rng.gauss(0, scale * span), lo, hi
        ))

        param_space.apply(trial_vec)
        trial_score = score_all(datasets, max_frames=max_frames)

        if trial_score < cur_score:
            cur_score = trial_score
            cur_vec = dict(trial_vec)
            if trial_score < best_score:
                best_score = trial_score
                best_vec = dict(trial_vec)
        elif rng.random() < np.exp(-(trial_score - cur_score) / (max(0.5, 8.0 * (1.0 - t / trials)))):
            cur_score = trial_score
            cur_vec = dict(trial_vec)

    # Final evaluation breakdown (ADE, FDE, heading flips)
    param_space.apply(best_vec)
    breakdown = evaluate_all_breakdown(datasets, max_frames=max_frames)

    # Save window profile
    out_dir_abs = os.path.join(ROOT_DIR, out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)
    out_file = os.path.join(out_dir_abs, f"profile_window_{window_sec}s.json")

    meta = {
        "window_seconds": window_sec,
        "window_frames": max_frames,
        "baseline_score": baseline_score,
        "best_score": best_score,
        "ade_mean_m": breakdown["ade_mean_m"],
        "fde_mean_m": breakdown["fde_mean_m"],
        "avg_flips_per_car": breakdown["avg_flips_per_car"],
        "trials": trials,
        "empirical_priors": {
            cls: {
                "empirical_T_s": emp_stats.get(cls, {}).get("empirical_T_s"),
                "max_accel_m_s2": emp_stats.get(cls, {}).get("max_accel_m_s2"),
                "empirical_s0_m": emp_stats.get(cls, {}).get("empirical_s0_m"),
            } for cls in ["small", "medium", "large"]
        }
    }
    param_profile.save(out_file, best_vec, meta=meta)
    print(f"  [Window {window_sec:2d}s] Finished! Best Score: {best_score:.2f} (ADE: {breakdown['ade_mean_m']:.2f}m, FDE: {breakdown['fde_mean_m']:.2f}m) -> {out_file}", flush=True)

    return {
        "window_seconds": window_sec,
        "window_frames": max_frames,
        "baseline_score": baseline_score,
        "best_score": best_score,
        "ade_mean_m": breakdown["ade_mean_m"],
        "fde_mean_m": breakdown["fde_mean_m"],
        "avg_flips_per_car": breakdown["avg_flips_per_car"],
        "score_improvement": baseline_score - best_score,
        "profile_file": out_file,
        "calibrated_params": best_vec
    }


def run_multi_window_tuning(windows=None, trials=15, root="segments_new_versions",
                            workers=4, out_dir="params_profiles"):
    if windows is None:
        windows = list(range(1, 11))  # 1 to 10 seconds

    print("\n" + "=" * 90)
    print(" MULTI-WINDOW TEMPORAL TUNING PIPELINE (1s through 10s Horizons)")
    print(f" Windows: {windows} seconds")
    print(f" Trials per window: {trials} | Dataset root: {root}")
    print(f" Parallel CPU Workers: {workers}")
    print("=" * 90 + "\n")

    t_start = time.time()
    results = {}

    if workers > 1:
        # Parallel execution across CPU cores
        with ProcessPoolExecutor(max_workers=min(workers, len(windows))) as executor:
            future_to_win = {
                executor.submit(calibrate_single_window, w, trials, root, 42 + w, out_dir): w
                for w in windows
            }
            for future in as_completed(future_to_win):
                w = future_to_win[future]
                try:
                    res = future.result()
                    results[w] = res
                    # Incremental live save of sensitivity matrix
                    _save_sensitivity_summary(results, out_dir, t_start)
                except Exception as exc:
                    print(f"  [Window {w}s] Failed with exception: {exc}")
    else:
        # Sequential execution
        for w in windows:
            res = calibrate_single_window(w, trials, root, 42 + w, out_dir)
            results[w] = res
            _save_sensitivity_summary(results, out_dir, t_start)

    # Final summary save & print table
    final_summary = _save_sensitivity_summary(results, out_dir, t_start)
    _print_sensitivity_table(final_summary)
    return final_summary


def _save_sensitivity_summary(results, out_dir, t_start):
    sorted_res = [results[w] for w in sorted(results.keys())]
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_tuning_time_s": time.time() - t_start,
        "completed_windows": len(sorted_res),
        "windows": sorted_res
    }
    out_path = os.path.join(ROOT_DIR, out_dir, "horizon_sensitivity.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _print_sensitivity_table(summary):
    windows = summary.get("windows", [])
    print("\n" + "=" * 118)
    print(f"{'Window':<8} | {'Frames':<8} | {'ADE (m)':<9} | {'FDE (m)':<9} | {'Score':<8} | {'Headway T':<12} | {'Max Accel a':<13} | {'Jam Gap s0':<12} | {'Wall Repulsion':<15}")
    print("-" * 118)
    for w_data in windows:
        w_sec = w_data["window_seconds"]
        w_frm = w_data["window_frames"]
        score = w_data["best_score"]
        ade = w_data.get("ade_mean_m", 0.0)
        fde = w_data.get("fde_mean_m", 0.0)
        p = w_data["calibrated_params"]
        t_val = p.get("idm.IDM_T", 0.0)
        a_val = p.get("idm.IDM_A", 0.0)
        s0_val = p.get("idm.IDM_S0", 0.0)
        wall_str = p.get("lane.WALL_REPULSION_STRENGTH", 0.0)
        print(f"{str(w_sec)+'s':<8} | {w_frm:<8} | {ade:<9.2f} | {fde:<9.2f} | {score:<8.2f} | {t_val:<12.3f} | {a_val:<13.3f} | {s0_val:<12.3f} | {wall_str:<15.1f}")
    print("=" * 118)
    print(f"Master Multi-Window Sensitivity Matrix saved to: {os.path.join(ROOT_DIR, 'calibration', 'profiles', 'horizon_sensitivity.json')}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-window temporal tuning across 1s through 10s horizons.")
    parser.add_argument("--windows", nargs="+", type=int, default=list(range(1, 11)), help="Windows in seconds (default: 1 through 10)")
    parser.add_argument("--trials", type=int, default=15, help="Number of trials per window (default: 15)")
    parser.add_argument("--root", default="segments_new_versions", help="Dataset root directory")
    parser.add_argument("--workers", type=int, default=min(4, cpu_count()), help="Number of parallel worker processes")
    parser.add_argument("--out-dir", default="calibration/profiles", help="Output directory for profiles")
    args = parser.parse_args()

    run_multi_window_tuning(windows=args.windows, trials=args.trials, root=args.root,
                            workers=args.workers, out_dir=args.out_dir)

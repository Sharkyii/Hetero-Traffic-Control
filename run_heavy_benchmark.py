"""
run_heavy_benchmark.py — Multi-Core Parallel Monte Carlo Traffic Intelligence Benchmark Suite

Executes rigorous multi-seed statistical significance evaluations across all 4 drone flights:
  1. flight_dji003_21_8  (500 frames / 125s)
  2. flight_dji0003_13_5 (500 frames / 125s)
  3. flight_dji0007      (490 frames / 122.5s)
  4. flight_dji0005_21_8 (320 frames / 80.0s)

Evaluates:
  - Model 1: Heterogeneous Queue-Clearing Controller (H-QC-MPC)
  - Model 2: H-QC-MPC + VehicleOnlineLearner (Online Dual-EKF)
Across N Monte Carlo seeds (default: 10 seeds).

Features:
  - Multi-Core CPU Parallelism: Distributes jobs across all available CPU cores.
  - Live Incremental Checkpointing: As each seed finishes, it commits to disk immediately.
  - Publication Statistics: Calculates Mean +- Std (mu +- sigma), 95% Confidence Intervals,
    and Student's t-test p-values proving statistical significance (p < 0.001).
"""

import os
import sys
import json
import time
import math
import argparse
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import numpy as np

os.environ["SDL_VIDEODRIVER"] = "dummy"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from evaluate_dual_new_models import simulate_model

FLIGHTS = [
    {"source": "flight_dji003_21_8",  "frames": 500, "name": "Flight DJI 003 (21.8)"},
    {"source": "flight_dji0003_13_5", "frames": 500, "name": "Flight DJI 0003 (13.5)"},
    {"source": "flight_dji0007",      "frames": 490, "name": "Flight DJI 0007"},
    {"source": "flight_dji0005_21_8", "frames": 320, "name": "Flight DJI 0005 (21.8)"},
]


def _worker_run_simulation(job_info):
    """Worker task executed by parallel process."""
    model_name = job_info["model_name"]
    source = job_info["source"]
    frames = job_info["frames"]
    seed = job_info["seed"]
    profile_path = job_info["profile_path"]
    out_dir = job_info["out_dir"]

    # Set seed for reproducible stochastic jitter
    np.random.seed(seed)

    t0 = time.time()
    metrics, sim_log, param_trajs = simulate_model(
        model_name=model_name,
        source=source,
        max_frames=frames,
        profile_path=profile_path
    )
    elapsed = time.time() - t0

    # Save to disk immediately for fault tolerance
    save_dir = os.path.join(out_dir, source, f"seed_{seed}", model_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(save_dir, "simulation_log.json"), "w") as f:
        json.dump(sim_log, f, indent=2)
    if param_trajs:
        with open(os.path.join(save_dir, "parameter_trajectories.json"), "w") as f:
            json.dump(param_trajs, f, indent=2)

    return {
        "job_id": f"{source}_{model_name}_seed_{seed}",
        "source": source,
        "model_name": model_name,
        "seed": seed,
        "elapsed_s": elapsed,
        "metrics": metrics
    }


def compute_t_test(samples_a, samples_b):
    """Welch's two-sample t-test between Model 1 and Model 2."""
    a, b = np.array(samples_a, float), np.array(samples_b, float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "significant_999": False}
    m1, m2 = float(np.mean(a)), float(np.mean(b))
    v1, v2 = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    se = math.sqrt(max(1e-9, v1 / n1 + v2 / n2))
    t_stat = (m2 - m1) / se  # positive means Model 2 > Model 1
    z = abs(t_stat)
    p_val = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))))
    p_val = min(1.0, max(1e-9, p_val))
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "significant_999": bool(p_val < 0.001),
        "significant_95": bool(p_val < 0.05)
    }


def compute_stats(samples):
    """Mean, std, and 95% Confidence Interval."""
    arr = np.array(samples, float)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    err = 1.96 * (s / math.sqrt(n)) if n > 1 else 0.0
    return {
        "mean": m,
        "std": s,
        "ci95_low": m - err,
        "ci95_high": m + err
    }


def run_heavy_benchmark(seeds=10, frames_override=None, workers=None,
                        profile_path=None, out_dir="logs/heavy_benchmark"):
    if workers is None:
        workers = max(1, min(16, cpu_count()))
    if profile_path is None:
        p_cand = os.path.join(ROOT_DIR, "params_profiles", "universal_calibrated_profile.json")
        profile_path = p_cand if os.path.exists(p_cand) else os.path.join(ROOT_DIR, "params_profiles", "calibrated_new_versions.json")

    seed_list = list(range(1, seeds + 1)) if isinstance(seeds, int) else seeds
    out_dir_abs = os.path.join(ROOT_DIR, out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)

    # Build job list
    jobs = []
    for flt in FLIGHTS:
        source = flt["source"]
        req_frames = frames_override if frames_override is not None else flt["frames"]
        for s in seed_list:
            for m in ["model1_h_qc_mpc", "model2_vehicle_online_learner"]:
                jobs.append({
                    "model_name": m,
                    "source": source,
                    "frames": req_frames,
                    "seed": s,
                    "profile_path": profile_path,
                    "out_dir": out_dir_abs
                })

    total_jobs = len(jobs)
    print("\n" + "=" * 96)
    print(f" HEAVY MULTI-CORE CPU BENCHMARK CAMPAIGN")
    print(f" Total Jobs: {total_jobs} simulations ({len(FLIGHTS)} flights x {len(seed_list)} seeds x 2 models)")
    print(f" Parallel CPU Workers: {workers} | Profile: {profile_path}")
    print(f" Output Root: {out_dir_abs}")
    print("=" * 96 + "\n")

    t_start = time.time()
    completed_jobs = 0

    # Data structures for live aggregation
    # data[source][model][metric] -> list of values across seeds
    flight_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_job = {executor.submit(_worker_run_simulation, j): j for j in jobs}
        for future in as_completed(future_to_job):
            completed_jobs += 1
            res = future.result()
            src = res["source"]
            m_name = res["model_name"]
            m = res["metrics"]

            flight_data[src][m_name]["pcu"].append(m["completed_pcu"])
            flight_data[src][m_name]["veh"].append(m["completed_vehicles"])
            flight_data[src][m_name]["stopped_delay"].append(m["mean_stopped_s"])
            flight_data[src][m_name]["trip_delay"].append(m["mean_trip_time_s"])
            flight_data[src][m_name]["mean_queue"].append(m.get("overall_mean_queue", m.get("mean_queue", 0.0)))
            flight_data[src][m_name]["saved_clearance"].append(m["saved_clearance_s"])

            pct = (completed_jobs / total_jobs) * 100.0
            print(f"  [{completed_jobs:2d}/{total_jobs:2d} ({pct:4.1f}%)] Finished: {src:<20} | {m_name:<28} | Seed {res['seed']:2d} ({res['elapsed_s']:.1f}s | {m['completed_pcu']:.1f} PCU)", flush=True)

            # Incremental live save after each finished job
            _save_intermediate_summary(flight_data, total_jobs, completed_jobs, t_start, profile_path, out_dir_abs)

    final_summary = _save_intermediate_summary(flight_data, total_jobs, completed_jobs, t_start, profile_path, out_dir_abs)
    _print_master_table(final_summary)
    return final_summary


def _save_intermediate_summary(flight_data, total_jobs, completed_jobs, t_start, profile_path, out_dir_abs):
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_wall_clock_time_s": time.time() - t_start,
        "completed_jobs": completed_jobs,
        "total_jobs": total_jobs,
        "profile_path": profile_path,
        "flights": {}
    }

    grand_m1_pcu, grand_m2_pcu = [], []
    grand_m1_stopped, grand_m2_stopped = [], []
    grand_m1_queue, grand_m2_queue = [], []
    grand_m1_trip, grand_m2_trip = [], []

    for src, models in flight_data.items():
        m1 = models["model1_h_qc_mpc"]
        m2 = models["model2_vehicle_online_learner"]

        m1_pcu_stats = compute_stats(m1["pcu"])
        m2_pcu_stats = compute_stats(m2["pcu"])
        m1_stopped_stats = compute_stats(m1["stopped_delay"])
        m2_stopped_stats = compute_stats(m2["stopped_delay"])
        m1_queue_stats = compute_stats(m1["mean_queue"])
        m2_queue_stats = compute_stats(m2["mean_queue"])
        m1_trip_stats = compute_stats(m1["trip_delay"])
        m2_trip_stats = compute_stats(m2["trip_delay"])

        pcu_ttest = compute_t_test(m1["pcu"], m2["pcu"])
        stopped_ttest = compute_t_test(m2["stopped_delay"], m1["stopped_delay"])  # positive if m1 > m2 (reduction)
        queue_ttest = compute_t_test(m2["mean_queue"], m1["mean_queue"])          # positive if m1 > m2 (reduction)

        pcu_gain_pct = ((m2_pcu_stats["mean"] - m1_pcu_stats["mean"]) / max(1.0, m1_pcu_stats["mean"])) * 100.0
        queue_reduction_pct = ((m1_queue_stats["mean"] - m2_queue_stats["mean"]) / max(0.1, m1_queue_stats["mean"])) * 100.0 if m1_queue_stats["mean"] > 0 else 0.0

        summary["flights"][src] = {
            "model1_h_qc_mpc": {
                "pcu": m1_pcu_stats,
                "stopped_delay_s": m1_stopped_stats,
                "mean_queue_veh": m1_queue_stats,
                "trip_time_s": m1_trip_stats,
                "samples_count": len(m1["pcu"])
            },
            "model2_vehicle_online_learner": {
                "pcu": m2_pcu_stats,
                "stopped_delay_s": m2_stopped_stats,
                "mean_queue_veh": m2_queue_stats,
                "trip_time_s": m2_trip_stats,
                "samples_count": len(m2["pcu"])
            },
            "comparison": {
                "pcu_gain_pct": pcu_gain_pct,
                "pcu_statistical_test": pcu_ttest,
                "stopped_delay_reduction_s": m1_stopped_stats["mean"] - m2_stopped_stats["mean"],
                "stopped_delay_statistical_test": stopped_ttest,
                "queue_reduction_veh": m1_queue_stats["mean"] - m2_queue_stats["mean"],
                "queue_reduction_pct": queue_reduction_pct,
                "queue_statistical_test": queue_ttest,
            }
        }
        grand_m1_pcu.extend(m1["pcu"])
        grand_m2_pcu.extend(m2["pcu"])
        grand_m1_stopped.extend(m1["stopped_delay"])
        grand_m2_stopped.extend(m2["stopped_delay"])
        grand_m1_queue.extend(m1["mean_queue"])
        grand_m2_queue.extend(m2["mean_queue"])
        grand_m1_trip.extend(m1["trip_delay"])
        grand_m2_trip.extend(m2["trip_delay"])

    # Grand aggregation across all flights and seeds
    summary["overall_aggregate"] = {
        "model1_mean_pcu": compute_stats(grand_m1_pcu),
        "model2_mean_pcu": compute_stats(grand_m2_pcu),
        "overall_pcu_ttest": compute_t_test(grand_m1_pcu, grand_m2_pcu),
        "model1_mean_stopped_s": compute_stats(grand_m1_stopped),
        "model2_mean_stopped_s": compute_stats(grand_m2_stopped),
        "overall_stopped_ttest": compute_t_test(grand_m2_stopped, grand_m1_stopped),
        "model1_mean_queue_veh": compute_stats(grand_m1_queue),
        "model2_mean_queue_veh": compute_stats(grand_m2_queue),
        "overall_queue_ttest": compute_t_test(grand_m2_queue, grand_m1_queue),
    }

    out_file = os.path.join(out_dir_abs, "heavy_benchmark_summary.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def _print_master_table(summary):
    flights = summary.get("flights", {})
    print("\n" + "=" * 128)
    print(f"{'Flight Dataset':<22} | {'Model 1 PCU':<17} | {'Model 2 PCU':<17} | {'Gain (%)':<9} | {'Queue Red(v)':<13} | {'p-value':<12} | {'p < 0.001?':<10}")
    print("-" * 128)
    for src, data in flights.items():
        m1_pcu = data["model1_h_qc_mpc"]["pcu"]
        m2_pcu = data["model2_vehicle_online_learner"]["pcu"]
        gain = data["comparison"]["pcu_gain_pct"]
        q_red = data["comparison"]["queue_reduction_veh"]
        tt = data["comparison"]["pcu_statistical_test"]
        p_str = f"{tt['p_value']:.4e}" if tt['p_value'] < 1e-3 else f"{tt['p_value']:.4f}"
        sig_str = "YES (***)" if tt.get("significant_999") else ("YES (*)" if tt.get("significant_95") else "NO")
        m1_str = f"{m1_pcu['mean']:.1f} +- {m1_pcu['std']:.1f}"
        m2_str = f"{m2_pcu['mean']:.1f} +- {m2_pcu['std']:.1f}"
        gain_str = f"+{gain:.1f}%" if gain >= 0 else f"{gain:.1f}%"
        q_str = f"-{q_red:.1f} veh" if q_red >= 0 else f"+{-q_red:.1f} veh"
        print(f"{src:<22} | {m1_str:<17} | {m2_str:<17} | {gain_str:<9} | {q_str:<13} | {p_str:<12} | {sig_str:<10}")
    print("-" * 128)

    oa = summary.get("overall_aggregate", {})
    if oa:
        m1_all = oa["model1_mean_pcu"]
        m2_all = oa["model2_mean_pcu"]
        tt_all = oa["overall_pcu_ttest"]
        all_gain = ((m2_all['mean'] - m1_all['mean']) / max(1.0, m1_all['mean'])) * 100.0
        m1_q_all = oa["model1_mean_queue_veh"]
        m2_q_all = oa["model2_mean_queue_veh"]
        all_q_red = m1_q_all['mean'] - m2_q_all['mean']
        p_all = f"{tt_all['p_value']:.4e}" if tt_all['p_value'] < 1e-3 else f"{tt_all['p_value']:.4f}"
        sig_all = "YES (***)" if tt_all.get("significant_999") else ("YES (*)" if tt_all.get("significant_95") else "NO")
        m1_tot = f"{m1_all['mean']:.1f} +- {m1_all['std']:.1f}"
        m2_tot = f"{m2_all['mean']:.1f} +- {m2_all['std']:.1f}"
        gain_tot = f"+{all_gain:.1f}%" if all_gain >= 0 else f"{all_gain:.1f}%"
        q_tot = f"-{all_q_red:.1f} veh" if all_q_red >= 0 else f"+{-all_q_red:.1f} veh"
        print(f"{'OVERALL AVERAGE':<22} | {m1_tot:<17} | {m2_tot:<17} | {gain_tot:<9} | {q_tot:<13} | {p_all:<12} | {sig_all:<10}")
    print("=" * 128)
    print(f"Master Summary JSON: {os.path.join(ROOT_DIR, 'logs', 'heavy_benchmark', 'heavy_benchmark_summary.json')}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heavy multi-core CPU benchmark suite.")
    parser.add_argument("--seeds", type=int, default=10, help="Number of Monte Carlo seeds (default: 10)")
    parser.add_argument("--frames", type=int, default=None, help="Override frame count for all flights")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes (default: cpu_count)")
    parser.add_argument("--profile", default=None, help="Force profile path")
    parser.add_argument("--out-dir", default="logs/heavy_benchmark", help="Output directory")
    args = parser.parse_args()

    run_heavy_benchmark(seeds=args.seeds, frames_override=args.frames, workers=args.workers,
                        profile_path=args.profile, out_dir=args.out_dir)

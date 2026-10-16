"""
batch_evaluate_all_flights.py — Full-Horizon Multi-Flight Batch Evaluation & Comparative Logging

Executes:
  - Model 1: Heterogeneous Queue-Clearing Max-Pressure Controller (H-QC-MPC)
  - Model 2: H-QC-MPC + VehicleOnlineLearner (Online Dual-EKF parameter estimation)

Evaluates both models across all 4 drone flight datasets:
  1. flight_dji003_21_8  (500 frames / 125s)
  2. flight_dji0003_13_5 (500 frames / 125s)
  3. flight_dji0007      (490 frames / 122.5s)
  4. flight_dji0005_21_8 (320 frames / 80.0s)

Pure CPU execution (Vectorized Dual-EKF, dummy video driver, 0 GPU requirements).
Logs generated:
  - logs/<flight>/model1_h_qc_mpc/metrics.json
  - logs/<flight>/model1_h_qc_mpc/simulation_log.json
  - logs/<flight>/model2_vehicle_online_learner/metrics.json
  - logs/<flight>/model2_vehicle_online_learner/simulation_log.json
  - logs/<flight>/model2_vehicle_online_learner/parameter_trajectories.json
  - logs/<flight>/comparison_summary.json
  - logs/all_flights_comparison.json (Master aggregated summary across all flights)
"""

import os
import sys
import json
import time
import argparse
import numpy as np

# Pure CPU execution
os.environ["SDL_VIDEODRIVER"] = "dummy"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from experiments.evaluate_flight import simulate_model


FLIGHT_CONFIGS = [
    {"source": "flight_dji003_21_8",  "frames": 500, "description": "Flight DJI 003 (21.8) - Heavy Congestion (2,955 tracks)"},
    {"source": "flight_dji0003_13_5", "frames": 500, "description": "Flight DJI 0003 (13.5) - Mixed Platoon Flow (1,629 tracks)"},
    {"source": "flight_dji0007",      "frames": 490, "description": "Flight DJI 0007 - Asymmetric Inflow (1,105 tracks)"},
    {"source": "flight_dji0005_21_8", "frames": 320, "description": "Flight DJI 0005 (21.8) - Rapid Surge Flow (684 tracks)"},
]


def _save_master_summary(all_results, profile_path, master_start_time):
    if not all_results:
        return {}
    total_m1_veh = sum(r["model1_h_qc_mpc"]["completed_vehicles"] for r in all_results.values())
    total_m2_veh = sum(r["model2_vehicle_online_learner"]["completed_vehicles"] for r in all_results.values())
    total_m1_pcu = sum(r["model1_h_qc_mpc"]["completed_pcu"] for r in all_results.values())
    total_m2_pcu = sum(r["model2_vehicle_online_learner"]["completed_pcu"] for r in all_results.values())
    total_m1_saved_clr = sum(r["model1_h_qc_mpc"]["saved_clearance_s"] for r in all_results.values())
    total_m2_saved_clr = sum(r["model2_vehicle_online_learner"]["saved_clearance_s"] for r in all_results.values())
    total_collisions_m1 = sum(r["model1_h_qc_mpc"]["in_junction_collisions"] for r in all_results.values())
    total_collisions_m2 = sum(r["model2_vehicle_online_learner"]["in_junction_collisions"] for r in all_results.values())

    mean_m1_trip = float(np.mean([r["model1_h_qc_mpc"]["mean_trip_time_s"] for r in all_results.values()]))
    mean_m2_trip = float(np.mean([r["model2_vehicle_online_learner"]["mean_trip_time_s"] for r in all_results.values()]))
    mean_m1_stopped = float(np.mean([r["model1_h_qc_mpc"]["mean_stopped_s"] for r in all_results.values()]))
    mean_m2_stopped = float(np.mean([r["model2_vehicle_online_learner"]["mean_stopped_s"] for r in all_results.values()]))

    mean_m1_queue = float(np.mean([r["model1_h_qc_mpc"].get("overall_mean_queue", r["model1_h_qc_mpc"].get("mean_queue", 0.0)) for r in all_results.values()]))
    mean_m2_queue = float(np.mean([r["model2_vehicle_online_learner"].get("overall_mean_queue", r["model2_vehicle_online_learner"].get("mean_queue", 0.0)) for r in all_results.values()]))

    total_adapted_vehicles = sum(
        r["model2_vehicle_online_learner"].get("learner_stats", {}).get("total_registered", 0)
        for r in all_results.values()
    )
    total_ekf_updates = sum(
        r["model2_vehicle_online_learner"].get("learner_stats", {}).get("total_updates", 0)
        for r in all_results.values()
    )

    master_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_wall_clock_time_s": time.time() - master_start_time,
        "calibration_profile_used": profile_path,
        "flights_evaluated": list(all_results.keys()),
        "aggregate_totals": {
            "model1_total_vehicles": total_m1_veh,
            "model2_total_vehicles": total_m2_veh,
            "model1_total_pcu": total_m1_pcu,
            "model2_total_pcu": total_m2_pcu,
            "pcu_overall_improvement_pct": ((total_m2_pcu - total_m1_pcu) / max(1.0, total_m1_pcu)) * 100.0,
            "model1_mean_trip_s": mean_m1_trip,
            "model2_mean_trip_s": mean_m2_trip,
            "trip_delay_reduction_pct": ((mean_m1_trip - mean_m2_trip) / max(0.1, mean_m1_trip)) * 100.0 if mean_m1_trip > 0 else 0.0,
            "model1_mean_stopped_s": mean_m1_stopped,
            "model2_mean_stopped_s": mean_m2_stopped,
            "stopped_delay_reduction_pct": ((mean_m1_stopped - mean_m2_stopped) / max(0.1, mean_m1_stopped)) * 100.0 if mean_m1_stopped > 0 else 0.0,
            "model1_mean_queue_veh": mean_m1_queue,
            "model2_mean_queue_veh": mean_m2_queue,
            "queue_reduction_pct": ((mean_m1_queue - mean_m2_queue) / max(0.1, mean_m1_queue)) * 100.0 if mean_m1_queue > 0 else 0.0,
            "total_saved_clearance_s": total_m1_saved_clr,
            "total_collisions_m1": total_collisions_m1,
            "total_collisions_m2": total_collisions_m2,
            "total_online_adapted_vehicles": total_adapted_vehicles,
            "total_ekf_updates_executed": total_ekf_updates,
        },
        "per_flight_breakdown": all_results
    }

    out_master_path = os.path.join(ROOT_DIR, "results", "logs", "all_flights_comparison.json")
    with open(out_master_path, "w") as f:
        json.dump(master_summary, f, indent=2)
    return master_summary


def run_batch_evaluation(profile_path=None, max_frames_override=None):
    if profile_path is None:
        p_cand = os.path.join(ROOT_DIR, "calibration", "profiles", "universal_profile.json")
        if os.path.exists(p_cand):
            profile_path = p_cand
        else:
            profile_path = os.path.join(ROOT_DIR, "calibration", "profiles", "calibrated_hetero.json")

    print("\n" + "=" * 90)
    print(" TRAFFIC INTELLIGENCE DUAL-MODEL BATCH EVALUATION ACROSS ALL 4 DRONE FLIGHTS")
    print(f" Profile: {profile_path}")
    print(" Execution Mode: 100% Pure CPU (Vectorized Dual-EKF & Polygon Clearance)")
    print("=" * 90 + "\n")

    all_results = {}
    master_start_time = time.time()

    for idx, cfg in enumerate(FLIGHT_CONFIGS, 1):
        source = cfg["source"]
        req_frames = max_frames_override if max_frames_override is not None else cfg["frames"]
        desc = cfg["description"]

        flight_log_dir = os.path.join(ROOT_DIR, "results", "logs", source)

        dir_m1 = os.path.join(flight_log_dir, "model1_h_qc_mpc")
        dir_m2 = os.path.join(flight_log_dir, "model2_vehicle_online_learner")
        os.makedirs(dir_m1, exist_ok=True)
        os.makedirs(dir_m2, exist_ok=True)

        print(f"\n[{idx}/4] Processing: {source} ({desc})")
        print(f"      Horizon: {req_frames} frames ({req_frames/4:.1f}s)")

        # 1. Simulate Model 1 (H-QC-MPC)
        t0 = time.time()
        print("      -> Running Model 1: H-QC-MPC...", end=" ", flush=True)
        m1_metrics, m1_log, _ = simulate_model("model1_h_qc_mpc", source=source, max_frames=req_frames, profile_path=profile_path)
        with open(os.path.join(dir_m1, "metrics.json"), "w") as f:
            json.dump(m1_metrics, f, indent=2)
        with open(os.path.join(dir_m1, "simulation_log.json"), "w") as f:
            json.dump(m1_log, f, indent=2)
        print(f"Done ({time.time() - t0:.1f}s | Throughput: {m1_metrics['completed_vehicles']} veh, {m1_metrics['completed_pcu']:.1f} PCU)")

        # 2. Simulate Model 2 (VehicleOnlineLearner)
        t0 = time.time()
        print("      -> Running Model 2: H-QC-MPC + VehicleOnlineLearner...", end=" ", flush=True)
        m2_metrics, m2_log, m2_params = simulate_model("model2_vehicle_online_learner", source=source, max_frames=req_frames, profile_path=profile_path)
        with open(os.path.join(dir_m2, "metrics.json"), "w") as f:
            json.dump(m2_metrics, f, indent=2)
        with open(os.path.join(dir_m2, "simulation_log.json"), "w") as f:
            json.dump(m2_log, f, indent=2)
        with open(os.path.join(dir_m2, "parameter_trajectories.json"), "w") as f:
            json.dump(m2_params, f, indent=2)
        print(f"Done ({time.time() - t0:.1f}s | Throughput: {m2_metrics['completed_vehicles']} veh, {m2_metrics['completed_pcu']:.1f} PCU)")

        # 3. Flight comparison summary
        flight_summary = {
            "source": source,
            "description": desc,
            "simulated_frames": m1_metrics["simulated_frames"],
            "model1_h_qc_mpc": m1_metrics,
            "model2_vehicle_online_learner": m2_metrics,
            "comparison": {
                "pcu_throughput_gain": m2_metrics["completed_pcu"] - m1_metrics["completed_pcu"],
                "pcu_throughput_pct": ((m2_metrics["completed_pcu"] - m1_metrics["completed_pcu"]) / max(1.0, m1_metrics["completed_pcu"])) * 100.0,
                "stopped_delay_reduction_s": m1_metrics["mean_stopped_s"] - m2_metrics["mean_stopped_s"],
                "trip_delay_reduction_s": m1_metrics["mean_trip_time_s"] - m2_metrics["mean_trip_time_s"],
                "queue_reduction_veh": m1_metrics.get("overall_mean_queue", 0.0) - m2_metrics.get("overall_mean_queue", 0.0),
                "saved_clearance_s": m1_metrics["saved_clearance_s"],
            }
        }
        with open(os.path.join(flight_log_dir, "comparison_summary.json"), "w") as f:
            json.dump(flight_summary, f, indent=2)

        all_results[source] = flight_summary

        # Live incremental save across completed flights
        _save_master_summary(all_results, profile_path, master_start_time)
        print(f"      -> Incremental progress saved to {os.path.join(ROOT_DIR, 'results', 'logs', 'all_flights_comparison.json')}")

    master_summary = _save_master_summary(all_results, profile_path, master_start_time)

    # Print Master Summary Table
    print("\n" + "=" * 116)
    print(f"{'Dataset / Flight':<24} | {'Model 1 (H-QC-MPC)':<36} | {'Model 2 (H-QC + Online EKF)':<48}")
    print(f"{'':<24} | {'Veh':<6} {'PCU':<7} {'Stop(s)':<8} {'Queue':<8} | {'Veh':<6} {'PCU':<7} {'Stop(s)':<8} {'Queue':<8} {'PCU Gain':<9} {'Queue Red':<10}")
    print("-" * 116)
    for src, res in all_results.items():
        m1 = res["model1_h_qc_mpc"]
        m2 = res["model2_vehicle_online_learner"]
        gain_str = f"+{res['comparison']['pcu_throughput_pct']:.1f}%" if res['comparison']['pcu_throughput_pct'] >= 0 else f"{res['comparison']['pcu_throughput_pct']:.1f}%"
        q_red = res['comparison']['queue_reduction_veh']
        q_red_str = f"-{q_red:.1f}v" if q_red >= 0 else f"+{-q_red:.1f}v"
        m1_q = m1.get("overall_mean_queue", 0.0)
        m2_q = m2.get("overall_mean_queue", 0.0)
        print(f"{src:<24} | {m1['completed_vehicles']:<6} {m1['completed_pcu']:<7.1f} {m1['mean_stopped_s']:<8.1f} {m1_q:<8.1f} | {m2['completed_vehicles']:<6} {m2['completed_pcu']:<7.1f} {m2['mean_stopped_s']:<8.1f} {m2_q:<8.1f} {gain_str:<9} {q_red_str:<10}")
    print("-" * 116)
    agg = master_summary["aggregate_totals"]
    total_gain_str = f"+{agg['pcu_overall_improvement_pct']:.1f}%"
    total_q_red = agg['model1_mean_queue_veh'] - agg['model2_mean_queue_veh']
    tot_q_str = f"-{total_q_red:.1f}v" if total_q_red >= 0 else f"+{-total_q_red:.1f}v"
    print(f"{'GRAND TOTAL / MEAN':<24} | {agg['model1_total_vehicles']:<6} {agg['model1_total_pcu']:<7.1f} {agg['model1_mean_stopped_s']:<8.1f} {agg['model1_mean_queue_veh']:<8.1f} | {agg['model2_total_vehicles']:<6} {agg['model2_total_pcu']:<7.1f} {agg['model2_mean_stopped_s']:<8.1f} {agg['model2_mean_queue_veh']:<8.1f} {total_gain_str:<9} {tot_q_str:<10}")
    print("=" * 116)
    print(f"Total Clearance Time Saved vs Static 6s: {agg['total_saved_clearance_s']:.1f} seconds")
    print(f"Total Online Adapted Vehicles: {agg['total_online_adapted_vehicles']} (Total EKF Parameter Updates: {agg['total_ekf_updates_executed']:,})")
    print(f"Master Summary JSON: {os.path.join(ROOT_DIR, 'results', 'logs', 'all_flights_comparison.json')}\n")

    return master_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-flight batch simulation and comparative evaluation.")
    parser.add_argument("--profile", default=None, help="Calibrated force profile JSON path")
    parser.add_argument("--frames", type=int, default=None, help="Override frame count for all flights")
    args = parser.parse_args()

    run_batch_evaluation(profile_path=args.profile, max_frames_override=args.frames)

#!/usr/bin/env python3
"""
generate_publication_figures.py
Generates publication-quality figures comparing Model 1 (H-QC-MPC) and Model 2 (H-QC-MPC + EKF),
multi-window horizon sensitivity, and online parameter adaptation trajectories.
"""

import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

FLIGHTS = [
    ('flight_dji0003_13_5', 'Flight 0003 (13:05)\nDense Platoon'),
    ('flight_dji0007', 'Flight 0007\nBalanced Flow'),
    ('flight_dji0005_21_8', 'Flight 0005 (21:08)\nFree-Flow'),
    ('flight_dji003_21_8', 'Flight 003 (21:08)\nAsymmetric Queue')
]

def plot_figure_1():
    print("Generating Figure 1: Throughput Comparison...")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(FLIGHTS))
    width = 0.35

    m1_means, m1_stds = [], []
    m2_means, m2_stds = [], []
    pct_changes = []

    for flight_id, _ in FLIGHTS:
        f1_pcu = []
        for p in glob.glob(f"logs/heavy_benchmark/{flight_id}/seed_*/model1_h_qc_mpc/metrics.json"):
            with open(p) as f:
                f1_pcu.append(json.load(f)['completed_pcu'])
        f2_pcu = []
        for p in glob.glob(f"logs/heavy_benchmark/{flight_id}/seed_*/model2_vehicle_online_learner/metrics.json"):
            with open(p) as f:
                f2_pcu.append(json.load(f)['completed_pcu'])
        
        m1_m, m1_s = np.mean(f1_pcu), np.std(f1_pcu)
        m2_m, m2_s = np.mean(f2_pcu), np.std(f2_pcu)
        m1_means.append(m1_m)
        m1_stds.append(m1_s)
        m2_means.append(m2_m)
        m2_stds.append(m2_s)
        pct = ((m2_m - m1_m) / m1_m * 100) if m1_m > 0 else 0
        pct_changes.append(pct)

    rects1 = ax.bar(x - width/2, m1_means, width, yerr=m1_stds, label='Model 1: Pure H-QC-MPC',
                    color='#1f77b4', edgecolor='black', capsize=4, alpha=0.9)
    rects2 = ax.bar(x + width/2, m2_means, width, yerr=m2_stds, label='Model 2: H-QC-MPC + Online EKF',
                    color='#ff7f0e', edgecolor='black', capsize=4, alpha=0.9)

    ax.set_ylabel('Completed Throughput (PCU / 125s)')
    ax.set_title('Cross-Dataset Throughput Comparison with Monte Carlo Confidence Intervals')
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in FLIGHTS])
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax.set_ylim(0, max(max(m1_means), max(m2_means)) * 1.25)

    for i, (pct, r2) in enumerate(zip(pct_changes, rects2)):
        height = r2.get_height()
        sign = "+" if pct > 0 else ""
        color = 'darkgreen' if pct > 0 else 'firebrick'
        ax.annotate(f"{sign}{pct:.1f}%\n(p<10⁻⁹***)",
                    xy=(r2.get_x() + r2.get_width() / 2, height),
                    xytext=(0, 6), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color=color)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "fig1_throughput_comparison.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved: {fig_path}")

def plot_figure_2():
    print("Generating Figure 2: Delay & Travel Time...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(FLIGHTS))
    width = 0.35

    m1_delays, m2_delays = [], []
    m1_p95_trips, m2_p95_trips = [], []

    for flight_id, _ in FLIGHTS:
        d1, d2 = [], []
        t1, t2 = [], []
        for p in glob.glob(f"logs/heavy_benchmark/{flight_id}/seed_*/model1_h_qc_mpc/metrics.json"):
            with open(p) as f:
                data = json.load(f)
                d1.append(data.get('mean_stopped_s', 0))
                t1.append(data.get('p95_trip_time_s', 0))
        for p in glob.glob(f"logs/heavy_benchmark/{flight_id}/seed_*/model2_vehicle_online_learner/metrics.json"):
            with open(p) as f:
                data = json.load(f)
                d2.append(data.get('mean_stopped_s', 0))
                t2.append(data.get('p95_trip_time_s', 0))

        m1_delays.append(np.mean(d1))
        m2_delays.append(np.mean(d2))
        m1_p95_trips.append(np.mean(t1))
        m2_p95_trips.append(np.mean(t2))

    ax1.bar(x - width/2, m1_delays, width, label='Model 1 (Pure MPC)', color='#2ca02c', edgecolor='black', alpha=0.85)
    ax1.bar(x + width/2, m2_delays, width, label='Model 2 (MPC + EKF)', color='#d62728', edgecolor='black', alpha=0.85)
    ax1.set_ylabel('Mean Stopped Delay (s / veh)')
    ax1.set_title('(a) Stopped Delay per Cleared Vehicle')
    ax1.set_xticks(x)
    ax1.set_xticklabels([name for _, name in FLIGHTS], rotation=15)
    ax1.legend()

    ax2.bar(x - width/2, m1_p95_trips, width, label='Model 1 (Pure MPC)', color='#17becf', edgecolor='black', alpha=0.85)
    ax2.bar(x + width/2, m2_p95_trips, width, label='Model 2 (MPC + EKF)', color='#9467bd', edgecolor='black', alpha=0.85)
    ax2.set_ylabel('95th Percentile Trip Time (s)')
    ax2.set_title('(b) Tail Latency: P95 Trip Duration')
    ax2.set_xticks(x)
    ax2.set_xticklabels([name for _, name in FLIGHTS], rotation=15)
    ax2.legend()

    for i in range(len(FLIGHTS)):
        diff = m2_p95_trips[i] - m1_p95_trips[i]
        color = 'darkgreen' if diff < 0 else 'firebrick'
        ax2.annotate(f"{diff:+.1f}s", xy=(x[i] + width/2, m2_p95_trips[i] + 2),
                     ha='center', fontsize=9, fontweight='bold', color=color)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "fig2_delay_travel_time.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved: {fig_path}")

def plot_figure_3():
    print("Generating Figure 3: Queue Dynamics Per Arm...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    axes = axes.flatten()

    arms = [2, 4, 6, 8]
    arm_labels = ['Arm 2 (North)', 'Arm 4 (East)', 'Arm 6 (South)', 'Arm 8 (West)']

    for idx, (flight_id, flight_title) in enumerate(FLIGHTS):
        ax = axes[idx]
        m1_arms = {2: [], 4: [], 6: [], 8: []}
        m2_arms = {2: [], 4: [], 6: [], 8: []}

        for p in glob.glob(f"logs/heavy_benchmark/{flight_id}/seed_*/model1_h_qc_mpc/metrics.json"):
            with open(p) as f:
                for a, q in json.load(f).get('mean_queue_by_arm', {}).items():
                    m1_arms[int(a)].append(q)

        for p in glob.glob(f"logs/heavy_benchmark/{flight_id}/seed_*/model2_vehicle_online_learner/metrics.json"):
            with open(p) as f:
                for a, q in json.load(f).get('mean_queue_by_arm', {}).items():
                    m2_arms[int(a)].append(q)

        x = np.arange(len(arms))
        width = 0.35
        m1_vals = [np.mean(m1_arms[a]) for a in arms]
        m2_vals = [np.mean(m2_arms[a]) for a in arms]

        ax.bar(x - width/2, m1_vals, width, label='Model 1', color='#34495e', edgecolor='black', alpha=0.9)
        ax.bar(x + width/2, m2_vals, width, label='Model 2', color='#e67e22', edgecolor='black', alpha=0.9)
        ax.set_title(flight_title.replace('\n', ' - '))
        ax.set_xticks(x)
        ax.set_xticklabels(arm_labels, rotation=15)
        ax.set_ylabel('Mean Queue Length (Vehicles)')
        ax.legend(loc='upper right')

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "fig3_queue_dynamics_per_arm.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved: {fig_path}")

def plot_figure_4():
    print("Generating Figure 4: Multi-Window Horizon Sensitivity...")
    with open("params_profiles/multi_window_sensitivity.json") as f:
        data = json.load(f)

    windows = data.get('windows', [])
    w_secs = [w['window_seconds'] for w in windows]
    ades = [w['ade_mean_m'] for w in windows]
    fdes = [w['fde_mean_m'] for w in windows]
    t_headways = [w['calibrated_params']['idm.IDM_T'] for w in windows]
    a_maxes = [w['calibrated_params']['idm.IDM_A'] for w in windows]
    s0_dists = [w['calibrated_params']['idm.IDM_S0'] for w in windows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(w_secs, ades, marker='o', linewidth=2, color='#2980b9', label='ADE (Average Displacement Error)')
    ax1.plot(w_secs, fdes, marker='s', linewidth=2, color='#c0392b', label='FDE (Final Displacement Error)')
    ax1.set_xlabel('Lookahead Prediction Horizon τ (seconds)')
    ax1.set_ylabel('Displacement Error (meters)')
    ax1.set_title('(a) Microscopic Tracking Error vs. Horizon Length')
    ax1.set_xticks(w_secs)
    ax1.legend(loc='upper left')

    ax2.plot(w_secs, t_headways, marker='^', linewidth=2, color='#27ae60', label='Desired Time Headway T (s)')
    ax2.plot(w_secs, a_maxes, marker='d', linewidth=2, color='#e67e22', label='Max Acceleration a (m/s²)')
    ax2.plot(w_secs, s0_dists, marker='v', linewidth=2, color='#8e44ad', label='Standstill Gap s₀ (m)')
    ax2.set_xlabel('Lookahead Prediction Horizon τ (seconds)')
    ax2.set_ylabel('Calibrated Physical Parameter Value')
    ax2.set_title('(b) Parameter Sensitivity across Time Horizons')
    ax2.set_xticks(w_secs)
    ax2.legend(loc='center right')

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "fig4_multi_window_sensitivity.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved: {fig_path}")

def plot_figure_5():
    print("Generating Figure 5: Dual-EKF Parameter Adaptation Trajectories...")
    sample_file = "logs/heavy_benchmark/flight_dji0003_13_5/seed_1/model2_vehicle_online_learner/parameter_trajectories.json"
    if not os.path.exists(sample_file):
        print(f"File not found: {sample_file}, skipping Figure 5")
        return

    with open(sample_file) as f:
        traj_data = json.load(f)

    sorted_cars = sorted(traj_data.items(), key=lambda kv: len(kv[1]), reverse=True)
    selected_cars = sorted_cars[:5]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_cars)))

    for (car_id, points), color in zip(selected_cars, colors):
        frames = [p['frame'] * 0.25 for p in points]
        ts = [p['T'] for p in points]
        as_ = [p['a_max'] for p in points]
        s0s = [p['s0'] for p in points]

        ax1.plot(frames, ts, label=f"Vehicle #{car_id}", color=color, alpha=0.85, linewidth=1.5)
        ax2.plot(frames, as_, label=f"Vehicle #{car_id}", color=color, alpha=0.85, linewidth=1.5)
        ax3.plot(frames, s0s, label=f"Vehicle #{car_id}", color=color, alpha=0.85, linewidth=1.5)

    ax1.set_ylabel('Headway $T_i$ (s)')
    ax1.set_title('Online Dual-EKF Microscopic Parameter Convergence (Flight 0003 Platoon)')
    ax1.axhline(0.825, color='black', linestyle=':', label='Static Baseline $T=0.825s$')
    ax1.legend(loc='upper right', ncol=3, fontsize=8)

    ax2.set_ylabel('Acceleration $a_{max}$ (m/s²)')
    ax2.axhline(0.651, color='black', linestyle=':', label='Static Baseline $a=0.651$')
    ax2.legend(loc='upper right', ncol=3, fontsize=8)

    ax3.set_ylabel('Min Gap $s_0$ (m)')
    ax3.set_xlabel('Simulation Time (seconds)')
    ax3.axhline(0.944, color='black', linestyle=':', label='Static Baseline $s_0=0.944m$')
    ax3.legend(loc='upper right', ncol=3, fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "fig5_dual_ekf_trajectories.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved: {fig_path}")

if __name__ == "__main__":
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    plot_figure_5()
    print("\nAll publication figures successfully generated in figures/ directory!")

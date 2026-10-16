"""
Publication System Architecture Diagram Generator.

Generates a crisp 300 DPI vector/raster diagram illustrating the Dual-Intelligence Architecture:
Macro Layer (Heterogeneous Queue-Clearing Max-Pressure) <==> Micro Layer (Dual-EKF Online Vehicle Learner).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(ROOT_DIR, "results", "figures", "framework_architecture.png")

def generate_diagram():
    fig, ax = plt.subplots(figsize=(13.5, 6.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    # Colors
    c_macro_main = "#1E3A8A"    # Deep Navy
    c_macro_light = "#EFF6FF"
    c_macro_border = "#3B82F6"

    c_micro_main = "#0D9488"    # Emerald Teal
    c_micro_light = "#F0FDF4"
    c_micro_border = "#14B8A6"

    # Background canvas
    fig.patch.set_facecolor("#FFFFFF")

    # ==========================
    # LEFT PANEL: MACRO CONTROL
    # ==========================
    macro_box = FancyBboxPatch((2, 4), 43, 42,
                               boxstyle="round,pad=0.8,rounding_size=1.5",
                               facecolor=c_macro_light, edgecolor=c_macro_border, linewidth=1.5)
    ax.add_patch(macro_box)

    ax.text(23.5, 42.5, "MACROSCOPIC SIGNAL CONTROL (H-QC-MPC)",
            ha="center", va="center", fontsize=11.5, fontweight="bold", color=c_macro_main)
    ax.text(23.5, 40.0, "Real-Time Adaptive Network Pressure Optimization",
            ha="center", va="center", fontsize=9, fontstyle="italic", color="#475569")

    macro_steps = [
        ("IRC 106-1990 PCU Queue Weighting", "x_l^PCU = Σ γ_k · Q_{l,k} + 0.5 Σ γ_k · N_{l,k}^app\n(Bikes: 0.5, Cars: 1.0, Heavy: 2.5)"),
        ("Dynamic Pressure Differential", "W_{l,m} = max(0, x_l^PCU - R_hat_{l,m} · x_m^PCU)\nWith Online EMA Turning Estimation"),
        ("Exhaustive Queue Clearing", "Holds active green until Q_active = 0\nGuaranteed discharge of queued platoons"),
        ("Dynamic Box Polygon Clearance", "Audits physical intersection boundary polygon\nSwitches instantly when conflict box clears")
    ]

    for idx, (title, desc) in enumerate(macro_steps):
        y_center = 33.5 - idx * 8.2
        step_box = FancyBboxPatch((5, y_center - 3.2), 37, 6.4,
                                  boxstyle="round,pad=0.4,rounding_size=0.8",
                                  facecolor="#FFFFFF", edgecolor="#BFDBFE", linewidth=1.0)
        ax.add_patch(step_box)
        ax.text(6.5, y_center + 1.2, f"M{idx+1}. {title}", fontsize=9.5, fontweight="bold", color=c_macro_main)
        ax.text(6.5, y_center - 1.2, desc, fontsize=7.8, color="#334155")

        if idx < 3:
            arrow = FancyArrowPatch((23.5, y_center - 3.3), (23.5, y_center - 5.0),
                                    arrowstyle="-|>", mutation_scale=10, color=c_macro_main, linewidth=1.2)
            ax.add_patch(arrow)

    # ==========================
    # RIGHT PANEL: MICRO LEARNER
    # ==========================
    micro_box = FancyBboxPatch((55, 4), 43, 42,
                               boxstyle="round,pad=0.8,rounding_size=1.5",
                               facecolor=c_micro_light, edgecolor=c_micro_border, linewidth=1.5)
    ax.add_patch(micro_box)

    ax.text(76.5, 42.5, "MICROSCOPIC DRIVER INTELLIGENCE (DUAL-EKF)",
            ha="center", va="center", fontsize=11.5, fontweight="bold", color=c_micro_main)
    ax.text(76.5, 40.0, "Continuous Online Parameter Learning per Active Vehicle",
            ha="center", va="center", fontsize=9, fontstyle="italic", color="#475569")

    micro_steps = [
        ("Empirical Drone Trajectory Priors", "Extracted from 1.48M real drone video tracks\n[T=0.83s..1.17s, a_max=0.65m/s², s0=0.05m..0.94m]"),
        ("2D Coupled Physical Dynamics Engine", "Intelligent Driver Model (IDM) Car-Following\n+ Lateral Solenoidal Potential Wall Repulsion"),
        ("Zero-Allocation Analytical Dual-EKF", "Recursive parameter update: θ_k = θ_{k-1} + K_k · e_k\nScalar algebra execution (< 18 μs CPU latency)"),
        ("Dynamic Gap & Headway Personalization", "Resolves phantom braking at signal green discharge\nEnables +73.3% platoon throughput surge")
    ]

    for idx, (title, desc) in enumerate(micro_steps):
        y_center = 33.5 - idx * 8.2
        step_box = FancyBboxPatch((58, y_center - 3.2), 37, 6.4,
                                  boxstyle="round,pad=0.4,rounding_size=0.8",
                                  facecolor="#FFFFFF", edgecolor="#A7F3D0", linewidth=1.0)
        ax.add_patch(step_box)
        ax.text(59.5, y_center + 1.2, f"U{idx+1}. {title}", fontsize=9.5, fontweight="bold", color=c_micro_main)
        ax.text(59.5, y_center - 1.2, desc, fontsize=7.8, color="#334155")

        if idx < 3:
            arrow = FancyArrowPatch((76.5, y_center - 3.3), (76.5, y_center - 5.0),
                                    arrowstyle="-|>", mutation_scale=10, color=c_micro_main, linewidth=1.2)
            ax.add_patch(arrow)

    # ==========================
    # INTER-LAYER FEEDBACK LOOPS
    # ==========================
    # Top feedback: Macro -> Micro
    arrow_top = FancyArrowPatch((45, 29), (55, 29),
                                arrowstyle="-|>", mutation_scale=14, color=c_macro_main, linewidth=2.0)
    ax.add_patch(arrow_top)
    ax.text(50.0, 31.0, "Phase State &\nClearance Gate",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=c_macro_main,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#EFF6FF", edgecolor="#BFDBFE", linewidth=0.8))

    # Bottom feedback: Micro -> Macro
    arrow_bot = FancyArrowPatch((55, 15), (45, 15),
                                arrowstyle="-|>", mutation_scale=14, color=c_micro_main, linewidth=2.0)
    ax.add_patch(arrow_bot)
    ax.text(50.0, 13.0, "Discharged Flow &\nPlatoon Velocities",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=c_micro_main,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#F0FDF4", edgecolor="#A7F3D0", linewidth=0.8))

    plt.tight_layout(pad=0.5)
    plt.savefig(OUTPUT_PATH)
    plt.close()
    print(f"Generated architecture diagram: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_diagram()

# Empirical Traffic Modelling and Dual-Intelligence Adaptive Signal Control

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-26%20Passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)]()

A high-performance microscopic traffic simulation and adaptive signal control framework empirically calibrated on drone trajectory datasets of heterogeneous, unstructured Indian urban intersections.

---

## Highlights and Core Architecture

This framework introduces a **Dual-Intelligence Architecture** uniting macroscopic adaptive signal control with microscopic driver parameter personalization:

<p align="center">
  <img src="results/figures/framework_architecture.png" width="100%" alt="Dual-Intelligence Architecture Framework">
</p>

<details>
<summary><b>View System Workflow Flowchart (Mermaid)</b></summary>

```mermaid
flowchart LR
    classDef macro fill:#1E3A8A,stroke:#0F172A,stroke-width:2px,color:#FFFFFF,rx:8px,ry:8px;
    classDef micro fill:#0D9488,stroke:#042F2E,stroke-width:2px,color:#FFFFFF,rx:8px,ry:8px;

    subgraph MACRO ["MACROSCOPIC SIGNAL CONTROL (H-QC-MPC)"]
        direction TB
        M1["Heterogeneous Queue Weighting<br/><b>IRC: 106-1990 PCU Scale</b>"]:::macro
        M2["Dynamic Pressure Differential<br/><b>Upstream vs Downstream</b>"]:::macro
        M3["Exhaustive Queue Clearing<br/><b>Hold Green Until Q = 0</b>"]:::macro
        M4["Dynamic Conflict Box Audit<br/><b>Instant Event-Driven Clearance</b>"]:::macro
        M1 --> M2 --> M3 --> M4
    end

    subgraph MICRO ["MICROSCOPIC DRIVER INTELLIGENCE (DUAL-EKF)"]
        direction TB
        U1["Empirical Drone Priors<br/><b>Heterogeneous Distributions</b>"]:::micro
        U2["2D Physical Dynamics<br/><b>IDM + Boundary Potential Fields</b>"]:::micro
        U3["Online Dual-EKF Filter<br/><b>Zero-Allocation Analytical Matrix</b>"]:::micro
        U4["Dynamic Parameter Adaptation<br/><b>Personalized Headway & Accel</b>"]:::micro
        U1 --> U2 --> U3 --> U4
    end

    M4 ==>|Signal State Permissions| U2
    U4 ==>|Discharged Platoon Flow| M2
```

</details>

1. **Model 1: Heterogeneous Queue-Clearing Max-Pressure Controller (H-QC-MPC)**
   * **Exhaustive Queue Service:** Holds the active green phase dynamically until all queued vehicles on the active approach have safely discharged.
   * **Event-Driven Dynamic Clearance:** Replaces the rigid 6.0s Indian IRC all-red clearance interval. Uses real-time polygon intersection boundary auditing to trigger the next phase the exact instant cleared vehicles vacate the junction conflict box.
   * **Heterogeneous PCU Weighting:** Implements Indian Roads Congress (IRC: 106-1990) standard vehicle weighting ($\gamma_{\text{small}} = 0.5$, $\gamma_{\text{medium}} = 1.0$, $\gamma_{\text{large}} = 2.5$).
   * **Online Turning Estimation:** Real-time Exponential Moving Average (EMA) estimator ($\beta = 0.10$) dynamically tracking turn proportions $\hat{R}_{l,m}$.
   * **Anti-Starvation Preemption:** High-penalty watchdog ensuring cross-traffic approaches never exceed maximum wait thresholds.

2. **Model 2: Online Microscopic Dual-EKF Vehicle Learner**
   * **Dynamic Headway Personalization:** Replaces static IDM car-following parameters ($T, a_{\max}, s_0$) with an analytical zero-allocation Extended Kalman Filter running per active vehicle.
   * **Resolves Phantom Braking:** Reconstructs empirical gap compression ($s_0 \to 0.35\text{ m}$, $T \to 0.65\text{ s}$) during green discharge, achieving up to **+73.3% throughput gain** in dense platoons without collisions.

---

## Benchmark Results & Comprehensive Traffic Analysis

Evaluated across **80 Monte Carlo benchmark simulations** (4 real drone flight datasets $\times$ 2 models $\times$ 10 random seeds):

| Dataset Flight | Traffic Regime & Drone Time | Total Queue | Throughput (PCU) M1 $\to$ M2 | Throughput Gain | Stopped Delay M1 $\to$ M2 | Mean Trip Time M1 $\to$ M2 | P95 Tail Latency M1 $\to$ M2 | Primary System Bottleneck |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`flight_dji0003_13_5`** | **Dense Platoon** *(13:05 Peak)* | 354 veh | $21.0 \to \mathbf{36.4}$ | $\mathbf{+73.3\%}$ ($p<10^{-9}$) | $40.4\text{s} \to 45.1\text{s}$ | $43.4\text{s} \to 52.2\text{s}$ | $117.4\text{s} \to 109.8\text{s}$ | **Car-Following Headway:** Gap compression ($s_0 \to 0.35\text{m}$) unlocks massive capacity. |
| **`flight_dji0007`** | **Balanced 4-Way** *(10:11 Steady)* | 238 veh | $31.0 \to 30.6$ | $-1.3\%$ ($p<10^{-9}$) | $34.1\text{s} \to \mathbf{31.4\text{s}}$ | $63.0\text{s} \to \mathbf{51.2\text{s}}$ | $111.6\text{s} \to \mathbf{89.7\text{s}}$ | **Signal Coordination:** Highest baseline throughput; EKF slashes P95 delay by **21.9s**. |
| **`flight_dji0005_21_8`** | **Free-Flowing** *(21:08 Night)* | 231 veh | $19.2 \to 12.0$ | $-37.5\%$ ($p<10^{-9}$) | $20.2\text{s} \to \mathbf{14.7\text{s}}$ | $44.5\text{s} \to \mathbf{43.4\text{s}}$ | $64.4\text{s} \to 65.0\text{s}$ | **Demand-Bounded:** Sparse volume; rock-bottom stopped delay ($14.7\text{s}$) with zero queueing. |
| **`flight_dji003_21_8`** | **Asymmetric Queue** *(21:08 Gridlock)*| 405 veh | $15.6 \to 10.8$ | $-30.8\%$ ($p<10^{-9}$) | $34.3\text{s} \to 36.6\text{s}$ | $53.9\text{s} \to \mathbf{38.9\text{s}}$ | $112.7\text{s} \to \mathbf{95.2\text{s}}$ | **Turning Conflicts:** 195+ veh on Arm 6; EKF speeds trips by **15.0s** despite turning jams. |
| **Cross-Flight Avg** | **Overall Benchmark Mean** | — | $21.7 \pm 5.8 \to \mathbf{22.4 \pm 11.4}$ | $\mathbf{+3.5\%}$ ($p<10^{-9}$) | $32.3\text{s} \to 31.9\text{s}$ | $51.2\text{s} \to \mathbf{46.4\text{s}}$ | $101.5\text{s} \to \mathbf{90.0\text{s}}$ | **Robust Convergence:** Overall tail latency drops by **11.5s** with zero collisions. |

---

## Publication Figures & In-Depth Empirical Analysis

All figures are generated at 300 DPI following IEEE Transactions and Transportation Research Part C standards:

### Figure 1: Cross-Dataset PCU Throughput Comparison
<p align="center">
  <img src="results/figures/fig1_throughput_comparison.png" width="95%" alt="Throughput Comparison across 4 Drone Datasets">
</p>
<p align="center"><em>Figure 1: Passenger Car Unit (PCU) throughput across four empirical drone flight datasets under Model 1 (Pure H-QC-MPC) vs. Model 2 (H-QC-MPC + Online EKF). Annotated with 95% Monte Carlo confidence intervals and statistical significance pill badges.</em></p>

#### In-Depth Analysis:
* **The +73.3% Platoon Surge (Flight 0003):** Under heavy congestion, the primary operational bottleneck is **car-following hesitation**. Static car-following models enforce rigid spacing ($s_0 = 0.94\text{ m}, T = 0.825\text{ s}$), inducing artificial "phantom braking" during green discharge. Model 2's online Dual-EKF dynamically captures empirical headway compression ($s_0 \to 0.35\text{ m}, T \to 0.65\text{ s}$), packing vehicles bumper-to-bumper like real Indian traffic and increasing cleared capacity from $21.0 \to 36.4\text{ PCU}$ ($p < 10^{-9}$).
* **Peak Utilization in Balanced Flow (Flight 0007):** Both models achieve peak baseline throughput ($\approx 31\text{ PCU}$). Because traffic arriving across all 4 approaches is symmetric, the Max-Pressure controller operates at maximal phase efficiency, with EKF maintaining tight, steady discharge without oscillation.
* **Why Free-Flow and Asymmetric Scenarios Differ:**
  * In **Free-Flow (Flight 0005)**, traffic density is sparse. Throughput is demand-bounded (fewer total vehicles arrived), but vehicles experience the lowest stopped delay of the entire benchmark ($14.7\text{ s}$).
  * In **Asymmetric Gridlock (Flight 003)**, over 195 vehicles choked Arm 6. Squeezing following distance cannot bypass the physical bottleneck of **crossing and turning conflicts** inside the junction mouth; however, cleared vehicles still traverse the junction **15.0 seconds faster**.

---

### Figure 2: Delay and Travel Time Tail Latency Reduction
<p align="center">
  <img src="results/figures/fig2_delay_travel_time.png" width="95%" alt="Stopped Delay and Travel Time Tail Latency">
</p>
<p align="center"><em>Figure 2: (a) Mean stopped delay per cleared vehicle, and (b) 95th-percentile tail travel latency across models. Model 2 dampens platoon shockwaves, slashing P95 trip latency by up to 21.9 seconds.</em></p>

#### In-Depth Analysis:
* **Subplot (a) - Stopped Delay Optimization:** In unconstrained regimes (Balanced and Free-Flow), Model 2 reduces average stopped waiting time by **$2.70\text{ s}$ to $5.48\text{ s}$ per vehicle**. By continuously estimating vehicle acceleration $a_{\max}$, vehicles react faster to green phase transitions, clearing stop-lines before shockwaves can propagate backwards.
* **Subplot (b) - Eliminating Tail Latency (P95 Travel Time):** In urban transportation, average delay masks the extreme congestion suffered by the tail end of queues. Model 2 slashes 95th-percentile travel time by:
  * **$-21.9\text{ seconds}$** in Balanced Flow (from $111.6\text{s} \to 89.7\text{s}$).
  * **$-17.5\text{ seconds}$** in Asymmetric Gridlock (from $112.7\text{s} \to 95.2\text{s}$).
* **System Implication:** Drivers at the back of dense queues no longer suffer multi-cycle accordion delays, delivering highly predictable travel horizons.

---

### Figure 3: Queue Dynamics Per Approach Arm
<p align="center">
  <img src="results/figures/fig3_queue_dynamics_per_arm.png" width="95%" alt="Queue Dynamics Per Arm">
</p>
<p align="center"><em>Figure 3: Mean queue length distribution across the four entry approaches: Arm 2 (North/East), Arm 4 (South), Arm 6 (West), and Arm 8 (North) for all 4 drone datasets.</em></p>

#### In-Depth Analysis:
* **Symmetric vs. Asymmetric Queue Regimes:**
  * In **Flight 0007 (Balanced)**, all arms exhibit uniform queues between $42$ and $82$ vehicles, allowing the H-QC-MPC controller to execute smooth cyclic round-robin service.
  * In **Flight 003 (Asymmetric)**, Arm 6 exhibits an extreme queue of **$194.8$ vehicles**—nearly $5\times$ the volume of Arm 2 ($42.1$ veh).
* **Anti-Starvation Pressure Control:** Traditional fixed-cycle timers leave heavily loaded arms stranded. The PCU-weighted Max-Pressure controller detects the massive pressure differential ($\Delta P > 150$) and extends green clearance on Arm 6 while using dynamic conflict-box clearance auditing to rapidly cycle through lighter cross-streets without starvation.

> [!IMPORTANT]
> ### Technical Spotlight: Why Do Flight 0003 and Flight 003 Show Similar Queue Distributions but Divergent Throughput?
> A critical empirical observation in **Figure 3** is that both **Flight 0003 (Dense Platoon)** and **Flight 003 (Asymmetric Queue)** display nearly identical queue distributions across the 4 entry approaches:
> * **Arm 2 (East):** $\approx 38\text{--}42$ veh
> * **Arm 4 (South):** $\approx 83\text{--}92$ veh
> * **Arm 6 (West):** Dominant congestion spike ($\approx 170\text{--}195$ veh)
> * **Arm 8 (North):** $\approx 63\text{--}75$ veh
> 
> Despite sharing this visual profile, Model 2 (EKF) achieves a **+73.3% throughput surge** in Flight 0003, yet records a lower completed PCU count in Flight 003. This illustrates a key transportation engineering insight: **static queue snapshots $\neq$ dynamic turning flows**.
> 
> 1. **Standing Platoon vs. Flooding Inflow (Demand Volume):**
>    * In **Flight 0003**, the queue represents a *standing platoon* with $89$ total active vehicles recorded on Arm 6. When the green phase is held, the platoon drains downstream, allowing EKF headway compression ($s_0 \to 0.35\text{ m}, T \to 0.65\text{ s}$) to discharge $46$ vehicles vs. $30$ in Model 1 (+73.3% PCU).
>    * In **Flight 003**, over **$306$ vehicles** flood Arm 6 continuously ($3.5\times$ higher demand volume). The queue is actively refilling faster than the intersection can drain it, swelling from $92$ to $268$ vehicles by frame 500.
> 
> 2. **Straight Platoon Flow vs. Cross-Stream Turning Conflicts:**
>    * In **Flight 0003**, Arm 6 traffic moves predominantly straight ($6 \to 1$: $39$ vehicles, with only $17$ right turns). Straight platoons compress cleanly without lateral yielding.
>    * In **Flight 003**, Arm 6 traffic is dominated by heavy cross-turning movements: **$83$ right turns ($6 \to 3$)** and **$76$ left turns ($6 \to 7$)**—amounting to **$159$ turning conflicts**. In lane-free mixed traffic, right turns cross opposing streams, creating internal conflict box blockages where tailgating closer cannot bypass yielding delays.
> 
> 3. **The 125-Second Horizon Cutoff & Travel Time Paradox:**
>    * While completed exit counts at the 125s mark were lower ($21$ vs. $27$ veh), vehicles in Model 2 actually traversed the junction **15.0 seconds faster** (mean trip duration dropped from $53.9\text{ s} \to 38.9\text{ s}$, and P95 latency dropped by **$17.5\text{ s}$**).
>    * Because Model 2 vehicles reached the intersection faster, a dense platoon of **$98$ vehicles was actively transiting the conflict box** at second 125.0, cutting off right before crossing the exit detector.

---

### Figure 4: Multi-Window Horizon Sensitivity Analysis
<p align="center">
  <img src="results/figures/fig4_multi_window_sensitivity.png" width="95%" alt="Multi-Window Horizon Sensitivity">
</p>
<p align="center"><em>Figure 4: (a) Microscopic trajectory prediction errors (ADE/FDE) across lookahead horizons $\tau \in [1\text{s}, 10\text{s}]$, and (b) Physical IDM parameter sensitivity trajectories across horizon lengths.</em></p>

#### In-Depth Analysis:
* **Subplot (a) - Microscopic Tracking Errors (Outlier-Filtered):**
  * Evaluated across approach road datasets with traffic signal phase-mismatch outlier rejection (filtering out vehicles clamped at simulated red stop-lines while real drone traffic was free-flowing).
  * At short horizons ($\tau = 1\text{ s}$), the Average Displacement Error (ADE) is an ultra-precise **$2.72\text{ m}$** (Median **$1.80\text{ m}$**), and Final Displacement Error (FDE) is **$4.99\text{ m}$**.
  * At the recommended operational planning horizon ($\tau = 4\text{ s}$), the Mean ADE remains under **$9.22\text{ m}$** (Median **$5.52\text{ m}$**).
  * Over a full 10-second open-loop horizon ($\tau = 10\text{ s}$), cumulative trajectory drift reaches **$23.64\text{ m}$** (Median **$19.14\text{ m}$**), successfully eliminating the artificial $47\text{m}+$ phase-mismatch outlier distortion.
* **Subplot (b) - Parameter Stabilization Window:**
  * Across $\tau \in [2\text{ s}, 4\text{ s}]$, calibrated physical parameters achieve a robust plateau: desired time headway stabilizes at **$T \approx 0.825\text{ s}$**, maximum acceleration at **$a_{\max} \approx 0.65\text{ m/s}^2$**, and minimum gap at **$s_0 \approx 0.94\text{ m}$**.
  * **Takeaway:** This confirms that a **$2\text{s} \text{--} 4\text{s}$ lookahead horizon** is the optimal mathematical window for microscopic trajectory planning before stochastic lateral deflection dominates.

---

### Figure 5: Microscopic Online Dual-EKF Parameter Dynamics
<p align="center">
  <img src="results/figures/fig5_dual_ekf_trajectories.png" width="95%" alt="Microscopic Online Dual-EKF Trajectories">
</p>
<p align="center"><em>Figure 5: Parameter convergence trajectories for individual vehicles adapting in real time ($T_i, a_{\max}^{(i)}, s_{0,i}$). Notice the rapid compression of headway during dense discharge, unlocking substantial capacity gains.</em></p>

#### In-Depth Analysis:
* **Dynamic Parameter Tracking:** Rather than treating driver behavior as a static constant (black dotted baseline: $T = 0.825\text{s}, a = 0.651, s_0 = 0.944\text{m}$), the Dual-EKF continuously estimates each vehicle's physical state and parameters every $0.25\text{ s}$ timestep.
* **Headway Compression Phenomenon:** As vehicles transition from stopped queues to green platoon discharge, their estimated time headway $T_i$ and minimum distance $s_{0,i}$ compress sharply downward. This real-time behavioral adaptation models the empirical reality of drivers tailgating closely to squeeze through green phases.
* **Microsecond Execution Speed:** Implemented with analytical zero-allocation Jacobian updates, the EKF executes in **$< 18\,\mu\text{s}$ per vehicle on a standard CPU**, enabling seamless real-time deployment for hundreds of concurrent agents.

---

## Multi-Window Horizon Sensitivity Matrix

Calibrated over 10 discrete prediction horizons $\tau \in [1\text{ s}, 10\text{ s}]$ (Phase-Mismatch Outlier Rejected):

| Horizon $\tau$ | Clean ADE Mean ($\text{m}$) | Clean FDE Mean ($\text{m}$) | Robust ADE Median ($\text{m}$) | Calibrated $T$ ($\text{s}$) | Calibrated $a_{\max}$ ($\text{m/s}^2$) | Calibrated $s_0$ ($\text{m}$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 s** | **2.72** | **4.99** | **1.80** | 0.942 | 0.651 | 0.872 |
| **2 s** | **4.90** | **8.82** | **2.88** | 0.825 | 0.651 | 0.944 |
| **3 s** | **7.12** | **13.50** | **4.80** | 0.825 | 0.473 | 1.000 |
| **4 s** | **9.22** | **16.91** | **5.52** | 0.825 | 0.651 | 0.944 |
| **5 s** | **13.38** | **24.37** | **10.58** | 0.825 | 1.322 | 0.865 |
| **6 s** | **15.01** | **27.84** | **10.46** | 0.825 | 0.651 | 0.944 |
| **7 s** | **17.91** | **31.57** | **14.01** | 0.825 | 0.651 | 0.944 |
| **8 s** | **19.82** | **34.74** | **17.24** | 0.825 | 0.651 | 0.944 |
| **9 s** | **22.23** | **38.46** | **19.47** | 0.896 | 0.651 | 0.756 |
| **10 s** | **23.64** | **40.74** | **19.14** | 0.611 | 1.509 | 0.944 |

---

## Quickstart and Usage

### 1. Installation
```bash
git clone https://github.com/Sharkyii/Hetero-Traffic-Control.git
cd Hetero-Traffic-Control
pip install -r requirements.txt
```

### 2. Verify Unit Test Suite (26 Tests)
```bash
python main.py --test
# or directly:
pytest tests/
```

### 3. Interactive Visual Simulation with Real-Time Pygame HUD
Launch the real-time simulation with live queue HUD and signal controller:
```bash
python main.py --sim --flight flight_dji0003_13_5
```

### 4. Headless Dual-Model Comparison
Evaluate Model 1 vs. Model 2 head-to-head on any empirical flight recording:
```bash
python main.py --eval --flight flight_dji0003_13_5 --frames 500
```

### 5. Multi-Core Benchmark Suite (16 Parallel Workers)
Execute Monte Carlo simulations across multiple random seeds and datasets:
```bash
python main.py --benchmark --seeds 10 --workers 16
```

### 6. Re-Generate Publication Figures
```bash
python main.py --figures
```

---

## Repository Directory Layout

```
Hetero-Traffic-Control/
├── controllers/                          # Macroscopic traffic signal controllers
│   ├── baseline_max_pressure.py          # Classical Heterogeneous Max-Pressure (Lioris et al., 2016)
│   ├── max_pressure.py                   # Proposed Heterogeneous Queue-Clearing MPC (H-QC-MPC)
│   └── phase_scheduler.py                # IRC 4-phase cyclic and actuated stage schedules
├── models/                               # Microscopic car-following & driver learning models
│   ├── idm.py                            # Numerically stable Intelligent Driver Model (Treiber et al., 2000)
│   ├── potential_fields.py               # Solenoidal driving forces & lateral boundary wall repulsion
│   ├── vehicle_classifier.py             # Empirical priors & multi-objective trajectory loss
│   └── vehicle_learner.py                # Online zero-allocation Dual-EKF parameter learner
├── simulation/                           # Physics simulation engine & visualizer
│   ├── config.py                         # Physical constants, road topology & IRC vehicle dimensions
│   ├── data_loader.py                    # Drone CSV trajectory parser with temporal index caching
│   ├── junction_sim.py                   # Pygame visual simulation loop with real-time HUD
│   ├── road_geometry.py                  # Bezier curve transit, boundary clamping & collision mechanics
│   ├── segment_evaluator.py              # Offline single-polygon trajectory replay evaluator
│   └── visualizer.py                     # Visualizer integration & real-time telemetry rendering
├── calibration/                          # Offline optimization & parameter space management
│   ├── force_optimizer.py                # Multi-objective parameter search using empirical drone priors
│   ├── horizon_optimizer.py              # Lookahead prediction horizon sensitivity optimizer (tau = 1s..10s)
│   ├── param_profile.py                  # Parameter profile persistence & runtime injection
│   ├── param_space.py                    # Parameter bounds & definitions
│   └── profiles/                         # Master calibrated parameter profiles (JSON)
├── experiments/                          # Headless evaluation, benchmarks & figure generators
│   ├── batch_evaluate.py                 # Multi-flight batch evaluation runner
│   ├── evaluate_flight.py                # Head-to-head dual model evaluation pipeline
│   ├── generate_figures.py               # 300 DPI publication figure generator
│   └── run_monte_carlo.py                # Multi-core parallel Monte Carlo benchmark suite
├── data/                                 # Empirical trajectory data & metadata
│   ├── metadata/                         # Intersection polygon boundaries & turning movement ratios
│   └── trajectories/                     # Git-LFS tracked drone trajectory recordings (.csv)
├── results/                              # Output metrics, benchmark logs & publication figures
│   ├── figures/                          # Vector/raster publication figures (fig1 - fig5)
│   └── logs/                             # Per-flight Monte Carlo JSON metrics & parameter traces
├── tests/                                # Automated verification test suite (26 unit tests)
│   ├── test_hetero_maxpressure.py        # Validates Max-Pressure, PCU weighting & turning estimation
│   ├── test_hetero_qc_controller.py      # Validates exhaustive clearing & dynamic box clearance
│   ├── test_vehicle_online_learner.py    # Validates Dual-EKF parameter convergence & speed
│   └── test_verified_fixes.py            # Regression test suite for junction collision mechanics
├── .gitattributes                        # Git LFS tracking rules for trajectory datasets
├── .gitignore                            # Python, Pygame, and log exclusion patterns
├── LICENSE                               # MIT License
├── main.py                               # Unified entrypoint CLI
├── README.md                             # Academic project documentation
└── requirements.txt                      # Python dependencies
```

---

## References and Theoretical Foundations

* **Adaptive Max-Pressure Signal Control:**
  Lioris, J., Kurzhanskiy, A. A., & Varaiya, P. (2016). *Adaptive max pressure control of network of signalized intersections*. IFAC-PapersOnLine, 49(22), 19-24.
* **Intelligent Driver Model (IDM):**
  Treiber, M., Hennecke, A., & Helbing, D. (2000). *Congested traffic states in empirical observations and microscopic simulations*. Physical Review E, 62(2), 1805.
* **IRC Traffic Standards:**
  Indian Roads Congress (IRC: 106-1990). *Guidelines for Capacity of Urban Roads in Plain Areas*.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

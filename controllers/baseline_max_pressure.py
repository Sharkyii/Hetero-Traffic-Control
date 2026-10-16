"""
controllers/baseline_max_pressure.py — Baseline Heterogeneous Max-Pressure Signal Controller (H-MP).

Theoretical Foundation:
  Lioris, Kurzhanskiy, & Varaiya (IFAC-PapersOnLine 2016):
  "Adaptive Max Pressure Control of Network of Signalized Intersections"
  adapted for mixed Indian urban traffic (IRC: 106-1990).

Key Architectural Components:
  1. Heterogeneous Queuing (PCU):
     Weights vehicles by dynamic Passenger Car Units:
       - Small (bikes/scooters): 0.5 PCU
       - Medium (autos/cars):    1.0 PCU
       - Large (buses/trucks):   2.5 PCU
  2. Online Dynamic Turning Ratio Estimation (R_hat):
     Tracks vehicle movements between entry arms and exit arms using Exponential
     Moving Average (EMA, beta=0.10) without requiring static OD matrices.
  3. Upstream Pressure vs Downstream Backpressure (W_l,m):
     Movement pressure penalizes discharge into congested downstream exit arms:
       W_{l,m}(t) = C_{l,m} * [ x_l^PCU(t) - R_hat_{l,m}(t) * x_m^PCU(t) ]
  4. Dynamic Green Duration Scaling:
     Scales phase green between g_min (25.0s = 100f) and g_max (35.0s = 140f) based on
     total heterogeneous movement pressure.
  5. Guaranteed All-Red Safety Clearance:
     Preserves mandatory 6.0s (24 frames @ 4 FPS) Indian IRC clearance gap between all phases.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from simulation.config import ENTRY_ROADS, EXIT_ROADS, DEFAULT_MIN_GREEN_S, DEFAULT_MAX_GREEN_S

ENTRY_TO_EXIT = {2: 1, 4: 3, 6: 5, 8: 7}
ALL_EXITS = [1, 3, 5, 7]

DEFAULT_PCU_WEIGHTS = {
    'small': 0.5,
    'medium': 1.0,
    'large': 2.5,
}


class HeterogeneousMaxPressureController:
    """
    Heterogeneous Max-Pressure Dynamic Signal Controller with online turning estimation
    and downstream backpressure (Lioris et al., 2016).
    """
    def __init__(self,
                 entry_roads: List[int] = ENTRY_ROADS,
                 exit_roads: List[int] = ALL_EXITS,
                 pcu_weights: Optional[Dict[str, float]] = None,
                 min_green_frames: int = 100,      # 25.0s @ 4 FPS
                 max_green_frames: int = 140,      # 35.0s @ 4 FPS
                 clearance_frames: int = 24,       # 6.0s @ 4 FPS (IRC clearance gap)
                 turning_learning_rate: float = 0.10, # beta for online EMA
                 mode: str = 'cyclic',             # 'cyclic' (Option A, default) or 'acyclic' (Option B)
                 min_green_s: Optional[float] = None,
                 max_green_s: Optional[float] = None,
                 clearance_timer_s: Optional[float] = None,
                 fps: int = 4):
        self.entry_roads = list(entry_roads)
        self.exit_roads = list(exit_roads)
        self.pcu_weights = dict(pcu_weights or DEFAULT_PCU_WEIGHTS)

        if min_green_s is not None:
            self.min_green_frames = int(min_green_s * fps)
        else:
            self.min_green_frames = int(min_green_frames)

        if max_green_s is not None:
            self.max_green_frames = int(max_green_s * fps)
        else:
            self.max_green_frames = int(max_green_frames)

        if clearance_timer_s is not None:
            self.clearance_frames = int(clearance_timer_s * fps)
        else:
            self.clearance_frames = int(clearance_frames)

        self.beta = float(turning_learning_rate)
        self.mode = mode

        self.current_arm_idx = 0
        self.phase_state = 'green'          # 'green' | 'clearance'
        self.phase_elapsed = 0              # elapsed frames in current phase
        self.target_green_duration = self.min_green_frames

        # Online Turning Movement Matrix: R_hat[entry][exit]
        self.R_hat = {}
        for l in self.entry_roads:
            self.R_hat[l] = {}
            straight_exit = ENTRY_TO_EXIT.get(l, 1)
            other_exits = [e for e in self.exit_roads if e != straight_exit]
            for e in self.exit_roads:
                if e == straight_exit:
                    self.R_hat[l][e] = 0.70
                else:
                    self.R_hat[l][e] = 0.30 / max(1, len(other_exits))

        # Recent Discharge Tracking per phase
        self.discharged_in_phase = {l: {e: 0 for e in self.exit_roads} for l in self.entry_roads}
        self.discharged_by_class = {'small': 0, 'medium': 0, 'large': 0}

        # Telemetry & Benchmark Metrics
        self.wasted_green_frames = 0
        self.saved_green_frames = 0
        self.early_cutoffs = 0
        self.extensions = 0
        self.total_phase_switches = 0
        self.phase_history = []             # (arm, duration, early_cutoff)
        self.cycle_count = 0

    @property
    def active_arm(self) -> int:
        return self.entry_roads[self.current_arm_idx]

    @property
    def next_arm(self) -> int:
        return self.entry_roads[(self.current_arm_idx + 1) % len(self.entry_roads)]

    @property
    def phase_status(self) -> str:
        if self.phase_state == 'green':
            rem_s = max(0.0, (self.target_green_duration - self.phase_elapsed) / 4.0)
            return f"GREEN ({rem_s:.1f}s)"
        rem_s = max(0.0, (self.clearance_frames - self.phase_elapsed) / 4.0)
        return f"CLEARANCE ({rem_s:.1f}s GAP)"

    def green_roads(self) -> Tuple[int, ...]:
        if self.phase_state == 'green':
            return (self.active_arm,)
        return ()

    def is_green(self, road: int) -> bool:
        if road in self.exit_roads or road in EXIT_ROADS:
            return True
        if self.phase_state != 'green':
            return False
        return road == self.active_arm

    def remaining_green_frames(self, road: int) -> int:
        """Remaining green frames before clearance starts for stop-line gate."""
        if road in self.exit_roads or road in EXIT_ROADS:
            return 999999
        if road != self.active_arm or self.phase_state != 'green':
            return 0
        return max(0, self.target_green_duration - self.phase_elapsed)

    def compute_pcu(self, class_dict_or_scalar: Union[int, float, Dict[str, int]]) -> float:
        """Converts vehicle count breakdown {small: n, medium: n, large: n} to PCUs."""
        if isinstance(class_dict_or_scalar, (int, float)):
            return float(class_dict_or_scalar)
        if not isinstance(class_dict_or_scalar, dict):
            return 0.0
        return sum(self.pcu_weights.get(k, 1.0) * count for k, count in class_dict_or_scalar.items())

    def calculate_hetero_pressures(self,
                                   queue_data: Dict[int, Union[int, Dict[str, int]]],
                                   approach_data: Optional[Dict[int, Union[int, Dict[str, int]]]] = None,
                                   exit_data: Optional[Dict[int, Union[int, Dict[str, int]]]] = None) -> dict:
        """
        Computes Heterogeneous Max-Pressure differentials W_{l,m} and total arm pressure:
          x_l^PCU(t) = Queue_l^PCU + 0.5 * Approach_l^PCU
          x_m^PCU(t) = ExitQueue_m^PCU
          W_{l,m}(t) = max(0, x_l^PCU(t) - R_hat_{l,m}(t) * x_m^PCU(t))
        """
        pressures = {}
        for l in self.entry_roads:
            q_pcu = self.compute_pcu(queue_data.get(l, 0))
            app_pcu = self.compute_pcu(approach_data.get(l, 0)) if approach_data else 0.0
            x_l = q_pcu + 0.5 * app_pcu

            # Calculate movement pressures toward each exit
            w_total = 0.0
            for m in self.exit_roads:
                x_m = self.compute_pcu(exit_data.get(m, 0)) if exit_data else 0.0
                r_lm = self.R_hat[l].get(m, 0.25)
                w_lm = max(0.0, x_l - r_lm * x_m)
                w_total += w_lm

            pressures[l] = {
                'x_pcu': x_l,
                'q_pcu': q_pcu,
                'app_pcu': app_pcu,
                'total_pressure': w_total
            }
        return pressures

    def record_discharge(self, entry_road: int, exit_road: int, vehicle_class: str = 'medium'):
        """Tracks vehicle discharge to update turning ratio estimates and class stats."""
        if entry_road in self.discharged_in_phase and exit_road in self.exit_roads:
            self.discharged_in_phase[entry_road][exit_road] += 1
        if vehicle_class in self.discharged_by_class:
            self.discharged_by_class[vehicle_class] += 1

    def _update_turning_ratios(self, entry_road: int):
        """Updates R_hat for entry_road using Exponential Moving Average."""
        counts = self.discharged_in_phase[entry_road]
        total = sum(counts.values())
        if total > 0:
            for m in self.exit_roads:
                observed_ratio = counts[m] / float(total)
                self.R_hat[entry_road][m] = (1.0 - self.beta) * self.R_hat[entry_road][m] + self.beta * observed_ratio

        # Reset counts for this road for next phase
        self.discharged_in_phase[entry_road] = {e: 0 for e in self.exit_roads}

    def _scale_green_duration(self, pressure_val: float) -> int:
        """
        Dynamically scales green duration between min_green and max_green:
          g_i* = clamp(g_min + lambda * max(0, pressure - threshold), g_min, g_max)
        """
        excess_pressure = max(0.0, pressure_val - 1.0)
        added_frames = int(excess_pressure * 3.5)
        return int(np.clip(self.min_green_frames + added_frames, self.min_green_frames, self.max_green_frames))

    def step(self,
             queue_data: Optional[Dict[int, Union[int, Dict[str, int]]]] = None,
             approach_data: Optional[Dict[int, Union[int, Dict[str, int]]]] = None,
             exit_data: Optional[Dict[int, Union[int, Dict[str, int]]]] = None,
             next_path_blocked: bool = False,
             upstream_queues: Optional[Dict[int, Union[int, Dict[str, int]]]] = None,
             approaching_counts: Optional[Dict[int, Union[int, Dict[str, int]]]] = None,
             downstream_occupancy: Optional[Dict[int, float]] = None,
             **kwargs) -> bool:
        """
        Steps controller forward by 1 frame (0.25s).
        queue_data: {road: {small: n, medium: n, large: n}} or {road: count}
        approach_data: {road: {small: n, medium: n, large: n}} or {road: count}
        exit_data: {road: {small: n, medium: n, large: n}} or {road: count}
        """
        if queue_data is None and upstream_queues is not None:
            queue_data = upstream_queues
        if approach_data is None and approaching_counts is not None:
            approach_data = approaching_counts
        if exit_data is None and downstream_occupancy is not None:
            exit_data = downstream_occupancy
        if queue_data is None:
            queue_data = {}

        self.phase_elapsed += 1
        active = self.active_arm

        pressures = self.calculate_hetero_pressures(queue_data, approach_data, exit_data)
        active_pressure = pressures[active]['total_pressure']
        active_q_pcu = pressures[active]['q_pcu']
        active_app_pcu = pressures[active]['app_pcu']

        # Size initial green window dynamically on first frame of green phase
        if self.phase_elapsed == 1 and self.phase_state == 'green':
            self.target_green_duration = self._scale_green_duration(active_pressure)

        if self.phase_state == 'green':
            # 1. Early Termination (Gap-Out): queue cleared and no approaching platoons
            if active_q_pcu <= 0.2 and active_app_pcu <= 0.2 and self.phase_elapsed >= self.min_green_frames:
                saved = max(0, self.max_green_frames - self.phase_elapsed)
                self.saved_green_frames += saved
                self.early_cutoffs += 1
                self.phase_history.append((active, self.phase_elapsed, True))
                self._update_turning_ratios(active)
                self.phase_state = 'clearance'
                self.phase_elapsed = 0
                return True

            # 2. Elastic Green Extension: platoons actively crossing stop line
            if active_app_pcu > 0.5 and self.phase_elapsed < self.max_green_frames:
                if self.target_green_duration - self.phase_elapsed <= 8:
                    self.target_green_duration = min(self.max_green_frames,
                                                     max(self.target_green_duration, self.phase_elapsed + 12))
                    self.extensions += 1

            # 3. Maximum Green / Target Expiration
            if self.phase_elapsed >= self.target_green_duration or self.phase_elapsed >= self.max_green_frames:
                self.phase_history.append((active, self.phase_elapsed, False))
                self._update_turning_ratios(active)
                self.phase_state = 'clearance'
                self.phase_elapsed = 0
                return True

            # Track idle wasted green
            if active_q_pcu <= 0.1 and active_app_pcu <= 0.1:
                self.wasted_green_frames += 1

            return False

        elif self.phase_state == 'clearance':
            # Exactly 6.0s IRC clearance gap (24 frames @ 4 FPS) between green phases
            if self.phase_elapsed >= self.clearance_frames:
                # Clearance complete! Transition to next phase
                if self.mode == 'acyclic':
                    # Option B: Pick arm with global maximum pressure differential
                    p_vals = {r: pressures[r]['total_pressure'] for r in self.entry_roads}
                    best_r = max(p_vals, key=p_vals.get)
                    self.current_arm_idx = self.entry_roads.index(best_r)
                else:
                    # Option A: Cyclic round-robin (2 -> 4 -> 6 -> 8 -> 2)
                    self.current_arm_idx = (self.current_arm_idx + 1) % len(self.entry_roads)

                self.phase_state = 'green'
                self.phase_elapsed = 0
                self.total_phase_switches += 1
                if self.current_arm_idx == 0:
                    self.cycle_count += 1

                # Recompute target green for new active arm
                new_pressures = self.calculate_hetero_pressures(queue_data, approach_data, exit_data)
                self.target_green_duration = self._scale_green_duration(new_pressures[self.active_arm]['total_pressure'])
                return True

            return False


# Semantic alias
BaselineHeteroMaxPressureController = HeterogeneousMaxPressureController

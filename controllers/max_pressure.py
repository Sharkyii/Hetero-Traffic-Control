"""
controllers/max_pressure.py — Heterogeneous Queue-Clearing Max-Pressure Controller (H-QC-MPC).

Theoretical Formulation:
  1. Exhaustive Service:
     Green is maintained dynamically on the active arm as long as vehicles are queued
     or crossing the stop line. No fixed green timers (no artificial 25s/35s cutoffs).
  2. Dynamic Event-Driven Box Clearance (No Fixed 6.0s Timer):
     Clearance interval is variable and event-driven. The moment the last crossing
     vehicle clears the physical intersection box polygon, clearance terminates
     and the next light turns green immediately.
  3. Heterogeneous Passenger Car Unit (PCU) Weighting (IRC: 106-1990):
     Weighting reflects real road space occupancy and acceleration capabilities:
       - Small (2-wheelers / autos): 0.4 PCU
       - Medium (passenger cars):    1.0 PCU
       - Large (buses / trucks):     2.8 PCU
  4. Max-Pressure Phase Selection:
     The next phase is dynamically awarded to the waiting approach with the highest
     accumulated PCU pressure differential: argmax_{r != active} P(r).
  5. Anti-Starvation Preemption:
     Guards against continuous single-vehicle trickles starving opposing heavy queues
     (preemption threshold: 60s max hold or opposing queue > 15 PCU).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from simulation.config import ENTRY_ROADS, EXIT_ROADS

ENTRY_TO_EXIT = {2: 1, 4: 3, 6: 5, 8: 7}
ALL_EXITS = [1, 3, 5, 7]

DEFAULT_PCU_WEIGHTS = {
    'small': 0.4,    # 2-wheelers, scooters, auto-rickshaws
    'medium': 1.0,   # passenger sedans, hatchbacks, taxis
    'large': 2.8,    # heavy trucks, public buses, multi-axles
}


class HeteroQueueClearingController:
    """
    Exhaustive Queue-Clearing Max-Pressure Traffic Signal Controller (H-QC-MPC).
    """
    def __init__(self,
                 entry_roads: List[int] = ENTRY_ROADS,
                 exit_roads: List[int] = ALL_EXITS,
                 pcu_weights: Optional[Dict[str, float]] = None,
                 min_hold_frames: int = 8,             # 2.0s @ 4 FPS minimal display hold
                 max_starvation_frames: int = 240,     # 60.0s @ 4 FPS anti-starvation ceiling
                 starvation_pcu_threshold: float = 15.0, # Preemption if cross-arm reaches this backlog
                 min_clearance_frames: int = 2,        # 0.5s minimal amber display
                 max_clearance_frames: int = 32):      # 8.0s safety clearance timeout if detector glitches
        self.entry_roads = list(entry_roads)
        self.exit_roads = list(exit_roads)
        self.pcu_weights = dict(pcu_weights or DEFAULT_PCU_WEIGHTS)
        self.min_hold_frames = int(min_hold_frames)
        self.max_starvation_frames = int(max_starvation_frames)
        self.starvation_pcu_threshold = float(starvation_pcu_threshold)
        self.min_clearance_frames = int(min_clearance_frames)
        self.max_clearance_frames = int(max_clearance_frames)

        self.current_arm_idx = 0
        self.phase_state = 'green'          # 'green' | 'clearance'
        self.phase_elapsed = 0              # elapsed ticks in current state
        self.preempted = False

        # Telemetry & Performance Tracking
        self.saved_clearance_frames = 0
        self.total_phase_switches = 0
        self.total_green_frames_by_arm = {r: 0 for r in self.entry_roads}
        self.preemption_events = 0
        self.phase_history = []             # (arm, duration, reason)
        self.recent_clearance_durations = []

    @property
    def active_arm(self) -> int:
        return self.entry_roads[self.current_arm_idx]

    @property
    def phase_status(self) -> str:
        dur_s = self.phase_elapsed / 4.0
        if self.phase_state == 'green':
            return f"GREEN (Road {self.active_arm}) [{dur_s:.1f}s active]"
        return f"CLEARING BOX [{dur_s:.1f}s elapsed]"

    def compute_pcu(self, count_or_dict: Union[int, float, Dict[str, int]]) -> float:
        """Convert vehicle count or class dictionary into total Passenger Car Units (PCU)."""
        if isinstance(count_or_dict, (int, float)):
            return float(count_or_dict)
        if isinstance(count_or_dict, dict):
            return float(sum(
                count * self.pcu_weights.get(cls, 1.0)
                for cls, count in count_or_dict.items()
            ))
        return 0.0

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

    def calculate_pressures(self,
                            queue_data: Dict[int, Union[int, Dict[str, int]]],
                            approach_data: Optional[Dict[int, Union[int, Dict[str, int]]]] = None,
                            exit_data: Optional[Dict[int, Union[int, Dict[str, int]]]] = None) -> Dict[int, float]:
        """
        Compute heterogeneous PCU pressure for all entry arms:
          Pressure(r) = PCU_queue(r) + 0.5 * PCU_approach(r) - 0.2 * PCU_exit(downstream)
        """
        pressures = {}
        for r in self.entry_roads:
            q_pcu = self.compute_pcu(queue_data.get(r, 0))
            app_pcu = self.compute_pcu(approach_data.get(r, 0)) if approach_data else 0.0

            downstream_exit = ENTRY_TO_EXIT.get(r)
            exit_pcu = self.compute_pcu(exit_data.get(downstream_exit, 0)) if (exit_data and downstream_exit) else 0.0

            pressures[r] = float(max(0.0, q_pcu + 0.5 * app_pcu - 0.2 * exit_pcu))
        return pressures

    def step(self,
             queue_data: Optional[Dict[int, Union[int, Dict[str, int]]]] = None,
             approach_data: Optional[Dict[int, Union[int, Dict[str, int]]]] = None,
             exit_data: Optional[Dict[int, Union[int, Dict[str, int]]]] = None,
             box_occupied: bool = False,
             upstream_queues: Optional[Dict[int, Union[int, Dict[str, int]]]] = None,
             approaching_counts: Optional[Dict[int, Union[int, Dict[str, int]]]] = None,
             downstream_occupancy: Optional[Dict[int, float]] = None,
             cars_in_box_count: Optional[int] = None,
             **kwargs) -> bool:
        """
        Step the controller forward by 1 frame (0.25s).

        Supports both (queue_data, approach_data, exit_data, box_occupied)
        and (upstream_queues, approaching_counts, downstream_occupancy, cars_in_box_count).
        """
        if queue_data is None and upstream_queues is not None:
            queue_data = upstream_queues
        if approach_data is None and approaching_counts is not None:
            approach_data = approaching_counts
        if exit_data is None and downstream_occupancy is not None:
            exit_data = downstream_occupancy
        if cars_in_box_count is not None:
            box_occupied = (cars_in_box_count > 0)
        if queue_data is None:
            queue_data = {}

        self.phase_elapsed += 1
        active = self.active_arm

        if self.phase_state == 'green':
            self.total_green_frames_by_arm[active] += 1

            active_q_pcu = self.compute_pcu(queue_data.get(active, 0))
            active_app_pcu = self.compute_pcu(approach_data.get(active, 0)) if approach_data else 0.0
            total_active_demand = active_q_pcu + active_app_pcu

            # Calculate waiting pressures on competing arms
            pressures = self.calculate_pressures(queue_data, approach_data, exit_data)
            competing_max_pcu = max([pressures[r] for r in self.entry_roads if r != active], default=0.0)

            # 1. Anti-Starvation Preemption Check
            should_preempt = False
            preempt_reason = ""
            if self.phase_elapsed >= self.max_starvation_frames:
                should_preempt = True
                preempt_reason = f"Max hold limit ({self.phase_elapsed / 4:.1f}s)"
            elif competing_max_pcu >= self.starvation_pcu_threshold and total_active_demand <= 1.0:
                should_preempt = True
                preempt_reason = f"Trickle preemption (Cross queue {competing_max_pcu:.1f} PCU)"

            if should_preempt:
                self.preempted = True
                self.preemption_events += 1
                self.phase_history.append((active, self.phase_elapsed, f"Preempted: {preempt_reason}"))
                self.phase_state = 'clearance'
                self.phase_elapsed = 0
                return True

            # 2. Exhaustive Queue Clearing Check
            if total_active_demand <= 0.0 and self.phase_elapsed >= self.min_hold_frames:
                self.phase_history.append((active, self.phase_elapsed, "Exhausted Queue"))
                self.phase_state = 'clearance'
                self.phase_elapsed = 0
                return True

            return False

        elif self.phase_state == 'clearance':
            box_cleared = (not box_occupied and self.phase_elapsed >= self.min_clearance_frames)
            timeout = (self.phase_elapsed >= self.max_clearance_frames)

            if box_cleared or timeout:
                clearance_dur = self.phase_elapsed
                self.recent_clearance_durations.append(clearance_dur)

                # Saved frames vs historical 6.0s (24 frames) fixed buffer
                saved = max(0, 24 - clearance_dur)
                self.saved_clearance_frames += saved

                # 3. Next Phase Selection: Max-Pressure
                pressures = self.calculate_pressures(queue_data, approach_data, exit_data)
                candidates = [r for r in self.entry_roads if r != active]

                best_arm = max(candidates, key=lambda r: (
                    pressures.get(r, 0.0),
                    -((self.entry_roads.index(r) - self.current_arm_idx) % len(self.entry_roads))
                ))
                self.current_arm_idx = self.entry_roads.index(best_arm)

                self.phase_state = 'green'
                self.phase_elapsed = 0
                self.total_phase_switches += 1
                self.preempted = False
                return True

            return False

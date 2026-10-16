"""
tests/test_hetero_qc_controller.py — Unit Tests for HeteroQueueClearingController (H-QC-MPC)

Verifies:
  1. Accurate PCU weighting for small (0.4), medium (1.0), and large (2.8).
  2. Exhaustive green hold: green stays active as long as queued cars remain.
  3. Immediate phase transition the instant queue and approach demand reach zero.
  4. Dynamic box clearance:
     - Waits while box_occupied == True.
     - Ends immediately when box_occupied == False (e.g. in 1.5s instead of fixed 6.0s).
  5. Max-Pressure phase selection (highest PCU arm gets next green).
  6. Anti-starvation preemption under continuous trickle demand.
  7. Strict all-red enforcement during clearance intervals.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.max_pressure import (
    HeteroQueueClearingController,
    ENTRY_ROADS, ALL_EXITS, DEFAULT_PCU_WEIGHTS
)


def test_hetero_pcu_weighting():
    """Verify PCU weighting matches physical road space."""
    ctrl = HeteroQueueClearingController()

    mix = {'small': 5, 'medium': 4, 'large': 2}
    # 5 * 0.4 + 4 * 1.0 + 2 * 2.8 = 2.0 + 4.0 + 5.6 = 11.6 PCU
    pcu = ctrl.compute_pcu(mix)
    assert abs(pcu - 11.6) < 1e-6, f"Expected 11.6 PCU, got {pcu}"


def test_exhaustive_green_hold():
    """Verify green light holds indefinitely as long as active queue > 0."""
    ctrl = HeteroQueueClearingController(min_hold_frames=8, max_starvation_frames=500)
    assert ctrl.phase_state == 'green'
    active = ctrl.active_arm

    # Step for 120 frames (30 seconds) with constant queue
    for _ in range(120):
        ctrl.step(queue_data={active: {'medium': 8}}, box_occupied=False)
        assert ctrl.phase_state == 'green', "Green must NOT cut off while vehicles are waiting"
        assert ctrl.is_green(active), f"Active arm {active} must remain green"


def test_immediate_cutoff_on_queue_clearing():
    """Verify controller transitions to clearance immediately when queue clears."""
    ctrl = HeteroQueueClearingController(min_hold_frames=8)
    active = ctrl.active_arm

    # 10 frames with queue
    for _ in range(10):
        ctrl.step(queue_data={active: {'medium': 3}}, box_occupied=False)
    assert ctrl.phase_state == 'green'

    # Queue drops to zero
    ctrl.step(queue_data={active: {'medium': 0}}, box_occupied=False)
    assert ctrl.phase_state == 'clearance', "Must transition to clearance immediately when queue reaches 0"


def test_dynamic_box_clearance_and_time_saved():
    """
    Verify clearance waits while box is occupied, and terminates the instant
    the box is clear (saving time vs 6.0s fixed timer).
    """
    ctrl = HeteroQueueClearingController(min_hold_frames=4, min_clearance_frames=2)
    active = ctrl.active_arm

    # Step through green with queue until min_hold
    for _ in range(4):
        ctrl.step(queue_data={active: 1}, box_occupied=False)
    assert ctrl.phase_state == 'green'

    # Queue clears -> transitions to clearance
    ctrl.step(queue_data={active: 0}, box_occupied=True)
    assert ctrl.phase_state == 'clearance'

    # Vehicle is crossing inside junction box: box_occupied=True for 6 frames (1.5s)
    for _ in range(6):
        ctrl.step(queue_data={active: 0}, box_occupied=True)
        assert ctrl.phase_state == 'clearance', "Clearance must hold while vehicle is in box"

    # Vehicle exits box! box_occupied becomes False
    ctrl.step(queue_data={active: 0}, box_occupied=False)
    assert ctrl.phase_state == 'green', "Green must activate immediately after vehicle exits box"

    # Total clearance was ~7 frames (1.75s). Saved 24 - 7 = 17 frames vs 6.0s fixed gap!
    assert ctrl.saved_clearance_frames >= 15, f"Expected >= 15 saved frames, got {ctrl.saved_clearance_frames}"


def test_max_pressure_phase_selection():
    """Verify that when green switches, the arm with the highest PCU queue wins green."""
    ctrl = HeteroQueueClearingController(min_hold_frames=4, min_clearance_frames=2)
    active = ctrl.active_arm  # Starts at road 2

    # Clear active arm
    for _ in range(5):
        ctrl.step(queue_data={active: 0}, box_occupied=False)
    assert ctrl.phase_state == 'clearance'

    # Road 4 has 2 small (0.8 PCU), Road 6 has 4 large (11.2 PCU), Road 8 has 1 car (1.0 PCU)
    queues = {
        2: 0,
        4: {'small': 2},
        6: {'large': 4},
        8: {'medium': 1}
    }

    # Clear the box
    ctrl.step(queue_data=queues, box_occupied=False)
    assert ctrl.phase_state == 'green'
    # Road 6 has 11.2 PCU (highest) -> Road 6 MUST be selected!
    assert ctrl.active_arm == 6, f"Expected Road 6 to win green, got Road {ctrl.active_arm}"


def test_anti_starvation_preemption():
    """Verify trickle on green arm is preempted if cross arm queue exceeds 15 PCU."""
    ctrl = HeteroQueueClearingController(
        min_hold_frames=8,
        starvation_pcu_threshold=15.0
    )
    active = ctrl.active_arm

    # Active arm has a trickle of 1 car (1.0 PCU), but Road 4 has a huge queue of 6 large trucks (16.8 PCU)
    queues = {
        active: {'medium': 1},
        4: {'large': 6}
    }

    ctrl.step(queue_data=queues, box_occupied=False)
    assert ctrl.phase_state == 'clearance', "Trickle must be preempted when cross-arm exceeds 15 PCU"
    assert ctrl.preempted is True


def test_all_red_safety_during_clearance():
    """Verify strict all-red enforcement during clearance intervals (zero green roads)."""
    ctrl = HeteroQueueClearingController()
    active = ctrl.active_arm

    # Move into clearance
    for _ in range(10):
        ctrl.step(queue_data={active: 0}, box_occupied=True)

    assert ctrl.phase_state == 'clearance'
    assert ctrl.green_roads() == (), "No road may be green during clearance"
    for r in ENTRY_ROADS:
        assert not ctrl.is_green(r), f"Entry road {r} must be RED during clearance"

"""
tests/test_hetero_maxpressure.py — Unit Tests for Heterogeneous Max-Pressure Signal Controller

Verifies:
  1. Accurate PCU weighting for mixed traffic compositions (IRC: 106-1990).
  2. Convergence of online turning ratio estimator (R_hat) via EMA.
  3. Downstream backpressure dampening on congested exit approaches.
  4. Strict preservation of the 6.0s (24 frames @ 4 FPS) all-red clearance interval.
  5. Dynamic green duration scaling bounded by [g_min, g_max].
  6. Early gap-out termination upon queue evacuation.
  7. Elastic green extension for incoming vehicle platoons.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.baseline_max_pressure import (
    HeterogeneousMaxPressureController,
    ENTRY_ROADS, ALL_EXITS, DEFAULT_PCU_WEIGHTS
)


def test_pcu_weighting():
    """Verify IRC 106-1990 PCU factor calculation for mixed vehicles."""
    ctrl = HeterogeneousMaxPressureController()

    mix = {'small': 10, 'medium': 5, 'large': 2}
    # 10 * 0.5 + 5 * 1.0 + 2 * 2.5 = 5.0 + 5.0 + 5.0 = 15.0 PCU
    pcu = ctrl.compute_pcu(mix)
    assert abs(pcu - 15.0) < 1e-6, f"Expected 15.0 PCU, got {pcu}"

    # Scalar backward compatibility
    assert ctrl.compute_pcu(8.0) == 8.0


def test_downstream_backpressure():
    """Verify that exit arm congestion dampens movement pressure."""
    ctrl = HeterogeneousMaxPressureController()

    queue_uncongested = {2: {'medium': 10}}   # 10 PCU
    exit_empty = {1: {'medium': 0}}
    pressures_free = ctrl.calculate_hetero_pressures(queue_uncongested, exit_data=exit_empty)
    w_free = pressures_free[2]['total_pressure']

    exit_blocked = {1: {'medium': 8}}         # 8 PCU downstream queue
    pressures_blocked = ctrl.calculate_hetero_pressures(queue_uncongested, exit_data=exit_blocked)
    w_blocked = pressures_blocked[2]['total_pressure']

    assert w_blocked < w_free, f"Backpressure must reduce movement pressure: {w_blocked} < {w_free}"


def test_online_turning_ratio_estimation():
    """Verify that dynamic online EMA updates turning movement proportions."""
    ctrl = HeterogeneousMaxPressureController(turning_learning_rate=0.20)

    # Initially dominant straight (Exit 1 for Entry 2)
    assert ctrl.R_hat[2][1] == 0.70

    # Discharge 20 vehicles all turning into Exit 3 (left turn)
    for _ in range(20):
        ctrl.record_discharge(entry_road=2, exit_road=3, vehicle_class='medium')

    ctrl._update_turning_ratios(entry_road=2)

    # R_hat[2][3] should increase significantly from ~0.10
    assert ctrl.R_hat[2][3] > 0.20, f"Turning ratio for exit 3 should increase: {ctrl.R_hat[2][3]}"
    # Sum of turning probabilities across all exits should stay close to 1.0
    total_ratio = sum(ctrl.R_hat[2].values())
    assert abs(total_ratio - 1.0) < 0.15, f"R_hat should remain normalized: sum={total_ratio}"


def test_clearance_gap_guarantee_6_seconds():
    """Verify that the 6.0s (24 frames @ 4 FPS) all-red buffer is strictly enforced between all green phases."""
    ctrl = HeterogeneousMaxPressureController(
        min_green_frames=100,
        max_green_frames=140,
        clearance_frames=24
    )

    assert ctrl.phase_state == 'green'
    first_arm = ctrl.active_arm

    # Run through entire green until clearance
    for _ in range(150):
        ctrl.step(queue_data={first_arm: {'medium': 10}})
        if ctrl.phase_state == 'clearance':
            break

    assert ctrl.phase_state == 'clearance', "Phase must transition to clearance after green expiration"

    # Step through clearance and verify NO entry road is green for all 24 frames
    clearance_ticks = 0
    while ctrl.phase_state == 'clearance':
        assert ctrl.green_roads() == (), "No road may be green during all-red clearance"
        for r in ENTRY_ROADS:
            assert not ctrl.is_green(r), f"Entry road {r} must be RED during clearance"
        ctrl.step(queue_data={first_arm: {'medium': 10}})
        clearance_ticks += 1

    assert clearance_ticks == 24, f"Clearance phase must last exactly 24 frames (6.0s), lasted {clearance_ticks}"
    assert ctrl.phase_state == 'green', "Controller must switch to green after clearance"
    assert ctrl.active_arm != first_arm, "Next green phase must serve next arm in cycle"


def test_dynamic_green_scaling():
    """Verify dynamic green window scales with PCU pressure within bounds [100, 140]."""
    ctrl = HeterogeneousMaxPressureController(min_green_frames=100, max_green_frames=140)

    # Low queue -> min green
    g_low = ctrl._scale_green_duration(pressure_val=0.5)
    assert g_low == 100, f"Expected min green 100, got {g_low}"

    # Moderate queue -> intermediate green
    g_med = ctrl._scale_green_duration(pressure_val=6.0)
    assert 100 < g_med < 140, f"Expected intermediate green, got {g_med}"

    # Heavy queue -> max green
    g_high = ctrl._scale_green_duration(pressure_val=25.0)
    assert g_high == 140, f"Expected max green 140, got {g_high}"


def test_early_gap_out_cutoff():
    """Verify early cutoff occurs when PCU queue empties after min_green."""
    ctrl = HeterogeneousMaxPressureController(min_green_frames=100, max_green_frames=140)

    # Empty queue from start
    empty_q = {r: 0 for r in ENTRY_ROADS}

    for f in range(1, 101):
        ctrl.step(queue_data=empty_q)
        if f < 100:
            assert ctrl.phase_state == 'green', f"Must hold green until min_green (frame {f})"

    # At frame 101, queue is empty -> triggers early cutoff
    ctrl.step(queue_data=empty_q)
    assert ctrl.phase_state == 'clearance', "Must trigger early cutoff into clearance"
    assert ctrl.early_cutoffs == 1, "Must increment early_cutoffs counter"
    assert ctrl.saved_green_frames > 0, "Must record saved green frames"


def test_elastic_green_extension():
    """Verify approaching platoons extend green when approaching target expiration."""
    ctrl = HeterogeneousMaxPressureController(min_green_frames=100, max_green_frames=140)

    # Run to frame 95
    for _ in range(95):
        ctrl.step(queue_data={ctrl.active_arm: {'medium': 2}})

    initial_target = ctrl.target_green_duration

    # Incoming platoon appears in passage zone
    approach_platoon = {ctrl.active_arm: {'medium': 4}}
    ctrl.step(queue_data={ctrl.active_arm: {'medium': 0}}, approach_data=approach_platoon)

    assert ctrl.target_green_duration >= initial_target, "Approaching platoon should extend green duration"
    assert ctrl.phase_state == 'green', "Must remain green during elastic extension"

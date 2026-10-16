"""
tests/test_vehicle_online_learner.py — Unit Tests for Online Self-Optimizing Vehicle Learner

Verifies:
  1. Contextual prior differentiation across classes and geometric dimensions.
  2. Recursive Dual-EKF parameter convergence towards true driver behavior.
  3. Bounded feasibility projection (no NaN, no unphysical values).
  4. High-performance execution on pure CPU (< 50 microseconds per update).
  5. Vehicle lifecycle management (registration, active tracking, archival).
"""

import time
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vehicle_learner import VehicleOnlineLearner, PARAM_BOUNDS, CLASS_PRIORS


def test_contextual_prior_differentiation():
    """Verify that different vehicle classes and dimensions receive customized priors."""
    learner = VehicleOnlineLearner()

    # Small two-wheeler (short length, high agility)
    p_small = learner.register_vehicle(1, 'small', length_px=30.0, width_px=15.0, v_entry=50.0)
    # Large heavy truck (long length, sluggish)
    p_large = learner.register_vehicle(2, 'large', length_px=180.0, width_px=45.0, v_entry=15.0)

    # Two-wheeler should have smaller headway T and higher acceleration a_max
    assert p_small['IDM_T'] < p_large['IDM_T'], f"Bike headway ({p_small['IDM_T']}) must be < Truck ({p_large['IDM_T']})"
    assert p_small['IDM_A'] > p_large['IDM_A'], f"Bike accel ({p_small['IDM_A']}) must be > Truck ({p_large['IDM_A']})"
    assert p_small['IDM_S0'] < p_large['IDM_S0'], f"Bike gap ({p_small['IDM_S0']}) must be < Truck ({p_large['IDM_S0']})"


def test_ekf_parameter_convergence():
    """Verify that Dual-EKF adjusts a vehicle's a_max when observed acceleration is consistently higher."""
    learner = VehicleOnlineLearner(learning_rate=1.0)
    p_init = learner.register_vehicle(10, 'medium', length_px=80.0, width_px=35.0, v_entry=30.0)
    init_a = p_init['IDM_A']

    # Vehicle is observed accelerating aggressively: 2.5 m/s^2 when predicted would be ~1.2 m/s^2
    for step in range(25):
        p_updated = learner.update_vehicle(
            veh_id=10,
            v=4.0,           # 4 m/s speed
            v0=12.0,         # desired 12 m/s
            g=15.0,          # 15m gap
            dv=0.0,          # matching speeds
            observed_accel=2.5 # aggressive driver
        )

    # a_max should have increased towards 2.5
    assert p_updated['IDM_A'] > init_a, f"Expected a_max to increase from {init_a}, got {p_updated['IDM_A']}"


def test_parameter_bounds_enforcement():
    """Verify that parameter updates remain strictly within physical limits under extreme residuals."""
    learner = VehicleOnlineLearner(learning_rate=5.0) # Intentionally high learning rate
    learner.register_vehicle(99, 'medium')

    # Subject to massive negative shock (-20 m/s^2) and massive positive shock (+30 m/s^2)
    for _ in range(20):
        p = learner.update_vehicle(99, v=5.0, v0=10.0, g=5.0, dv=-5.0, observed_accel=-20.0)
        assert PARAM_BOUNDS['IDM_T'][0] <= p['IDM_T'] <= PARAM_BOUNDS['IDM_T'][1]
        assert PARAM_BOUNDS['IDM_A'][0] <= p['IDM_A'] <= PARAM_BOUNDS['IDM_A'][1]
        assert PARAM_BOUNDS['IDM_S0'][0] <= p['IDM_S0'] <= PARAM_BOUNDS['IDM_S0'][1]

    for _ in range(20):
        p = learner.update_vehicle(99, v=5.0, v0=10.0, g=5.0, dv=5.0, observed_accel=30.0)
        assert PARAM_BOUNDS['IDM_T'][0] <= p['IDM_T'] <= PARAM_BOUNDS['IDM_T'][1]
        assert PARAM_BOUNDS['IDM_A'][0] <= p['IDM_A'] <= PARAM_BOUNDS['IDM_A'][1]
        assert PARAM_BOUNDS['IDM_S0'][0] <= p['IDM_S0'] <= PARAM_BOUNDS['IDM_S0'][1]


def test_cpu_runtime_performance():
    """Verify that EKF parameter estimation executes in under 50 microseconds per vehicle on CPU."""
    learner = VehicleOnlineLearner()
    learner.register_vehicle(42, 'medium')

    n_trials = 1000
    t0 = time.perf_counter()
    for _ in range(n_trials):
        learner.update_vehicle(42, v=6.0, v0=12.0, g=10.0, dv=0.5, observed_accel=1.2)
    elapsed = time.perf_counter() - t0

    time_per_update_us = (elapsed / n_trials) * 1e6
    assert time_per_update_us < 50.0, f"Expected < 50 us per update, got {time_per_update_us:.2f} us"


def test_lifecycle_archival():
    """Verify registration, tracking, and post-exit archival."""
    learner = VehicleOnlineLearner()
    learner.register_vehicle(101, 'small')
    assert 101 in learner.vehicles

    learner.update_vehicle(101, v=3.0, v0=10.0, g=8.0, dv=0.0, observed_accel=1.5)
    learner.deregister_vehicle(101)

    assert 101 not in learner.vehicles
    assert 101 in learner.history
    assert 'IDM_T' in learner.history[101]

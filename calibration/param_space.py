"""
calibration/param_space.py — Parameter Space and Live Module Dynamic Configuration.

Defines tunable physical parameter bounds and runtime setters for:
  - Intelligent Driver Model (IDM) car-following parameters (a_max, b_max, T, s0)
  - Lateral potential field parameters (drive force scale, wall repulsion strength/range)
  - Vehicle class-specific spatial clearance and repulsion profiles
"""

import models.idm as idm_mod
import models.potential_fields as pot_mod
from simulation.config import VEHICLE_CLASSES

def _set_idm(attr, val):
    setattr(idm_mod, attr, float(val))

def _set_pot(attr, val):
    setattr(pot_mod, attr, float(val))

def _vp_get(cls, key):
    return lambda: VEHICLE_CLASSES.get(cls, {}).get(key, 0.0)

def _vp_set(cls, key):
    def set_(v):
        if cls in VEHICLE_CLASSES:
            VEHICLE_CLASSES[cls][key] = float(v)
    return set_

PARAMS = [
    ("idm.IDM_A",  lambda: getattr(idm_mod, "IDM_A", 0.8),  lambda v: _set_idm("IDM_A", v),  0.3, 4.0),
    ("idm.IDM_B",  lambda: getattr(idm_mod, "IDM_B", 2.5),  lambda v: _set_idm("IDM_B", v),  0.5, 5.0),
    ("idm.IDM_T",  lambda: getattr(idm_mod, "IDM_T", 0.6),  lambda v: _set_idm("IDM_T", v),  0.05, 2.0),
    ("idm.IDM_S0", lambda: getattr(idm_mod, "IDM_S0", 0.3), lambda v: _set_idm("IDM_S0", v), 0.05, 1.0),

    ("lane.DRIVE_FORCE_SCALE",       lambda: getattr(pot_mod, "DRIVE_FORCE_SCALE", 100.0),
     lambda v: _set_pot("DRIVE_FORCE_SCALE", v), 20.0, 4000.0),
    ("lane.WALL_REPULSION_STRENGTH", lambda: getattr(pot_mod, "WALL_REPULSION_STRENGTH", 400.0),
     lambda v: _set_pot("WALL_REPULSION_STRENGTH", v), 10.0, 8000.0),
    ("lane.WALL_REPULSION_RANGE",    lambda: getattr(pot_mod, "WALL_REPULSION_RANGE", 1.5),
     lambda v: _set_pot("WALL_REPULSION_RANGE", v), 0.2, 4.0),
]

for _cls in ("small", "medium", "large"):
    PARAMS.append((f"vp.{_cls}.repulsion_strength", _vp_get(_cls, "repulsion_strength"),
                   _vp_set(_cls, "repulsion_strength"), 200.0, 250000.0))
    PARAMS.append((f"vp.{_cls}.repulsion_range", _vp_get(_cls, "repulsion_range"),
                   _vp_set(_cls, "repulsion_range"), 0.02, 5.0))

def capture() -> dict:
    """Snapshot of the parameter values currently live on the modules."""
    return {name: get() for name, get, _, _, _ in PARAMS}

def apply(values: dict):
    """Write parameter values onto live modules."""
    for name, get, set_, low, high in PARAMS:
        if name in values:
            set_(values[name])

"""
param_space.py - defines which parameters are tunable, their bounds, and how
to read/write them on the live modules. Shared by tune_params.py (search) and
param_profile.py (capture/apply saved profiles).
"""
import settings
from Forces import idm1, lane_forces
try:
    from Forces import idm1_verified
except ImportError:
    idm1_verified = None


def _set_idm(attr, val):
    setattr(idm1, attr, val)
    if idm1_verified is not None:
        setattr(idm1_verified, attr, val)


def _vp_get(cls, key):
    return lambda: settings.VEHICLE_PARAMS[cls][key]


def _vp_set(cls, key):
    def set_(v):
        settings.VEHICLE_PARAMS[cls][key] = v
    return set_


PARAMS = [
    ("idm.IDM_A",  lambda: idm1.IDM_A,  lambda v: _set_idm("IDM_A", v),  0.3, 4.0),
    ("idm.IDM_B",  lambda: idm1.IDM_B,  lambda v: _set_idm("IDM_B", v),  0.5, 5.0),
    ("idm.IDM_T",  lambda: idm1.IDM_T,  lambda v: _set_idm("IDM_T", v),  0.05, 2.0),
    ("idm.IDM_S0", lambda: idm1.IDM_S0, lambda v: _set_idm("IDM_S0", v), 0.05, 1.0),

    ("lane.DRIVE_FORCE_SCALE",       lambda: lane_forces.DRIVE_FORCE_SCALE,
     lambda v: setattr(lane_forces, "DRIVE_FORCE_SCALE", v), 20.0, 4000.0),
    ("lane.WALL_REPULSION_STRENGTH", lambda: lane_forces.WALL_REPULSION_STRENGTH,
     lambda v: setattr(lane_forces, "WALL_REPULSION_STRENGTH", v), 10.0, 8000.0),
    ("lane.WALL_REPULSION_RANGE",    lambda: lane_forces.WALL_REPULSION_RANGE,
     lambda v: setattr(lane_forces, "WALL_REPULSION_RANGE", v), 0.2, 4.0),
]
# Upper bounds on repulsion_strength/range were raised from the original
# 15000/1.5 -- at the old 1.5m range cap, repulsion decays to ~0 well before
# a real following gap (several metres), so it could never be strong enough
# to compete with F_drive and let a blocked car find a gap around a slow
# leader; confirmed manually (see KNOWN_ISSUES.md-style test) that ~15x
# strength + ~3m range measurably helps a stuck car make progress. Widened
# so the search can actually explore that regime instead of being fenced
# out of it, not because those exact multipliers are known-good.
for _cls in ("small", "medium", "large"):
    PARAMS.append((f"vp.{_cls}.repulsion_strength", _vp_get(_cls, "repulsion_strength"),
                   _vp_set(_cls, "repulsion_strength"), 200.0, 250000.0))
    PARAMS.append((f"vp.{_cls}.repulsion_range", _vp_get(_cls, "repulsion_range"),
                   _vp_set(_cls, "repulsion_range"), 0.02, 5.0))


def capture() -> dict:
    """Snapshot of the parameter values currently live on the modules."""
    return {name: get() for name, get, _, _, _ in PARAMS}


def apply(values: dict) -> None:
    """Push a {name: value} dict (e.g. loaded from a profile JSON) onto the modules."""
    by_name = {name: (set_, lo, hi) for name, _, set_, lo, hi in PARAMS}
    for name, v in values.items():
        if name not in by_name:
            continue
        set_, lo, hi = by_name[name]
        set_(max(lo, min(hi, float(v))))

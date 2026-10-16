"""
param_profile.py - save/load named parameter profiles as JSON under
new_sim/params_profiles/. The hardcoded values in settings.py, Forces/idm1.py
and Forces/lane_forces.py are never edited by this -- "current" always stays
exactly as it is in source. Profiles are an opt-in overlay applied in memory
at runtime (e.g. `python evaluate.py --profile params_profiles/best.json`).
"""
import json
import os
import time

import param_space

PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params_profiles")
BASELINE_PATH = os.path.join(PROFILE_DIR, "baseline.json")


def ensure_baseline_captured():
    """Write the original/current parameter values once, if not already saved.
    This is the permanent record of 'current' -- never overwritten."""
    os.makedirs(PROFILE_DIR, exist_ok=True)
    if os.path.exists(BASELINE_PATH):
        return
    save(BASELINE_PATH, param_space.capture(), meta={"note": "original values captured from source, before any tuning"})


def save(path, values: dict, meta: dict = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"params": values, "meta": meta or {}, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load(path) -> dict:
    with open(path) as f:
        data = json.load(f)
    return data["params"]


def apply_from_file(path):
    """Load a profile JSON and apply it on top of the current in-memory values."""
    param_space.apply(load(path))

"""
calibration/param_profile.py — Named Calibration Parameter Profile Persistence.

Saves and applies JSON parameter profiles onto simulation and model modules at runtime.
"""

import json
import os
import time
from calibration import param_space

PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")
BASELINE_PATH = os.path.join(PROFILE_DIR, "universal_profile.json")


def save(path: str, values: dict, meta: dict = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"params": values, "meta": meta or {}, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return data.get("params", data)


def apply_from_file(path: str):
    """Load a profile JSON and apply it on top of the current in-memory values."""
    if os.path.exists(path):
        param_space.apply(load(path))

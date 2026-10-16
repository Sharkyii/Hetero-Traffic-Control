"""
Drone Trajectory Dataset Loader.

Parses empirical vehicle tracking CSVs, organizes trajectory points by frame and track ID,
and identifies stationary/parked vehicles to isolate dynamic traffic queues.
"""

import csv
import os
from typing import Dict, List, Set, Tuple
import numpy as np
from simulation.config import METERS_PER_PIXEL

STATIC_MOVEMENT_THRESHOLD = 50.0  # Pixels
STATIC_SPEED_THRESHOLD = 0.5      # Empirical speed units


class TrajectoryDataLoader:
    """
    Loads, parses, and indexes drone tracking data from CSV format.
    """
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.cars_by_frame: Dict[int, List[dict]] = {}
        self.cars_by_id: Dict[int, List[dict]] = {}
        self.max_frame = 0
        self.min_x = float("inf")
        self.max_x = float("-inf")
        self.min_y = float("inf")
        self.max_y = float("-inf")
        self.static_ids: Set[int] = set()

        self._load_data()
        self._find_static_cars()

    def _load_data(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Trajectory CSV file not found: {self.csv_path}")

        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    frame_id = row.get("frame", "")
                    track_id = int(row["track_id"])
                    x = float(row["x"])
                    y = float(row["y"])
                    speed = float(row.get("robust_speed", row.get("speed", 0.0)))
                    w = float(row.get("w", 20.0))
                    h = float(row.get("h", 40.0))

                    self.min_x = min(self.min_x, x)
                    self.max_x = max(self.max_x, x)
                    self.min_y = min(self.min_y, y)
                    self.max_y = max(self.max_y, y)

                    try:
                        frame_num = int(row.get("frame_num", frame_id.split("_")[1].split(".")[0]))
                    except Exception:
                        frame_num = 0

                    self.max_frame = max(self.max_frame, frame_num)

                    car_data = {
                        "track_id": track_id,
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "speed": speed,
                        "robust_speed": speed,
                        "frame_id": frame_id,
                        "frame_num": frame_num,
                    }

                    if frame_num not in self.cars_by_frame:
                        self.cars_by_frame[frame_num] = []
                    self.cars_by_frame[frame_num].append(car_data)

                    if track_id not in self.cars_by_id:
                        self.cars_by_id[track_id] = []
                    self.cars_by_id[track_id].append(car_data)

                except (ValueError, KeyError):
                    continue

        for tid in self.cars_by_id:
            self.cars_by_id[tid].sort(key=lambda c: c["frame_num"])

    def get_cars_for_frame(self, frame_num: int) -> List[dict]:
        return self.cars_by_frame.get(frame_num, [])

    def get_new_cars_for_frame(self, frame_num: int) -> List[dict]:
        frame_cars = self.get_cars_for_frame(frame_num)
        new_cars = []
        for car in frame_cars:
            tid = car["track_id"]
            if self.cars_by_id[tid][0]["frame_num"] == frame_num:
                new_cars.append(car)
        return new_cars

    def _find_static_cars(self):
        self.static_ids = set()
        for tid, trajectory in self.cars_by_id.items():
            if not trajectory:
                continue
            max_speed = max(p["speed"] for p in trajectory)
            if max_speed < STATIC_SPEED_THRESHOLD:
                self.static_ids.add(tid)
                continue

            start_pos = np.array([trajectory[0]["x"], trajectory[0]["y"]])
            end_pos = np.array([trajectory[-1]["x"], trajectory[-1]["y"]])
            displacement = float(np.linalg.norm(end_pos - start_pos))
            if displacement < STATIC_MOVEMENT_THRESHOLD:
                self.static_ids.add(tid)
DataLoader = TrajectoryDataLoader

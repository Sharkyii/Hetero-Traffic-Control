
import csv
import os
import numpy as np
from settings import *
from constants import *

class DataLoader:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.cars_by_frame = {}
        self.cars_by_id = {}
        self.max_frame = 0
        self.min_x = float('inf')
        self.max_x = float('-inf')
        self.min_y = float('inf')
        self.max_y = float('-inf')
        self.dimensions_by_id = {} # track_id -> (w, h)
        self._load_data()
        ann_path = os.path.join(os.path.dirname(csv_path), "selected_area_annotations.json")
        if not os.path.exists(ann_path):
            ann_path = "selected_area_annotations.json"  # back-compat: root-level runs
        self._load_dimensions(ann_path)
        self._find_static_cars()

    def _load_dimensions(self, json_path):
        import json
        if not os.path.exists(json_path):
            return
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        for item in data['items']:
            for ann in item['annotations']:
                if ann['type'] == 'bbox':
                    tid = ann['attributes']['track_id']
                    if tid not in self.dimensions_by_id:
                        self.dimensions_by_id[tid] = (ann['bbox'][2], ann['bbox'][3])

    def _load_data(self):
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    frame_id = row['frame']
                    track_id = int(row['track_id'])
                    x = float(row['x'])
                    y = float(row['y'])
                    # Roboust speed seems to be the most reliable speed metric in the dataset
                    speed = float(row['robust_speed']) 
                    
                    # Update coordinate bounds
                    self.min_x = min(self.min_x, x)
                    self.max_x = max(self.max_x, x)
                    self.min_y = min(self.min_y, y)
                    self.max_y = max(self.max_y, y)

                    # Extract numeric frame index if possible (assuming format frame_XXXXXX.PNG)
                    try:
                        frame_num = int(row.get('frame_num', frame_id.split('_')[1].split('.')[0]))
                        self.max_frame = max(self.max_frame, frame_num)
                    except:
                        frame_num = 0 # Default if parsing fails

                    car_data = {
                        'track_id': track_id,
                        'x': x,
                        'y': y,
                        'w': float(row.get('w', DEFAULT_CAR_W)),
                        'h': float(row.get('h', DEFAULT_CAR_H)),
                        'speed': speed,
                        'robust_speed': speed, # Store both as speed and robust_speed
                        'frame_id': frame_id,
                        'frame_num': frame_num,
                        'time': float(row.get('time', 0.0))
                    }

                    # Organize by frame
                    if frame_num not in self.cars_by_frame:
                        self.cars_by_frame[frame_num] = []
                    self.cars_by_frame[frame_num].append(car_data)

                    # Organize by track ID (to find start frame)
                    if track_id not in self.cars_by_id:
                        self.cars_by_id[track_id] = []
                    self.cars_by_id[track_id].append(car_data)

                except ValueError as e:
                    continue # Skip invalid rows

        # Sort data for each car by frame to track trajectory
        for tid in self.cars_by_id:
            self.cars_by_id[tid].sort(key=lambda c: c['frame_num'])

    def get_cars_for_frame(self, frame_num):
        """Get all cars present in a specific frame."""
        return self.cars_by_frame.get(frame_num, [])

    def get_new_cars_for_frame(self, frame_num):
        """Get cars that APPEAR for the first time in this frame."""
        frame_cars = self.get_cars_for_frame(frame_num)
        new_cars = []
        for car in frame_cars:
            track_id = car['track_id']
            # Check if this is the first frame for this car
            if self.cars_by_id[track_id][0]['frame_num'] == frame_num:
                new_cars.append(car)
        return new_cars

    def get_bounds(self):
        return (self.min_x, self.max_x, self.min_y, self.max_y)

    def _find_static_cars(self):
        """Identify cars that never move significantly."""
        self.static_ids = set()
        MOVEMENT_THRESHOLD = STATIC_MOVEMENT_THRESHOLD # Pixels
        SPEED_THRESHOLD = STATIC_SPEED_THRESHOLD # Unit speed
        
        for tid, trajectory in self.cars_by_id.items():
            if not trajectory:
                continue
                
            # Check 1: Max Speed
            max_speed = max(p['speed'] for p in trajectory)
            if max_speed < SPEED_THRESHOLD:
                self.static_ids.add(tid)
                continue
                
            # Check 2: Total Displacement
            start_pos = np.array([trajectory[0]['x'], trajectory[0]['y']])
            end_pos = np.array([trajectory[-1]['x'], trajectory[-1]['y']])
            displacement = np.linalg.norm(end_pos - start_pos)
            
            if displacement < MOVEMENT_THRESHOLD:
                self.static_ids.add(tid)
                
        print(f"Identified {len(self.static_ids)} static cars out of {len(self.cars_by_id)}.")

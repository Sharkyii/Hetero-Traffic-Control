"""
Process Polygon
===============
Step 2: Reads polygon_points.json (from point_selector.py),
crops all frames to the polygon area, and copies selected_area_data.csv.

Run from new_sim/:
    python process_polygon.py
    python process_polygon.py --images OneDrive_4_20-06-2026/images --csv segments/od4_seg1/full_data.csv \
        --polygon segments/od4_seg1/polygon_points.json --out-folder segments/od4_seg1/selected_area_cropped \
        --out-csv segments/od4_seg1/selected_area_data.csv --out-json segments/od4_seg1/selected_area_annotations.json \
        --frame-start 0 --frame-end 300
"""

import argparse
import cv2
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from shapely.geometry import Polygon, Point

POLYGON_COLOR   = (0, 255, 255)
POLYGON_THICK   = 3
RED_LINE_COLOR  = (0, 0, 255)
RED_LINE_THICK  = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images",     default=r"images1\images")
    ap.add_argument("--csv",        default="selected_area_data.csv", help="trajectory CSV (e.g. from convert_cvat.py)")
    ap.add_argument("--polygon",    default="polygon_points.json")
    ap.add_argument("--out-folder", default="selected_area_cropped")
    ap.add_argument("--out-csv",    default="selected_area_data.csv")
    ap.add_argument("--out-json",   default="selected_area_annotations.json")
    ap.add_argument("--frame-start", type=int, default=None)
    ap.add_argument("--frame-end",   type=int, default=None, help="exclusive upper bound")
    args = ap.parse_args()

    IMAGES_FOLDER, TRAJECTORY_CSV, POLYGON_JSON = args.images, args.csv, args.polygon
    OUTPUT_FOLDER, OUTPUT_CSV, OUTPUT_JSON = args.out_folder, args.out_csv, args.out_json

    # ── load polygon + red line ───────────────────────────────────────────────
    if not os.path.exists(POLYGON_JSON):
        print(f"ERROR: {POLYGON_JSON} not found. Run point_selector.py first.")
        return

    with open(POLYGON_JSON) as f:
        pdata = json.load(f)

    polygon_pts = pdata["polygon_points"]   # [[x,y], ...]
    red_line    = pdata.get("red_line", []) # [[x,y], [x,y]]
    img_w       = pdata["image_dimensions"]["width"]
    img_h       = pdata["image_dimensions"]["height"]

    poly_np      = np.array(polygon_pts, dtype=np.int32)
    poly_shapely = Polygon(poly_np)

    xs = [p[0] for p in polygon_pts]
    ys = [p[1] for p in polygon_pts]
    x_min, x_max = max(0, min(xs)), min(img_w, max(xs))
    y_min, y_max = max(0, min(ys)), min(img_h, max(ys))

    print(f"Polygon: {polygon_pts}")
    print(f"Red line: {red_line}")
    print(f"Crop box: ({x_min},{y_min}) -> ({x_max},{y_max})")

    # ── output folder ─────────────────────────────────────────────────────────
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)

    # ── load trajectory CSV ───────────────────────────────────────────────────
    df = pd.read_csv(TRAJECTORY_CSV)
    print(f"Loaded {len(df)} rows from CSV, frames {df['frame_num'].min()}–{df['frame_num'].max()}")

    # ── process images ────────────────────────────────────────────────────────
    annotations = {"items": [], "trajectories": {}}
    track_dets  = {}

    for fname in sorted(os.listdir(IMAGES_FOLDER)):
        if not fname.lower().endswith(('.png', '.jpg')):
            continue

        # parse frame number
        stem = os.path.splitext(fname)[0]  # "frame_000111"
        try:
            frame_num = int(stem.split('_')[1])
        except (IndexError, ValueError):
            continue

        if args.frame_start is not None and frame_num < args.frame_start:
            continue
        if args.frame_end is not None and frame_num >= args.frame_end:
            continue

        img = cv2.imread(os.path.join(IMAGES_FOLDER, fname))
        if img is None:
            continue

        h_img, w_img = img.shape[:2]

        # scale polygon if image size differs from JSON
        scale = w_img / img_w if img_w > 0 else 1.0
        if abs(scale - 1.0) > 0.01:
            sp = [[int(p[0]*scale), int(p[1]*scale)] for p in polygon_pts]
            rl = [[int(p[0]*scale), int(p[1]*scale)] for p in red_line]
            sx_min = max(0, int(x_min*scale))
            sy_min = max(0, int(y_min*scale))
            sx_max = min(w_img, int(x_max*scale))
            sy_max = min(h_img, int(y_max*scale))
        else:
            sp = polygon_pts
            rl = red_line
            sx_min, sy_min, sx_max, sy_max = x_min, y_min, x_max, y_max

        # mask + crop
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, [np.array(sp, dtype=np.int32)], (255, 255, 255))
        masked = cv2.bitwise_and(img, mask)
        cropped = masked[sy_min:sy_max, sx_min:sx_max].copy()

        # draw polygon outline on crop
        adj_poly = [[p[0]-sx_min, p[1]-sy_min] for p in sp]
        cv2.polylines(cropped, [np.array(adj_poly, dtype=np.int32)],
                      True, POLYGON_COLOR, POLYGON_THICK)

        # draw red line on crop
        if len(rl) == 2:
            p1 = (rl[0][0]-sx_min, rl[0][1]-sy_min)
            p2 = (rl[1][0]-sx_min, rl[1][1]-sy_min)
            cv2.line(cropped, p1, p2, RED_LINE_COLOR, RED_LINE_THICK)

        out_name = f"frame_{frame_num:06d}_selected_area.jpg"
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, out_name), cropped)

        # ── collect car detections for this frame ─────────────────────────────
        frame_df = df[df['frame_num'] == frame_num]
        frame_anns = []
        for _, row in frame_df.iterrows():
            tid = int(row['track_id'])
            x, y = float(row['x']), float(row['y'])
            w = float(row.get('w', 20))
            h = float(row.get('h', 10))
            if poly_shapely.contains(Point(x, y)):
                frame_anns.append({
                    'type': 'bbox',
                    'bbox': [x - w/2, y - h/2, w, h],
                    'attributes': {'track_id': tid}
                })
                if tid not in track_dets:
                    track_dets[tid] = []
                track_dets[tid].append({
                    'frame_num': frame_num,
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'speed': float(row.get('robust_speed', 0))
                })

        if frame_anns:
            annotations['items'].append({
                'frame': fname,
                'annotations': frame_anns,
                'car_count': len(frame_anns)
            })

    # ── save annotations JSON ─────────────────────────────────────────────────
    for tid, dets in track_dets.items():
        annotations['trajectories'][str(tid)] = sorted(dets, key=lambda d: d['frame_num'])

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(annotations, f, indent=2)
    print(f"Saved {OUTPUT_JSON}")

    # ── copy/filter CSV to new_sim ────────────────────────────────────────────
    valid_ids = set(track_dets.keys())
    df_out = df[df['track_id'].isin(valid_ids)].copy()
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV} ({len(df_out)} rows, {len(valid_ids)} tracks)")
    print(f"Cropped images saved to {OUTPUT_FOLDER}/")
    print("\nDone. Next: run sim.py")


if __name__ == "__main__":
    main()

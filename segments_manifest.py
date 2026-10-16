"""
segments_manifest.py - the list of green-to-red segments to process, one
entry per polygon/analysis run, derived from the frame ranges given for each
OneDrive folder. `light_turn_red_frame` is the split boundary itself (the
frame index where the light actually turns red for that segment) -- set to
None where not yet known.

Each entry maps to its own working directory under segments/<name>/ so
segments never share or overwrite each other's CSV/polygon/cropped frames.
"""

SEGMENTS = [
    {
        "name": "od1_seg1",
        "source_dir": "OneDrive_1_20-06-2026",
        "annotation_json": "OneDrive_1_20-06-2026/annotations_filtered.json",
        "images_dir": "OneDrive_1_20-06-2026/images",
        "frame_start": 0,
        "frame_end": 183,
        "light_turn_red_frame": None,  # TBD -- not given for this folder
    },
    {
        "name": "od4_seg1",
        "source_dir": "OneDrive_4_20-06-2026",
        "annotation_json": "OneDrive_4_20-06-2026/jsons/selected_annotations.json",
        "images_dir": "OneDrive_4_20-06-2026/images",
        "frame_start": 0,
        "frame_end": 300,
        "light_turn_red_frame": 300,
    },
    {
        "name": "od4_seg2",
        "source_dir": "OneDrive_4_20-06-2026",
        "annotation_json": "OneDrive_4_20-06-2026/jsons/selected_annotations.json",
        "images_dir": "OneDrive_4_20-06-2026/images",
        "frame_start": 300,
        "frame_end": 750,
        "light_turn_red_frame": 300,
    },
    {
        "name": "od5_seg1",
        "source_dir": "OneDrive_5_20-06-2026",
        "annotation_json": "OneDrive_5_20-06-2026/jsons/selected_annotations.json",
        "images_dir": "OneDrive_5_20-06-2026/images",
        "frame_start": 0,
        "frame_end": 140,
        "light_turn_red_frame": 140,
    },
    {
        "name": "od5_seg2",
        "source_dir": "OneDrive_5_20-06-2026",
        "annotation_json": "OneDrive_5_20-06-2026/jsons/selected_annotations.json",
        "images_dir": "OneDrive_5_20-06-2026/images",
        "frame_start": 140,
        "frame_end": 550,
        "light_turn_red_frame": 140,
    },
    {
        "name": "od6_seg1",
        "source_dir": "OneDrive_6_20-06-2026",
        "annotation_json": "OneDrive_6_20-06-2026/Json/default.json",
        "images_dir": "OneDrive_6_20-06-2026/images",
        "frame_start": 0,
        "frame_end": 136,
        "light_turn_red_frame": 136,
    },
    {
        "name": "od6_seg2",
        "source_dir": "OneDrive_6_20-06-2026",
        "annotation_json": "OneDrive_6_20-06-2026/Json/default.json",
        "images_dir": "OneDrive_6_20-06-2026/images",
        "frame_start": 136,
        "frame_end": 1160,
        "light_turn_red_frame": 136,
    },
]

# SOURCES - one entry per raw OneDrive recording, full frame range, no
# seg1/seg2 split. Used by the per-road workflow (add_polygon.py,
# point_picker.html): a road's frame range is given directly and can span
# past where the old seg1/seg2 guesses cut off, so each road pulls straight
# from a single unsplit CSV covering the whole recording instead.
SOURCES = [
    {
        "name": "od1",
        "annotation_json": "OneDrive_1_20-06-2026/annotations_filtered.json",
        "images_dir": "OneDrive_1_20-06-2026/images",
    },
    {
        "name": "od4",
        "annotation_json": "OneDrive_4_20-06-2026/jsons/selected_annotations.json",
        "images_dir": "OneDrive_4_20-06-2026/images",
    },
    {
        "name": "od5",
        "annotation_json": "OneDrive_5_20-06-2026/jsons/selected_annotations.json",
        "images_dir": "OneDrive_5_20-06-2026/images",
    },
    {
        "name": "od6",
        "annotation_json": "OneDrive_6_20-06-2026/Json/default.json",
        "images_dir": "OneDrive_6_20-06-2026/images",
    },
]

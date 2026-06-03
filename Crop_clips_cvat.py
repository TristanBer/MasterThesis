"""
CVAT-Based ROI Cropping Script for Ablation Study A2
=====================================================

This script re-extracts video clips from the raw game footage, applying
bounding-box-based cropping derived from CVAT annotations. For each
annotated setting action (track), the script computes a unified bounding
box across all annotated frames, adds configurable padding, and applies
that fixed ROI to every frame in the extracted clip.

The output directory mirrors the structure of Master_Dataset_Extracted/
so that the existing training pipeline (R2Plus1D_train.py, dataset.py)
can be used without modification — only ROOT_DIR needs to be changed.

Usage:
    python crop_clips_cvat.py

Frame Mapping:
    CVAT annotations reference absolute frame numbers in the raw footage.
    The clipping script computes:
        actual_start = first_annotated_frame - int(fps * 0.5)
    Then extracts MAX_FRAMES frames with frame_step (2 for 60fps, 1 for 30fps).
    The crop region is computed in the original resolution (1920x1080) and
    applied before resizing to TARGET_SIZE.

References:
    Grabner, H., & Van Gool, L. (2013). Visual Object Tracking with
        Spatially Regularized Correlation Filters. CVPR.
"""

import cv2
import xml.etree.ElementTree as ET
import os
import numpy as np

# --- CONFIGURATION ---
SSD_PATH = r"D:"
GLOBAL_OUTPUT_DIR = os.path.join(SSD_PATH, "Master_Dataset_Cropped_CVAT")
VIDEO_DIR = r"D:\BeachVolleyballData"

# All 3 games with CVAT annotations
# Format: (xml_folder_name, video_filename, game_prefix)
GAMES_TO_PROCESS = [
    ("KUN-GRÜ", "2025-07-13 KUN-GRÜ.mp4", "KUN_GRÜ"),
    ("HAR-AHR", "2025-09-05 HAR-AHR.mp4", "HAR_AHR"),
    ("BOR-BEH", "2025-07-11 BOR-BEH.mp4", "BOR_BEH"),
]

TARGET_SIZE = (448, 448)
TARGET_FPS = 30
MAX_FRAMES = 60

# ROI padding as a fraction of the bounding box dimensions.
# 0.3 = 30% padding on each side, giving a region roughly 1.6x the bbox.
# This accounts for player movement between annotated and unannotated frames.
ROI_PADDING = 0.3

# Minimum ROI size in pixels (original resolution) to prevent overly tight crops
# on small or distant players. Ensures the model receives sufficient spatial context.
MIN_ROI_SIZE = 250


def compute_union_bbox(track):
    """
    Compute the union bounding box across all visible (non-outside) frames
    in a CVAT track.

    Args:
        track: XML Element representing a CVAT track.

    Returns:
        Tuple (x_min, y_min, x_max, y_max) in original video coordinates,
        or None if no visible boxes exist.
    """
    x_mins, y_mins, x_maxs, y_maxs = [], [], [], []

    for box in track.findall('box'):
        if box.get('outside') == '1':
            continue
        x_mins.append(float(box.get('xtl')))
        y_mins.append(float(box.get('ytl')))
        x_maxs.append(float(box.get('xbr')))
        y_maxs.append(float(box.get('ybr')))

    if not x_mins:
        return None

    return min(x_mins), min(y_mins), max(x_maxs), max(y_maxs)


def pad_and_clamp_roi(bbox, frame_width, frame_height, padding=ROI_PADDING,
                      min_size=MIN_ROI_SIZE):
    """
    Add padding to a bounding box and clamp to frame boundaries.
    Also enforces a minimum ROI size and makes the ROI square to avoid
    aspect ratio distortion when resizing to TARGET_SIZE.

    Args:
        bbox: Tuple (x_min, y_min, x_max, y_max).
        frame_width: Width of the original video frame.
        frame_height: Height of the original video frame.
        padding: Fractional padding (e.g. 0.3 = 30% on each side).
        min_size: Minimum side length in pixels.

    Returns:
        Tuple (x1, y1, x2, y2) as integer pixel coordinates.
    """
    x_min, y_min, x_max, y_max = bbox
    w = x_max - x_min
    h = y_max - y_min

    # Add padding proportional to bbox dimensions
    pad_x = w * padding
    pad_y = h * padding

    x_min -= pad_x
    y_min -= pad_y
    x_max += pad_x
    y_max += pad_y

    # Enforce minimum ROI size
    roi_w = x_max - x_min
    roi_h = y_max - y_min

    if roi_w < min_size:
        deficit = min_size - roi_w
        x_min -= deficit / 2
        x_max += deficit / 2

    if roi_h < min_size:
        deficit = min_size - roi_h
        y_min -= deficit / 2
        y_max += deficit / 2

    # Make ROI square (take the larger dimension)
    roi_w = x_max - x_min
    roi_h = y_max - y_min
    if roi_w > roi_h:
        deficit = roi_w - roi_h
        y_min -= deficit / 2
        y_max += deficit / 2
    elif roi_h > roi_w:
        deficit = roi_h - roi_w
        x_min -= deficit / 2
        x_max += deficit / 2

    # Clamp to frame boundaries
    x1 = max(0, int(round(x_min)))
    y1 = max(0, int(round(y_min)))
    x2 = min(frame_width, int(round(x_max)))
    y2 = min(frame_height, int(round(y_max)))

    return x1, y1, x2, y2


def extract_cropped_clips():
    """
    Main extraction loop. For each annotated game, parses the CVAT XML,
    computes per-track union bounding boxes, and extracts cropped clips.
    """
    if not os.path.exists(GLOBAL_OUTPUT_DIR):
        os.makedirs(GLOBAL_OUTPUT_DIR)
        print(f"Output directory created: {GLOBAL_OUTPUT_DIR}")

    total_clips = 0
    total_skipped = 0

    for xml_folder, video_name, game_prefix in GAMES_TO_PROCESS:
        print(f"\n{'='*60}")
        print(f"Processing game: {game_prefix}")
        print(f"{'='*60}")

        # Resolve video path (same fallback logic as original clipping script)
        video_path = os.path.join(VIDEO_DIR, video_name)
        xml_path = os.path.join(SSD_PATH, "annotated_games", xml_folder, "annotations.xml")

        local_video_path = os.path.join(SSD_PATH, "annotated_games", xml_folder, video_name)
        if os.path.exists(local_video_path):
            video_path = local_video_path
            print(f"  Video found in local folder: {video_path}")
        elif not os.path.exists(video_path) and os.path.exists(VIDEO_DIR):
            for f in os.listdir(VIDEO_DIR):
                if xml_folder.replace(" ", "").lower() in f.replace(" ", "").lower() and f.endswith(".mp4"):
                    video_path = os.path.join(VIDEO_DIR, f)
                    print(f"  Video auto-resolved: {f}")
                    break

        if not os.path.exists(video_path):
            print(f"  WARNING: Video not found at {video_path}. Skipping...")
            continue
        if not os.path.exists(xml_path):
            print(f"  WARNING: XML not found at {xml_path}. Skipping...")
            continue

        # Parse CVAT annotations
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get original video resolution from CVAT metadata
        orig_width = int(root.find('.//original_size/width').text)
        orig_height = int(root.find('.//original_size/height').text)
        print(f"  Original resolution: {orig_width}x{orig_height}")

        # Open video
        cap = cv2.VideoCapture(video_path)
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps == 0:
            orig_fps = 30

        source_fps = round(orig_fps)
        frame_step = 2 if source_fps == 60 else 1
        print(f"  Source FPS: {source_fps}, frame_step: {frame_step}")

        game_clips = 0

        for track in root.findall('track'):
            track_id = track.get('id', '0')

            # Get action label from first box attribute
            first_box = track.find('box')
            if first_box is None:
                continue

            attr = first_box.find(".//attribute[@name='type']")
            if attr is not None:
                label = attr.text
            else:
                label = track.get('label', 'unknown')

            label = label.replace(" ", "_")

            # Compute union bounding box for this track
            union_bbox = compute_union_bbox(track)
            if union_bbox is None:
                print(f"  Track {track_id}: No visible boxes. Skipping.")
                total_skipped += 1
                continue

            # Apply padding and clamping
            roi = pad_and_clamp_roi(union_bbox, orig_width, orig_height)
            x1, y1, x2, y2 = roi
            roi_w, roi_h = x2 - x1, y2 - y1

            # Compute clip start frame (identical logic to original clipping script)
            start_frame = int(first_box.get('frame'))
            offset = int(orig_fps * 0.5)
            actual_start = max(0, start_frame - offset)

            # Create output directory and file path
            label_dir = os.path.join(GLOBAL_OUTPUT_DIR, label)
            os.makedirs(label_dir, exist_ok=True)

            output_name = f"{game_prefix}_set_{track_id}.mp4"
            output_path = os.path.join(label_dir, output_name)

            # Extract and crop frames
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, TARGET_FPS, TARGET_SIZE)

            cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start)

            frames_written = 0
            for _ in range(MAX_FRAMES):
                ret, frame = cap.read()
                if not ret:
                    break

                # Crop to ROI in original resolution
                cropped = frame[y1:y2, x1:x2]

                # Resize cropped region to target size
                resized = cv2.resize(cropped, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
                out.write(resized)
                frames_written += 1

                # Skip frames for 60fps sources (same as original script)
                if frame_step == 2:
                    cap.grab()

            out.release()
            game_clips += 1
            total_clips += 1

            if game_clips <= 3 or game_clips % 20 == 0:
                print(f"  Track {track_id:>3s} ({label:>25s}): "
                      f"ROI=({x1},{y1})-({x2},{y2}) [{roi_w}x{roi_h}px], "
                      f"frames={frames_written}")

        cap.release()
        print(f"  -> {game_clips} clips extracted for {game_prefix}")

    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"  Total clips extracted: {total_clips}")
    print(f"  Total tracks skipped:  {total_skipped}")
    print(f"  Output directory:      {GLOBAL_OUTPUT_DIR}")
    print(f"{'='*60}")

    # Print directory summary
    print(f"\nDataset structure:")
    for class_dir in sorted(os.listdir(GLOBAL_OUTPUT_DIR)):
        class_path = os.path.join(GLOBAL_OUTPUT_DIR, class_dir)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) if f.endswith('.mp4')])
            print(f"  {class_dir}: {count} clips")


if __name__ == "__main__":
    extract_cropped_clips()
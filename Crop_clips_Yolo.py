"""
YOLOv8-Based ROI Cropping Script for Ablation Study A2
=======================================================

This script processes all existing extracted clips (from Master_Dataset_Extracted)
and produces cropped versions using YOLOv8 person detection. Unlike the CVAT-based
script (crop_clips_cvat.py), which relies on manual annotations and is limited to
5 games, this script can process all 21 games automatically.

For each clip, the script:
    1. Runs YOLOv8 on every frame to detect persons and the sports ball
    2. Identifies the setter as the person closest to the ball in each frame
    3. Computes a union bounding box across the identified setter in all frames
    4. Applies padding, enforces a minimum size, and squares the ROI
    5. Crops every frame to the ROI and resizes to TARGET_SIZE
    6. Saves the cropped clip with the same filename to a parallel directory

Setter identification strategy:
    - If ball is detected: select the person whose bbox center is closest to the ball
    - If no ball detected: fall back to the person closest to frame center
    - If no person detected: copy the original clip unchanged

The output directory mirrors the structure of Master_Dataset_Extracted/ so that
the existing training pipeline can be used without modification.

Requirements:
    pip install ultralytics

Usage:
    python crop_clips_yolo.py

References:
    Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8.
    https://github.com/ultralytics/ultralytics
"""

import cv2
import os
import numpy as np
from ultralytics import YOLO

# --- CONFIGURATION ---
INPUT_DIR = "/workspace/Master_Dataset_Extracted"
OUTPUT_DIR = "/workspace/Master_Dataset_Cropped_YOLO"

TARGET_SIZE = (448, 448)
TARGET_FPS = 30

# YOLOv8 model size: 'yolov8n.pt' (nano, fastest) to 'yolov8x.pt' (extra-large, most accurate)
# Nano is sufficient for person detection; use 's' or 'm' if detection quality is poor.
YOLO_MODEL = "yolov8n.pt"

# COCO class IDs: 0 = person, 32 = sports ball
# Both are detected; the person closest to the ball is identified as the setter.
TARGET_CLASSES = [0, 32]  # person + sports ball

# Minimum detection confidence threshold
CONFIDENCE_THRESHOLD = 0.3

# ROI padding as a fraction of the union bounding box dimensions.
ROI_PADDING = 0.15

# Detection window: only these frames are used to identify the setter.
# The setting action typically occurs in frames 20-40 of the 60-frame clip.
# By restricting detection to this window, the script avoids picking different
# players across frames and produces a tighter, more consistent crop.
DETECTION_WINDOW = (20, 40)

# Minimum ROI size in pixels (in the 448x448 clip coordinate space)
# Proportionally equivalent to 250px in 1920px original ≈ 58px in 448px
# But we set a higher minimum to ensure sufficient spatial context.
MIN_ROI_SIZE = 100

# Fallback: if no person is detected in any frame, skip cropping
# and copy the original clip unchanged. This ensures no data is lost.
FALLBACK_COPY_ORIGINAL = True


def compute_union_bbox_yolo(detections_per_frame, frame_width, frame_height):
    """
    Compute the union bounding box across all person detections in all frames.

    Args:
        detections_per_frame: List of lists, where each inner list contains
            (x1, y1, x2, y2) tuples for detected persons in that frame.
        frame_width: Width of the clip frames.
        frame_height: Height of the clip frames.

    Returns:
        Tuple (x_min, y_min, x_max, y_max) or None if no detections.
    """
    all_boxes = [box for frame_dets in detections_per_frame for box in frame_dets]
    if not all_boxes:
        return None

    x_min = min(b[0] for b in all_boxes)
    y_min = min(b[1] for b in all_boxes)
    x_max = max(b[2] for b in all_boxes)
    y_max = max(b[3] for b in all_boxes)

    return x_min, y_min, x_max, y_max


def pad_and_clamp_roi(bbox, frame_width, frame_height, padding=ROI_PADDING,
                      min_size=MIN_ROI_SIZE):
    """
    Add padding, enforce minimum size, square the ROI, and clamp to frame boundaries.
    Identical logic to the CVAT script for consistency.
    """
    x_min, y_min, x_max, y_max = bbox
    w = x_max - x_min
    h = y_max - y_min

    pad_x = w * padding
    pad_y = h * padding

    x_min -= pad_x
    y_min -= pad_y
    x_max += pad_x
    y_max += pad_y

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

    x1 = max(0, int(round(x_min)))
    y1 = max(0, int(round(y_min)))
    x2 = min(frame_width, int(round(x_max)))
    y2 = min(frame_height, int(round(y_max)))

    return x1, y1, x2, y2


def detect_setter_in_clip(model, frames):
    """
    Run YOLOv8 inference on all frames and identify the setter as the
    person whose bounding box center is closest to the detected ball.

    For frames where the ball is detected: keeps the closest person to the ball.
    For frames where no ball is detected: keeps the person closest to frame center.
    For frames where no person is detected: returns empty list.

    Args:
        model: Loaded YOLO model.
        frames: List of numpy arrays (H, W, 3).

    Returns:
        List of lists, where each inner list contains at most one
        (x1, y1, x2, y2) tuple for the identified setter.
    """
    detections_per_frame = []

    # Batch inference for efficiency
    results = model.predict(
        source=frames,
        classes=TARGET_CLASSES,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
        device='cuda',
    )

    for result in results:
        frame_dets = []
        if result.boxes is None or len(result.boxes) == 0:
            detections_per_frame.append(frame_dets)
            continue

        # Separate persons and balls
        persons = []
        balls = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            coords = (float(x1), float(y1), float(x2), float(y2))
            if cls_id == 0:
                persons.append(coords)
            elif cls_id == 32:
                balls.append(coords)

        if not persons:
            detections_per_frame.append(frame_dets)
            continue

        if balls:
            # Use the highest-confidence ball (first in list after NMS)
            ball = balls[0]
            ball_cx = (ball[0] + ball[2]) / 2
            ball_cy = (ball[1] + ball[3]) / 2

            # Find the person closest to the ball center
            best_person = None
            best_dist = float('inf')
            for p in persons:
                p_cx = (p[0] + p[2]) / 2
                p_cy = (p[1] + p[3]) / 2
                dist = ((p_cx - ball_cx) ** 2 + (p_cy - ball_cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_person = p
            frame_dets.append(best_person)
        else:
            # Fallback: no ball detected, keep the person closest to frame center
            frame_h, frame_w = frames[0].shape[:2]
            center_x, center_y = frame_w / 2, frame_h / 2
            best_person = min(persons, key=lambda p: (
                ((p[0]+p[2])/2 - center_x)**2 + ((p[1]+p[3])/2 - center_y)**2
            ))
            frame_dets.append(best_person)

        detections_per_frame.append(frame_dets)

    return detections_per_frame


def load_clip_frames(video_path):
    """
    Load all frames from an extracted clip.

    Returns:
        List of numpy arrays (H, W, 3) in BGR format, or empty list on failure.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_cropped_clip(frames, roi, output_path):
    """
    Crop all frames to the ROI and save as a new video clip.
    """
    x1, y1, x2, y2 = roi
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, TARGET_FPS, TARGET_SIZE)

    for frame in frames:
        cropped = frame[y1:y2, x1:x2]
        resized = cv2.resize(cropped, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
        out.write(resized)

    out.release()


def process_all_clips():
    """
    Main processing loop. Iterates over all clips in the input directory,
    detects persons with YOLOv8, computes ROIs, and saves cropped clips.
    """
    # Load YOLOv8 model
    print(f"Loading YOLOv8 model: {YOLO_MODEL}")
    model = YOLO(YOLO_MODEL)
    print(f"Model loaded successfully.\n")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Output directory created: {OUTPUT_DIR}")

    total_clips = 0
    total_no_detection = 0
    total_cropped = 0

    # Iterate over class directories
    class_dirs = sorted([d for d in os.listdir(INPUT_DIR)
                         if os.path.isdir(os.path.join(INPUT_DIR, d))])

    for class_name in class_dirs:
        class_input_dir = os.path.join(INPUT_DIR, class_name)
        class_output_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        clip_files = sorted([f for f in os.listdir(class_input_dir)
                             if f.endswith(('.mp4', '.avi'))])

        print(f"\n{'='*60}")
        print(f"Class: {class_name} ({len(clip_files)} clips)")
        print(f"{'='*60}")

        for clip_idx, clip_name in enumerate(clip_files):
            clip_path = os.path.join(class_input_dir, clip_name)
            output_path = os.path.join(class_output_dir, clip_name)

            # Load clip frames
            frames = load_clip_frames(clip_path)
            if not frames:
                print(f"  WARNING: Could not load {clip_name}. Skipping.")
                continue

            frame_h, frame_w = frames[0].shape[:2]
            total_clips += 1

            # Only run detection on the setting action window (frames 20-40)
            # to identify the setter consistently, then apply that crop to all frames
            win_start, win_end = DETECTION_WINDOW
            win_start = min(win_start, len(frames))
            win_end = min(win_end, len(frames))
            window_frames = frames[win_start:win_end]

            if not window_frames:
                window_frames = frames  # fallback if clip is shorter than expected

            # Run YOLOv8 detection — identify the setter (person closest to ball)
            detections = detect_setter_in_clip(model, window_frames)

            # Compute union bounding box from window detections only
            union_bbox = compute_union_bbox_yolo(detections, frame_w, frame_h)

            if union_bbox is None:
                total_no_detection += 1
                if FALLBACK_COPY_ORIGINAL:
                    # Copy original clip unchanged
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, TARGET_FPS, TARGET_SIZE)
                    for frame in frames:
                        resized = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
                        out.write(resized)
                    out.release()

                    if clip_idx < 3 or clip_idx % 50 == 0:
                        print(f"  {clip_name}: NO PERSON DETECTED -> copied original")
                continue

            # Compute padded ROI
            roi = pad_and_clamp_roi(union_bbox, frame_w, frame_h)
            x1, y1, x2, y2 = roi

            # Count frames with at least one detection (within the window)
            frames_with_det = sum(1 for d in detections if d)

            # Save cropped clip (all frames, not just window)
            save_cropped_clip(frames, roi, output_path)
            total_cropped += 1

            if clip_idx < 3 or clip_idx % 50 == 0:
                print(f"  {clip_name}: ROI=({x1},{y1})-({x2},{y2}) "
                      f"[{x2-x1}x{y2-y1}px], "
                      f"detections in {frames_with_det}/{len(window_frames)} window frames")

        print(f"  -> {class_name}: {len(clip_files)} clips processed")

    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"  Total clips processed:        {total_clips}")
    print(f"  Clips cropped (person found):  {total_cropped}")
    print(f"  Clips without detection:       {total_no_detection} "
          f"({'copied unchanged' if FALLBACK_COPY_ORIGINAL else 'skipped'})")
    print(f"  Output directory:              {OUTPUT_DIR}")
    print(f"{'='*60}")

    # Print dataset summary
    print(f"\nDataset structure:")
    for class_dir in sorted(os.listdir(OUTPUT_DIR)):
        class_path = os.path.join(OUTPUT_DIR, class_dir)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) if f.endswith('.mp4')])
            print(f"  {class_dir}: {count} clips")


if __name__ == "__main__":
    process_all_clips()

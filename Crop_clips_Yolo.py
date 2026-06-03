"""
YOLOv8-Based ROI Cropping Script for Ablation Study A2
=======================================================

This script processes all existing extracted clips (from Master_Dataset_Extracted)
and produces cropped versions using YOLOv8 person detection. Unlike the CVAT-based
script (crop_clips_cvat.py), which relies on manual annotations and is limited to
5 games, this script can process all 21 games automatically.

For each clip, the script:
    1. Runs YOLOv8 on every frame to detect "person" instances
    2. Computes a union bounding box across all detected persons in all frames
    3. Applies padding, enforces a minimum size, and squares the ROI
    4. Crops every frame to the ROI and resizes to TARGET_SIZE
    5. Saves the cropped clip with the same filename to a parallel directory

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
INPUT_DIR = r"D:\Master_Dataset_Extracted"
OUTPUT_DIR = r"D:\Master_Dataset_Cropped_YOLO"

TARGET_SIZE = (448, 448)
TARGET_FPS = 30

# YOLOv8 model size: 'yolov8n.pt' (nano, fastest) to 'yolov8x.pt' (extra-large, most accurate)
# Nano is sufficient for person detection; use 's' or 'm' if detection quality is poor.
YOLO_MODEL = "yolov8n.pt"

# COCO class IDs: 0 = person, 32 = sports ball
# We primarily detect the person; the ball is too small and inconsistently detected.
TARGET_CLASSES = [0]  # person only

# Minimum detection confidence threshold
CONFIDENCE_THRESHOLD = 0.3

# ROI padding as a fraction of the union bounding box dimensions.
ROI_PADDING = 0.3

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


def detect_persons_in_clip(model, frames):
    """
    Run YOLOv8 inference on all frames of a clip.

    Args:
        model: Loaded YOLO model.
        frames: List of numpy arrays (H, W, 3).

    Returns:
        List of lists, where each inner list contains (x1, y1, x2, y2) tuples
        for detected persons in that frame.
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
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                frame_dets.append((float(x1), float(y1), float(x2), float(y2)))
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

            # Run YOLOv8 detection on all frames
            detections = detect_persons_in_clip(model, frames)

            # Compute union bounding box
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

            # Count frames with at least one detection
            frames_with_det = sum(1 for d in detections if d)

            # Save cropped clip
            save_cropped_clip(frames, roi, output_path)
            total_cropped += 1

            if clip_idx < 3 or clip_idx % 50 == 0:
                print(f"  {clip_name}: ROI=({x1},{y1})-({x2},{y2}) "
                      f"[{x2-x1}x{y2-y1}px], "
                      f"detections in {frames_with_det}/{len(frames)} frames")

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
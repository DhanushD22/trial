# processor/video_processor.py

import cv2
import os
import json
import time
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog


from config.settings import (
    DETECTION_DIR,
    REPORT_DIR,
    WAGON_INTERVAL_SEC,
    IOU_THRESHOLD,
    DEFECT_CLASSES,
)


# ================================
# IOU FUNCTION
# ================================
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


# ================================
# MAIN PROCESSING FUNCTION
# ================================
def process_video(video_path, session_id, predictor):

    print(f"\n📹 Processing video: {video_path}")
    print(f"🆔 Session ID: {session_id}")

    # Create output folders
    session_detection_dir = os.path.join(DETECTION_DIR, session_id)
    os.makedirs(session_detection_dir, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    report_file = os.path.join(REPORT_DIR, f"{session_id}.json")

    # Setup metadata (same as simulate.py)
    MetadataCatalog.get("rail_defects_train").thing_classes = [
        "crack",
        "chip",
        "missing_spring",
        "spring",
        "normal",
    ]
    metadata = MetadataCatalog.get("rail_defects_train")

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Could not open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"FPS: {fps}")
    print(f"Total Frames: {total_frames}")
    print("🔍 Starting analysis...\n")

    train_report = {"train": {}}

    frame_count = 0
    current_wagon = 1
    current_wagon_defects = []
    current_wagon_logged_defects = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time_sec = frame_count / fps

        # Wagon simulation
        if current_time_sec >= current_wagon * WAGON_INTERVAL_SEC:
            if current_wagon_defects:
                train_report["train"][f"wagon{current_wagon}"] = current_wagon_defects

            current_wagon += 1
            current_wagon_defects = []
            current_wagon_logged_defects = []

            print(f"\n🚃 New wagon: wagon{current_wagon}")

        # Inference
        outputs = predictor(frame)
        instances = outputs["instances"].to("cpu")

        if len(instances) == 0:
            continue

        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()

        timestamp_sec = round(current_time_sec, 2)
        timestamp_str = time.strftime("%M:%S", time.gmtime(current_time_sec))

        detected_defects_this_frame = []

        for i in range(len(classes)):
            class_name = metadata.thing_classes[classes[i]]

            if class_name in DEFECT_CLASSES:
                detected_defects_this_frame.append({
                    "timestamp_sec": timestamp_sec,
                    "timestamp": timestamp_str,
                    "frame": frame_count,
                    "class": class_name,
                    "confidence": float(round(scores[i], 3)),
                    "bbox": boxes[i].tolist(),
                    "image_path": ""
                })

        if not detected_defects_this_frame:
            continue

        # Duplicate suppression
        to_log = []

        for defect in detected_defects_this_frame:
            is_duplicate = False

            for logged in current_wagon_logged_defects:
                if (
                    defect["class"] == logged["class"]
                    and iou(defect["bbox"], logged["bbox"]) >= IOU_THRESHOLD
                ):
                    is_duplicate = True
                    break

            if not is_duplicate:
                to_log.append(defect)

        # Update memory
        for new_defect in to_log:
            current_wagon_logged_defects.append({
                "class": new_defect["class"],
                "bbox": new_defect["bbox"]
            })

        if not to_log:
            continue

        # Visualize only when new defect appears
        v = Visualizer(
            frame[:, :, ::-1],
            metadata=metadata,
            scale=1.0,
            instance_mode=ColorMode.IMAGE,
        )
        v = v.draw_instance_predictions(instances)
        annotated_frame = v.get_image()[:, :, ::-1]

        image_name = f"frame_{frame_count:06d}_wagon{current_wagon}.jpg"
        image_path = os.path.join(session_detection_dir, image_name)

        cv2.imwrite(image_path, annotated_frame)

        # Relative path for JSON
        rel_path = f"detections/{session_id}/{image_name}"

        for defect in to_log:
            defect["image_path"] = rel_path
            current_wagon_defects.append(defect)

        print(
            f"Frame {frame_count} | wagon{current_wagon} | "
            f"New defects logged: {len(to_log)}"
        )

    # Save last wagon
    if current_wagon_defects:
        train_report["train"][f"wagon{current_wagon}"] = current_wagon_defects

    cap.release()

    # Save report
    with open(report_file, "w") as f:
        json.dump(train_report, f, indent=4)

    total_defects = sum(len(v) for v in train_report["train"].values())

    print("\n" + "=" * 60)
    print("✅ Analysis Complete")
    print(f"Total wagons: {current_wagon}")
    print(f"Total unique defects: {total_defects}")
    print(f"Report saved at: {report_file}")
    print("=" * 60)

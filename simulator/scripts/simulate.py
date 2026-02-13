import cv2
import os
import json
import time
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# ================================
# CONFIGURATION
# ================================
VIDEO_NAME = "vid3.mp4"
INPUT_VIDEO_DIR = "../inputs"
OUTPUT_BASE_DIR = "../../outputs/detections"
REPORT_DIR = "../../outputs/reports"
LOG_DIR = "../logs"
VIDEO_TAG = "vid3"

OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, VIDEO_TAG)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

REPORT_FILE = os.path.join(REPORT_DIR, f"train_report_{VIDEO_TAG}.json")
SIM_LOG_FILE = os.path.join(LOG_DIR, f"simulation_{VIDEO_TAG}.log")

WAGON_INTERVAL_SEC = 4.0
IOU_THRESHOLD = 0.3   # lower threshold for stability
DEFECT_CLASSES = {"crack", "chip", "missing_spring"}

# ================================
# MODEL SETUP (GPU ENABLED)
# ================================
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
cfg.MODEL.WEIGHTS = "../../models/checkpoints/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50

if torch.cuda.is_available():
    print("✅ Using GPU:", torch.cuda.get_device_name(0))
    cfg.MODEL.DEVICE = "cuda"
    torch.backends.cudnn.benchmark = True
else:
    print("⚠ GPU not available, using CPU")
    cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)

MetadataCatalog.get("rail_defects_train").thing_classes = [
    "crack", "chip", "missing_spring", "spring", "normal"
]
metadata = MetadataCatalog.get("rail_defects_train")

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
# VIDEO PROCESSING
# ================================
video_path = os.path.join(INPUT_VIDEO_DIR, VIDEO_NAME)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\nVideo: {VIDEO_NAME}")
print(f"FPS: {fps:.2f}")
print(f"Total Frames: {total_frames}")
print(f"Duration: {total_frames/fps:.2f} seconds")
print("Starting analysis...\n")

train_report = {"train": {}}

with open(SIM_LOG_FILE, "a") as log_f:

    def log_print(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

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

            log_print(f"\n🚃 New wagon: wagon{current_wagon}")

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

        # Duplicate suppression (position-based)
        to_log = []

        for defect in detected_defects_this_frame:
            is_duplicate = False
            for logged in current_wagon_logged_defects:
                if (
                    defect["class"] == logged["class"] and
                    iou(defect["bbox"], logged["bbox"]) >= IOU_THRESHOLD
                ):
                    is_duplicate = True
                    break

            if not is_duplicate:
                to_log.append(defect)

        # Update memory AFTER filtering
        for new_defect in to_log:
            current_wagon_logged_defects.append({
                "class": new_defect["class"],
                "bbox": new_defect["bbox"]
            })

        if not to_log:
            continue

        # Visualize and save only when new defect appears
        v = Visualizer(
            frame[:, :, ::-1],
            metadata=metadata,
            scale=1.0,
            instance_mode=ColorMode.IMAGE,
        )
        v = v.draw_instance_predictions(instances)
        annotated_frame = v.get_image()[:, :, ::-1]

        image_name = f"frame_{frame_count:06d}_wagon{current_wagon}.jpg"
        image_path = os.path.join(OUTPUT_DIR, image_name)
        cv2.imwrite(image_path, annotated_frame)

        rel_path = f"detections/{VIDEO_TAG}/{image_name}"

        for defect in to_log:
            defect["image_path"] = rel_path
            current_wagon_defects.append(defect)

        log_print(
            f"Frame {frame_count} | wagon{current_wagon} | "
            f"New defects logged: {len(to_log)}"
        )

    # Save last wagon
    if current_wagon_defects:
        train_report["train"][f"wagon{current_wagon}"] = current_wagon_defects

cap.release()

# ================================
# SAVE REPORT
# ================================
with open(REPORT_FILE, "w") as f:
    json.dump(train_report, f, indent=4)

total_defects = sum(len(v) for v in train_report["train"].values())

print("\n" + "="*60)
print("Simulation & Analysis Complete")
print(f"Total wagons simulated: {current_wagon}")
print(f"Total unique defects logged: {total_defects}")
print(f"Report saved at: {REPORT_FILE}")
print(f"Log saved at: {SIM_LOG_FILE}")
print("="*60)

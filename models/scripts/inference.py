import cv2
import os
import json
import time
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# ================================
# CONFIGURATION
# ================================
VIDEO_NAME = "vid3.mp4"
INPUT_VIDEO_DIR = "../../input_video"
OUTPUT_BASE_DIR = "../../outputs/detections/vid3" # Change "vid3" to match your video name.
VIDEO_TAG = "vid3"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, VIDEO_TAG)
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUTPUT_DIR, "detection_log.json")

# ================================
# MODEL SETUP
# ================================
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
cfg.MODEL.WEIGHTS = "../checkpoints/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.35   # lowered for better video recall
cfg.MODEL.DEVICE = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
predictor = DefaultPredictor(cfg)

MetadataCatalog.get("rail_defects_train").thing_classes = [
    "crack", "chip", "missing_spring", "spring", "normal"
]
metadata = MetadataCatalog.get("rail_defects_train")

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
print(f"Video: {VIDEO_NAME}")
print(f"  - FPS: {fps:.1f}")
print(f"  - Total frames: {total_frames}")
print(f"  - Approx. duration: {total_frames/fps:.1f} seconds")
print(f"  - Output folder: {OUTPUT_DIR}")
print("Starting analysis...\n")

frame_count = 0
process_every_n_frames = 1  # every frame
detection_log = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % process_every_n_frames != 0:
        continue

    # Run inference
    outputs = predictor(frame)
    instances = outputs["instances"].to("cpu")

    if len(instances) > 0:
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()

        timestamp_sec = frame_count / fps
        timestamp_str = time.strftime("%M:%S", time.gmtime(timestamp_sec))

        # Visualize
        v = Visualizer(
            frame[:, :, ::-1],
            metadata=metadata,
            scale=1.0,
            instance_mode=ColorMode.IMAGE,
        )
        v = v.draw_instance_predictions(instances)
        annotated_frame = v.get_image()[:, :, ::-1]

        # Save annotated frame
        out_image_path = os.path.join(
            OUTPUT_DIR,
            f"frame_{frame_count:06d}_{timestamp_str}.jpg"
        )
        cv2.imwrite(out_image_path, annotated_frame)

        # Log all detections (including normal/spring)
        for i in range(len(classes)):
            class_name = metadata.thing_classes[classes[i]]
            detection_log.append({
                "frame": frame_count,
                "timestamp_sec": round(timestamp_sec, 2),
                "timestamp": timestamp_str,
                "class": class_name,
                "confidence": float(round(scores[i], 3)),
                "bbox": boxes[i].tolist(),
                "image_file": os.path.basename(out_image_path)
            })

        print(f"Frame {frame_count:6d} ({timestamp_str}) → {len(classes)} detections saved")

    # Progress update
    if frame_count % 100 == 0:
        percent = (frame_count / total_frames) * 100
        print(f"  Processed {frame_count}/{total_frames} frames ({percent:.1f}%)")

cap.release()

# ================================
# SAVE STRUCTURED JSON LOG
# ================================
final_log = {
    "video": {
        "filename": VIDEO_NAME,
        "fps": float(fps),
        "total_frames": total_frames,
        "duration_seconds": round(total_frames / fps, 2),
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
    },
    "model": {
        "weights": "model_final.pth",
        "confidence_threshold": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
        "classes": metadata.thing_classes
    },
    "detections": detection_log
}

with open(LOG_FILE, "w") as f:
    json.dump(final_log, f, indent=4)

print("\n" + "="*60)
print("Video processing complete.")
print(f"Annotated frames saved in: {OUTPUT_DIR}")
print(f"Total frames processed: {frame_count}")
print(f"Total detections logged: {len(detection_log)}")
print(f"Log file: {LOG_FILE}")
print("="*60)
import cv2
import os
import time
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# Config and predictor setup (same as infer_single.py)
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
cfg.MODEL.WEIGHTS = "../checkpoints/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Adjust (0.3-0.7) based on tests
cfg.MODEL.DEVICE = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
predictor = DefaultPredictor(cfg)

# Class names
MetadataCatalog.get("rail_defects_train").thing_classes = ["crack", "chip", "missing_spring", "spring", "normal"]

# Paths
INPUT_VIDEO_DIR = "../../input_video"  # Your new folder
OUTPUT_DIR = "../../outputs/detections/vid3" # Change "vid3" to match your video name.
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Video file (change this to your .mp4 name)
video_file = "vid3.mp4"  # e.g., your file name
video_path = os.path.join(INPUT_VIDEO_DIR, video_file)

# Process video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

frame_count = 0
fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Assume 30 if unknown
process_every_n_frames = 1  # Skip frames for speed (adjust: 1 = every frame, 10 = lighter)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % process_every_n_frames != 0:
        continue  # Skip this frame

    # Run model
    outputs = predictor(frame)
    instances = outputs["instances"].to("cpu")

    if len(instances) > 0:  # If any detections
        # Visualize
        v = Visualizer(frame[:, :, ::-1],
                       metadata=MetadataCatalog.get("rail_defects_train"),
                       scale=1.0,
                       instance_mode=ColorMode.IMAGE)
        v = v.draw_instance_predictions(instances)
        annotated_frame = v.get_image()[:, :, ::-1]

        # Timestamp (e.g., 00:05)
        timestamp_sec = frame_count / fps
        timestamp_str = time.strftime("%M:%S", time.gmtime(timestamp_sec))

        # Save if defects or normal detected
        detected_classes = [MetadataCatalog.get("rail_defects_train").thing_classes[cls.item()] for cls in instances.pred_classes]
        defect_str = "_".join(set(detected_classes))  # e.g., "crack_normal"
        out_file = f"frame_{timestamp_str}_{defect_str}.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_file)
        cv2.imwrite(out_path, annotated_frame)
        print(f"Saved detection: {out_path}")

cap.release()
print("Video processing complete.")
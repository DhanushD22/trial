# config/settings.py

import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RAW_VIDEO_DIR = os.path.join(OUTPUT_DIR, "raw_videos")
DETECTION_DIR = os.path.join(OUTPUT_DIR, "detections")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

LOG_DIR = os.path.join(BASE_DIR, "logs")

# ================================
# Camera Settings
# ================================
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# ================================
# Model Settings
# ================================
MODEL_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
MODEL_WEIGHTS = "/home/dhanush/Desktop/loc/models/checkpoints/model_final.pth"
NUM_CLASSES = 5
SCORE_THRESHOLD = 0.5

# ================================
# Wagon Simulation
# ================================
WAGON_INTERVAL_SEC = 4.0
IOU_THRESHOLD = 0.3

DEFECT_CLASSES = {"crack", "chip", "missing_spring"}


def generate_session_id():
    return datetime.now().strftime("session_%Y%m%d_%H%M%S")

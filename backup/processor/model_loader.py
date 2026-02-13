# processor/model_loader.py

import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from config.settings import (
    MODEL_CONFIG,
    MODEL_WEIGHTS,
    NUM_CLASSES,
    SCORE_THRESHOLD,
)

def load_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_CONFIG))

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.WEIGHTS = MODEL_WEIGHTS
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD

    if torch.cuda.is_available():
        print("✅ Using GPU:", torch.cuda.get_device_name(0))
        cfg.MODEL.DEVICE = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        print("⚠ Using CPU")
        cfg.MODEL.DEVICE = "cpu"

    predictor = DefaultPredictor(cfg)
    return predictor


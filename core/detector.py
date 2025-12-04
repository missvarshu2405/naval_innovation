from ultralytics import YOLO
import numpy as np
from config.settings import CONFIG

class ObjectDetector:
    """
    High-level wrapper for YOLOv8 object detection.
    Provides:
    - Model loading
    - Prediction on frames
    - Confidence filtering
    - Cleaned output dictionaries
    """

    def __init__(self, model_path=None):
        model_path = model_path or CONFIG.get("model_path")
        print(f"[INFO] Loading model from {model_path}...")
        self.model = YOLO(model_path)

        self.conf_thres = CONFIG.get("confidence_threshold")
        self.iou_thres = CONFIG.get("iou_threshold")

    def detect(self, frame):
        """
        Runs YOLO detection and returns structured results.
        """
        results = self.model.predict(frame, conf=self.conf_thres, iou=self.iou_thres, verbose=False)
        if len(results) == 0:
            return []

        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf)
            cls = int(box.cls)

            detections.append({
                "box": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls
            })

        return detections

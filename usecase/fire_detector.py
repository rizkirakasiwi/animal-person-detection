from typing import List, Dict

import cv2
import cvzone
import numpy as np

from core.helper.detector import ObjectDetector
from core.helper.extensions import to_camel_case
from usecase.base_detector import BaseDetector as Base


class FireDetector(Base):
    def __init__(self, min_conf: float = 0.25, show_label: bool = True):
        self.detector = ObjectDetector(
            "model/fire_v5.pt", allowed_classes=[]
        )
        self.min_conf = min_conf
        self.show_label = show_label
        self.color_map = {
            "fire": (0, 0, 255),
            "smoke": (255, 0, 0),
        }

    def detect(self, frame: np.ndarray) -> List[Dict]:
        detections = self.detector.detect_plg(frame)

        if not detections:
            return []

        has_overlay = False
        overlay = frame.copy()

        for det in detections:
            conf = det["confidence"]
            if conf < self.min_conf:
                continue

            cls_name = det["class_name"]
            polygon = np.array([det["polygon"]], dtype=np.int32)
            color = self.color_map.get(cls_name, (0, 255, 0))  # fallback to green

            # Fill and outline
            cv2.fillPoly(overlay, polygon, color)
            cv2.polylines(frame, polygon, isClosed=True, color=color, thickness=1)

            # Optional label
            if self.show_label:
                x, y = det["polygon"][0]
                cvzone.putTextRect(
                    frame,
                    to_camel_case(cls_name),
                    (max(20, x), max(20, y)),
                    colorR=color,
                    scale=1,
                    thickness=1,
                )

            has_overlay = True

        if has_overlay:
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        return detections

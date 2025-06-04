import numpy as np
from typing import Optional, List, Dict
from helper.detector import ObjectDetector
from usecase.base_detector import BaseDetector as Base
from helper.extensions import to_camel_case
import cv2
import cvzone


class FireDetector(Base):
    def __init__(self, min_conf: float = 0.25, show_label: bool = True):
        allowed_classes = []  # Or specify class indices you want
        self.detector = ObjectDetector(
            "model/fire_v5.pt", allowed_classes=allowed_classes
        )
        self.min_conf = min_conf
        self.show_label = show_label

    def detect(self, frame: np.ndarray) -> List[Dict]:
        color = {
            "fire": (0, 0, 255),  # Red for fire
            "smoke": (255, 0, 0),  # Gray for smoke
        }
        detections = self.detector.detect_plg(frame)
        overlay = frame.copy()

        for det in detections:
            polygon = np.array([det["polygon"]], dtype=np.int32)  # ✅ wrap in list + convert to np array
            cls_name = det["class_name"]
            conf = det["confidence"]

            if conf >= self.min_conf:
                cv2.fillPoly(overlay, polygon, color[cls_name])
                cv2.polylines(frame, polygon, isClosed=True, color=color[cls_name], thickness=1)

                if self.show_label:
                    x, y = det["polygon"][0]
                    cvzone.putTextRect(
                        frame,
                        to_camel_case(cls_name),
                        (max(20, x), max(20, y)),
                        colorR=color[cls_name],
                        scale=1,
                        thickness=1,
                    )

        # 🧪 Blend the overlay (with shading) into the frame
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        return detections

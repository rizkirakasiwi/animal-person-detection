from typing import List, Dict

import cv2
import cvzone
import numpy as np

from core.helper.detector import ObjectDetector
from core.helper.extensions import to_camel_case
from usecase.base_detector import BaseDetector as Base


class PPEDetector(Base):
    def __init__(self, min_conf: float = 0.25, show_label: bool = True):
        self.detector = ObjectDetector(
            "model/reliable_model/ppe3n.pt",
            allowed_classes=[],  # You can specify if you want to limit detection classes
        )
        self.min_conf = min_conf
        self.show_label = show_label

    def detect(self, frame: np.ndarray) -> List[Dict]:
        color = {
            "helmet": (0, 255, 0),  # Green for helmet
            "vest": (0, 255, 0),  # Green for vest
            "no-helmet": (0, 0, 255),  # Red for no helmet
            "no-vest": (0, 0, 255),  # Red for no vest
            "person": (255, 0, 0)  # Blue for person
        }

        detector = self.detector.detect(frame)

        for det in detector:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox
            cls_name = det["class_name"]
            conf = det["confidence"]
            if conf > self.min_conf:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color[cls_name], 1)
                if self.show_label:
                    cvzone.putTextRect(frame, to_camel_case(cls_name), (max(20, x1), max(20, y1)),
                                       colorR=color[cls_name],
                                       scale=1, thickness=1)

        critical_detections = [
            d for d in detector if d["class_name"] in {"no-helmet", "no-vest"} and d["confidence"] > self.min_conf
        ]
        return critical_detections

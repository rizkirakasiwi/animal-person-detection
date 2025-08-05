from typing import Dict, List

import cv2
import cvzone
import numpy as np

from core.helper.detector import ObjectDetector
from core.helper.extensions import to_camel_case
from usecase.base_detector import BaseDetector as Base


class PPEDetector(Base):
    def __init__(self, min_conf: float = 0.25, show_label: bool = True):
        self.detector = ObjectDetector(
            "model/reliable_model/safetynet-y11-v1.pt",
            allowed_classes=[],  # You can specify if you want to limit detection classes
        )
        self.min_conf = min_conf
        self.show_label = show_label

    def detect(self, frame: np.ndarray) -> List[Dict]:
        color = {
            "SafetyShoe": (0, 255, 0),      # Green
            "boot": (0, 255, 0),            # Green
            "faceMask": (0, 255, 255),      # Yellow
            "glove": (255, 0, 255),         # Magenta
            "gloves": (255, 0, 255),        # Magenta
            "goggle": (255, 255, 0),        # Cyan
            "hardhat": (0, 255, 0),         # Green
            "head": (255, 128, 0),          # Orange
            "helmet": (0, 255, 0),          # Green
            "vest": (0, 255, 0),            # Green
            "object": (128, 128, 128),      # Gray
            "person": (255, 0, 0),          # Blue

            "no vest": (0, 0, 255),         # Red
            "no_faceMask": (0, 0, 255),     # Red
            "no_gloves": (0, 0, 255),       # Red
            "no_helmet": (0, 0, 255),       # Red
            "no_vest": (0, 0, 255),         # Red
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

                    human_readable_names = {
                        "SafetyShoe": "Safety Shoe",
                        "boot": "Boot",
                        "faceMask": "Face Mask",
                        "glove": "Glove",
                        "gloves": "Gloves",
                        "goggle": "Goggle",
                        "hardhat": "Hard Hat",
                        "head": "Head",
                        "helmet": "Helmet",
                        "vest": "Vest",
                        "object": "Object",
                        "person": "Person",
                        "no vest": "No Vest",
                        "no_faceMask": "No Face Mask",
                        "no_gloves": "No Gloves",
                        "no_helmet": "No Helmet",
                        "no_vest": "No Vest"
                    }
                    display_name = human_readable_names.get(cls_name, to_camel_case(cls_name))
                    cvzone.putTextRect(frame, display_name, (max(20, x1), max(20, y1)),
                                       colorR=color[cls_name],
                                       scale=1, thickness=1)

        critical_detections = [
            d for d in detector if d["class_name"] in {
                "no vest", "no_faceMask", "no_gloves", "no_helmet", "no_vest"
            } and d["confidence"] > self.min_conf
        ]
        return critical_detections

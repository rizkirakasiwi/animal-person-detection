import numpy as np
from typing import Optional, List, Dict
from core.helper.detector import ObjectDetector
from usecase.base_detector import BaseDetector as Base
import cv2
from core.helper.extensions import to_camel_case
import cvzone

class PPEDetector(Base):
    def __init__(self):
        self.detector = ObjectDetector(
            "model/reliable_model/ppe3n.pt",
            allowed_classes=[],  # You can specify if you want to limit detection classes
        )

    def detect(self, frame: np.ndarray) -> List[Dict]:
        min_conf = min_conf if min_conf is not None else 0.3
        show_label = show_label if show_label is not None else True

        color = {
            "helmet": (0, 255, 0),  # Green for helmet
            "vest": (0, 255, 0),    # Green for vest
            "no-helmet": (0, 0, 255),  # Red for no helmet
            "no-vest": (0, 0, 255),    # Red for no vest
            "person": (255, 0, 0)   # Blue for person
        }

        detector = self.detector.detect(frame)

        for det in detector:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox
            cls_name = det["class_name"]
            conf = det["confidence"]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color[cls_name], 1)
            cvzone.putTextRect(frame, to_camel_case(cls_name), (max(20, x1), max(20, y1)), colorR=color[cls_name], scale=1, thickness=1)
            
        return detector
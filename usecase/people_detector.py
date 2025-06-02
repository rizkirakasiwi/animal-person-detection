import numpy as np
from typing import Optional, List, Dict
from helper.detector import ObjectDetector
from usecase.base_detector import BaseDetector
from helper.extensions import to_camel_case
import cv2
import cvzone

class PeopleDetector(BaseDetector):
    def __init__(self):
        self.human_detector = ObjectDetector(
            "model/reliable_model/person1n.pt", allowed_classes=[]
        )

    def detect(
        self,
        frame: np.ndarray,
        min_conf: Optional[float] = None,
        show_label: Optional[bool] = None,
    ) -> List[Dict]:
        min_conf = min_conf if min_conf is not None else 0.3
        show_label = show_label if show_label is not None else True

        color = {
            "person": (0, 255, 0),  # Green for human
        }

        detector = self.human_detector.detect(frame)
        for (x1, y1, x2, y2), cls_id, cls_name, _ in detector:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color[cls_name], 1)
            person_label = "person" if cls_name == "human" else cls_name
            cvzone.putTextRect(frame, to_camel_case(person_label), (max(20, x1), max(20, y1)), colorR=color[cls_name], scale=1, thickness=1)
            
        return detector

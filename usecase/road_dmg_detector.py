import numpy as np
from typing import List, Dict
from helper.detector import ObjectDetector
from usecase.base_detector import BaseDetector as Base
from helper.extensions import to_camel_case
import cv2
import cvzone


class RoadDmgDetector(Base):
    def __init__(self, min_conf: float = 0.25, show_label: bool = True):
        allowed_classes = []
        self.animal_detector = ObjectDetector(
            "model/road_damage_v1.pt", allowed_classes=allowed_classes
        )
        self.min_conf = min_conf
        self.show_label = show_label

    def detect(
        self,
        frame: np.ndarray,
    ) -> List[Dict]:
        color = (0, 0, 255)

        detector = self.animal_detector.detect(frame)
        for (x1, y1, x2, y2), cls_id, cls_name, conf in detector:
            if conf >= self.min_conf:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                if self.show_label:
                    cvzone.putTextRect(
                        frame,
                        to_camel_case(cls_name),
                        (max(20, x1), max(20, y1)),
                        colorR=color,
                        scale=1,
                        thickness=1,
                    )
        return detector

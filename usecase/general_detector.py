import numpy as np
from typing import List, Dict
from core.helper.detector import ObjectDetector
from usecase.base_detector import BaseDetector as Base
from core.helper.extensions import to_camel_case
import cv2
import cvzone


class GeneralDetector(Base):
    def __init__(self, min_conf: float = 0.25, show_label: bool = True):
        allowed_classes = []
        self.animal_detector = ObjectDetector(
            "model/reliable_model/general_v2.pt", allowed_classes=allowed_classes
        )
        self.min_conf = min_conf
        self.show_label = show_label

    def detect(self, frame: np.ndarray) -> List[Dict]:
        color = {
            "bird": (51, 87, 255),  # orange-red
            "cat": (255, 193, 51),  # sky blue-ish
            "chick": (102, 255, 255),  # yellow
            "chicken": (51, 183, 255),  # orange
            "cow": (173, 68, 142),  # purple
            "dog": (219, 152, 52),  # blue
            "duck": (113, 204, 46),  # green
            "duckling": (215, 228, 163),  # mint
            "guinea pig": (0, 84, 211),  # dark orange
            "horse": (45, 82, 160),  # brown
            "kitten": (180, 105, 255),  # pink
            "lamb": (199, 195, 189),  # light gray
            "rabbit": (34, 126, 230),  # carrot
            "sheep": (241, 240, 236),
            "person": (0, 255, 0),  # near white
        }

        detector = self.animal_detector.detect(frame)
        for det in detector:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox
            cls_name = det["class_name"]
            conf = det["confidence"]
            if conf >= self.min_conf:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color[cls_name], 1)
                if self.show_label:
                    cvzone.putTextRect(
                        frame,
                        to_camel_case(cls_name),
                        (max(20, x1), max(20, y1)),
                        colorR=color[cls_name],
                        scale=1,
                        thickness=1,
                    )
        return detector

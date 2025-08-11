from typing import Dict, List

import cv2
import cvzone
import numpy as np

from core.helper.detector import ObjectDetector
from core.helper.extensions import to_camel_case
from usecase.base_detector import BaseDetector as Base


class BadParkingDetector(Base):
    def __init__(self, min_conf: float = 0.25, show_label: bool = False, frame_size: tuple = (640, 480)):
        self.detector = ObjectDetector(
            "model/bp-yl11-v3.pt", allowed_classes=[]
        )
        self.min_conf = min_conf
        self.show_label = show_label
        self.frame_size = frame_size

    def check_collision(self, car_polygon, line_polygon):
        car_pts = np.array(car_polygon, dtype=np.int32)
        line_pts = np.array(line_polygon, dtype=np.int32)

        # Quick bounding box check first (fast elimination)
        car_rect = cv2.boundingRect(car_pts)
        line_rect = cv2.boundingRect(line_pts)

        if (car_rect[0] > line_rect[0] + line_rect[2] or
            car_rect[0] + car_rect[2] < line_rect[0] or
            car_rect[1] > line_rect[1] + line_rect[3] or
            car_rect[1] + car_rect[3] < line_rect[1]):
            return False

        # Create masks for both polygons
        mask_size = (max(car_rect[1] + car_rect[3], line_rect[1] + line_rect[3]) + 10,
                    max(car_rect[0] + car_rect[2], line_rect[0] + line_rect[2]) + 10)

        car_mask = np.zeros(mask_size, dtype=np.uint8)
        line_mask = np.zeros(mask_size, dtype=np.uint8)

        cv2.fillPoly(car_mask, [car_pts], 255) # type: ignore
        cv2.fillPoly(line_mask, [line_pts], 255) # type: ignore

        # Check if masks intersect
        intersection = cv2.bitwise_and(car_mask, line_mask)
        return np.any(intersection)

    def detect(self, frame: np.ndarray) -> List[Dict]:
        detections = self.detector.detect_plg(frame)

        if not detections:
            return []

        self.frame_size = (frame.shape[1], frame.shape[0])

        # Separate cars and lines
        cars = [det for det in detections if det["class_name"] == "car" and det["confidence"] >= self.min_conf]
        lines = [det for det in detections if det["class_name"] == "lines" and det["confidence"] >= self.min_conf]

        # Create a set to track violating cars for O(1) lookup
        violating_cars = set()
        detect_violation = False

        # Check each car against all lines
        for i, car in enumerate(cars):
            for line in lines:
                if self.check_collision(car["polygon"], line["polygon"]):
                    violating_cars.add(i)
                    detect_violation = True
                    break  # No need to check other lines for this car

        has_overlay = False
        overlay = frame.copy()

        for det in detections:
            conf = det["confidence"]
            if conf < self.min_conf:
                continue

            cls_name = det["class_name"]
            polygon = np.array([det["polygon"]], dtype=np.int32)

            # Determine color based on class and violation status
            if cls_name == "car":
                car_index = next((i for i, car in enumerate(cars) if car == det), -1)
                if car_index in violating_cars:
                    color = (0, 0, 255)
                    cv2.fillPoly(overlay, polygon, color) # type: ignore
                    cv2.polylines(frame, polygon, isClosed=True, color=color, thickness=1) # type: ignore
            else:
                color = (0, 255, 0)
                cv2.fillPoly(overlay, polygon, color) # type: ignore
                cv2.polylines(frame, polygon, isClosed=True, color=color, thickness=1) # type: ignore


            # Optional label
            if self.show_label:
                x, y = det["polygon"][0]
                cvzone.putTextRect(
                    frame,
                    to_camel_case(cls_name),
                    (max(20, x), max(20, y)),
                    colorR=(0, 255, 0),
                    scale=1,
                    thickness=1,
                )

            has_overlay = True

        if has_overlay:
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # Return violating cars only
        return [cars[i] for i in violating_cars] if detect_violation else []

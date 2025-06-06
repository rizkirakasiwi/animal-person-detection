import cv2
import numpy as np
from datetime import datetime
from typing import List

import cvzone
from usecase.base_detector import BaseDetector
from core.helper.recorder import VideoRecorder
from core.helper.capture import ImageCapture
from core.report.report import Report
from core.helper.message import Message


class DetectionEngine:
    def __init__(
            self,
            detectors: List[BaseDetector],
            recorder: VideoRecorder,
            capture: ImageCapture,
            report: Report,
            frame_size=(1280, 720),
            start_threshold: int = 30,
            show_timestamp: bool = True,
    ):
        self.detectors = detectors
        self.recorder = recorder
        self.capture = capture
        self.report = report

        self.frame_width, self.frame_height = frame_size
        self.start_threshold = start_threshold
        self.show_timestamp = show_timestamp

        self.frame_buffer = 0
        self.recording = False

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        base_frame = frame.copy()

        all_detections = self._run_detectors(frame)

        if all_detections:
            self._handle_trigger(base_frame, all_detections)
        else:
            self._handle_no_detections(detections=all_detections)

        if self.show_timestamp:
            self._overlay_timestamp(frame)

        if self.recording:
            self.recorder.write(base_frame)

        return frame

    def _run_detectors(self, base_frame: np.ndarray) -> List[dict]:
        all_detections = []

        for detector in self.detectors:
            # Pass the actual frame copy into detector, where annotations are made
            detections = detector.detect(base_frame)
            if detections:
                all_detections.extend(detections)

        return all_detections

    def _handle_trigger(self, frame: np.ndarray, detections: List[dict]):
        self.frame_buffer += 1

        if not self.recording and self.frame_buffer >= self.start_threshold:
            def on_image_saved(image_path):
                caption = Message.generate_message(detections)
                self.report.send_notif(message=caption, image=image_path)

            self.capture.capture(frame.copy(), callback=on_image_saved)
            self.recorder.start()
            self.recording = True

    def _handle_no_detections(self, detections: List[dict]):
        self.frame_buffer = 0
        if self.recording:
            path = self.recorder.stop()
            caption = Message.generate_message(detections)
            self.report.send_video_async(video=path, message=caption)
            self.recording = False

    def _overlay_timestamp(self, frame: np.ndarray):
        now = datetime.now()
        timestamp = now.strftime("%d %m %Y %H:%M:%S")
        cvzone.putTextRect(frame, timestamp, (50, 50), scale=1, thickness=2)

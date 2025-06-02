import cv2
from datetime import datetime
from typing import List
import numpy as np

from usecase.base_detector import BaseDetector
from helper.recorder import VideoRecorder
from helper.capture import ImageCapture
from report.report import Report
import cvzone


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

        self.frame_buffer = 0
        self.recording = False

        self.show_timestamp = show_timestamp

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))

        has_detection = any(len(det.detect(frame)) > 0 for det in self.detectors)

        if has_detection:
            self.frame_buffer += 1
            if not self.recording and self.frame_buffer >= self.start_threshold:
                img = self.capture.capture(frame)
                self.report.send_notif("Some object detected", img)
                self.recorder.start()
                self.recording = True
        else:
            self.frame_buffer = 0
            if self.recording:
                self.capture.stop()
                self.recorder.stop()
                self.recording = False

        if self.show_timestamp:           
            now = datetime.now()
            formatted = now.strftime("%d %m %Y %H:%M:%S")
            cvzone.putTextRect(frame, formatted, (50, 50), scale=1, thickness=2)

        if self.recording:
            self.recorder.write(frame)

        return frame

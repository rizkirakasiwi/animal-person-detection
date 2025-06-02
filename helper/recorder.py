import cv2
import threading
import queue
import os
from helper.output_path import get_output_path

class VideoRecorder:
    def __init__(self, frame_size, fps, output_dir="videos", enable=True):
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.frame_size = frame_size
        self.fps = fps
        self.output_dir = output_dir
        self.enable = enable
        self.writer = None
        self.queue = queue.Queue()
        self.thread = None
        self.running = False

    def _write_frames(self):
        while self.running or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=1)
                if self.writer:
                    self.writer.write(frame)
            except queue.Empty:
                continue

    def start(self):
        if not self.enable:
            return

        if not self.writer:
            os.makedirs(self.output_dir, exist_ok=True)
            path = get_output_path(path=self.output_dir, extension="mp4")
            self.writer = cv2.VideoWriter(path, self.fourcc, self.fps, self.frame_size)

        self.running = True
        self.thread = threading.Thread(target=self._write_frames)
        self.thread.start()

    def write(self, frame):
        if self.enable and self.running:
            self.queue.put(frame)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.writer:
            self.writer.release()
            self.writer = None

import cv2
import threading
import queue
import os
from helper.output_path import get_output_path

class ImageCapture:
    def __init__(self, enable: bool = True):
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._save_images)
        self.running = False
        self.enable = enable
        self.thread.start()

    def _save_images(self):
        while self.running or not self.queue.empty():
            try:
                frame, callback = self.queue.get(timeout=1)
                path = get_output_path(path="images", extension="jpg")
                cv2.imwrite(path, frame)
                if callback:
                    callback(path)
            except queue.Empty:
                continue

    def capture(self, frame, callback=None):
        if self.enable:
            """Capture a frame in the background. Callback will be called with the saved image path."""
            self.running = True
            self.queue.put((frame.copy(), callback))

    def stop(self):
        self.running = False
        self.thread.join()

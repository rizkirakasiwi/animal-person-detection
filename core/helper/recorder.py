import cv2
import threading
import queue
import os
from core.helper.output_path import get_output_path


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
        self.stop_event = threading.Event()
        self.path = None

    def _write_frames(self):
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=1)
                if self.writer:
                    self.writer.write(frame)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VideoRecorder] Error writing frame: {e}")

    def start(self):
        if not self.enable:
            return

        if not self.writer:
            os.makedirs(self.output_dir, exist_ok=True)
            self.path = get_output_path(path=self.output_dir, extension="mp4")
            self.writer = cv2.VideoWriter(self.path, self.fourcc, self.fps, self.frame_size)

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._write_frames, name="VideoRecorderWorker", daemon=True)
        self.thread.start()
        print("[VideoRecorder] Recording started.")

    def write(self, frame):
        if self.enable and not self.stop_event.is_set():
            self.queue.put(frame)

    def stop(self):
        if self.stop_event.is_set():
            return None

        print("[VideoRecorder] Stopping...")
        self.stop_event.set()
        if self.thread:
            self.thread.join()

        if self.writer:
            self.writer.release()
            self.writer = None

        print(f"[VideoRecorder] Stopped recording. Video saved at: {self.path}")
        return self.path

    def release(self):
        self.stop()
        print("[VideoRecorder] was stopped by the user")

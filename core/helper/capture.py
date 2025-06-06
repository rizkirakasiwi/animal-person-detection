import cv2
import queue
import threading
import traceback
from core.helper.output_path import get_output_path  # Adjust this import to match your project


class ImageCapture:
    def __init__(self, enable: bool = True):
        self.enable = enable
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, name="ImageCaptureWorker", daemon=True)
        self.thread.start()

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                # Wait up to 1 second for new items
                frame, callback = self.queue.get(timeout=1)
            except queue.Empty:
                continue  # No item, loop again

            try:
                path = get_output_path("images", "jpg")
                success = cv2.imwrite(path, frame)

                if success:
                    print(f"[ImageCapture] Image saved to {path}")
                    if callback:
                        callback(path)
                else:
                    print("[ImageCapture] Failed to save image.")
            except Exception as e:
                print(f"[ImageCapture] Error: {e}")
                traceback.print_exc()
            finally:
                self.queue.task_done()

    def capture(self, frame, callback=None):
        if self.enable:
            self.queue.put((frame.copy(), callback))
            print("[ImageCapture] Image capture queued.")

    def shutdown(self):
        """Signal the worker thread to stop and wait for it to finish."""
        print("[ImageCapture] Shutting down...")
        self.queue.join()  # Ensure all items are processed before stopping
        self.stop_event.set()
        self.thread.join()
        print("[ImageCapture] Shutdown complete.")

import argparse
import cv2
from core.helper.recorder import VideoRecorder
from core.report.report import Report
from core.helper.capture import ImageCapture
from core.engine.detection_engine import DetectionEngine
from usecase.fire_detector import FireDetector
from usecase.general_detector import GeneralDetector

def main(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        return

    frame_width, frame_height = (1280, 720)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # Instantiate dependencies
    recorder = VideoRecorder((frame_width, frame_height), fps)
    capture = ImageCapture()
    report = Report()

    detectors = [GeneralDetector(), FireDetector()]

    engine = DetectionEngine(
        detectors=detectors,
        recorder=recorder,
        capture=capture,
        report=report,
        frame_size=(frame_width, frame_height),
        start_threshold=30,
        show_timestamp=False,
    )

    # Main loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = engine.process_frame(frame)
        cv2.imshow("Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    recorder.release()
    capture.shutdown()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run object detection on a video file.")
    parser.add_argument("--video", required=True, help="Path to the video file")
    args = parser.parse_args()

    main(args.video)

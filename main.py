import argparse

import cv2

from core.engine.detection_engine import DetectionEngine
from core.helper.capture import ImageCapture
from core.helper.recorder import VideoRecorder
from core.report.report import Report
from usecase.fire_detector import FireDetector
from usecase.general_detector import GeneralDetector
from usecase.ppe_detector import PPEDetector
from usecase.road_dmg_detector import RoadDmgDetector


def main(video_path: str, use_case: str = "palm_security"):
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

    use_case_map = {
        "palm_security": [GeneralDetector(), FireDetector()],
        "ppe": [PPEDetector()],
        "road_damage": [RoadDmgDetector()],
    }

    detectors = use_case_map.get(use_case)

    if detectors is None:
        raise ValueError(f"Unknown usecase: {use_case}")

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
    parser.add_argument(
        "--usecase",
        required=True,
        choices=["palm_security", "ppe", "road_damage"],
        help="Choose usecase: palm_security, ppe, or road_damage",
    )
    args = parser.parse_args()

    main(args.video, args.usecase)

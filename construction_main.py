import cv2
from helper.recorder import VideoRecorder
from report.report import Report
from helper.capture import ImageCapture
from usecase.ppe_detector import PPEDetector
from engine.detection_engine import DetectionEngine

cap = cv2.VideoCapture("assets/videos/work.mp4")
frame_width, frame_height = (1280, 720)
fps = cap.get(cv2.CAP_PROP_FPS)

# Instantiate dependencies
recorder = VideoRecorder((frame_width, frame_height), fps, enable=False)
capture = ImageCapture(enable=False) 
report = Report()

detectors = [PPEDetector()]

engine = DetectionEngine(
    detectors=detectors,
    recorder=recorder,
    capture=capture,
    report=report,
    frame_size=(frame_width, frame_height),
    start_threshold=30,
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
recorder.stop()
capture.stop()
cv2.destroyAllWindows()

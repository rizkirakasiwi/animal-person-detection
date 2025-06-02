from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path, allowed_classes):
        self.model = YOLO(model_path)
        self.allowed_classes = allowed_classes

    def detect(self, frame):
        results = self.model(frame, stream=True, device=-1)
        detections = []

        for result in results:
            class_names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if len(self.allowed_classes) > 0:
                    if cls_id in self.allowed_classes:
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections.append(
                            ((x1, y1, x2, y2),
                             cls_id,
                             class_names[cls_id],
                             conf)
                        )
                else:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append(
                        ((x1, y1, x2, y2), cls_id, class_names[cls_id], conf))
        return detections

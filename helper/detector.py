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

    def detect_plg(self, frame):
        results = self.model(frame, stream=True, device=-1)
        detections = []

        for result in results:
            class_names = result.names

            # Check if result contains masks (only in segmentation models)
            if result.masks is not None:
                for seg, cls_id, conf in zip(result.masks.xy, result.boxes.cls, result.boxes.conf):
                    cls_id = int(cls_id)
                    if len(self.allowed_classes) > 0 and cls_id not in self.allowed_classes:
                        continue

                    polygon = [(int(x), int(y)) for x, y in seg]

                    detections.append({
                        "polygon": polygon,
                        "class_id": cls_id,
                        "class_name": class_names[cls_id],
                        "confidence": float(conf)
                    })
            else:
                # Fallback for models without masks (returns rectangle polygon)
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if len(self.allowed_classes) > 0 and cls_id not in self.allowed_classes:
                        continue

                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

                    detections.append({
                        "polygon": polygon,
                        "class_id": cls_id,
                        "class_name": class_names[cls_id],
                        "confidence": conf
                    })

        return detections


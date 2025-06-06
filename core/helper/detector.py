from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path, allowed_classes, stream: bool = False):
        self.model = YOLO(model_path)
        self.allowed_classes = set(allowed_classes) if allowed_classes else set()
        self.stream = stream

    def detect(self, frame):
        results = self.model(frame, stream=self.stream, device=-1)
        if not results:
            return []

        detections = []
        class_names = results[0].names if results else []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if self.allowed_classes and cls_id not in self.allowed_classes:
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "class_id": cls_id,
                    "class_name": class_names[cls_id],
                    "confidence": conf
                })

        return detections


    def detect_plg(self, frame):
        results = self.model(frame, stream=self.stream, device=-1)
        if not results:
            return []

        detections = []
        class_names = results[0].names if results else []

        for result in results:
            # Check if result contains masks (only in segmentation models)
            if result.masks is not None:
                for seg, cls_id, conf in zip(result.masks.xy, result.boxes.cls, result.boxes.conf):
                    cls_id = int(cls_id)
                    if len(self.allowed_classes) > 0 and cls_id not in self.allowed_classes:
                        continue

                    polygon = [(int(x), int(y)) for x, y in seg]

                    detections.append({
                        "bbox": None,  # No bounding box for masks
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
                        "bbox": (x1, y1, x2, y2),
                        "polygon": polygon,
                        "class_id": cls_id,
                        "class_name": class_names[cls_id],
                        "confidence": conf
                    })

        return detections


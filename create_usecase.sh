#!/bin/bash

# Function to get detection type
get_detection_type() {
    echo ""
    echo "Select detection type:"
    echo "1) Detection (bounding boxes)"
    echo "2) Segmentation (polygons)"
    echo ""

    sleep 3
    read -p "Enter choice (1 or 2): " choice

    case $choice in
        1) echo "detection" ;;
        2) echo "segmentation" ;;
        *) echo "Invalid choice. Defaulting to detection." >&2 && echo "detection" ;;
    esac
}

# Function to get usecase name
enter_usecase_name() {
    while true; do
        read -p "Enter usecase name (ex: example_usecase.py): " usecase_name

        if [[ -z $usecase_name ]]; then
            echo "Usecase name must be not empty"
            continue
        fi

        # Add .py extension if not present
        if [[ ! $usecase_name == *.py ]]; then
            usecase_name="$usecase_name.py"
        fi

        # Check if file exists
        if [[ -f "usecase/$usecase_name" ]]; then
            read -p "Usecase with name $usecase_name already exists, want to override? (y/n): " override
            if [[ $override =~ ^[Yy]$ ]]; then
                break
            fi
        else
            break
        fi
    done

    echo "$usecase_name"
}

# Function to get class name
enter_class_name() {
    read -p "Enter class name (ex: ExampleDetector): " class_name

    if [[ -z $class_name ]]; then
        echo "Class name must be not empty"
        exit 1
    fi

    echo "$class_name"
}

# Function to get usecase key
enter_usecase_key() {
    read -p "Enter usecase key for run_detection.py (ex: example_case): " usecase_key

    if [[ -z $usecase_key ]]; then
        echo "Usecase key must be not empty"
        exit 1
    fi

    echo "$usecase_key"
}

# Function to create detection template
create_detection_template() {
    local class_name=$1
    local model_name=$2
    local filename=$3

    cat > "usecase/$filename" << EOF
import numpy as np
from typing import List, Dict
from core.helper.detector import ObjectDetector
from usecase.base_detector import BaseDetector as Base
from core.helper.extensions import to_camel_case
import cv2
import cvzone


class $class_name(Base):
    def __init__(self, min_conf: float = 0.25, show_label: bool = True):
        allowed_classes = []  # Add your class IDs here
        self.detector = ObjectDetector(
            "model/$model_name", allowed_classes=allowed_classes
        )
        self.min_conf = min_conf
        self.show_label = show_label

    def detect(self, frame: np.ndarray) -> List[Dict]:
        color = (255, 0, 0)  # Blue color for all detections

        detections = self.detector.detect(frame)
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox
            cls_name = det["class_name"]
            conf = det["confidence"]

            if conf >= self.min_conf:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

                if self.show_label:
                    cvzone.putTextRect(
                        frame,
                        to_camel_case(cls_name),
                        (max(20, x1), max(20, y1)),
                        colorR=color,
                        scale=1,
                        thickness=1,
                    )
        return detections
EOF
}

# Function to create segmentation template
create_segmentation_template() {
    local class_name=$1
    local model_name=$2
    local filename=$3

    cat > "usecase/$filename" << EOF
from typing import Dict, List

import cv2
import cvzone
import numpy as np

from core.helper.detector import ObjectDetector
from core.helper.extensions import to_camel_case
from usecase.base_detector import BaseDetector as Base


class $class_name(Base):
    def __init__(self, min_conf: float = 0.25, show_label: bool = True):
        self.detector = ObjectDetector(
            "model/$model_name", allowed_classes=[]
        )
        self.min_conf = min_conf
        self.show_label = show_label

    def detect(self, frame: np.ndarray) -> List[Dict]:
        color = (0, 0, 255)  # Red color for all detections
        detections = self.detector.detect_plg(frame)

        if not detections:
            return []

        has_overlay = False
        overlay = frame.copy()

        for det in detections:
            conf = det["confidence"]
            if conf < self.min_conf:
                continue

            cls_name = det["class_name"]
            polygon = np.array([det["polygon"]], dtype=np.int32)

            # Fill and outline
            cv2.fillPoly(overlay, polygon, color) #type: ignore
            cv2.polylines(frame, polygon, isClosed=True, color=color, thickness=1) #type: ignore

            # Optional label
            if self.show_label:
                x, y = det["polygon"][0]
                cvzone.putTextRect(
                    frame,
                    to_camel_case(cls_name),
                    (max(20, x), max(20, y)),
                    colorR=color,
                    scale=1,
                    thickness=1,
                )

            has_overlay = True

        if has_overlay:
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        return detections
EOF
}

# Function to update run_detection.py
update_run_detection() {
    local class_name=$1
    local filename=$2
    local usecase_key=$3

    # Add import
    module_name=${filename%.py}
    sed -i "/from usecase\.fire_detector import FireDetector/a from usecase.$module_name import $class_name" run_detection.py

    # Add to use_case_map
    sed -i "/\"palm_security\": \[GeneralDetector(), FireDetector()\]/a \\        \"$usecase_key\": [$class_name()]," run_detection.py
}

# Main function
create_usecase() {
    echo "=== Usecase Creator ==="
    echo "Make sure you are placing your model into /model folder"

    read -p "Enter your model name (ex: best.pt): " model_name

    if [[ -z $model_name ]]; then
        echo "Model name must be not empty"
        exit 1
    fi

    # Check if model exists
    if [[ ! -f "model/$model_name" ]]; then
        echo "Warning: Model file 'model/$model_name' not found!"
        read -p "Continue anyway? (y/n): " continue_anyway
        if [[ ! $continue_anyway =~ ^[Yy]$ ]]; then
            echo "Exiting..."
            exit 1
        fi
    fi

    detection_type=$(get_detection_type)
    usecase_name=$(enter_usecase_name)
    class_name=$(enter_class_name)
    usecase_key=$(enter_usecase_key)

    echo ""
    echo "Creating usecase with:"
    echo "  Type: $detection_type"
    echo "  File: $usecase_name"
    echo "  Class: $class_name"
    echo "  Model: $model_name"
    echo "  Key: $usecase_key"
    echo ""

    # Create the detector file
    if [[ $detection_type == "segmentation" ]]; then
        create_segmentation_template "$class_name" "$model_name" "$usecase_name"
    else
        create_detection_template "$class_name" "$model_name" "$usecase_name"
    fi

    # Update run_detection.py
    update_run_detection "$class_name" "$usecase_name" "$usecase_key"

    echo "✅ Usecase created successfully!"
    echo "📁 File: usecase/$usecase_name"
    echo "🔧 Updated: run_detection.py"
    echo ""
    echo "Next steps:"
    echo "1. (Optional) Edit usecase/$usecase_name to customize allowed_classes if needed"
    echo "2. (Optional) Edit detection color per class"
    echo "3. Test with: python main.py --video path_to_video --usecase $usecase_key"
}

# Run the main function
create_usecase

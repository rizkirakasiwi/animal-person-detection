#!/usr/bin/env python3
"""
Usecase Creator - Python version
Automatically creates new detection/segmentation usecases and updates run_detection.py
"""

import os
import re
from pathlib import Path


def get_detection_type():
    """Get detection type from user input"""
    print("\nSelect detection type:")
    print("1) Detection (bounding boxes)")
    print("2) Segmentation (polygons)")
    print()
    
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "1":
            return "detection"
        elif choice == "2":
            return "segmentation"
        else:
            print("Invalid choice. Please enter 1 or 2.")


def get_usecase_name():
    """Get usecase filename from user input"""
    while True:
        usecase_name = input("Enter usecase name (ex: example_usecase.py): ").strip()
        
        if not usecase_name:
            print("Usecase name must be not empty")
            continue
        
        # Add .py extension if not present
        if not usecase_name.endswith('.py'):
            usecase_name += '.py'
        
        usecase_path = Path("usecase") / usecase_name
        
        # Check if file exists
        if usecase_path.exists():
            override = input(f"Usecase with name {usecase_name} already exists, want to override? (y/n): ").strip().lower()
            if override in ['y', 'yes']:
                break
        else:
            break
    
    return usecase_name


def get_class_name():
    """Get class name from user input"""
    while True:
        class_name = input("Enter class name (ex: ExampleDetector): ").strip()
        if class_name:
            return class_name
        print("Class name must be not empty")


def get_usecase_key():
    """Get usecase key from user input"""
    while True:
        usecase_key = input("Enter usecase key for run_detection.py (ex: example_case): ").strip()
        if usecase_key:
            return usecase_key
        print("Usecase key must be not empty")


def create_detection_template(class_name, model_name, filename):
    """Create detection template file"""
    template = f'''import numpy as np
from typing import List, Dict
from core.helper.detector import ObjectDetector
from usecase.base_detector import BaseDetector as Base
from core.helper.extensions import to_camel_case
import cv2
import cvzone


class {class_name}(Base):
    def __init__(self, min_conf: float = 0.25, show_label: bool = True):
        allowed_classes = []  # Add your class IDs here
        self.detector = ObjectDetector(
            "model/{model_name}", allowed_classes=allowed_classes
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
'''
    
    usecase_path = Path("usecase") / filename
    with open(usecase_path, 'w', encoding='utf-8') as f:
        f.write(template)


def create_segmentation_template(class_name, model_name, filename):
    """Create segmentation template file"""
    template = f'''from typing import Dict, List

import cv2
import cvzone
import numpy as np

from core.helper.detector import ObjectDetector
from core.helper.extensions import to_camel_case
from usecase.base_detector import BaseDetector as Base


class {class_name}(Base):
    def __init__(self, min_conf: float = 0.25, show_label: bool = True):
        self.detector = ObjectDetector(
            "model/{model_name}", allowed_classes=[]
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
'''
    
    usecase_path = Path("usecase") / filename
    with open(usecase_path, 'w', encoding='utf-8') as f:
        f.write(template)


def update_run_detection(class_name, filename, usecase_key):
    """Update run_detection.py with new import and usecase mapping"""
    run_detection_path = Path("run_detection.py")
    
    if not run_detection_path.exists():
        print("Warning: run_detection.py not found!")
        return
    
    # Read the file
    with open(run_detection_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    module_name = filename[:-3]  # Remove .py extension
    
    # Add import statement
    import_line = f"from usecase.{module_name} import {class_name}"
    
    # Find the fire_detector import line and add after it
    fire_import_pattern = r"(from usecase\.fire_detector import FireDetector)"
    if re.search(fire_import_pattern, content):
        content = re.sub(
            fire_import_pattern,
            r"\1\n" + import_line,
            content
        )
    else:
        # Fallback: add after other imports
        content = re.sub(
            r"(from usecase\.[^\n]+\n)",
            r"\1" + import_line + "\n",
            content,
            count=1
        )
    
    # Add to use_case_map
    usecase_entry = f'        "{usecase_key}": [{class_name}()],'
    
    # Find palm_security line and add after it
    palm_security_pattern = r'(\s*"palm_security": \[GeneralDetector\(\), FireDetector\(\)\],)'
    if re.search(palm_security_pattern, content):
        content = re.sub(
            palm_security_pattern,
            r"\1\n" + usecase_entry,
            content
        )
    else:
        # Fallback: add before closing brace of use_case_map
        content = re.sub(
            r'(\s*}[\s]*$)',
            f"    {usecase_entry}\n\\1",
            content,
            flags=re.MULTILINE
        )
    
    # Write back to file
    with open(run_detection_path, 'w', encoding='utf-8') as f:
        f.write(content)


def main():
    """Main function to create usecase"""
    print("=== Usecase Creator ===")
    print("Make sure you are placing your model into /model folder")
    
    # Get model name
    model_name = input("Enter your model name (ex: best.pt): ").strip()
    
    if not model_name:
        print("Model name must be not empty")
        return
    
    # Check if model exists
    model_path = Path("model") / model_name
    if not model_path.exists():
        print(f"Warning: Model file 'model/{model_name}' not found!")
        continue_anyway = input("Continue anyway? (y/n): ").strip().lower()
        if continue_anyway not in ['y', 'yes']:
            print("Exiting...")
            return
    
    # Get user inputs
    detection_type = get_detection_type()
    usecase_name = get_usecase_name()
    class_name = get_class_name()
    usecase_key = get_usecase_key()
    
    print()
    print("Creating usecase with:")
    print(f"  Type: {detection_type}")
    print(f"  File: {usecase_name}")
    print(f"  Class: {class_name}")
    print(f"  Model: {model_name}")
    print(f"  Key: {usecase_key}")
    print()
    
    # Create the detector file
    if detection_type == "segmentation":
        create_segmentation_template(class_name, model_name, usecase_name)
    else:
        create_detection_template(class_name, model_name, usecase_name)
    
    # Update run_detection.py
    update_run_detection(class_name, usecase_name, usecase_key)
    
    print("✅ Usecase created successfully!")
    print(f"📁 File: usecase/{usecase_name}")
    print("🔧 Updated: run_detection.py")
    print()
    print("Next steps:")
    print(f"1. (Optional) Edit usecase/{usecase_name} to customize allowed_classes if needed")
    print("2. (Optional) Edit detection color per class")
    print(f"3. Test with: python main.py --video path_to_video --usecase {usecase_key}")


if __name__ == "__main__":
    main()
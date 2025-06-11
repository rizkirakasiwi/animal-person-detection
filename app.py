import tempfile

import streamlit as st
import torch
import cv2

from run_detection import run_detection

torch.classes.__path__ = []


def __main():
    st.title("🚨 Object Detection Interface")

    use_case = st.selectbox("Select Use Case", ["palm_security", "ppe", "road_damage"])
    input_source = st.radio("Select Input Source", ["Webcam", "Upload Video"])

    if input_source == "Webcam":
        if use_case and st.button("Start Detection"):
            st.write("Using webcam for live detection...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Error: Cannot access webcam")
                return
            run_detection(cap, use_case)
    elif input_source == "Upload Video":
        video_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
        if video_file and st.button("Start Detection"):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            cap = cv2.VideoCapture(tfile.name)
            run_detection(cap, use_case)

if __name__ == "__main__":
    __main()
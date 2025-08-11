import tempfile

import cv2
import streamlit as st
import torch

from run_detection import run_detection

torch.classes.__path__ = []


def __main():
    st.title("🚨 Object Detection Interface")

    use_case = st.selectbox("Select Use Case", ["palm_security", "ppe", "road_damage", "bird_drop", "bad_parking"])
    input_source = st.radio(
        "Select Input Source",
        ["Camera (by Device ID)", "Upload Video", "Streaming Link"]
    )

    if input_source == "Camera (by Device ID)":
        device_id = st.number_input("Enter camera device ID (e.g., 0, 1)", min_value=0, step=1, value=0)
        if use_case and st.button("Start Detection"):
            st.write(f"🎥 Using camera device ID {device_id} for live detection...")
            cap = cv2.VideoCapture(device_id)
            if not cap.isOpened():
                st.error(f"❌ Error: Cannot access camera device ID {device_id}")
                return
            run_detection(cap, use_case)

    elif input_source == "Upload Video":
        video_file = st.file_uploader("📁 Upload a Video", type=["mp4", "avi", "mov"])
        if video_file and st.button("Start Detection"):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            cap = cv2.VideoCapture(tfile.name)
            run_detection(cap, use_case)

    elif input_source == "Streaming Link":
        stream_url = st.text_input("🔗 Enter streaming URL (e.g. RTSP/HTTP)")
        if stream_url and use_case and st.button("Start Detection"):
            st.write(f"📡 Connecting to stream: {stream_url}")
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                st.error("❌ Error: Cannot access the streaming link")
                return
            run_detection(cap, use_case)


if __name__ == "__main__":
    __main()

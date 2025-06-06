import tempfile

import streamlit as st
import torch

from run_detection import run_detection

torch.classes.__path__ = []


def __main():
    st.title("🚨 Object Detection Interface")

    use_case = st.selectbox("Select Use Case", ["palm_security", "ppe", "road_damage"])
    video_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

    if video_file and st.button("Start Detection"):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        run_detection(tfile.name, use_case)

if __name__ == "__main__":
    __main()
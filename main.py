import argparse
import cv2

from run_detection import run_detection

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
    if args.video == "0":
        args.video = 0
        
    cap = cv2.VideoCapture(args.video)

    run_detection(cap, args.usecase)

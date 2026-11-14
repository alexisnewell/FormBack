
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from src.video_processor import process_video


def main():

    print("FormBack Running Form Analysis")
    print("------------------------------")

    print("OpenCV:", cv2.__version__)
    print("MediaPipe:", mp.__version__)
    print("NumPy:", np.__version__)

    print("\nEnvironment successfully configured!")

    if len(sys.argv) < 2:
        print("\nPlease provide a video path.")
        print("Example: python main.py videos/video1_running.mp4")
        return

    video_path = Path(sys.argv[1])

    if not video_path.exists():
        print(f"\nError: Video not found: {video_path}")
        return

    print("\nStarting video processing...")
    print("----------------------------")
    print("Input video:", video_path)

    process_video(str(video_path))

    print("\nFormBack processing finished!")


if __name__ == "__main__":
    main()

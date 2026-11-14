import cv2
import csv
import argparse

from pathlib import Path

from src.pose_estimator import PoseEstimator


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_heavy.task"

OUTPUT_DIR = PROJECT_ROOT / "output"

# Landmarks below this visibility are drawn in orange (thin) instead of hidden,
# so you can see where the model is unsure rather than having limbs vanish.
VISIBILITY_DRAW = 0.5

GREEN = (0, 255, 0)        # BGR
RED = (0, 0, 255)
ORANGE = (0, 165, 255)


# MediaPipe body landmark connections.

CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (27, 29), (29, 31),
    (24, 26), (26, 28),
    (28, 30), (30, 32),
    (27, 31), (28, 32)
]


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def open_writer(path, fps, size):
    """Try H.264 first (much better quality), fall back to mp4v."""

    for code in ("avc1", "mp4v"):

        fourcc = cv2.VideoWriter_fourcc(*code)

        writer = cv2.VideoWriter(str(path), fourcc, fps, size)

        if writer.isOpened():
            print(f"Video codec: {code}")
            return writer

        writer.release()

    raise RuntimeError("Could not initialize video writer.")


def draw_skeleton(frame, landmarks, width, height):

    for start_id, end_id in CONNECTIONS:

        start = landmarks[start_id]
        end = landmarks[end_id]

        confident = (
            getattr(start, "visibility", 1.0) >= VISIBILITY_DRAW
            and getattr(end, "visibility", 1.0) >= VISIBILITY_DRAW
        )

        cv2.line(
            frame,
            (int(start.x * width), int(start.y * height)),
            (int(end.x * width), int(end.y * height)),
            GREEN if confident else ORANGE,
            3 if confident else 1
        )

    for landmark in landmarks:

        confident = (
            getattr(landmark, "visibility", 1.0) >= VISIBILITY_DRAW
        )

        cv2.circle(
            frame,
            (int(landmark.x * width), int(landmark.y * height)),
            5 if confident else 3,
            RED if confident else ORANGE,
            -1
        )


# --------------------------------------------------
# VIDEO PROCESSING
# --------------------------------------------------

def process_video(video_path):

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found: {MODEL_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_video = OUTPUT_DIR / (
        video_path.stem + "_annotated.mp4"
    )

    output_csv = OUTPUT_DIR / (
        video_path.stem + "_landmarks.csv"
    )

    # ----------------------------------------------
    # 1. OPEN VIDEO
    # ----------------------------------------------

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    duration = frame_count / fps

    print(f"Video: {video_path.name}")
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.2f}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Frames: {frame_count}")
    print(f"Model: {MODEL_PATH.name}")

    if duration < 10 or duration > 60:
        print(
            "Warning: Expected video duration is 10-60 seconds."
        )

    # ----------------------------------------------
    # 2. INITIALIZE VIDEO OUTPUT
    # ----------------------------------------------

    try:
        writer = open_writer(output_video, fps, (width, height))
    except RuntimeError:
        cap.release()
        raise

    frame_index = 0
    frames_without_pose = 0

    # ----------------------------------------------
    # 3. INITIALIZE POSE ESTIMATOR
    # ----------------------------------------------

    try:

        with PoseEstimator(MODEL_PATH) as estimator:

            with open(
                output_csv,
                "w",
                newline="",
                encoding="utf-8"
            ) as csv_file:

                csv_writer = csv.writer(csv_file)

                csv_writer.writerow([
                    "frame",
                    "timestamp_ms",
                    "landmark",
                    "x",
                    "y",
                    "z",
                    "visibility"
                ])

                # ----------------------------------
                # 4. PROCESS EACH FRAME
                # ----------------------------------

                while True:

                    success, frame = cap.read()

                    if not success:
                        break

                    timestamp_ms = int(
                        round(frame_index * 1000 / fps)
                    )

                    # Pose estimation runs on the clean frame, before drawing.
                    result = estimator.estimate(
                        frame,
                        timestamp_ms
                    )

                    # ----------------------------------
                    # 5. EXTRACT LANDMARKS
                    # ----------------------------------

                    if result.pose_landmarks:

                        landmarks = result.pose_landmarks[0]

                        for landmark_id, landmark in enumerate(
                            landmarks
                        ):

                            csv_writer.writerow([
                                frame_index,
                                timestamp_ms,
                                landmark_id,
                                landmark.x,
                                landmark.y,
                                landmark.z,
                                getattr(
                                    landmark,
                                    "visibility",
                                    ""
                                )
                            ])

                        # ----------------------------------
                        # 6. DRAW SKELETON
                        # ----------------------------------

                        draw_skeleton(frame, landmarks, width, height)

                    else:
                        frames_without_pose += 1

                    # ----------------------------------
                    # 7. ADD FRAME INFORMATION
                    # ----------------------------------

                    cv2.putText(
                        frame,
                        f"Frame: {frame_index}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2
                    )

                    # ----------------------------------
                    # 8. SAVE PROCESSED FRAME
                    # ----------------------------------

                    writer.write(frame)

                    frame_index += 1

                    if frame_index % 30 == 0:
                        print(
                            f"Processed {frame_index} frames..."
                        )

    finally:

        cap.release()
        writer.release()

    print("\nProcessing complete!")

    print(f"Frames with no pose detected: {frames_without_pose}")
    print(f"Annotated video: {output_video}")
    print(f"Landmark data: {output_csv}")


# --------------------------------------------------
# PROGRAM ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Running form video processor"
    )

    parser.add_argument(
        "video",
        help="Path to input running video"
    )

    args = parser.parse_args()

    process_video(args.video)
import argparse
import csv
import os
import time

from pathlib import Path

import cv2

from src.pose_estimator import PoseEstimator


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_full.task"

OUTPUT_DIR = PROJECT_ROOT / "output"

# Longest side (pixels) of the frame handed to the pose model.
DEFAULT_INFER_MAX_SIDE = 640

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

CSV_HEADER = [
    "frame",
    "timestamp_ms",
    "landmark",
    "x",
    "y",
    "z",
    "visibility"
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

def process_video(
    video_path,
    write_video=False,
    frame_stride=1,
    infer_max_side=DEFAULT_INFER_MAX_SIDE,
):
    """
    Extract pose landmarks from a video into output/<stem>_landmarks.csv.

    write_video:    also render output/<stem>_annotated.mp4 (slow; the
                    analysis does not need it).
    frame_stride:   run the model on every Nth frame only. Leave at 1 unless
                    you have checked the results: gait_analysis treats the
                    skipped frames as missing data, which lowers the
                    landmark-coverage quality score.
    infer_max_side: longest side of the frame given to the model.
    """

    video_path = Path(video_path)
    frame_stride = max(1, int(frame_stride))

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

    # Written first, then renamed, so a crash can never leave a half-finished
    # *_landmarks.csv behind for the pipeline to reuse as a cache.
    temp_csv = output_csv.with_suffix(".csv.tmp")

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
    print(f"Inference size: longest side {infer_max_side or 'original'}")

    if frame_stride > 1:
        print(f"Frame stride: {frame_stride} (every {frame_stride}th frame)")

    if duration < 10 or duration > 60:
        print(
            "Warning: Expected video duration is 10-60 seconds."
        )

    # ----------------------------------------------
    # 2. OPTIONAL ANNOTATED VIDEO
    # ----------------------------------------------

    writer = None

    if write_video:
        try:
            writer = open_writer(
                output_video, fps / frame_stride, (width, height)
            )
        except RuntimeError:
            cap.release()
            raise

    frame_index = 0
    processed = 0
    frames_without_pose = 0
    rows = []

    t_read = 0.0
    t_detect = 0.0
    t_start = time.perf_counter()

    # ----------------------------------------------
    # 3. PROCESS FRAMES
    # ----------------------------------------------

    try:

        with PoseEstimator(MODEL_PATH, infer_max_side) as estimator:

            while True:

                t0 = time.perf_counter()

                # grab() advances without the BGR conversion, so skipped
                # frames cost less than a full read().
                if not cap.grab():
                    break

                if frame_index % frame_stride:
                    frame_index += 1
                    t_read += time.perf_counter() - t0
                    continue

                success, frame = cap.retrieve()

                t_read += time.perf_counter() - t0

                if not success:
                    break

                timestamp_ms = int(
                    round(frame_index * 1000 / fps)
                )

                t0 = time.perf_counter()

                result = estimator.estimate(
                    frame,
                    timestamp_ms
                )

                t_detect += time.perf_counter() - t0

                # ------------------------------
                # EXTRACT LANDMARKS
                # ------------------------------

                if result.pose_landmarks:

                    landmarks = result.pose_landmarks[0]

                    rows.extend(
                        (
                            frame_index,
                            timestamp_ms,
                            landmark_id,
                            landmark.x,
                            landmark.y,
                            landmark.z,
                            getattr(landmark, "visibility", "")
                        )
                        for landmark_id, landmark in enumerate(landmarks)
                    )

                    if writer is not None:
                        draw_skeleton(frame, landmarks, width, height)

                else:
                    frames_without_pose += 1

                if writer is not None:

                    cv2.putText(
                        frame,
                        f"Frame: {frame_index}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2
                    )

                    writer.write(frame)

                processed += 1
                frame_index += 1

                if processed % 100 == 0:
                    print(
                        f"Processed {processed} frames..."
                    )

        # ------------------------------------------
        # 4. WRITE CSV ONCE, THEN PUBLISH IT
        # ------------------------------------------

        with open(
            temp_csv,
            "w",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(CSV_HEADER)
            csv_writer.writerows(rows)

        os.replace(temp_csv, output_csv)

    finally:

        cap.release()

        if writer is not None:
            writer.release()

        if temp_csv.exists():
            temp_csv.unlink()

    total = time.perf_counter() - t_start

    print("\nProcessing complete!")

    print(f"Frames processed: {processed}")
    print(f"Frames with no pose detected: {frames_without_pose}")

    if write_video:
        print(f"Annotated video: {output_video}")

    print(f"Landmark data: {output_csv}")

    if processed:
        print(
            f"Timing: {total:.1f}s total "
            f"({processed / total:.1f} frames/s) | "
            f"detection {t_detect:.1f}s | "
            f"video read {t_read:.1f}s | "
            f"other {max(0.0, total - t_detect - t_read):.1f}s"
        )


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

    parser.add_argument(
        "--annotate",
        action="store_true",
        help="also write the annotated video (slower)"
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="process every Nth frame (default 1 = all frames)"
    )

    parser.add_argument(
        "--infer-size",
        type=int,
        default=DEFAULT_INFER_MAX_SIDE,
        help="longest side of the frame given to the model, 0 = original"
    )

    args = parser.parse_args()

    process_video(
        args.video,
        write_video=args.annotate,
        frame_stride=args.stride,
        infer_max_side=args.infer_size,
    )
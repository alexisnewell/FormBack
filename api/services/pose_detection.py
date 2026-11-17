"""
Pose detection: video -> MediaPipe landmarks CSV (+ annotated video).

Stage 1 of the pipeline. Merges the old pose_estimator.py and
video_processor.py.

MediaPipe is imported lazily inside ``PoseEstimator`` so that importing this
module (and therefore starting the API) stays fast and does not require
MediaPipe to be installed until a video is actually processed.

Command line:
    python -m api.services.pose_detection videos/clip.mp4
"""

import argparse
import csv
import logging
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)

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
    (27, 31), (28, 32),
]

CSV_COLUMNS = ["frame", "timestamp_ms", "landmark", "x", "y", "z", "visibility"]


# --------------------------------------------------
# POSE ESTIMATOR
# --------------------------------------------------

class PoseEstimator:
    """Thin wrapper around MediaPipe's PoseLandmarker in VIDEO mode."""

    def __init__(self, model_path):
        import mediapipe as mp  # heavy import, kept lazy
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self._mp = mp

        base_options = python.BaseOptions(
            model_asset_path=str(model_path)
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.detector = vision.PoseLandmarker.create_from_options(options)

    def estimate(self, frame, timestamp_ms):
        """
        Detect body landmarks in one video frame.

        Input:
            frame: OpenCV BGR image
            timestamp_ms: Frame timestamp in milliseconds

        Returns:
            MediaPipe pose detection result
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        return self.detector.detect_for_video(mp_image, timestamp_ms)

    def close(self):
        """Release MediaPipe resources."""
        self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


# --------------------------------------------------
# DRAWING HELPERS
# --------------------------------------------------

def open_writer(path, fps, size):
    """Try H.264 first (much better quality), fall back to mp4v."""

    for code in ("avc1", "mp4v"):

        fourcc = cv2.VideoWriter_fourcc(*code)
        writer = cv2.VideoWriter(str(path), fourcc, fps, size)

        if writer.isOpened():
            logger.info("Video codec: %s", code)
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
            3 if confident else 1,
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
            -1,
        )


# --------------------------------------------------
# VIDEO PROCESSING
# --------------------------------------------------

def process_video(video_path, output_dir, model_path):
    """
    Run pose estimation over every frame of a video.

    Writes two files into ``output_dir``:
        <stem>_landmarks.csv   one row per landmark per frame
        <stem>_annotated.mp4   the video with the skeleton drawn on

    Returns a dict with both paths and frame counts.
    """

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    model_path = Path(model_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"MediaPipe model not found: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    output_video = output_dir / (video_path.stem + "_annotated.mp4")
    output_csv = output_dir / (video_path.stem + "_landmarks.csv")

    # ----------------------------------------------
    # 1. OPEN VIDEO
    # ----------------------------------------------

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps

    logger.info("Video: %s", video_path.name)
    logger.info("Resolution: %d x %d", width, height)
    logger.info("FPS: %.2f", fps)
    logger.info("Duration: %.2f seconds", duration)
    logger.info("Frames: %d", frame_count)
    logger.info("Model: %s", model_path.name)

    if duration < 10 or duration > 60:
        logger.warning("Expected video duration is 10-60 seconds.")

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
    # 3. PROCESS EVERY FRAME
    # ----------------------------------------------

    try:

        with PoseEstimator(model_path) as estimator:

            with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:

                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(CSV_COLUMNS)

                while True:

                    success, frame = cap.read()

                    if not success:
                        break

                    timestamp_ms = int(round(frame_index * 1000 / fps))

                    # Pose estimation runs on the clean frame, before drawing.
                    result = estimator.estimate(frame, timestamp_ms)

                    if result.pose_landmarks:

                        landmarks = result.pose_landmarks[0]

                        for landmark_id, landmark in enumerate(landmarks):

                            csv_writer.writerow([
                                frame_index,
                                timestamp_ms,
                                landmark_id,
                                landmark.x,
                                landmark.y,
                                landmark.z,
                                getattr(landmark, "visibility", ""),
                            ])

                        draw_skeleton(frame, landmarks, width, height)

                    else:
                        frames_without_pose += 1

                    cv2.putText(
                        frame,
                        f"Frame: {frame_index}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )

                    writer.write(frame)

                    frame_index += 1

                    if frame_index % 30 == 0:
                        logger.info("Processed %d frames...", frame_index)

    finally:

        cap.release()
        writer.release()

    logger.info("Processing complete. Frames with no pose: %d", frames_without_pose)

    return {
        "landmarks": output_csv,
        "annotated_video": output_video,
        "frames": frame_index,
        "frames_without_pose": frames_without_pose,
    }


# --------------------------------------------------
# PROGRAM ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Running form pose detection")
    parser.add_argument("video", help="Path to input running video")
    parser.add_argument("--output-dir", default=str(project_root / "output"))
    parser.add_argument(
        "--model",
        default=str(project_root / "models" / "pose_landmarker_heavy.task"),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    info = process_video(args.video, args.output_dir, args.model)
    print(f"Landmark data:   {info['landmarks']}")
    print(f"Annotated video: {info['annotated_video']}")

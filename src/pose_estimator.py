import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PoseEstimator:

    def __init__(self, model_path, infer_max_side=640):
        """
        infer_max_side: frames whose longest side is larger than this are
        shrunk before detection. MediaPipe downsamples internally anyway, so
        feeding it 1080p+ frames only adds color-conversion and copy cost.
        Landmarks are normalized (0-1), so they still map onto the original
        full-size frame. Use 0 or None to disable resizing.
        """

        self.infer_max_side = infer_max_side

        base_options = python.BaseOptions(
            model_asset_path=str(model_path)
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.detector = vision.PoseLandmarker.create_from_options(
            options
        )

    def estimate(self, frame, timestamp_ms):
        """
        Detect body landmarks in one video frame.

        Input:
            frame: OpenCV BGR image (not modified)
            timestamp_ms: Frame timestamp in milliseconds (must increase)

        Returns:
            MediaPipe pose detection result (normalized landmarks)
        """

        height, width = frame.shape[:2]
        longest = max(height, width)

        # Shrink first so the BGR->RGB conversion runs on fewer pixels.
        if self.infer_max_side and longest > self.infer_max_side:
            scale = self.infer_max_side / longest
            frame = cv2.resize(
                frame,
                (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_AREA
            )

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        return self.detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

    def close(self):
        """Release MediaPipe resources."""

        self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = Path("output/video1_running_landmarks.csv")
df = pd.read_csv(INPUT_FILE)

FPS = 25

CANDIDATES = [
    ("left", 11),
    ("left", 21),
    ("left", 38),
    ("left", 46),
    ("left", 55),
    ("left", 61),
    ("right", 15),
    ("right", 39),
    ("right", 51),
]

LANDMARKS = {
    "left": {"heel": 29, "toe": 31},
    "right": {"heel": 30, "toe": 32},
}

WINDOW = 2
MIN_VISIBILITY = 0.5

print("\nFORMBACK RAW FOOT STABILITY ANALYSIS")
print("------------------------------------")

for side, center_frame in CANDIDATES:

    print(f"\n{side.upper()} FOOT — CANDIDATE FRAME {center_frame}")

    start = center_frame - WINDOW
    end = center_frame + WINDOW

    frame_values = range(start, end + 1)

    for part, landmark_id in LANDMARKS[side].items():

        points = df[
            (df["landmark"] == landmark_id)
            & (df["frame"].between(start, end))
            & (df["visibility"] >= MIN_VISIBILITY)
        ].copy()

        points = points.sort_values("frame")

        if len(points) < 3:
            print(f"  {part}: insufficient valid observations")
            continue

        x_range = points["x"].max() - points["x"].min()
        y_range = points["y"].max() - points["y"].min()

        x_steps = points["x"].diff().abs()
        y_steps = points["y"].diff().abs()

        max_step = np.sqrt(
            x_steps.pow(2) + y_steps.pow(2)
        ).max()

        print(
            f"  {part}: "
            f"observations={len(points)}, "
            f"x_range={x_range:.4f}, "
            f"y_range={y_range:.4f}, "
            f"max_frame_step={max_step:.4f}"
        )

print("\n------------------------------------")
print("Analysis complete.")


import pandas as pd
from pathlib import Path

INPUT_FILE = Path("output/video1_running_landmarks.csv")

df = pd.read_csv(INPUT_FILE)

# Landmarks:
# 23 = left hip
# 24 = right hip
# 29 = left heel
# 30 = right heel
# 31 = left foot index
# 32 = right foot index

CANDIDATES = [
    ("left", 38),
    ("right", 51),
    ("left", 46)
]

LANDMARKS = {
    "left_hip": 23,
    "right_hip": 24,
    "left_heel": 29,
    "right_heel": 30,
    "left_toe": 31,
    "right_toe": 32
}

print("\nFORMBACK CANDIDATE MOTION INSPECTION")
print("------------------------------------")

for side, center_frame in CANDIDATES:

    print(f"\n{side.upper()} FOOT — FRAME {center_frame}")
    print("")

    start = center_frame - 3
    end = center_frame + 3

    for frame in range(start, end + 1):

        frame_data = df[df["frame"] == frame]

        print(f"FRAME {frame}")

        for name, landmark_id in LANDMARKS.items():

            point = frame_data[
                frame_data["landmark"] == landmark_id
            ]

            if point.empty:
                print(f"  {name}: MISSING")
                continue

            row = point.iloc[0]

            print(
                f"  {name}: "
                f"x={row['x']:.4f}, "
                f"y={row['y']:.4f}, "
                f"visibility={row['visibility']:.3f}"
            )

print("\n------------------------------------")
print("Inspection complete.")

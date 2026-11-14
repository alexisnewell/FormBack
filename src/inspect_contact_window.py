
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("output/video1_running_landmarks.csv")
df = pd.read_csv(INPUT_FILE)

FPS = 25
CENTER_FRAME = 51
START_FRAME = 45
END_FRAME = 57

LANDMARKS = {
    "right_heel": 30,
    "right_toe": 32,
    "left_heel": 29,
    "left_toe": 31,
    "left_hip": 23,
    "right_hip": 24,
}

print("\nFORMBACK CONTACT WINDOW INSPECTION")
print("----------------------------------")

for frame in range(START_FRAME, END_FRAME + 1):

    frame_data = df[df["frame"] == frame]

    print(f"\nFRAME {frame} | TIME {(frame / FPS):.3f}s")

    for name, landmark_id in LANDMARKS.items():

        point = frame_data[
            frame_data["landmark"] == landmark_id
        ]

        if point.empty:
            print(f"{name:12}: MISSING")
            continue

        row = point.iloc[0]

        print(
            f"{name:12}: "
            f"x={row['x']:.4f}, "
            f"y={row['y']:.4f}, "
            f"visibility={row['visibility']:.3f}"
        )

print("\n----------------------------------")
print("Inspection complete.")

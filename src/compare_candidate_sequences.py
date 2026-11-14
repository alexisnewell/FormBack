
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("output/video1_running_landmarks.csv")
df = pd.read_csv(INPUT_FILE)

LANDMARKS = {
    "left_heel": 29,
    "left_toe": 31,
    "right_heel": 30,
    "right_toe": 32,
    "left_hip": 23,
    "right_hip": 24,
}

START_FRAME = 55
END_FRAME = 63

print("\nFORMBACK — TWO-SIDED STANCE COMPARISON")
print("--------------------------------------")

for frame in range(START_FRAME, END_FRAME + 1):

    frame_data = df[df["frame"] == frame]

    def get_x(landmark_id):
        point = frame_data[
            frame_data["landmark"] == landmark_id
        ]

        if point.empty:
            return None

        return point.iloc[0]["x"]

    values = {
        name: get_x(landmark_id)
        for name, landmark_id in LANDMARKS.items()
    }

    if (
        values["left_hip"] is not None
        and values["right_hip"] is not None
    ):
        pelvis_x = (
            values["left_hip"] + values["right_hip"]
        ) / 2
    else:
        pelvis_x = None

    print(f"\nFRAME {frame}")

    for name in [
        "left_heel",
        "left_toe",
        "right_heel",
        "right_toe",
    ]:
        value = values[name]

        if value is None:
            print(f"{name:12}: MISSING")
        else:
            print(f"{name:12}: {value:.4f}")

    if pelvis_x is not None:
        print(f"{'pelvis_x':12}: {pelvis_x:.4f}")

print("\n--------------------------------------")
print("Comparison complete.")

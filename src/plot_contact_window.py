
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_FILE = Path("output/video1_running_landmarks.csv")
OUTPUT_FILE = Path("output/right_foot_contact_window.png")

df = pd.read_csv(INPUT_FILE)

START_FRAME = 45
END_FRAME = 57
FPS = 25

# Extract relevant landmarks
landmark_ids = {
    "right_heel_x": 30,
    "right_toe_x": 32,
    "left_hip_x": 23,
    "right_hip_x": 24,
}

data = {}

for name, landmark_id in landmark_ids.items():

    points = df[
        (df["landmark"] == landmark_id)
        & (df["frame"].between(START_FRAME, END_FRAME))
    ][["frame", "x"]].copy()

    data[name] = points.set_index("frame")["x"]

plot_df = pd.DataFrame(data)

plot_df["pelvis_x"] = (
    plot_df["left_hip_x"] + plot_df["right_hip_x"]
) / 2

plot_df["time_seconds"] = plot_df.index / FPS

plt.figure(figsize=(11, 6))

plt.plot(
    plot_df["time_seconds"],
    plot_df["right_heel_x"],
    marker="o",
    label="Right heel X"
)

plt.plot(
    plot_df["time_seconds"],
    plot_df["right_toe_x"],
    marker="o",
    label="Right toe X"
)

plt.plot(
    plot_df["time_seconds"],
    plot_df["pelvis_x"],
    marker="o",
    label="Pelvis X"
)

plt.axvspan(
    50 / FPS,
    53 / FPS,
    alpha=0.15,
    label="Potential stable-foot interval"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Horizontal image position (normalized)")
plt.title("FormBack — Right Foot Movement Around Candidate Contact")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=200)
plt.show()

print(f"\nGraph saved to: {OUTPUT_FILE}")

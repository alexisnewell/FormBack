
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load landmark data
csv_path = Path("output/video1_running_landmarks.csv")
df = pd.read_csv(csv_path)

# MediaPipe landmark IDs
LANDMARKS = {
    "Left heel": 29,
    "Right heel": 30,
    "Left foot index": 31,
    "Right foot index": 32
}

# Keep only the detected running section
fig, axes = plt.subplots(2, 1, figsize=(12, 9))

for name, landmark_id in LANDMARKS.items():

    data = df[df["landmark"] == landmark_id].copy()

    if data.empty:
        print(f"Warning: No data for {name}")
        continue

    # Convert timestamps to seconds
    data["time_seconds"] = data["timestamp_ms"] / 1000

    # Plot vertical foot position
    axes[0].plot(
        data["time_seconds"],
        data["y"],
        label=name
    )

    # Plot horizontal foot position
    axes[1].plot(
        data["time_seconds"],
        data["x"],
        label=name
    )

axes[0].set_title("Foot Vertical Movement")
axes[0].set_xlabel("Time (seconds)")
axes[0].set_ylabel("Normalized Y position")
axes[0].invert_yaxis()
axes[0].legend()
axes[0].grid(True)

axes[1].set_title("Foot Horizontal Movement")
axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Normalized X position")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()

output_path = Path("output/video1_running_foot_trajectory.png")
plt.savefig(output_path, dpi=150)
plt.show()

print(f"\nFoot trajectory saved to: {output_path}")

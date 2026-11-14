
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

VIDEO_FILE = Path("output/video1_running_annotated.mp4")
OUTPUT_FILE = Path("output/left_contact_frame_inspection.png")

FRAMES = [57, 58, 59, 60, 61, 62, 63, 64, 65]

cap = cv2.VideoCapture(str(VIDEO_FILE))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_FILE}")

fps = cap.get(cv2.CAP_PROP_FPS)

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
axes = axes.flatten()

for ax, frame_number in zip(axes, FRAMES):

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()

    if not success:
        ax.set_title(f"Frame {frame_number} - unavailable")
        ax.axis("off")
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    ax.imshow(frame_rgb)
    ax.set_title(
        f"Frame {frame_number} | {frame_number / fps:.2f}s"
    )
    ax.axis("off")

cap.release()

plt.suptitle(
    "FormBack — Right Foot Contact Candidate",
    fontsize=16
)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=180)
plt.show()

print(f"\nSaved inspection sheet: {OUTPUT_FILE}")

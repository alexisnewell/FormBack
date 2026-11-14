
import cv2
import pandas as pd
from pathlib import Path
import math

VIDEO_PATH = "output/video1_running_annotated.mp4"
EVENTS_PATH = "output/video1_gait_events.csv"
OUTPUT_PATH = "output/gait_event_inspection.png"

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    raise RuntimeError("Could not open annotated video.")

events = pd.read_csv(EVENTS_PATH)

if events.empty:
    raise RuntimeError("No candidate events found.")

frames = []

for _, event in events.iterrows():

    frame_number = int(event["frame"])
    side = event["side"]

    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = video.read()

    if not success:
        print(f"Could not read frame {frame_number}")
        continue

    # Resize frame for the inspection sheet.
    frame = cv2.resize(frame, (640, 360))

    label = (
        f"{side.upper()} CONTACT CANDIDATE | "
        f"Frame {frame_number} | "
        f"{event['time_seconds']:.2f}s"
    )

    cv2.rectangle(frame, (0, 0), (640, 35), (0, 0, 0), -1)

    cv2.putText(
        frame,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    frames.append(frame)

video.release()

if not frames:
    raise RuntimeError("No event frames could be extracted.")

columns = 3
rows = math.ceil(len(frames) / columns)

sheet = cv2.vconcat([
    cv2.hconcat(
        frames[i:i + columns]
        + [
            255 * (0 * frames[0])
            for _ in range(columns - len(frames[i:i + columns]))
        ]
    )
    for i in range(0, len(frames), columns)
])

Path("output").mkdir(exist_ok=True)

cv2.imwrite(OUTPUT_PATH, sheet)

print("\nGait event inspection complete.")
print("Events inspected:", len(frames))
print("Saved:", OUTPUT_PATH)

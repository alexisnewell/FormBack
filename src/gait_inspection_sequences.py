
import cv2
import pandas as pd
import math
from pathlib import Path

VIDEO_PATH = "output/video1_running_annotated.mp4"
EVENTS_PATH = "output/video1_gait_events.csv"
OUTPUT_PATH = "output/gait_event_sequences.png"

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    raise RuntimeError("Could not open annotated video.")

events = pd.read_csv(EVENTS_PATH)

# Five frames around each candidate event.
offsets = [-2, -1, 0, 1, 2]

tile_width = 320
tile_height = 180
label_height = 25

rows = []

for _, event in events.iterrows():

    event_frame = int(event["frame"])
    side = event["side"]

    tiles = []

    for offset in offsets:

        frame_number = event_frame + offset

        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = video.read()

        if not success:
            continue

        frame = cv2.resize(
            frame,
            (tile_width, tile_height)
        )

        label = (
            f"{side.upper()} | "
            f"Frame {frame_number} | "
            f"Offset {offset:+d}"
        )

        cv2.rectangle(
            frame,
            (0, 0),
            (tile_width, label_height),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            label,
            (5, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1
        )

        tiles.append(frame)

    if len(tiles) == len(offsets):

        rows.append(cv2.hconcat(tiles))

video.release()

if not rows:
    raise RuntimeError("No event sequences could be extracted.")

sheet = cv2.vconcat(rows)

Path("output").mkdir(exist_ok=True)

cv2.imwrite(OUTPUT_PATH, sheet)

print("\nGait sequence inspection complete.")
print("Candidate events:", len(rows))
print("Frames per event:", len(offsets))
print("Saved:", OUTPUT_PATH)

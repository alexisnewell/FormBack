"""
Contact sheet of frames around each detected gait event, for checking by eye.

Reads the <stem>_events.csv written by gait_analysis.py (columns: side,
event, frame, ...) and a video. Use the annotated video to see the skeleton:

    python -m src.video_processor clip.mp4 --annotate
    python -m src.gait_inspection_sequences output/clip_annotated.mp4

Frame numbers only line up if landmarks were extracted with the default
frame stride of 1.
"""

import argparse
from pathlib import Path

import cv2
import pandas as pd

LABEL_HEIGHT = 25

EVENT_SHORT = {"initial_contact": "IC", "toe_off": "TO"}


def default_events_path(video_path):

    stem = video_path.stem.replace("_annotated", "")

    return video_path.parent / f"{stem}_events.csv"


def select_events(events, side, kind, limit):

    if side:
        events = events[events["side"] == side]

    if kind:
        events = events[events["event"] == kind]

    if limit:
        events = events.head(limit)

    return events


def collect_frames(video, needed, tile_size):
    """Decode the video once, keeping only the needed frames, resized."""

    wanted = set(needed)
    last = max(wanted)
    frames = {}

    index = 0

    while index <= last:

        success, frame = video.read()

        if not success:
            break

        if index in wanted:
            frames[index] = cv2.resize(
                frame, tile_size, interpolation=cv2.INTER_AREA
            )

        index += 1

    return frames


def build_sheet(video_path, events_path, output_path, offsets,
                tile_height, side, kind, limit):

    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Keep the video's own aspect ratio (portrait clips stay portrait).
    tile_size = (round(tile_height * width / height), tile_height)

    events = pd.read_csv(events_path)

    if not {"frame", "side"}.issubset(events.columns):
        raise ValueError("Events CSV needs 'frame' and 'side' columns.")

    events = select_events(events, side, kind, limit)

    if events.empty:
        raise RuntimeError("No events match the selection.")

    needed = {
        int(event["frame"]) + offset
        for _, event in events.iterrows()
        for offset in offsets
    }

    frames = collect_frames(video, needed, tile_size)

    video.release()

    rows = []
    skipped = 0

    for _, event in events.iterrows():

        event_frame = int(event["frame"])

        kind_label = EVENT_SHORT.get(
            event.get("event", ""), str(event.get("event", ""))
        )

        tiles = []

        for offset in offsets:

            frame_number = event_frame + offset

            if frame_number not in frames:
                break

            tile = frames[frame_number].copy()

            label = (
                f"{str(event['side']).upper()} {kind_label} | "
                f"F{frame_number} | {offset:+d}"
            )

            cv2.rectangle(
                tile, (0, 0), (tile_size[0], LABEL_HEIGHT), (0, 0, 0), -1
            )

            cv2.putText(
                tile, label, (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
            )

            tiles.append(tile)

        if len(tiles) == len(offsets):
            rows.append(cv2.hconcat(tiles))
        else:
            skipped += 1

    if not rows:
        raise RuntimeError("No event sequences could be extracted.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), cv2.vconcat(rows))

    print("\nGait sequence inspection complete.")
    print(f"Events shown: {len(rows)} (skipped near video edges: {skipped})")
    print(f"Frames per event: {len(offsets)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Contact sheet of frames around each gait event."
    )

    parser.add_argument("video", help="Annotated (or original) video")
    parser.add_argument("--events", help="Events CSV (default: <stem>_events.csv beside the video)")
    parser.add_argument("--out", help="Output PNG (default: <stem>_event_sequences.png)")
    parser.add_argument("--side", choices=["left", "right"])
    parser.add_argument("--event", choices=["initial_contact", "toe_off"])
    parser.add_argument("--max-events", type=int, help="Only the first N events")
    parser.add_argument("--span", type=int, default=2, help="Frames either side of each event (default 2)")
    parser.add_argument("--tile-height", type=int, default=360)

    args = parser.parse_args()

    video_path = Path(args.video)

    events_path = (
        Path(args.events) if args.events else default_events_path(video_path)
    )

    output_path = (
        Path(args.out) if args.out
        else video_path.parent / (
            video_path.stem.replace("_annotated", "") + "_event_sequences.png"
        )
    )

    build_sheet(
        video_path,
        events_path,
        output_path,
        offsets=list(range(-args.span, args.span + 1)),
        tile_height=args.tile_height,
        side=args.side,
        kind=args.event,
        limit=args.max_events,
    )
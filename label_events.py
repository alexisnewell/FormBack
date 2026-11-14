"""
Frame-by-frame gait event labelling tool.

Step through the ORIGINAL video (not the annotated one) and press a key at each
initial contact and toe-off. Labels are saved to a CSV after every key press, in
the format validate_events.py expects:

    side,event,frame
    left,initial_contact,812

Usage:
    python label_events.py videos/16773916_1080_1920_120fps.mp4 labels_120fps.csv
    python label_events.py videos/clip.mp4 labels.csv --start 5 --end 14

Re-running with the same labels file resumes where you left off.

Keys
    d / a    next / previous frame
    w / s    forward / back 10 frames
    1        left  initial contact
    2        left  toe-off
    3        right initial contact
    4        right toe-off
    u        undo the last label
    x / Esc  save and quit

Definitions (keep them the same for every label):
    initial contact : first frame in which the foot touches the ground
    toe-off         : last frame in which the toe is still touching the ground

Do not look at the detector's output while labelling; that biases the labels.
Frame numbers are 0-based, matching video_processor.py and the landmarks CSV.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2

KEYS = {
    ord("1"): ("left", "initial_contact"),
    ord("2"): ("left", "toe_off"),
    ord("3"): ("right", "initial_contact"),
    ord("4"): ("right", "toe_off"),
}
SHORT = {
    ("left", "initial_contact"): "L contact",
    ("left", "toe_off"): "L toe-off",
    ("right", "initial_contact"): "R contact",
    ("right", "toe_off"): "R toe-off",
}
DISPLAY_HEIGHT = 900


def load_labels(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [(r["side"], r["event"], int(r["frame"])) for r in csv.DictReader(f)]


def save_labels(path, labels):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["side", "event", "frame"])
        for side, event, frame in sorted(labels, key=lambda r: r[2]):
            w.writerow([side, event, frame])


def read_frame(cap, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    return frame if ok else None


def render(frame, index, fps, labels, last_frame):
    h, w = frame.shape[:2]
    scale = min(1.0, DISPLAY_HEIGHT / h)
    img = cv2.resize(frame, (int(w * scale), int(h * scale)))

    def put(text, y, color=(255, 255, 255), size=0.7, thick=2):
        cv2.putText(img, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), thick + 3)
        cv2.putText(img, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thick)

    put(f"Frame {index} / {last_frame}   t = {index / fps:.3f} s", 30)

    here = [SHORT[(s, e)] for s, e, f in labels if f == index]
    for i, text in enumerate(here):
        put(f"LABELLED: {text}", 70 + 35 * i, (0, 255, 255), 0.9)

    counts = Counter((s, e) for s, e, _ in labels)
    put("L contact %d | L toe-off %d | R contact %d | R toe-off %d" % (
        counts[("left", "initial_contact")], counts[("left", "toe_off")],
        counts[("right", "initial_contact")], counts[("right", "toe_off")]),
        img.shape[0] - 40, (0, 255, 0), 0.55, 1)
    put("a/d +-1   s/w +-10   1 L contact  2 L toe-off  3 R contact  4 R toe-off  u undo  x quit",
        img.shape[0] - 15, (200, 200, 200), 0.5, 1)
    return img


def main():
    ap = argparse.ArgumentParser(description="Label gait events frame by frame.")
    ap.add_argument("video")
    ap.add_argument("labels", help="output CSV (resumes if it exists)")
    ap.add_argument("--start", type=float, default=0.0, help="start time (s)")
    ap.add_argument("--end", type=float, help="end time (s)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    first = max(0, int(args.start * fps))
    last = min(total - 1, int(args.end * fps)) if args.end else total - 1

    labels = load_labels(args.labels)
    if labels:
        print(f"Resuming with {len(labels)} existing labels.")
    index = first

    while True:
        frame = read_frame(cap, index)
        if frame is None:
            index = max(first, index - 1)
            continue

        cv2.imshow("label_events", render(frame, index, fps, labels, last))
        key = cv2.waitKey(0) & 0xFF

        if key == ord("d"):
            index += 1
        elif key == ord("a"):
            index -= 1
        elif key == ord("w"):
            index += 10
        elif key == ord("s"):
            index -= 10
        elif key in KEYS:
            side, event = KEYS[key]
            if (side, event, index) not in labels:
                labels.append((side, event, index))
                save_labels(args.labels, labels)
        elif key == ord("u") and labels:
            labels.pop()
            save_labels(args.labels, labels)
        elif key in (ord("x"), 27):
            break

        index = min(max(index, first), last)

    cap.release()
    cv2.destroyAllWindows()
    save_labels(args.labels, labels)

    counts = Counter((s, e) for s, e, _ in labels)
    print(f"Saved {len(labels)} labels to {args.labels}")
    for (side, event), n in sorted(counts.items()):
        print(f"  {side:5} {event:16} {n}")


if __name__ == "__main__":
    main()

"""
Compare detected gait events with hand-labelled ground truth.

Labels CSV (you create this by stepping through the video):
    side,event,frame
    left,initial_contact,41
    left,toe_off,49
    right,initial_contact,54
    ...

Usage:
    python validate_events.py output/video1_events.csv labels_video1.csv --fps 25
    python validate_events.py output/video1_events.csv labels_video1.csv --fps 25 --tolerance 4

Reports, per side and event type:
    recall      fraction of labelled events that were detected
    precision   fraction of detections that match a label
    bias        mean (detected - labelled); a consistent non-zero value
                means a fixed offset you can correct for
    mae         mean absolute error
Errors are shown in frames and milliseconds.
"""

import argparse

import numpy as np
import pandas as pd


def match(pred, truth, tolerance):
    """Greedy one-to-one nearest matching. Returns errors (pred - truth)."""
    pred = sorted(pred)
    used = set()
    errors = []
    for t in sorted(truth):
        best, best_d = None, None
        for j, p in enumerate(pred):
            if j in used:
                continue
            d = abs(p - t)
            if d <= tolerance and (best_d is None or d < best_d):
                best, best_d = j, d
        if best is not None:
            used.add(best)
            errors.append(pred[best] - t)
    return np.array(errors), len(pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("detected", help="<stem>_events.csv from gait_analysis")
    ap.add_argument("labels", help="hand-labelled CSV: side,event,frame")
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--tolerance", type=float, default=3.0,
                    help="max frames between detection and label to count as a match")
    args = ap.parse_args()

    det = pd.read_csv(args.detected)
    lab = pd.read_csv(args.labels)
    col = "frame_exact" if "frame_exact" in det.columns else "frame"
    ms = 1000 / args.fps

    print(f"{'side':6} {'event':16} {'labels':>6} {'found':>6} {'recall':>7} "
          f"{'prec':>6} {'bias(fr)':>9} {'bias(ms)':>9} {'MAE(fr)':>8} {'MAE(ms)':>8}")
    for side in ("left", "right"):
        for event in ("initial_contact", "toe_off"):
            t = lab[(lab.side == side) & (lab.event == event)]["frame"].tolist()
            p = det[(det.side == side) & (det.event == event)][col].tolist()
            if not t:
                continue
            err, n_pred = match(p, t, args.tolerance)
            recall = len(err) / len(t)
            prec = len(err) / n_pred if n_pred else 0.0
            if len(err):
                bias, mae = err.mean(), np.abs(err).mean()
                print(f"{side:6} {event:16} {len(t):6d} {n_pred:6d} {recall:7.0%} "
                      f"{prec:6.0%} {bias:9.2f} {bias * ms:9.1f} {mae:8.2f} {mae * ms:8.1f}")
            else:
                print(f"{side:6} {event:16} {len(t):6d} {n_pred:6d} {recall:7.0%} "
                      f"{prec:6.0%} {'-':>9} {'-':>9} {'-':>8} {'-':>8}")

    print("\nReading the result:")
    print("  - Low recall: missed events (check landmark visibility / camera view).")
    print("  - Low precision: spurious events (lower sensitivity or check the debug plot).")
    print("  - Consistent bias: add a fixed offset to event times.")
    print("  - High MAE with ~zero bias: noisy detection; try a higher frame rate.")


if __name__ == "__main__":
    main()
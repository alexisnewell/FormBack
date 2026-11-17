"""
Synthetic running landmarks with known ground truth, for tests.

The runner's pelvis stays fixed in the frame (as if the camera pans with them).
Each ankle swings forward and back relative to the pelvis in a way that has
known timing:

    initial contact  = forward extreme, at   t = offset + k * period
    toe-off          = backward extreme, at  t = offset + k * period + duty * period

The left and right legs are half a period apart, so with duty < 0.5 there is a
positive flight time of (0.5 - duty) * period.

Output matches the CSV written by pose_detection.process_video:
    frame, timestamp_ms, landmark, x, y, z, visibility
"""

import csv
import math
from pathlib import Path

import numpy as np

WIDTH = 1000
HEIGHT = 1000
LEG_PX = 400.0          # hip -> ankle, in pixels
AMPLITUDE = 0.35        # ankle swing, in leg lengths, either side of the pelvis


def _swing(phase, duty):
    """Ankle position relative to pelvis, in leg lengths.

    phase in [0, 1). +AMPLITUDE at phase 0 (contact), -AMPLITUDE at phase
    `duty` (toe-off), back to +AMPLITUDE at phase 1.
    """
    p = phase % 1.0
    if p < duty:
        theta = math.pi * p / duty
    else:
        theta = math.pi + math.pi * (p - duty) / (1 - duty)
    return AMPLITUDE * math.cos(theta)


def expected_events(seconds, period, duty, offset=0.0):
    """Ground-truth (initial_contact_s, toe_off_s) for one leg."""
    ic, to = [], []
    k = 0
    while True:
        t_ic = offset + k * period
        if t_ic > seconds:
            break
        ic.append(t_ic)
        to.append(t_ic + duty * period)
        k += 1
    return np.array(ic), np.array(to)


def make_landmarks_csv(path, seconds=20.0, fps=120, period=0.7, duty=0.4,
                       facing=1, left_visible=True, right_visible=True):
    """Write a landmarks CSV and return the number of frames.

    facing: +1 runner faces right, -1 faces left.
    right_visible=False hides the right leg (visibility 0) to mimic occlusion.
    """
    n = int(seconds * fps)
    rows = []

    for f in range(n):
        t = f / fps
        ts = int(round(f * 1000 / fps))

        px = 0.5 * WIDTH
        py = 0.45 * HEIGHT + 6.0 * math.sin(2 * math.pi * 2 * t / period)
        shoulder = (px + facing * 8.0, py - 0.62 * LEG_PX)

        points = {0: (shoulder[0] + facing * 10, shoulder[1] - 60)}   # nose
        points[11] = points[12] = shoulder

        for side, base in (("l", 23), ("r", 24)):
            offset = 0.0 if side == "l" else 0.5 * period
            rel = _swing((t - offset) / period, duty)

            hip = (px, py)
            ax = px + facing * rel * LEG_PX
            ay = py + LEG_PX * 0.92 * math.sqrt(max(0.0, 1 - rel ** 2))
            ankle = (ax, ay)

            # Knee: midpoint pushed slightly forward, giving a mild bend.
            mid = ((hip[0] + ankle[0]) / 2, (hip[1] + ankle[1]) / 2)
            knee = (mid[0] + facing * 0.06 * LEG_PX, mid[1])

            heel = (ax - facing * 12, ay + 14)
            toe = (ax + facing * 40, ay + 14)

            points[base] = hip
            points[base + 2] = knee
            points[base + 4] = ankle
            points[base + 6] = heel
            points[base + 8] = toe

        for lm, (x, y) in points.items():
            right_leg = lm in (24, 26, 28, 30, 32)
            left_leg = lm in (23, 25, 27, 29, 31)
            vis = 1.0
            if (right_leg and not right_visible) or (left_leg and not left_visible):
                vis = 0.0
            rows.append([f, ts, lm, x / WIDTH, y / HEIGHT, 0.0, vis])

    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "timestamp_ms", "landmark", "x", "y", "z", "visibility"])
        w.writerows(rows)
    return n

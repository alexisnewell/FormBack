"""
Joint-angle time series from MediaPipe landmarks.

Uses the same conditioning and conventions as gait_analysis.py, so numbers
here agree with the ones the app reports:

  * pixel coordinates from the real video size (no aspect distortion)
  * low-visibility landmarks masked, short gaps filled, tracks low-passed
  * knee flexion  = 180 - included hip-knee-ankle angle (0 = straight leg)
  * trunk lean    = shoulder-midpoint vs pelvis-midpoint line from vertical,
                    positive = leaning toward the running direction
  * hip flexion   = thigh swing + trunk lean (positive = flexion,
                    negative = extension); signed, unlike an included angle
  * ankle         = dorsiflexion (+) / plantarflexion (-) relative to a
                    perpendicular foot; approximate, because MediaPipe's
                    heel and toe landmarks are noisy

Usage:
    python -m src.biomechanics output/clip_landmarks.csv --video path/to/clip.mp4
    python -m src.biomechanics output/clip_landmarks.csv --width 1080 --height 1920
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.gait_analysis import (
    SIDES,
    build_tracks,
    estimate_fps,
    facing_sign,
    included_angle,
    load_landmarks,
    video_dimensions,
)


def forward_angle(vec, direction, from_up):
    """
    Angle of an image-space vector from vertical, in degrees.

    Positive means pointing toward the running direction. from_up=True
    measures from straight up (trunk), False from straight down (thigh).
    Image y grows downward, hence the sign flip.
    """

    dy = -vec[:, 1] if from_up else vec[:, 1]

    return np.degrees(np.arctan2(direction * vec[:, 0], dy))


def ankle_dorsiflexion(knee, ankle, heel, toe):
    """Dorsiflexion (+) / plantarflexion (-): 90 deg minus shank-foot angle."""

    shank = knee - ankle
    foot = toe - heel

    den = np.linalg.norm(shank, axis=1) * np.linalg.norm(foot, axis=1)
    cosine = np.clip((shank * foot).sum(axis=1) / den, -1.0, 1.0)

    return 90.0 - np.degrees(np.arccos(cosine))


def compute_angles(tracks, direction):
    """Return a dict of per-frame angle arrays from conditioned tracks."""

    pelvis = (tracks["l_hip"] + tracks["r_hip"]) / 2
    shoulder = (tracks["l_sh"] + tracks["r_sh"]) / 2

    trunk = forward_angle(shoulder - pelvis, direction, from_up=True)

    out = {"trunk_lean_deg": trunk}

    for side, p in SIDES.items():

        hip = tracks[f"{p}_hip"]
        knee = tracks[f"{p}_knee"]
        ankle = tracks[f"{p}_ank"]

        thigh = forward_angle(knee - hip, direction, from_up=False)

        out[f"{side}_knee_flexion_deg"] = 180.0 - included_angle(
            hip, knee, ankle
        )

        out[f"{side}_hip_flexion_deg"] = thigh + trunk

        out[f"{side}_ankle_dorsiflexion_deg"] = ankle_dorsiflexion(
            knee, ankle, tracks[f"{p}_heel"], tracks[f"{p}_toe"]
        )

    return out


def process_landmarks(csv_path, width, height):

    csv_path = Path(csv_path)

    if width <= 0 or height <= 0:
        raise ValueError("Video width and height must be positive.")

    df = load_landmarks(csv_path)
    fps = estimate_fps(df)

    tracks = build_tracks(df, width, height, fps)
    direction = facing_sign(tracks)

    angles = compute_angles(tracks, direction)

    n_frames = len(next(iter(angles.values())))

    table = pd.DataFrame({
        "frame": np.arange(n_frames),
        "time_seconds": np.arange(n_frames) / fps,
        **angles,
    })

    base = csv_path.stem.replace("_landmarks", "")
    output_csv = csv_path.parent / f"{base}_joint_angles.csv"
    output_plot = csv_path.parent / f"{base}_joint_angles.png"

    table.to_csv(output_csv, index=False)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    for ax, name, title in (
        (axes[0], "knee_flexion_deg", "Knee flexion (0 = straight)"),
        (axes[1], "hip_flexion_deg", "Hip flexion (+) / extension (-)"),
        (axes[2], "ankle_dorsiflexion_deg", "Ankle dorsiflexion (+) / plantarflexion (-), approximate"),
    ):
        for side in SIDES:
            ax.plot(table["time_seconds"], table[f"{side}_{name}"], label=side.capitalize())

        ax.set_ylabel("degrees")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (seconds)")
    fig.tight_layout()
    fig.savefig(output_plot, dpi=150)
    plt.close(fig)

    print("\nJoint-angle analysis complete.")
    print(f"Video size used: {width} x {height}, {fps:.1f} fps, "
          f"running direction: {'right' if direction > 0 else 'left'}")
    print(f"Angle data: {output_csv}")
    print(f"Graph: {output_plot}")

    print("\nMedian and 5th-95th percentile (valid frames only):")

    for column in angles:

        values = table[column].to_numpy()
        valid = values[np.isfinite(values)]

        if valid.size == 0:
            print(f"{column}: no valid frames")
            continue

        # Rounding first turns tiny negatives like -1e-15 into a clean 0.0.
        valid = np.round(valid, 1) + 0.0
        low, high = np.percentile(valid, [5, 95])

        print(
            f"{column}: median {np.median(valid):.1f}, "
            f"range {low:.1f} to {high:.1f} "
            f"({valid.size / n_frames:.0%} of frames valid)"
        )

    return table


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Calculate running joint angles from pose landmarks."
    )

    parser.add_argument("csv", help="Path to the landmark CSV")

    parser.add_argument(
        "--video",
        help="Original video (its size is read from the file)"
    )

    parser.add_argument("--width", type=int, help="Video width in pixels")
    parser.add_argument("--height", type=int, help="Video height in pixels")

    args = parser.parse_args()

    if args.video:
        width, height = video_dimensions(args.video)
    elif args.width and args.height:
        width, height = args.width, args.height
    else:
        parser.error("give --video, or both --width and --height")

    process_landmarks(args.csv, width, height)
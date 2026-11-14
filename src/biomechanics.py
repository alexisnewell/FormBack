
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_angle(a, b, c):
    """Calculate the angle ABC in degrees. B is the middle point."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator < 1e-8:
        return np.nan

    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine)))


def get_point(frame, landmark_id, width, height):
    """Return landmark coordinates converted from normalized values to pixels."""

    row = frame.loc[frame["landmark"] == landmark_id]

    if row.empty:
        return None

    row = row.iloc[0]

    x = row["x"]
    y = row["y"]

    if not np.isfinite(x) or not np.isfinite(y):
        return None

    # Convert normalized MediaPipe coordinates to pixel coordinates.
    x_pixel = x * width
    y_pixel = y * height

    return np.array([x_pixel, y_pixel], dtype=float)


def angle_from_frame(frame, a_id, b_id, c_id, width, height):
    """Calculate an angle from three landmarks in one frame."""

    a = get_point(frame, a_id, width, height)
    b = get_point(frame, b_id, width, height)
    c = get_point(frame, c_id, width, height)

    if a is None or b is None or c is None:
        return np.nan

    return calculate_angle(a, b, c)


def process_landmarks(csv_path, width, height):

    csv_path = Path(csv_path)

    data = pd.read_csv(csv_path)

    required = {
        "frame", "timestamp_ms", "landmark", "x", "y"
    }

    if not required.issubset(data.columns):
        raise ValueError(
            f"CSV must contain columns: {sorted(required)}"
        )

    if width <= 0 or height <= 0:
        raise ValueError("Video width and height must be positive.")

    data["landmark"] = pd.to_numeric(
        data["landmark"], errors="coerce"
    )
    data["x"] = pd.to_numeric(data["x"], errors="coerce")
    data["y"] = pd.to_numeric(data["y"], errors="coerce")

    results = []

    # MediaPipe landmark IDs:
    # Shoulder 11/12, hip 23/24, knee 25/26,
    # ankle 27/28, foot index 31/32.

    for frame_number, frame in data.groupby("frame", sort=True):

        frame = frame.drop_duplicates(
            subset="landmark", keep="last"
        )

        timestamp = frame["timestamp_ms"].iloc[0]

        row = {
            "frame": frame_number,
            "timestamp_ms": timestamp,
            "time_seconds": timestamp / 1000.0,
        }

        # Knee angles
        row["left_knee_angle"] = angle_from_frame(
            frame, 23, 25, 27, width, height
        )

        row["right_knee_angle"] = angle_from_frame(
            frame, 24, 26, 28, width, height
        )

        # Hip included angles
        row["left_hip_included_angle"] = angle_from_frame(
            frame, 11, 23, 25, width, height
        )

        row["right_hip_included_angle"] = angle_from_frame(
            frame, 12, 24, 26, width, height
        )

        # Approximate ankle angles using foot index
        row["left_ankle_proxy_angle"] = angle_from_frame(
            frame, 25, 27, 31, width, height
        )

        row["right_ankle_proxy_angle"] = angle_from_frame(
            frame, 26, 28, 32, width, height
        )

        results.append(row)

    angles = pd.DataFrame(results)

    output_dir = csv_path.parent

    base_name = csv_path.stem.replace("_landmarks", "")

    output_csv = output_dir / f"{base_name}_joint_angles.csv"
    output_plot = output_dir / f"{base_name}_joint_angles.png"

    angles.to_csv(output_csv, index=False)

    # Plot knee angles over time
    plt.figure(figsize=(12, 5))

    plt.plot(
        angles["time_seconds"],
        angles["left_knee_angle"],
        label="Left knee"
    )

    plt.plot(
        angles["time_seconds"],
        angles["right_knee_angle"],
        label="Right knee"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Included knee angle (degrees)")
    plt.title("Knee angles throughout the running video")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_plot, dpi=150)
    plt.close()

    print("\nBiomechanics analysis complete.")
    print(f"Video dimensions: {width} x {height}")
    print(f"Angle data: {output_csv}")
    print(f"Knee angle graph: {output_plot}")

    print("\nMean angles across valid frames:")

    angle_columns = [
        "left_knee_angle",
        "right_knee_angle",
        "left_hip_included_angle",
        "right_hip_included_angle",
        "left_ankle_proxy_angle",
        "right_ankle_proxy_angle",
    ]

    for column in angle_columns:
        print(
            f"{column}: {angles[column].mean():.2f} degrees"
        )

    return angles


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Calculate running joint angles from pose landmarks."
    )

    parser.add_argument(
        "csv",
        help="Path to the landmark CSV"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="Original video width in pixels"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=1280,
        help="Original video height in pixels"
    )

    args = parser.parse_args()

    process_landmarks(
        args.csv,
        args.width,
        args.height
    )

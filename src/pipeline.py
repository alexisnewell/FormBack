"""
One-call pipeline: video -> landmarks -> gait analysis -> structured result.

Command line:
    python -m src.pipeline videos/16773916_1080_1920_120fps.mp4
    python -m src.pipeline videos/clip.mp4 --toe-off-offset -6 --window 5 14
    python -m src.pipeline videos/clip.mp4 --force      # re-extract landmarks

From code (what an API endpoint would call):
    from src.pipeline import analyze_video
    result = analyze_video("videos/clip.mp4")
    result["status"]    # "ok", "low_quality" or "failed"
    result["summary"]   # metrics, quality, warnings

Landmark extraction is skipped when <stem>_landmarks.csv already exists in the
output folder, because it is by far the slowest step.
"""

import argparse
import json
from pathlib import Path

from src.gait_analysis import (
    analyze, plot_cycles, plot_debug, video_dimensions,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# Set this from the bias that validate_events.py reports on your labelled clips
# (negative = move toe-off earlier). 0 means "no correction".
DEFAULT_TOE_OFF_OFFSET = 0.0

MIN_QUALITY = 0.5       # below this the result is flagged low_quality
MIN_VALID_STRIDES = 3   # per leg


def analyze_video(
    video_path,
    toe_off_offset_frames=DEFAULT_TOE_OFF_OFFSET,
    window=None,
    visibility=0.5,
    force=False,
    output_dir=OUTPUT_DIR,
):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    landmarks = output_dir / f"{stem}_landmarks.csv"

    result = {
        "status": "failed",
        "video": str(video_path),
        "summary": None,
        "files": {},
        "error": None,
    }

    try:
        # 1. Landmarks (slow, so reuse when possible)
        if force or not landmarks.exists():
            if output_dir.resolve() != OUTPUT_DIR.resolve():
                raise ValueError(
                    f"No landmarks at {landmarks}. Extraction only writes to "
                    f"{OUTPUT_DIR}; run video_processor first or use that folder.")
            from src.video_processor import process_video  # heavy import, kept lazy
            process_video(video_path)

        # 2. Analysis
        width, height = video_dimensions(video_path)
        events, strides, summary, context = analyze(
            landmarks, width, height,
            visibility=visibility,
            toe_off_offset_frames=toe_off_offset_frames,
            window=tuple(window) if window else None,
        )

        # 3. Save everything
        files = {
            "events": output_dir / f"{stem}_events.csv",
            "strides": output_dir / f"{stem}_strides.csv",
            "summary": output_dir / f"{stem}_summary.json",
            "debug_plot": output_dir / f"{stem}_debug.png",
            "cycles_plot": output_dir / f"{stem}_cycles.png",
        }
        events.to_csv(files["events"], index=False)
        strides.to_csv(files["strides"], index=False)
        files["summary"].write_text(json.dumps(summary, indent=2, default=float))
        plot_debug(context, files["debug_plot"])
        plot_cycles(context, strides, files["cycles_plot"])

        # 4. Quality gate: callers should hide metrics when status != "ok"
        per_side = summary["per_side"]
        enough = all(
            (per_side[s].get("stride_time_s") or {}).get("n", 0) >= MIN_VALID_STRIDES
            for s in ("left", "right"))
        good = summary["quality"]["score"] >= MIN_QUALITY and enough

        result.update(
            status="ok" if good else "low_quality",
            summary=summary,
            files={k: str(v) for k, v in files.items()},
        )

    except Exception as exc:  # report instead of crashing a caller / API worker
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def main():
    ap = argparse.ArgumentParser(description="Run the full gait pipeline on a video.")
    ap.add_argument("video")
    ap.add_argument("--toe-off-offset", type=float, default=DEFAULT_TOE_OFF_OFFSET)
    ap.add_argument("--window", type=float, nargs=2, metavar=("START", "END"))
    ap.add_argument("--visibility", type=float, default=0.5)
    ap.add_argument("--force", action="store_true", help="re-extract landmarks")
    args = ap.parse_args()

    result = analyze_video(
        args.video, args.toe_off_offset, args.window, args.visibility, args.force)

    print(f"status: {result['status']}")
    if result["error"]:
        print("error:", result["error"])
        return

    summary = result["summary"]
    q = summary["quality"]
    print(f"cadence {summary['cadence_spm']} spm | quality {q['label']} ({q['score']})")
    for w in summary["warnings"]:
        print("  !", w)
    for name, path in result["files"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()

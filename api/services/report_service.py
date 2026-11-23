"""
Report service: runs an analysis and produces the report the API returns.

Everything between "a video file exists" and "the frontend has JSON" lives
here, in this order:

    PATHS           where videos, outputs and the model live
    ERRORS          raised here, turned into HTTP responses in main.py
    VIDEO FILES     id validation, file lookup, "is this really a video?"
    PIPELINE        video -> landmarks -> gait analysis -> files on disk
    JOB STORE       in-memory state of each analysis (swap for a DB later)
    REPORT BUILDING raw summary dict -> the clean models in schemas/analysis.py
    EXPORT          clean models -> downloadable PDF / CSV (see report_export.py)
    PUBLIC API      what the routes call

Command line (runs the full pipeline without the API):
    python -m api.services.report_service videos/clip.mp4
    python -m api.services.report_service videos/clip.mp4 --window 5 14
    python -m api.services.report_service videos/clip.mp4 --force
"""

import argparse
import json
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from api.schemas.analysis import (
    AnalysisInfo,
    AnalysisQuality,
    Confidence,
    Diagnostics,
    JobStatus,
    MetricStat,
    Metrics,
    OverallMetrics,
    QualityInfo,
    ReportFormat,
    ResultsResponse,
)
from api.services import report_export
from api.services.gait_analysis import analyze, plot_cycles, plot_debug


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIDEOS_DIR = PROJECT_ROOT / "videos"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_full.task"

VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

# Artefact name -> file name suffix written by analyze_video
ARTEFACT_SUFFIXES = {
    "events": "_events.csv",
    "strides": "_strides.csv",
    "summary": "_summary.json",
    "debug_plot": "_debug.png",
    "cycles_plot": "_cycles.png",
}

# Set this from the bias that validate_events.py reports on your labelled clips
# (negative = move toe-off earlier). 0 means "no correction".
DEFAULT_TOE_OFF_OFFSET = 0.0

MIN_QUALITY = 0.5       # below this the result is flagged low_quality
MIN_VALID_STRIDES = 3   # per leg


# --------------------------------------------------
# ERRORS
# --------------------------------------------------

class AnalysisError(Exception):
    status_code = 500
    detail = "Internal error."

    def __init__(self, detail: Optional[str] = None):
        self.detail = detail or self.detail
        super().__init__(self.detail)


class InvalidVideoId(AnalysisError):
    status_code = 422
    detail = "video_id must be a valid UUID."


class InvalidVideo(AnalysisError):
    status_code = 400
    detail = "The file could not be read as a video."


class VideoNotFound(AnalysisError):
    status_code = 404
    detail = "Video not found."


class AnalysisNotFound(AnalysisError):
    status_code = 404
    detail = "No analysis found for this video."


class AnalysisInProgress(AnalysisError):
    status_code = 409
    detail = "Analysis is already processing."


class ReportUnavailable(AnalysisError):
    status_code = 409
    detail = "No report is available for this analysis."


# --------------------------------------------------
# VIDEO FILES
# --------------------------------------------------

def normalize_video_id(video_id: str) -> str:
    """Return the canonical UUID string, or raise InvalidVideoId.

    Besides rejecting junk, this guarantees the id can never contain path
    separators before it is used to build a file name.
    """
    try:
        return str(uuid.UUID(video_id))
    except (ValueError, AttributeError, TypeError):
        raise InvalidVideoId()


def video_path(video_id: str) -> Path:
    return VIDEOS_DIR / f"{normalize_video_id(video_id)}.mp4"


def inspect_video(path: Path) -> Dict[str, float]:
    """Open the file with OpenCV and return basic properties.

    Raises InvalidVideo if it cannot be decoded. This catches renamed
    non-video files, which an extension check alone does not.
    """
    import cv2  # lazy: keeps module import light

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise InvalidVideo()

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if fps <= 0 or frames <= 0:
            raise InvalidVideo("The video has no readable frames.")

        ok, _ = cap.read()
        if not ok:
            raise InvalidVideo("The first frame could not be decoded.")

        return {
            "fps": round(fps, 2),
            "frame_count": frames,
            "duration_s": round(frames / fps, 2),
        }
    finally:
        cap.release()


def video_dimensions(path) -> Tuple[int, int]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def available_files(video_id: str) -> List[str]:
    return [
        name
        for name, suffix in ARTEFACT_SUFFIXES.items()
        if (OUTPUT_DIR / f"{video_id}{suffix}").exists()
    ]


# --------------------------------------------------
# PIPELINE
# --------------------------------------------------

def classify_quality(summary: Dict[str, Any]) -> AnalysisQuality:
    """The quality gate: callers should hide metrics unless this is 'ok'."""
    per_side = summary.get("per_side") or {}
    enough = all(
        ((per_side.get(side) or {}).get("stride_time_s") or {}).get("n", 0)
        >= MIN_VALID_STRIDES
        for side in ("left", "right")
    )
    score = (summary.get("quality") or {}).get("score", 0)

    if score >= MIN_QUALITY and enough:
        return AnalysisQuality.ok
    return AnalysisQuality.low_quality


def analyze_video(
    video_path,
    toe_off_offset_frames=DEFAULT_TOE_OFF_OFFSET,
    window=None,
    visibility=0.5,
    force=False,
    output_dir=None,
):
    """
    Video -> landmarks -> gait analysis -> files on disk.

    Returns a dict (never raises, so a worker thread cannot crash):
        status   "ok", "low_quality" or "failed"
        video    the input path
        summary  the metrics dict, or None
        files    {name: path} of everything written
        error    message when status is "failed"

    Landmark extraction is skipped when <stem>_landmarks.csv already exists in
    the output folder, because it is by far the slowest step.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir or OUTPUT_DIR)
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
            from api.services.pose_detection import process_video  # heavy
            process_video(video_path, output_dir, MODEL_PATH)

        # 2-3. Events and metrics
        width, height = video_dimensions(video_path)
        events, strides, summary, context = analyze(
            landmarks, width, height,
            visibility=visibility,
            toe_off_offset_frames=toe_off_offset_frames,
            window=tuple(window) if window else None,
        )

        # 4. Save everything
        files = {
            name: output_dir / f"{stem}{suffix}"
            for name, suffix in ARTEFACT_SUFFIXES.items()
        }
        events.to_csv(files["events"], index=False)
        strides.to_csv(files["strides"], index=False)
        files["summary"].write_text(json.dumps(summary, indent=2, default=float))
        plot_debug(context, files["debug_plot"])
        plot_cycles(context, strides, files["cycles_plot"])

        # 5. Quality gate: callers should hide metrics when status != "ok"
        result.update(
            status=classify_quality(summary).value,
            summary=summary,
            files={k: str(v) for k, v in files.items()},
        )

    except Exception as exc:  # report instead of crashing a caller / API worker
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


# --------------------------------------------------
# JOB STORE
# --------------------------------------------------

@dataclass
class Job:
    status: JobStatus
    analysis_status: Optional[AnalysisQuality] = None
    summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# In-memory for now. Guarded by a lock because run_analysis executes in a
# worker thread while requests are handled on others.
_jobs: Dict[str, Job] = {}
_lock = threading.Lock()


def _load_job_from_disk(video_id: str) -> Optional[Job]:
    """Rebuild a finished job from its summary file (survives restarts)."""
    summary_file = OUTPUT_DIR / f"{video_id}_summary.json"

    if not summary_file.exists():
        return None

    try:
        summary = json.loads(summary_file.read_text())
        return Job(
            status=JobStatus.completed,
            analysis_status=classify_quality(summary),
            summary=summary,
        )
    except Exception:
        return None


# --------------------------------------------------
# REPORT BUILDING
# --------------------------------------------------

SYMMETRY_SUFFIX = "_symmetry_index_pct"

# metric key -> (label, unit, text that identifies its entry in metric_confidence)
METRICS: Dict[str, Tuple[str, Optional[str], str]] = {
    "stride_time_s": ("Stride time", "s", "stride time"),
    "contact_time_s": ("Ground contact time", "s", "contact time"),
    "flight_time_s": ("Flight time", "s", "flight time"),
    "duty_factor": ("Duty factor", None, "duty factor"),
    "knee_flexion_at_contact_deg": (
        "Knee flexion at contact", "deg", "knee flexion"),
    "peak_stance_knee_flexion_deg": (
        "Peak stance knee flexion", "deg", "knee flexion"),
    "ankle_ahead_of_pelvis_leg_lengths": (
        "Ankle ahead of pelvis", "leg lengths", "ankle ahead"),
    "mean_trunk_lean_deg": ("Trunk lean", "deg", "trunk lean"),
    "vertical_oscillation_pct_leg": (
        "Vertical oscillation", "% leg", "vertical oscillation"),
    "foot_strike_angle_deg": (
        "Foot strike angle", "deg", "foot strike angle"),
}


def _confidence_for(
    metric_confidence: Dict[str, str], needle: str
) -> Tuple[Optional[Confidence], Optional[str]]:
    """Find the confidence entry for a metric.

    The pipeline groups metrics under one key ("contact time, flight time,
    duty factor") and writes "<level>: <note>" or just "<level>". Split that
    into a level and an optional note.
    """
    for group, text in (metric_confidence or {}).items():
        if needle in group:
            level, _, note = str(text).partition(":")
            try:
                return Confidence(level.strip().lower()), (note.strip() or None)
            except ValueError:
                return None, str(text)
    return None, None


def build_quality(summary: Dict[str, Any]) -> Optional[QualityInfo]:
    quality = summary.get("quality")
    return QualityInfo(**quality) if quality else None


def build_analysis_info(summary: Dict[str, Any]) -> AnalysisInfo:
    return AnalysisInfo(
        fps=summary.get("fps"),
        direction=summary.get("direction"),
        estimated_cycle_period_s=summary.get("estimated_cycle_period_s"),
        window_s=summary.get("window_s"),
        toe_off_offset_frames=summary.get("toe_off_offset_frames"),
    )


def build_metrics(summary: Dict[str, Any]) -> Metrics:
    metric_confidence = summary.get("metric_confidence") or {}

    per_side: Dict[str, Dict[str, MetricStat]] = {}
    for side, raw in (summary.get("per_side") or {}).items():
        stats: Dict[str, MetricStat] = {}
        for key, (label, unit, needle) in METRICS.items():
            value = (raw or {}).get(key)
            if not value:
                continue  # no usable strides for this metric
            confidence, note = _confidence_for(metric_confidence, needle)
            stats[key] = MetricStat(
                label=label,
                unit=unit,
                median=value["median"],
                iqr=value["iqr"],
                n=value["n"],
                confidence=confidence,
                confidence_note=note,
            )
        per_side[side] = stats

    symmetry = {
        key[: -len(SYMMETRY_SUFFIX)]: value
        for key, value in summary.items()
        if key.endswith(SYMMETRY_SUFFIX)
    }

    return Metrics(
        overall=OverallMetrics(
            cadence_spm=summary.get("cadence_spm"),
            symmetry_index_pct=symmetry,
            foot_strike_pattern=summary.get("foot_strike_pattern_approx"),
        ),
        per_side=per_side,
    )


def build_diagnostics(summary: Dict[str, Any]) -> Diagnostics:
    counts = {}
    for side, raw in (summary.get("per_side") or {}).items():
        counts[side] = int(((raw or {}).get("stride_time_s") or {}).get("n", 0))

    return Diagnostics(
        stride_counts=counts,
        knee_valid_stride_fraction=summary.get("knee_valid_stride_fraction"),
        nonpositive_flight_fraction=summary.get("nonpositive_flight_fraction"),
    )


# --------------------------------------------------
# PUBLIC API USED BY THE ROUTES
# --------------------------------------------------

def begin_analysis(video_id: str) -> Path:
    """Mark a job as processing and return the video path to analyse.

    The existence check and the state change happen under one lock, so two
    quick requests cannot both start the same job.
    """
    video_id = normalize_video_id(video_id)
    path = video_path(video_id)

    if not path.exists():
        raise VideoNotFound()

    with _lock:
        existing = _jobs.get(video_id)
        if existing and existing.status == JobStatus.processing:
            raise AnalysisInProgress()
        _jobs[video_id] = Job(status=JobStatus.processing)

    return path


def run_analysis(video_id: str, path: Path) -> None:
    """Run the pipeline and record the outcome. Executed as a background task."""
    try:
        result = analyze_video(path)

        if result["error"]:
            job = Job(status=JobStatus.failed, error=result["error"])
        else:
            job = Job(
                status=JobStatus.completed,
                analysis_status=AnalysisQuality(result["status"]),
                summary=result["summary"],
            )

    except Exception as exc:
        job = Job(status=JobStatus.failed, error=f"{type(exc).__name__}: {exc}")

    with _lock:
        _jobs[video_id] = job


def get_job(video_id: str) -> Job:
    """Current job, falling back to results already saved on disk."""
    video_id = normalize_video_id(video_id)

    with _lock:
        job = _jobs.get(video_id)

    if job is None:
        job = _load_job_from_disk(video_id)

    if job is None:
        raise AnalysisNotFound()

    return job


def build_results(video_id: str, job: Job) -> ResultsResponse:
    """Translate a Job into the public results payload.

    Only fields that apply to the job's state are set; the route serialises
    with response_model_exclude_unset so the rest are left out of the JSON.
    Metrics are withheld when the analysis is low quality, so a client cannot
    accidentally show numbers the pipeline does not stand behind.
    """
    video_id = normalize_video_id(video_id)
    base = {"video_id": video_id, "status": job.status}

    if job.status == JobStatus.processing:
        return ResultsResponse(**base)

    if job.status == JobStatus.failed:
        return ResultsResponse(**base, error=job.error)

    summary = job.summary or {}

    response = ResultsResponse(
        **base,
        analysis_status=job.analysis_status,
        quality=build_quality(summary),
        warnings=summary.get("warnings") or [],
        analysis=build_analysis_info(summary),
        diagnostics=build_diagnostics(summary),
        files=available_files(video_id),
    )

    if job.analysis_status == AnalysisQuality.ok:
        response.metrics = build_metrics(summary)

    return response


# --------------------------------------------------
# EXPORT
# --------------------------------------------------

EXPORT_MEDIA_TYPES = {
    ReportFormat.pdf: "application/pdf",
    ReportFormat.csv: "text/csv; charset=utf-8",
}


def export_report(video_id: str, fmt: ReportFormat) -> Tuple[bytes, str, str]:
    """Build a downloadable report. Returns (content, media_type, filename).

    Built from build_results(), so the quality gate applies automatically:
    there is no report (and no CSV) when the metrics were withheld.
    """
    video_id = normalize_video_id(video_id)
    job = get_job(video_id)

    if job.status == JobStatus.processing:
        raise AnalysisInProgress("Analysis is still processing.")
    if job.status == JobStatus.failed:
        raise ReportUnavailable("The analysis failed, so there is no report.")

    results = build_results(video_id, job)
    if results.metrics is None:
        raise ReportUnavailable(
            "This analysis was low quality, so no report is available. "
            "Try a clip with better framing and lighting."
        )

    if fmt == ReportFormat.csv:
        content = report_export.build_csv(results)
    else:
        charts = {
            name: OUTPUT_DIR / f"{video_id}{ARTEFACT_SUFFIXES[name]}"
            for name in ("cycles_plot", "debug_plot")
        }
        content = report_export.build_pdf(results, video_id, charts)

    filename = f"formback-report-{video_id[:8]}.{fmt.value}"
    return content, EXPORT_MEDIA_TYPES[fmt], filename


# --------------------------------------------------
# COMMAND LINE
# --------------------------------------------------

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

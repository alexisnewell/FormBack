"""
Analysis job logic, kept out of the route handlers.

Routes call into this module and never touch the job store directly. When jobs
move to a database later, only this file has to change.

Errors are raised as ``AnalysisError`` subclasses. ``api/main.py`` turns them
into HTTP responses, so the routes stay free of try/except noise.
"""

import json
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from api.schema.video import AnalysisQuality, JobStatus, ResultsResponse
from api.services import results_formatter as fmt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "uploads"
# Must match OUTPUT_DIR in src/pipeline.py.
OUTPUT_DIR = PROJECT_ROOT / "output"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Artefact name -> file name suffix written by src/pipeline.py
ARTEFACT_SUFFIXES = {
    "events": "_events.csv",
    "strides": "_strides.csv",
    "summary": "_summary.json",
    "debug_plot": "_debug.png",
    "cycles_plot": "_cycles.png",
}


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


# --------------------------------------------------
# HELPERS
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
    return UPLOAD_DIR / f"{normalize_video_id(video_id)}.mp4"


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


def available_files(video_id: str) -> List[str]:
    return [
        name
        for name, suffix in ARTEFACT_SUFFIXES.items()
        if (OUTPUT_DIR / f"{video_id}{suffix}").exists()
    ]


def _classify(summary: Dict[str, Any]) -> AnalysisQuality:
    """Re-apply the pipeline's quality gate to a summary read from disk."""
    from src.pipeline import MIN_QUALITY, MIN_VALID_STRIDES  # lazy

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


def _load_job_from_disk(video_id: str) -> Optional[Job]:
    """Rebuild a finished job from its summary file (survives restarts)."""
    summary_file = OUTPUT_DIR / f"{video_id}_summary.json"

    if not summary_file.exists():
        return None

    try:
        summary = json.loads(summary_file.read_text())
        return Job(
            status=JobStatus.completed,
            analysis_status=_classify(summary),
            summary=summary,
        )
    except Exception:
        return None


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
        from src.pipeline import analyze_video  # heavy import, kept lazy

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
        quality=fmt.build_quality(summary),
        warnings=summary.get("warnings") or [],
        analysis=fmt.build_analysis(summary),
        diagnostics=fmt.build_diagnostics(summary),
        files=available_files(video_id),
    )

    if job.analysis_status == AnalysisQuality.ok:
        response.metrics = fmt.build_metrics(summary)

    return response

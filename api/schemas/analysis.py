"""
Response models for the video / analysis endpoints.

These define the contract the frontend relies on. Change them deliberately.

Job lifecycle (``JobStatus``):
    processing -> completed | failed

A *completed* job carries a quality verdict (``AnalysisQuality``):
    ok           metrics are trustworthy and returned in ``metrics``
    low_quality  the video was analysed but the numbers are not reliable, so
                 ``metrics`` is omitted; show ``quality``, ``warnings`` and
                 ``diagnostics`` and ask for a better clip instead

The results endpoint omits fields that do not apply to the current state
(it never sends ``"error": null`` or an empty ``metrics`` while processing).
A field that is present with a null value means "measured, but no value".
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    processing = "processing"
    completed = "completed"
    failed = "failed"


class AnalysisQuality(str, Enum):
    ok = "ok"
    low_quality = "low_quality"


class Confidence(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"
    unvalidated = "unvalidated"


class ErrorResponse(BaseModel):
    """Body of every non-2xx response."""

    detail: str


# --------------------------------------------------
# UPLOAD / JOB CONTROL
# --------------------------------------------------

class UploadResponse(BaseModel):
    video_id: str = Field(description="UUID used in every later request.")
    filename: Optional[str] = Field(description="Original name of the uploaded file.")
    status: str = Field(default="uploaded")
    size_bytes: int
    fps: float = Field(description="Frame rate reported by the container.")
    frame_count: int
    duration_s: float


class AnalysisStartResponse(BaseModel):
    video_id: str
    status: JobStatus
    message: str


class StatusResponse(BaseModel):
    video_id: str
    status: JobStatus
    error: Optional[str] = Field(
        default=None, description="Set only when status is 'failed'."
    )


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

class QualityInfo(BaseModel):
    score: float = Field(description="0 to 1; below 0.5 is flagged low_quality.")
    label: str
    landmark_coverage: Optional[float] = None
    valid_stride_fraction: Optional[float] = None
    stride_time_cv: Optional[float] = None
    frames: Optional[int] = None


class AnalysisInfo(BaseModel):
    """How the video was analysed."""

    fps: Optional[float] = None
    direction: Optional[str] = Field(
        default=None, description="Direction the runner faces: 'left' or 'right'."
    )
    estimated_cycle_period_s: Optional[float] = None
    window_s: Optional[List[float]] = Field(
        default=None, description="[start, end] in seconds if a window was used."
    )
    toe_off_offset_frames: Optional[float] = None


class MetricStat(BaseModel):
    """One metric for one leg, with everything needed to display it."""

    label: str = Field(description="Human-readable name.")
    unit: Optional[str] = Field(default=None, description="e.g. 's', 'deg', '% leg'.")
    median: float
    iqr: float = Field(description="Interquartile range; spread across strides.")
    n: int = Field(description="Number of strides this value is based on.")
    confidence: Optional[Confidence] = None
    confidence_note: Optional[str] = Field(
        default=None, description="Why confidence is not 'high', when the pipeline says."
    )


class OverallMetrics(BaseModel):
    cadence_spm: Optional[float] = None
    symmetry_index_pct: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="Left/right asymmetry per metric, keyed like per_side "
        "metrics (plus 'step_time'). 0 is perfectly symmetric.",
    )
    foot_strike_pattern: Optional[str] = Field(
        default=None,
        description="'rearfoot', 'midfoot' or 'forefoot'; null when too "
        "variable to classify.",
    )


class Metrics(BaseModel):
    overall: OverallMetrics
    per_side: Dict[str, Dict[str, MetricStat]] = Field(
        description="side ('left'/'right') -> metric key -> stat. A metric "
        "with no usable strides is omitted."
    )


class Diagnostics(BaseModel):
    """Data-sufficiency numbers that explain how far to trust the metrics."""

    stride_counts: Dict[str, int] = Field(
        default_factory=dict, description="Valid strides per side."
    )
    knee_valid_stride_fraction: Optional[float] = None
    nonpositive_flight_fraction: Optional[float] = None


class ResultsResponse(BaseModel):
    """
    Fields present, by job status:

    processing  video_id, status                          (HTTP 202; poll again)
    failed      video_id, status, error
    completed   video_id, status, analysis_status, quality, warnings,
                analysis, diagnostics, files, and ``metrics`` only when
                analysis_status is 'ok'
    """

    video_id: str
    status: JobStatus
    analysis_status: Optional[AnalysisQuality] = None
    quality: Optional[QualityInfo] = None
    warnings: List[str] = Field(default_factory=list)
    analysis: Optional[AnalysisInfo] = None
    metrics: Optional[Metrics] = None
    diagnostics: Optional[Diagnostics] = None
    files: List[str] = Field(
        default_factory=list,
        description="Names of generated artefacts available for this video "
        "(events, strides, summary, debug_plot, cycles_plot).",
    )
    error: Optional[str] = None

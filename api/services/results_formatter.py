"""
Turns the raw summary dict from src/gait_analysis.py into the clean models in
api/schema/video.py.

Pure functions, no I/O. If the pipeline's summary format changes, this is the
only file that needs to follow.
"""

from typing import Any, Dict, Optional, Tuple

from api.schema.video import (
    AnalysisInfo,
    Confidence,
    Diagnostics,
    MetricStat,
    Metrics,
    OverallMetrics,
    QualityInfo,
)

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


def build_analysis(summary: Dict[str, Any]) -> AnalysisInfo:
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

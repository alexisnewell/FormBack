"""
Gait events: landmarks -> conditioned signals -> initial contact / toe-off.

Stage 2 of the pipeline (stage 1 is pose_detection.py, stage 3 is
gait_analysis.py).

1. Load MediaPipe landmarks, convert to pixels (no aspect distortion).
2. Condition each track: mask low visibility, fill short gaps only,
   zero-phase low-pass filter each continuous run.
3. Derive facing direction from toe-minus-heel (works with a panning camera).
4. Build one signal per leg: ankle x relative to pelvis x, in leg lengths.
   Because it is relative to the pelvis, camera pan and runner speed cancel out.
5. Initial contact  = forward extreme of that signal,
   Toe-off          = backward extreme (Zeni et al. style), with sub-frame
   parabolic refinement.

Entry point: ``detect_gait_events``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

# All-NaN slices are expected (occluded landmarks) and handled explicitly.
warnings.filterwarnings("ignore", category=RuntimeWarning)

# MediaPipe pose landmark ids
LM = {
    "nose": 0,
    "l_sh": 11, "r_sh": 12,
    "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26,
    "l_ank": 27, "r_ank": 28,
    "l_heel": 29, "r_heel": 30,
    "l_toe": 31, "r_toe": 32,
}
SIDES = {"left": "l", "right": "r"}

MIN_EXCURSION = 0.15            # min peak prominence, in leg lengths
DEFAULT_PERIOD_S = 0.7          # used when the cycle period cannot be estimated


# ----------------------------------------------------------------------
# Loading, conditioning, geometry and event detection
# ----------------------------------------------------------------------

def load_landmarks(path):
    df = pd.read_csv(path)
    required = {"frame", "timestamp_ms", "landmark", "x", "y", "visibility"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    for col in ("frame", "timestamp_ms", "landmark", "x", "y", "visibility"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df["visibility"].isna().all():
        df["visibility"] = 1.0
    return df.dropna(subset=["frame", "landmark"])


def estimate_fps(df):
    obs = (df[["frame", "timestamp_ms"]]
           .drop_duplicates("frame").sort_values("frame"))
    if len(obs) < 2:
        raise ValueError("Not enough frames to estimate FPS.")
    frames = obs["frame"].iloc[-1] - obs["frame"].iloc[0]
    seconds = (obs["timestamp_ms"].iloc[-1] - obs["timestamp_ms"].iloc[0]) / 1000
    if seconds <= 0:
        raise ValueError("Timestamps are not increasing.")
    return float(frames / seconds)


def fill_short_gaps(v, max_gap):
    """Linearly fill NaN runs of <= max_gap that have valid neighbours."""
    v = v.copy()
    n = len(v)
    isnan = np.isnan(v)
    i = 0
    while i < n:
        if isnan[i]:
            j = i
            while j < n and isnan[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= max_gap:
                v[i:j] = np.interp(np.arange(i, j), [i - 1, j], [v[i - 1], v[j]])
            i = j
        else:
            i += 1
    return v


def condition(v, fps, max_gap, cutoff_hz):
    """Fill short gaps, then low-pass each continuous run (zero phase)."""
    v = fill_short_gaps(v, max_gap)
    out = np.full_like(v, np.nan)
    b, a = butter(2, cutoff_hz / (fps / 2))
    padlen = 3 * max(len(a), len(b))
    idx = np.flatnonzero(np.isfinite(v))
    if idx.size == 0:
        return out
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        if len(run) > padlen:
            out[run] = filtfilt(b, a, v[run])
    return out


def build_tracks(df, width, height, fps, vis_thr=0.5, max_gap=4):
    """Return {name: (n_frames, 2) array} of conditioned pixel coordinates."""
    n = int(df["frame"].max()) + 1
    cutoff = min(10.0, 0.4 * fps)
    tracks = {}
    for name, idx in LM.items():
        sub = (df[df["landmark"] == idx]
               .drop_duplicates("frame", keep="last")
               .set_index("frame").reindex(range(n)))
        vis = sub["visibility"].fillna(0).to_numpy(float)
        cols = []
        for axis, scale in (("x", width), ("y", height)):
            v = sub[axis].to_numpy(float) * scale
            v[vis < vis_thr] = np.nan
            cols.append(condition(v, fps, max_gap, cutoff))
        tracks[name] = np.column_stack(cols)
    return tracks


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------

def included_angle(a, b, c):
    """Angle at b (degrees) for arrays of points shaped (n, 2)."""
    ba, bc = a - b, c - b
    den = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1)
    cos = np.clip((ba * bc).sum(-1) / den, -1, 1)
    return np.degrees(np.arccos(cos))


def leg_length(T):
    lengths = []
    for p in SIDES.values():
        lengths.append(np.linalg.norm(T[f"{p}_hip"] - T[f"{p}_knee"], axis=1)
                       + np.linalg.norm(T[f"{p}_knee"] - T[f"{p}_ank"], axis=1))
    L = np.nanmedian(np.concatenate(lengths))
    if not np.isfinite(L) or L <= 0:
        raise ValueError("Could not estimate leg length (legs not visible).")
    return float(L)


def facing_sign(T):
    """+1 if the runner faces right in the image, -1 if left (toe ahead of heel)."""
    dx = np.concatenate([T["l_toe"][:, 0] - T["l_heel"][:, 0],
                         T["r_toe"][:, 0] - T["r_heel"][:, 0]])
    m = np.nanmedian(dx)
    if not np.isfinite(m) or m == 0:
        raise ValueError("Could not determine running direction.")
    return float(np.sign(m))


# ----------------------------------------------------------------------
# Event detection
# ----------------------------------------------------------------------

def estimate_period(sig, fps, lo=0.4, hi=1.4):
    """Single-leg cycle period (s) from autocorrelation; NaN if unclear."""
    x = np.nan_to_num(sig - np.nanmean(sig))
    if not np.any(x):
        return np.nan
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac = ac / ac[0]
    lags = np.arange(len(ac)) / fps
    m = (lags >= lo) & (lags <= hi)
    return float(lags[m][np.argmax(ac[m])]) if m.any() else np.nan


def refine_peak(y, p):
    """Sub-frame peak position via a parabola through p-1, p, p+1."""
    a, b, c = y[p - 1], y[p], y[p + 1]
    den = a - 2 * b + c
    return float(p) if den == 0 else float(p + 0.5 * (a - c) / den)


def detect_events(rel, period, fps):
    """Return (initial_contact_frames, toe_off_frames) as float arrays."""
    if not np.isfinite(rel).sum() > 10:
        return np.array([]), np.array([])
    x = np.where(np.isfinite(rel), rel, np.nanmean(rel))
    dist = max(2, int(0.7 * period * fps))

    def peaks(sig):
        idx, _ = find_peaks(sig, distance=dist, prominence=MIN_EXCURSION)
        keep = [p for p in idx
                if 0 < p < len(rel) - 1 and np.isfinite(rel[p - 1:p + 2]).all()]
        return np.array([refine_peak(sig, p) for p in keep])

    return peaks(x), peaks(-x)


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

@dataclass
class GaitSignals:
    """Everything stage 3 needs, produced once per clip."""

    fps: float
    tracks: Dict[str, np.ndarray]      # landmark name -> (n_frames, 2) pixels
    pelvis: np.ndarray                 # (n_frames, 2)
    shoulder: np.ndarray               # (n_frames, 2)
    leg: float                         # leg length in pixels
    direction: float                   # +1 facing right, -1 facing left
    sig: Dict[str, Dict[str, np.ndarray]]   # side -> {"rel", "knee_flexion"}
    period: float                      # single-leg cycle period, seconds
    events: Dict[str, Tuple[np.ndarray, np.ndarray]]   # side -> (ic, toe_off)


def detect_gait_events(landmark_csv, width, height, fps=None, visibility=0.5,
                       toe_off_offset_frames=0.0) -> GaitSignals:
    """
    Landmarks CSV -> signals and gait events (frame numbers, sub-frame).

    toe_off_offset_frames: shift every toe-off by this many frames
        (negative = earlier). Calibrate it from validate_events.py bias.
    """
    df = load_landmarks(landmark_csv)
    fps = fps or estimate_fps(df)
    T = build_tracks(df, width, height, fps, visibility)

    pelvis = (T["l_hip"] + T["r_hip"]) / 2
    shoulder = (T["l_sh"] + T["r_sh"]) / 2
    leg = leg_length(T)
    direction = facing_sign(T)

    sig = {}
    for side, p in SIDES.items():
        sig[side] = {
            "rel": direction * (T[f"{p}_ank"][:, 0] - pelvis[:, 0]) / leg,
            "knee_flexion": 180 - included_angle(
                T[f"{p}_hip"], T[f"{p}_knee"], T[f"{p}_ank"]),
        }

    periods = [estimate_period(sig[s]["rel"], fps) for s in sig]
    period = (float(np.nanmedian(periods)) if np.isfinite(periods).any()
              else DEFAULT_PERIOD_S)

    events = {side: detect_events(sig[side]["rel"], period, fps) for side in sig}
    if toe_off_offset_frames:
        events = {s: (ic, to + toe_off_offset_frames)
                  for s, (ic, to) in events.items()}

    return GaitSignals(fps=fps, tracks=T, pelvis=pelvis, shoulder=shoulder,
                       leg=leg, direction=direction, sig=sig, period=period,
                       events=events)


def events_to_frame(events, fps) -> pd.DataFrame:
    """Flatten {side: (ic, toe_off)} into one time-sorted table."""
    rows = []
    for side, (ic, to) in events.items():
        for kind, frames in (("initial_contact", ic), ("toe_off", to)):
            for f in frames:
                rows.append({"side": side, "event": kind,
                             "frame": int(round(f)),
                             "frame_exact": round(float(f), 2),
                             "time_seconds": round(float(f) / fps, 4)})
    return pd.DataFrame(
        rows, columns=["side", "event", "frame", "frame_exact", "time_seconds"]
    ).sort_values("time_seconds").reset_index(drop=True)

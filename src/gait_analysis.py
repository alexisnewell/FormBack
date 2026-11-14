"""
FormBack gait analysis: one pelvis-referenced pipeline.

Replaces gait_events.py, gait_processor.py and gait_classifier.py.

Pipeline
--------
1. Load MediaPipe landmarks, convert to pixels (no aspect distortion).
2. Condition each track: mask low visibility, fill short gaps only,
   zero-phase low-pass filter each continuous run.
3. Derive facing direction from toe-minus-heel (works with a panning camera).
4. Build one signal per leg: ankle x relative to pelvis x, in leg lengths.
   Because it is relative to the pelvis, camera pan and runner speed cancel out.
5. Initial contact  = forward extreme of that signal,
   Toe-off          = backward extreme (Zeni et al. style), with sub-frame
   parabolic refinement.
6. Validate each stride, compute metrics, score the clip's quality.

Usage
-----
    python -m src.gait_analysis output/video1_running_landmarks.csv \
        --video output/video1_running_annotated.mp4

Outputs (in --output-dir):
    <stem>_events.csv    side, event, frame, frame_exact, time_seconds
                         (compatible with gait_inspection_sequences.py)
    <stem>_strides.csv   one row per stride with all metrics
    <stem>_summary.json  medians/IQR per side, symmetry, cadence, quality
    <stem>_debug.png     signals with detected events for visual validation
    <stem>_cycles.png    all valid strides overlaid, normalised 0-100% of stride
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

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

# Plausibility limits for a running stride (one leg, contact to next contact)
DUTY_RANGE = (0.15, 0.60)       # contact_time / stride_time
STRIDE_TOLERANCE = 0.25         # allowed deviation from the side's median
MIN_EXCURSION = 0.15            # min peak prominence, in leg lengths
MAX_STANCE_KNEE_DEG = 70.0      # stance knee flexion above this is a tracking glitch
MAX_KNEE_SPEED_DEG_S = 1000.0   # faster knee change in early stance = glitch
KNEE_VALID_WARN = 0.70          # warn if fewer strides than this have clean knees
FOOT_STRIKE_MAX_IQR = 8.0       # deg; above this, foot strike is not classified
NONPOSITIVE_FLIGHT_WARN = 0.10  # warn if more strides than this overlap

# How far to trust each group of metrics (shown in the summary JSON)
METRIC_CONFIDENCE = {
    "cadence, stride time, step symmetry, trunk lean": "high",
    "contact time, flight time, duty factor":
        "unvalidated: depends on toe-off timing; check with validate_events.py",
    "knee flexion (contact, peak stance)":
        "medium: 2D view, far-leg occlusion, sensitive to contact timing",
    "ankle ahead of pelvis":
        "medium: changes with speed; use a steady section (--window)",
    "vertical oscillation":
        "low: landmark jitter and camera bob inflate it",
    "foot strike angle": "low: heel/toe landmarks are noisy",
}


# ----------------------------------------------------------------------
# Loading and conditioning
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
# Strides and metrics
# ----------------------------------------------------------------------

def foot_strike_class(angle):
    if not np.isfinite(angle):
        return None
    return "rearfoot" if angle > 8 else ("forefoot" if angle < 0 else "midfoot")


def build_strides(T, sig, events, pelvis, shoulder, leg, direction, fps):
    n = len(pelvis)
    rows = []
    for side, p in SIDES.items():
        ic, to = events[side]
        other = "right" if side == "left" else "left"
        ic_other = events[other][0]

        lean = np.degrees(np.arctan2(
            direction * (shoulder[:, 0] - pelvis[:, 0]),
            pelvis[:, 1] - shoulder[:, 1]))

        for i in range(len(ic) - 1):
            nxt = to[(to > ic[i]) & (to < ic[i + 1])]
            if nxt.size == 0:
                continue
            t_ic, t_to, t_next = ic[i] / fps, nxt[0] / fps, ic[i + 1] / fps
            stride_t, contact_t = t_next - t_ic, t_to - t_ic

            # The opposite leg's contact inside this stride. Flight is the gap
            # from this toe-off to that contact; negative means the feet
            # overlapped, which is not possible in running (toe-off detected late).
            mid = ic_other[(ic_other > ic[i]) & (ic_other < ic[i + 1])]
            flight = (mid[0] - nxt[0]) / fps if mid.size == 1 else np.nan

            f0, f1 = int(round(ic[i])), int(round(ic[i + 1]))
            ft = min(int(round(nxt[0])), n - 1)
            f1 = min(f1, n - 1)

            knee = sig[side]["knee_flexion"]
            # Peak loading is early/mid stance. The tail of the detected contact
            # can include early swing, so only use the first 60% of it.
            f_mid = min(f0 + max(2, int(round(0.6 * (ft - f0)))), n - 1)
            stance_knee = knee[f0:f_mid + 1]
            # Reject strides whose knee track jumps (e.g. the other leg crossing
            # in front and landmarks swapping): impossible depth or speed.
            speed = np.abs(np.diff(stance_knee)) * fps
            knee_ok = bool(
                stance_knee.size > 2
                and np.isfinite(stance_knee).all()
                and np.nanmax(stance_knee) <= MAX_STANCE_KNEE_DEG
                and np.nanmax(speed) <= MAX_KNEE_SPEED_DEG_S)

            heel = np.nanmean(T[f"{p}_heel"][f0:f0 + 2], axis=0)
            toe = np.nanmean(T[f"{p}_toe"][f0:f0 + 2], axis=0)
            # Positive = toe higher than heel at contact (heel-strike like).
            fsa = float(np.degrees(np.arctan2(heel[1] - toe[1],
                                              abs(toe[0] - heel[0]))))

            py = pelvis[f0:f1 + 1, 1]
            if py.size > 2 and np.isfinite(py).all():
                py = py - np.linspace(py[0], py[-1], py.size)
                osc = float((py.max() - py.min()) / leg * 100)
            else:
                osc = np.nan

            rows.append({
                "side": side,
                "ic_frame": ic[i],
                "toe_off_frame": nxt[0],
                "next_ic_frame": ic[i + 1],
                "stride_time_s": stride_t,
                "contact_time_s": contact_t,
                "flight_raw_s": flight,
                "flight_time_s": flight if flight > 0 else np.nan,
                "duty_factor": contact_t / stride_t,
                "knee_ok": knee_ok,
                "knee_flexion_at_contact_deg": float(knee[f0]) if knee_ok else np.nan,
                "peak_stance_knee_flexion_deg":
                    float(np.nanmax(stance_knee)) if knee_ok else np.nan,
                "ankle_ahead_of_pelvis_leg_lengths": float(sig[side]["rel"][f0]),
                "mean_trunk_lean_deg": float(np.nanmean(lean[f0:f1 + 1])),
                "vertical_oscillation_pct_leg": osc,
                "foot_strike_angle_deg": fsa,
            })

    strides = pd.DataFrame(rows)
    if strides.empty:
        return strides

    median_stride = strides.groupby("side")["stride_time_s"].transform("median")
    strides["valid"] = (
        strides["duty_factor"].between(*DUTY_RANGE)
        & ((strides["stride_time_s"] - median_stride).abs() / median_stride
           <= STRIDE_TOLERANCE)
    )
    return strides


def robust(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    q1, q3 = np.percentile(v, [25, 75])
    return {"median": round(float(np.median(v)), 3),
            "iqr": round(float(q3 - q1), 3), "n": int(v.size)}


def symmetry_index(a, b):
    if a is None or b is None or (a + b) == 0:
        return None
    return round(float(100 * abs(a - b) / (0.5 * (a + b))), 1)


METRICS = [
    "stride_time_s", "contact_time_s", "flight_time_s", "duty_factor",
    "knee_flexion_at_contact_deg", "peak_stance_knee_flexion_deg",
    "ankle_ahead_of_pelvis_leg_lengths", "mean_trunk_lean_deg",
    "vertical_oscillation_pct_leg", "foot_strike_angle_deg",
]


def summarize(strides, events_df, tracks, fps, n_frames):
    warnings_out = []
    summary = {"fps": round(fps, 2), "per_side": {}, "warnings": warnings_out}

    # Cadence and step symmetry from alternating contacts of both legs
    ics = (events_df[events_df["event"] == "initial_contact"]
           .sort_values("time_seconds"))
    t, s = ics["time_seconds"].to_numpy(), ics["side"].to_numpy()
    steps = {"left": [], "right": []}
    for i in range(len(t) - 1):
        if s[i] != s[i + 1]:
            steps[s[i]].append(t[i + 1] - t[i])
    all_steps = steps["left"] + steps["right"]
    summary["cadence_spm"] = (round(60 / float(np.median(all_steps)), 1)
                              if all_steps else None)
    summary["step_time_symmetry_index_pct"] = symmetry_index(
        np.median(steps["left"]) if steps["left"] else None,
        np.median(steps["right"]) if steps["right"] else None)

    valid = strides[strides["valid"]] if not strides.empty else strides
    for side in SIDES:
        sub = valid[valid["side"] == side] if not valid.empty else valid
        summary["per_side"][side] = {m: robust(sub[m]) for m in METRICS} if len(sub) else {}
        if len(sub) < 3:
            warnings_out.append(f"Only {len(sub)} valid {side}-leg strides; "
                                "results for this leg are unreliable.")

    ps = summary["per_side"]
    for m in ("contact_time_s", "knee_flexion_at_contact_deg"):
        a = (ps["left"].get(m) or {}).get("median")
        b = (ps["right"].get(m) or {}).get("median")
        summary[f"{m}_symmetry_index_pct"] = symmetry_index(a, b)

    if len(valid):
        kfrac = float(valid["knee_ok"].mean())
        summary["knee_valid_stride_fraction"] = round(kfrac, 2)
        if kfrac < KNEE_VALID_WARN:
            warnings_out.append(
                f"Knee tracking glitches in {1 - kfrac:.0%} of strides; knee metrics "
                "use only the clean ones. A heavier pose model may help.")
        fsa = valid["foot_strike_angle_deg"].dropna()
        spread = (float(np.subtract(*np.percentile(fsa, [75, 25])))
                  if len(fsa) else np.inf)
        if spread <= FOOT_STRIKE_MAX_IQR:
            summary["foot_strike_pattern_approx"] = foot_strike_class(fsa.median())
        else:
            summary["foot_strike_pattern_approx"] = None
            warnings_out.append(
                f"Foot-strike angle too variable (IQR {spread:.0f} deg) to classify; "
                "not reported.")

        raw = valid["flight_raw_s"].dropna()
        bad = float((raw <= 0).mean()) if len(raw) else 0.0
        summary["nonpositive_flight_fraction"] = round(bad, 2)
        if bad > NONPOSITIVE_FLIGHT_WARN:
            warnings_out.append(
                f"{bad:.0%} of strides show overlapping feet (negative flight time). "
                "Toe-off is probably detected late, so contact time is overestimated "
                "and flight time underestimated. Validate with labels and use "
                "--toe-off-offset.")
    summary["metric_confidence"] = METRIC_CONFIDENCE

    # Quality score
    key = ["l_hip", "r_hip", "l_knee", "r_knee", "l_ank", "r_ank", "l_sh", "r_sh"]
    coverage = float(np.mean([np.isfinite(tracks[k][:, 0]).mean() for k in key]))
    valid_frac = float(strides["valid"].mean()) if not strides.empty else 0.0
    cv = (float(valid["stride_time_s"].std() / valid["stride_time_s"].mean())
          if len(valid) > 2 else 1.0)
    regularity = float(np.clip(1 - cv / 0.15, 0, 1))
    score = 0.3 * coverage + 0.4 * valid_frac + 0.3 * regularity
    summary["quality"] = {
        "score": round(score, 2),
        "label": "good" if score >= 0.75 else "fair" if score >= 0.5 else "poor",
        "landmark_coverage": round(coverage, 2),
        "valid_stride_fraction": round(valid_frac, 2),
        "stride_time_cv": round(cv, 3),
        "frames": n_frames,
    }

    resolution_ms = 1000 / fps
    if fps < 50:
        warnings_out.append(
            f"{fps:.0f} fps gives about +/-{resolution_ms:.0f} ms timing error. "
            "Contact and flight times are coarse; record at 60+ fps if possible.")
    if cv > 0.10 and len(valid) > 2:
        warnings_out.append("Stride timing is irregular; check the debug plot.")
    return summary


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def analyze(landmark_csv, width, height, fps=None, visibility=0.5,
            toe_off_offset_frames=0.0, window=None):
    """
    toe_off_offset_frames: shift every toe-off by this many frames
        (negative = earlier). Calibrate it from validate_events.py bias.
    window: (start_s, end_s) to summarise only a steady section of the clip.
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
    period = float(np.nanmedian(periods)) if np.isfinite(periods).any() else 0.7

    events = {side: detect_events(sig[side]["rel"], period, fps) for side in sig}
    if toe_off_offset_frames:
        events = {s: (ic, to + toe_off_offset_frames)
                  for s, (ic, to) in events.items()}

    rows = []
    for side, (ic, to) in events.items():
        for kind, frames in (("initial_contact", ic), ("toe_off", to)):
            for f in frames:
                rows.append({"side": side, "event": kind,
                             "frame": int(round(f)), "frame_exact": round(float(f), 2),
                             "time_seconds": round(float(f) / fps, 4)})
    events_df = pd.DataFrame(
        rows, columns=["side", "event", "frame", "frame_exact", "time_seconds"]
    ).sort_values("time_seconds").reset_index(drop=True)

    strides = build_strides(T, sig, events, pelvis, shoulder, leg, direction, fps)
    if window and not strides.empty:
        t0, t1 = window
        strides = strides[(strides["ic_frame"] / fps >= t0)
                          & (strides["next_ic_frame"] / fps <= t1)].reset_index(drop=True)
        events_df = events_df[(events_df["time_seconds"] >= t0)
                              & (events_df["time_seconds"] <= t1)].reset_index(drop=True)
    summary = summarize(strides, events_df, T, fps, len(pelvis))
    summary["direction"] = "right" if direction > 0 else "left"
    summary["estimated_cycle_period_s"] = round(period, 3)
    summary["toe_off_offset_frames"] = toe_off_offset_frames
    summary["window_s"] = list(window) if window else None
    return events_df, strides, summary, {"sig": sig, "events": events, "fps": fps}


def plot_debug(context, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fps = context["fps"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, side in zip(axes, SIDES):
        rel = context["sig"][side]["rel"]
        t = np.arange(len(rel)) / fps
        ic, to = context["events"][side]
        ax.plot(t, rel, lw=1.6, label="ankle x relative to pelvis")
        ax.scatter(ic / fps, np.interp(ic, np.arange(len(rel)), np.nan_to_num(rel)),
                   marker="o", s=70, label="initial contact", zorder=3)
        ax.scatter(to / fps, np.interp(to, np.arange(len(rel)), np.nan_to_num(rel)),
                   marker="x", s=70, label="toe-off", zorder=3)
        for c in ic:
            nxt = to[to > c]
            if nxt.size:
                ax.axvspan(c / fps, nxt[0] / fps, alpha=0.15)
        ax.set_title(f"{side.title()} leg")
        ax.set_ylabel("Leg lengths (forward +)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cycles(context, strides, path):
    """Overlay every valid stride, time-normalised 0-100%, with the median.

    Use this to judge whether the knee curve and the contact / toe-off timing
    look physiologically sensible and consistent from stride to stride.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if strides.empty:
        return
    valid = strides[strides["valid"]]
    grid = np.linspace(0, 1, 101)
    panels = [("rel", "Ankle x relative to pelvis (leg lengths)"),
              ("knee_flexion", "Knee flexion (deg)")]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for col, side in enumerate(SIDES):
        sub = valid[valid["side"] == side]
        for row, (key, label) in enumerate(panels):
            ax = axes[row, col]
            y = context["sig"][side][key]
            curves, rejected = [], []
            for _, r in sub.iterrows():
                a, b = int(round(r["ic_frame"])), int(round(r["next_ic_frame"]))
                seg = y[a:b + 1]
                if len(seg) < 5 or not np.isfinite(seg).all():
                    continue
                c = np.interp(grid, np.linspace(0, 1, len(seg)), seg)
                if key == "knee_flexion" and not r["knee_ok"]:
                    rejected.append(c)
                else:
                    curves.append(c)
            for c in curves:
                ax.plot(grid * 100, c, color="tab:blue", alpha=0.15, lw=1)
            for c in rejected:
                ax.plot(grid * 100, c, color="tab:red", alpha=0.25, lw=1)
            if rejected:
                ax.plot([], [], color="tab:red", alpha=0.6,
                        label=f"{len(rejected)} rejected (knee glitch)")
            if curves:
                ax.plot(grid * 100, np.median(curves, axis=0), color="black", lw=2.2,
                        label=f"median of {len(curves)} strides")
            if len(sub):
                ax.axvline(sub["duty_factor"].median() * 100, color="tab:orange",
                           ls="--", label="median toe-off")
            ax.axvline(0, color="tab:green", ls=":", label="initial contact")
            ax.set_title(f"{side.title()} leg")
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
            if row == 0:
                ax.legend(loc="upper right", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("% of stride (contact to next contact)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def video_dimensions(video):
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def main():
    ap = argparse.ArgumentParser(description="Running gait analysis from landmarks.")
    ap.add_argument("landmarks", help="<name>_landmarks.csv from video_processor")
    ap.add_argument("--video", help="Video used to read width/height")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--fps", type=float, help="Override FPS (default: from CSV)")
    ap.add_argument("--visibility", type=float, default=0.5)
    ap.add_argument("--toe-off-offset", type=float, default=0.0,
                    help="shift toe-offs by N frames (negative = earlier); "
                         "set from validate_events.py bias")
    ap.add_argument("--window", type=float, nargs=2, metavar=("START", "END"),
                    help="only summarise strides between these times (seconds)")
    ap.add_argument("--output-dir", default="output")
    args = ap.parse_args()

    if args.video:
        width, height = video_dimensions(args.video)
    elif args.width and args.height:
        width, height = args.width, args.height
    else:
        width, height = 720, 1280
        print("Warning: no --video or --width/--height; assuming 720x1280.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.landmarks).stem.replace("_landmarks", "")

    events_df, strides, summary, ctx = analyze(
        args.landmarks, width, height, args.fps, args.visibility,
        args.toe_off_offset, tuple(args.window) if args.window else None)

    events_df.to_csv(out / f"{stem}_events.csv", index=False)
    strides.to_csv(out / f"{stem}_strides.csv", index=False)
    (out / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2, default=float))
    plot_debug(ctx, out / f"{stem}_debug.png")
    plot_cycles(ctx, strides, out / f"{stem}_cycles.png")

    q = summary["quality"]
    print(f"FPS {summary['fps']} | facing {summary['direction']} | "
          f"cadence {summary['cadence_spm']} spm | "
          f"quality {q['label']} ({q['score']})")
    for w in summary["warnings"]:
        print("  !", w)
    print(f"Saved results to {out}/")


if __name__ == "__main__":
    main()
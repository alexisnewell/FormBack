"""
Gait analysis: gait events -> strides -> metrics -> quality score.

Stage 3 of the pipeline. Stage 2 (gait_events.py) finds the events; this
module validates each stride, computes the metrics, summarises them per leg
and scores the clip's quality. It also draws the two diagnostic plots.

Entry point: ``analyze``.

Outputs (written by report_service.py):
    <stem>_events.csv    side, event, frame, frame_exact, time_seconds
    <stem>_strides.csv   one row per stride with all metrics
    <stem>_summary.json  medians/IQR per side, symmetry, cadence, quality
    <stem>_debug.png     signals with detected events for visual validation
    <stem>_cycles.png    all valid strides overlaid, normalised 0-100% of stride
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from api.services.gait_events import (
    SIDES, detect_gait_events, events_to_frame,
)

# Plausibility limits for a running stride (one leg, contact to next contact)
DUTY_RANGE = (0.15, 0.60)       # contact_time / stride_time
STRIDE_TOLERANCE = 0.25         # allowed deviation from the side's median
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
    Landmarks CSV -> (events_df, strides, summary, plot_context).

    toe_off_offset_frames: shift every toe-off by this many frames
        (negative = earlier). Calibrate it from validate_events.py bias.
    window: (start_s, end_s) to summarise only a steady section of the clip.
    """
    g = detect_gait_events(landmark_csv, width, height, fps, visibility,
                           toe_off_offset_frames)
    fps = g.fps
    events_df = events_to_frame(g.events, fps)

    strides = build_strides(g.tracks, g.sig, g.events, g.pelvis, g.shoulder,
                            g.leg, g.direction, fps)
    if window and not strides.empty:
        t0, t1 = window
        strides = strides[(strides["ic_frame"] / fps >= t0)
                          & (strides["next_ic_frame"] / fps <= t1)].reset_index(drop=True)
        events_df = events_df[(events_df["time_seconds"] >= t0)
                              & (events_df["time_seconds"] <= t1)].reset_index(drop=True)

    summary = summarize(strides, events_df, g.tracks, fps, len(g.pelvis))
    summary["direction"] = "right" if g.direction > 0 else "left"
    summary["estimated_cycle_period_s"] = round(g.period, 3)
    summary["toe_off_offset_frames"] = toe_off_offset_frames
    summary["window_s"] = list(window) if window else None
    return events_df, strides, summary, {"sig": g.sig, "events": g.events, "fps": fps}


# ----------------------------------------------------------------------
# Diagnostic plots
# ----------------------------------------------------------------------

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

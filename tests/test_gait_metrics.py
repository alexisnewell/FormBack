"""Tests for api/services/gait_analysis.py: strides, metrics, quality, plots."""

import math

import numpy as np
import pandas as pd
import pytest

from api.services import gait_analysis as ga
from api.services import gait_events as ge
from tests.synthetic import HEIGHT, LEG_PX, WIDTH

FPS = 120
PERIOD = 0.7
DUTY = 0.4

# Timing is read off a smoothed signal, so allow about 2 frames.
TIME_TOL = 0.015


def analyze(landmarks, **kw):
    return ga.analyze(landmarks(**kw.pop("clip", {})), WIDTH, HEIGHT, **kw)


# ------------------------------------------------------------ small helpers

def test_robust_median_iqr_and_count():
    out = ga.robust([1, 2, 3, 4, 5])
    assert out == {"median": 3.0, "iqr": 2.0, "n": 5}


def test_robust_ignores_nan_and_inf():
    assert ga.robust([1, np.nan, 3, np.inf]) == {"median": 2.0, "iqr": 1.0, "n": 2}


def test_robust_empty_is_none():
    assert ga.robust([]) is None
    assert ga.robust([np.nan, np.nan]) is None


def test_symmetry_index():
    assert ga.symmetry_index(10, 10) == 0.0
    assert ga.symmetry_index(10, 12) == pytest.approx(18.2)
    assert ga.symmetry_index(12, 10) == ga.symmetry_index(10, 12)   # order-free


@pytest.mark.parametrize("a,b", [(None, 1), (1, None), (None, None), (5, -5)])
def test_symmetry_index_returns_none_when_undefined(a, b):
    assert ga.symmetry_index(a, b) is None


@pytest.mark.parametrize("angle,expected", [
    (15, "rearfoot"), (8.01, "rearfoot"), (8, "midfoot"), (3, "midfoot"),
    (0, "midfoot"), (-0.1, "forefoot"), (-12, "forefoot"),
    (float("nan"), None),
])
def test_foot_strike_class(angle, expected):
    assert ga.foot_strike_class(angle) == expected


# ------------------------------------------------------------------ strides

def test_strides_match_known_timing(landmarks):
    _, strides, _, _ = analyze(landmarks, clip=dict(period=PERIOD, duty=DUTY))

    assert set(strides["side"]) == {"left", "right"}
    assert strides["stride_time_s"].to_numpy() == pytest.approx(PERIOD, abs=0.003)
    assert strides["contact_time_s"].to_numpy() == pytest.approx(DUTY * PERIOD, abs=TIME_TOL)
    assert strides["flight_time_s"].dropna().to_numpy() == pytest.approx(
        (0.5 - DUTY) * PERIOD, abs=TIME_TOL)
    assert strides["duty_factor"].to_numpy() == pytest.approx(DUTY, abs=0.03)


def test_clean_clip_has_every_stride_valid(landmarks):
    _, strides, _, _ = analyze(landmarks)
    assert strides["valid"].all() and strides["knee_ok"].all()
    assert (strides["flight_raw_s"] > 0).all()
    counts = strides["side"].value_counts()
    assert counts["left"] >= 25 and counts["right"] >= 25


@pytest.mark.parametrize("facing", [1, -1])
def test_trunk_lean_is_direction_corrected(landmarks, facing):
    # shoulder sits 8 px ahead of the pelvis, 0.62 legs above it
    expected = math.degrees(math.atan2(8.0, 0.62 * LEG_PX))
    _, strides, _, _ = analyze(landmarks, clip=dict(facing=facing))
    assert strides["mean_trunk_lean_deg"].to_numpy() == pytest.approx(expected, abs=0.15)


def test_knee_and_foot_strike_on_clean_clip(landmarks):
    _, strides, summary, _ = analyze(landmarks)
    assert strides["knee_flexion_at_contact_deg"].between(5, 40).all()
    assert strides["foot_strike_angle_deg"].abs().max() < 1.0      # flat foot
    assert summary["foot_strike_pattern_approx"] == "midfoot"


def test_vertical_oscillation_is_small_but_measured(landmarks):
    _, strides, _, _ = analyze(landmarks)
    osc = strides["vertical_oscillation_pct_leg"]
    assert osc.notna().all() and osc.between(0.5, 10).all()


def test_stride_with_impossible_knee_is_rejected(landmarks):
    g = ge.detect_gait_events(landmarks(), WIDTH, HEIGHT)
    g.sig["left"]["knee_flexion"][:] = 120          # beyond MAX_STANCE_KNEE_DEG
    strides = ga.build_strides(g.tracks, g.sig, g.events, g.pelvis, g.shoulder,
                               g.leg, g.direction, g.fps)
    left, right = strides[strides.side == "left"], strides[strides.side == "right"]
    assert not left["knee_ok"].any()
    assert left["knee_flexion_at_contact_deg"].isna().all()
    assert right["knee_ok"].all()                   # other leg unaffected


def test_knee_spike_in_stance_is_rejected(landmarks):
    g = ge.detect_gait_events(landmarks(), WIDTH, HEIGHT)
    first_ic = int(round(g.events["left"][0][2]))
    g.sig["left"]["knee_flexion"][first_ic + 1] += 80   # one-frame jump
    strides = ga.build_strides(g.tracks, g.sig, g.events, g.pelvis, g.shoulder,
                               g.leg, g.direction, g.fps)
    left = strides[strides.side == "left"].reset_index(drop=True)
    assert (~left["knee_ok"]).sum() >= 1
    assert left["knee_ok"].sum() >= len(left) - 3


def test_overlong_contact_marks_strides_invalid(landmarks):
    _, strides, summary, _ = analyze(landmarks, clip=dict(duty=0.7))
    assert not strides["valid"].any()               # duty > DUTY_RANGE upper bound
    assert summary["quality"]["label"] == "poor"
    assert any("Only 0 valid" in w for w in summary["warnings"])


# ------------------------------------------------------------------ summary

def test_summary_headline_numbers(landmarks):
    _, _, s, _ = analyze(landmarks)
    assert s["cadence_spm"] == pytest.approx(60 / (PERIOD / 2), abs=0.5)
    assert s["step_time_symmetry_index_pct"] == pytest.approx(0, abs=0.5)
    assert s["contact_time_s_symmetry_index_pct"] == pytest.approx(0, abs=0.5)
    assert s["knee_flexion_at_contact_deg_symmetry_index_pct"] == pytest.approx(0, abs=1)
    assert s["direction"] == "right"
    assert s["estimated_cycle_period_s"] == pytest.approx(PERIOD, abs=0.01)
    assert s["fps"] == pytest.approx(120, abs=0.1)
    assert s["window_s"] is None and s["toe_off_offset_frames"] == 0.0


def test_summary_per_side_has_every_metric(landmarks):
    _, _, s, _ = analyze(landmarks)
    for side in ("left", "right"):
        assert set(s["per_side"][side]) == set(ga.METRICS)
        stat = s["per_side"][side]["stride_time_s"]
        assert stat["median"] == pytest.approx(PERIOD, abs=0.003)
        assert stat["n"] >= 25 and stat["iqr"] < 0.005


def test_clean_clip_scores_good_with_no_warnings(landmarks):
    _, _, s, _ = analyze(landmarks)
    q = s["quality"]
    assert q["label"] == "good" and q["score"] >= 0.95
    assert q["landmark_coverage"] == 1.0 and q["valid_stride_fraction"] == 1.0
    assert q["frames"] == 20 * FPS
    assert s["warnings"] == []


def test_metric_confidence_is_attached(landmarks):
    _, _, s, _ = analyze(landmarks)
    assert s["metric_confidence"] == ga.METRIC_CONFIDENCE


def test_low_fps_warns_about_timing_resolution(landmarks):
    _, _, s, _ = analyze(landmarks, clip=dict(fps=30))
    assert any("30 fps" in w for w in s["warnings"])


def test_late_toe_off_triggers_overlap_warning(landmarks):
    # duty 0.5 means the feet just touch; detection bias pushes flight <= 0
    _, _, s, _ = analyze(landmarks, clip=dict(duty=0.5))
    assert s["nonpositive_flight_fraction"] > ga.NONPOSITIVE_FLIGHT_WARN
    assert any("overlapping feet" in w for w in s["warnings"])


def test_hidden_leg_degrades_gracefully(landmarks):
    events, strides, s, _ = analyze(landmarks, clip=dict(right_visible=False))
    assert strides.empty and events.empty
    assert s["cadence_spm"] is None
    assert s["quality"]["label"] == "poor" and s["quality"]["score"] < 0.5
    assert s["per_side"] == {"left": {}, "right": {}}


# ------------------------------------------------------------ options

def test_window_limits_strides_and_events(landmarks):
    path = landmarks()
    _, full, _, _ = ga.analyze(path, WIDTH, HEIGHT)
    ev, win, s, _ = ga.analyze(path, WIDTH, HEIGHT, window=(5.0, 12.0))

    assert 0 < len(win) < len(full)
    assert (win["ic_frame"] / FPS >= 5.0).all()
    assert (win["next_ic_frame"] / FPS <= 12.0).all()
    assert ev["time_seconds"].between(5.0, 12.0).all()
    assert s["window_s"] == [5.0, 12.0]


def test_window_outside_clip_is_empty_not_an_error(landmarks):
    _, strides, s, _ = ga.analyze(landmarks(), WIDTH, HEIGHT, window=(100, 200))
    assert strides.empty and s["cadence_spm"] is None
    assert s["quality"]["label"] == "poor"


def test_toe_off_offset_shortens_contact_and_lengthens_flight(landmarks):
    path = landmarks()
    _, base, _, _ = ga.analyze(path, WIDTH, HEIGHT)
    _, early, s, _ = ga.analyze(path, WIDTH, HEIGHT, toe_off_offset_frames=-2.0)

    shift = 2.0 / FPS
    assert (base["contact_time_s"].median() - early["contact_time_s"].median()
            ) == pytest.approx(shift, abs=0.002)
    assert (early["flight_time_s"].median() - base["flight_time_s"].median()
            ) == pytest.approx(shift, abs=0.002)
    assert s["toe_off_offset_frames"] == -2.0


def test_explicit_fps_overrides_estimate(landmarks):
    _, _, s, _ = ga.analyze(landmarks(), WIDTH, HEIGHT, fps=60.0)
    assert s["fps"] == 60.0


# ------------------------------------------------------------------- plots

def test_plots_are_written(landmarks, tmp_path):
    _, strides, _, ctx = analyze(landmarks)
    debug, cycles = tmp_path / "debug.png", tmp_path / "cycles.png"
    ga.plot_debug(ctx, debug)
    ga.plot_cycles(ctx, strides, cycles)
    for f in (debug, cycles):
        assert f.exists() and f.stat().st_size > 5_000
        assert f.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_cycles_with_no_strides_writes_nothing(landmarks, tmp_path):
    _, strides, _, ctx = analyze(landmarks, clip=dict(right_visible=False))
    out = tmp_path / "cycles.png"
    ga.plot_cycles(ctx, strides, out)
    assert not out.exists()

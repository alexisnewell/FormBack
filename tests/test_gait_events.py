"""Tests for api/services/gait_events.py: landmarks -> signals -> events."""

import numpy as np
import pandas as pd
import pytest

from api.services import gait_events as ge
from tests.synthetic import HEIGHT, WIDTH, expected_events

FPS = 120
PERIOD = 0.7
DUTY = 0.4


def sine(period=PERIOD, seconds=10.0, fps=FPS, amplitude=0.35):
    t = np.arange(int(seconds * fps)) / fps
    return amplitude * np.cos(2 * np.pi * t / period)


# ---------------------------------------------------------------- loading

def test_load_landmarks_requires_columns(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"frame": [0], "landmark": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="Missing CSV columns"):
        ge.load_landmarks(p)


def test_load_landmarks_all_nan_visibility_becomes_one(tmp_path):
    p = tmp_path / "ok.csv"
    pd.DataFrame({"frame": [0, 0], "timestamp_ms": [0, 0], "landmark": [1, 2],
                  "x": [0.1, 0.2], "y": [0.3, 0.4],
                  "visibility": ["", ""]}).to_csv(p, index=False)
    assert (ge.load_landmarks(p)["visibility"] == 1.0).all()


def test_estimate_fps_from_timestamps():
    df = pd.DataFrame({"frame": range(121),
                       "timestamp_ms": [round(i * 1000 / 120) for i in range(121)]})
    assert ge.estimate_fps(df) == pytest.approx(120, rel=1e-3)


def test_estimate_fps_rejects_bad_input():
    with pytest.raises(ValueError, match="Not enough frames"):
        ge.estimate_fps(pd.DataFrame({"frame": [0], "timestamp_ms": [0]}))
    with pytest.raises(ValueError, match="not increasing"):
        ge.estimate_fps(pd.DataFrame({"frame": [0, 1], "timestamp_ms": [5, 5]}))


# ------------------------------------------------------------ conditioning

def test_fill_short_gaps_fills_only_short_interior_gaps():
    v = np.array([1.0, np.nan, np.nan, 4.0, np.nan, np.nan, np.nan, np.nan, 9.0])
    out = ge.fill_short_gaps(v, max_gap=2)
    assert out[1:3] == pytest.approx([2.0, 3.0])        # linear fill
    assert np.isnan(out[4:8]).all()                     # gap of 4 > max_gap
    assert out[0] == 1.0 and out[8] == 9.0


def test_fill_short_gaps_leaves_edge_gaps_alone():
    v = np.array([np.nan, np.nan, 3.0, 4.0, np.nan])
    out = ge.fill_short_gaps(v, max_gap=5)
    assert np.isnan(out[:2]).all() and np.isnan(out[4])


def test_fill_short_gaps_does_not_modify_input():
    v = np.array([1.0, np.nan, 3.0])
    ge.fill_short_gaps(v, 2)
    assert np.isnan(v[1])


def test_condition_smooths_noise_and_keeps_long_gaps_nan():
    rng = np.random.default_rng(0)
    clean = np.sin(np.linspace(0, 6 * np.pi, 600))
    noisy = clean + rng.normal(0, 0.1, clean.size)
    noisy[300:340] = np.nan                              # 40-frame gap
    out = ge.condition(noisy, FPS, max_gap=4, cutoff_hz=10)
    assert out.shape == noisy.shape
    assert np.isnan(out[305:335]).all()
    ok = np.isfinite(out)
    assert np.abs(out[ok] - clean[ok]).mean() < np.abs(noisy[ok] - clean[ok]).mean()


def test_condition_all_nan_returns_all_nan():
    out = ge.condition(np.full(200, np.nan), FPS, 4, 10)
    assert np.isnan(out).all()


# ---------------------------------------------------------------- geometry

def test_included_angle():
    a = np.array([[0.0, 1.0]]); b = np.array([[0.0, 0.0]])
    assert ge.included_angle(a, b, np.array([[1.0, 0.0]]))[0] == pytest.approx(90)
    assert ge.included_angle(a, b, np.array([[0.0, -1.0]]))[0] == pytest.approx(180)


def _tracks(toe_dx, n=50):
    z = np.zeros((n, 2))
    T = {k: z.copy() for k in ge.LM}
    for p in ("l", "r"):
        T[f"{p}_toe"][:, 0] = toe_dx
        T[f"{p}_hip"][:] = [0, 0]
        T[f"{p}_knee"][:] = [0, 50]
        T[f"{p}_ank"][:] = [0, 100]
    return T


def test_facing_sign():
    assert ge.facing_sign(_tracks(+30)) == 1.0
    assert ge.facing_sign(_tracks(-30)) == -1.0
    with pytest.raises(ValueError, match="running direction"):
        ge.facing_sign(_tracks(0))


def test_leg_length():
    assert ge.leg_length(_tracks(10)) == pytest.approx(100.0)
    T = _tracks(10)
    for k in ("l_hip", "l_knee", "l_ank", "r_hip", "r_knee", "r_ank"):
        T[k][:] = np.nan
    with pytest.raises(ValueError, match="leg length"):
        ge.leg_length(T)


# --------------------------------------------------------- event detection

def test_estimate_period_recovers_cycle_length():
    assert ge.estimate_period(sine(), FPS) == pytest.approx(PERIOD, abs=0.01)


def test_estimate_period_is_nan_for_flat_signal():
    assert np.isnan(ge.estimate_period(np.zeros(500), FPS))


def test_refine_peak_finds_subframe_vertex():
    x = np.arange(21, dtype=float)
    y = -(x - 10.3) ** 2
    assert ge.refine_peak(y, 10) == pytest.approx(10.3)


def test_refine_peak_flat_returns_input_index():
    assert ge.refine_peak(np.ones(5), 2) == 2.0


def test_detect_events_alternate_and_match_period():
    ic, to = ge.detect_events(sine(), PERIOD, FPS)
    assert len(ic) >= 12 and abs(len(ic) - len(to)) <= 1
    assert np.diff(ic) / FPS == pytest.approx(PERIOD, abs=0.01)
    assert np.diff(to) / FPS == pytest.approx(PERIOD, abs=0.01)
    # initial contact (forward extreme) is followed by toe-off (backward extreme)
    for c in ic[:-1]:
        nxt = to[to > c]
        assert nxt.size and (nxt[0] - c) / FPS == pytest.approx(PERIOD / 2, abs=0.01)


def test_detect_events_positions_are_extremes():
    sig = sine()
    ic, to = ge.detect_events(sig, PERIOD, FPS)
    assert (np.interp(ic, np.arange(sig.size), sig) > 0.34).all()
    assert (np.interp(to, np.arange(sig.size), sig) < -0.34).all()


def test_detect_events_ignores_wiggles_below_min_excursion():
    assert ge.MIN_EXCURSION > 0.1
    ic, to = ge.detect_events(sine(amplitude=0.05), PERIOD, FPS)
    assert ic.size == 0 and to.size == 0


def test_detect_events_needs_enough_finite_samples():
    sig = np.full(100, np.nan)
    sig[:8] = [0, 1, 0, 1, 0, 1, 0, 1]
    ic, to = ge.detect_events(sig, PERIOD, FPS)
    assert ic.size == 0 and to.size == 0


def test_detect_events_skips_peaks_next_to_gaps():
    sig = sine()
    ic_all, _ = ge.detect_events(sig, PERIOD, FPS)
    target = int(round(ic_all[3]))
    gapped = sig.copy()
    gapped[target + 1] = np.nan                      # NaN beside the peak
    ic, _ = ge.detect_events(gapped, PERIOD, FPS)
    assert len(ic) == len(ic_all) - 1


# ------------------------------------------------------------- end to end

def test_detect_gait_events_matches_ground_truth(landmarks):
    g = ge.detect_gait_events(landmarks(period=PERIOD, duty=DUTY), WIDTH, HEIGHT)
    assert g.fps == pytest.approx(120, rel=1e-3)
    assert g.period == pytest.approx(PERIOD, abs=0.01)
    assert g.direction == 1.0

    for side, offset in (("left", 0.0), ("right", PERIOD / 2)):
        ic, to = g.events[side]
        exp_ic, exp_to = expected_events(20.0, PERIOD, DUTY, offset)
        # every detected event is within ~1 frame of a true one
        assert max(min(abs(i / FPS - e) for e in exp_ic) for i in ic) < 1.5 / FPS
        assert max(min(abs(t / FPS - e) for e in exp_to) for t in to) < 1.5 / FPS


def test_legs_are_half_a_cycle_apart(landmarks):
    g = ge.detect_gait_events(landmarks(), WIDTH, HEIGHT)
    left, right = g.events["left"][0], g.events["right"][0]
    lag = [(right[right > l][0] - l) / FPS for l in left[:-1] if (right > l).any()]
    assert np.median(lag) == pytest.approx(PERIOD / 2, abs=0.01)


@pytest.mark.parametrize("facing", [1, -1])
def test_direction_follows_facing(landmarks, facing):
    g = ge.detect_gait_events(landmarks(facing=facing), WIDTH, HEIGHT)
    assert g.direction == facing
    # same events regardless of direction: the signal is direction-normalised
    ref = ge.detect_gait_events(landmarks(facing=1), WIDTH, HEIGHT)
    assert g.events["left"][0] == pytest.approx(ref.events["left"][0], abs=0.5)


def test_toe_off_offset_shifts_only_toe_offs(landmarks):
    path = landmarks()
    base = ge.detect_gait_events(path, WIDTH, HEIGHT)
    shifted = ge.detect_gait_events(path, WIDTH, HEIGHT, toe_off_offset_frames=-2.0)
    for side in ("left", "right"):
        assert shifted.events[side][0] == pytest.approx(base.events[side][0])
        assert shifted.events[side][1] == pytest.approx(base.events[side][1] - 2.0)


def test_hidden_hip_gives_no_events_instead_of_crashing(landmarks):
    g = ge.detect_gait_events(landmarks(right_visible=False), WIDTH, HEIGHT)
    for side in ("left", "right"):
        assert g.events[side][0].size == 0


# ----------------------------------------------------------- events table

def test_events_to_frame_is_sorted_and_labelled():
    events = {"left": (np.array([10.0, 94.0]), np.array([44.0])),
              "right": (np.array([52.4]), np.array([86.6]))}
    df = ge.events_to_frame(events, FPS)
    assert list(df.columns) == ["side", "event", "frame", "frame_exact", "time_seconds"]
    assert df["time_seconds"].is_monotonic_increasing
    assert len(df) == 5
    row = df[(df.side == "right") & (df.event == "initial_contact")].iloc[0]
    assert row["frame"] == 52 and row["frame_exact"] == 52.4
    assert row["time_seconds"] == pytest.approx(52.4 / FPS, abs=1e-4)


def test_events_to_frame_empty_keeps_columns():
    df = ge.events_to_frame({"left": (np.array([]), np.array([])),
                             "right": (np.array([]), np.array([]))}, FPS)
    assert df.empty and list(df.columns) == [
        "side", "event", "frame", "frame_exact", "time_seconds"]

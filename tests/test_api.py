import json
import pathlib
import types
import uuid

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.services import report_service as service

REAL = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "real_summary.json").read_text()
)
GOOD_SUMMARY = REAL
LOW_SUMMARY = {**REAL, "quality": {**REAL["quality"], "score": 0.2, "label": "poor"}}


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated video/output dirs, empty job store, fake (instant) pipeline."""
    uploads, outputs = tmp_path / "videos", tmp_path / "output"
    uploads.mkdir(), outputs.mkdir()
    monkeypatch.setattr(service, "VIDEOS_DIR", uploads)
    monkeypatch.setattr(service, "OUTPUT_DIR", outputs)
    monkeypatch.setattr(service, "_jobs", {})

    pipeline = types.SimpleNamespace()
    pipeline.next_result = {"status": "ok", "summary": GOOD_SUMMARY, "error": None}
    pipeline.analyze_video = lambda path: dict(pipeline.next_result)
    # indirection so tests can swap pipeline.analyze_video after setup
    monkeypatch.setattr(service, "analyze_video", lambda path: pipeline.analyze_video(path))

    return types.SimpleNamespace(uploads=uploads, outputs=outputs, pipeline=pipeline)


@pytest.fixture
def client(env):
    return TestClient(app)


def make_mp4(path, frames=10):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 64))
    for _ in range(frames):
        w.write(np.zeros((64, 64, 3), dtype=np.uint8))
    w.release()


def upload_real(client, tmp_path):
    p = tmp_path / "clip.mp4"
    make_mp4(p)
    with open(p, "rb") as f:
        return client.post("/api/videos/upload", files={"file": ("clip.mp4", f, "video/mp4")})


def test_upload_rejects_non_mp4(client):
    r = client.post("/api/videos/upload", files={"file": ("a.txt", b"x")})
    assert r.status_code == 400


def test_upload_rejects_fake_mp4_and_cleans_up(client, env):
    r = client.post("/api/videos/upload", files={"file": ("a.mp4", b"not a video")})
    assert r.status_code == 400
    assert list(env.uploads.iterdir()) == []


def test_upload_ok(client, env, tmp_path):
    r = upload_real(client, tmp_path)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "uploaded" and body["frame_count"] == 10
    assert (env.uploads / f"{body['video_id']}.mp4").exists()


def test_invalid_id_is_422(client):
    for path in ("status", "results"):
        assert client.get(f"/api/videos/not-a-uuid/{path}").status_code == 422
    assert client.post("/api/videos/not-a-uuid/analyze").status_code == 422


def test_analyze_unknown_video_404(client):
    assert client.post(f"/api/videos/{uuid.uuid4()}/analyze").status_code == 404


def test_status_and_results_without_analysis_404(client):
    vid = uuid.uuid4()
    assert client.get(f"/api/videos/{vid}/status").status_code == 404
    assert client.get(f"/api/videos/{vid}/results").status_code == 404


def test_full_flow_ok(client, env, tmp_path):
    vid = upload_real(client, tmp_path).json()["video_id"]
    (env.outputs / f"{vid}_strides.csv").write_text("x")

    r = client.post(f"/api/videos/{vid}/analyze")
    assert r.status_code == 200 and r.json()["status"] == "processing"

    # TestClient runs background tasks before returning
    assert client.get(f"/api/videos/{vid}/status").json()["status"] == "completed"

    r = client.get(f"/api/videos/{vid}/results")
    body = r.json()
    assert r.status_code == 200
    assert body["analysis_status"] == "ok"
    assert body["metrics"]["overall"]["cadence_spm"] == 191.2
    assert body["files"] == ["strides"]
    assert "summary" not in body


def test_low_quality_withholds_metrics(client, env, tmp_path):
    env.pipeline.next_result = {"status": "low_quality", "summary": LOW_SUMMARY, "error": None}
    vid = upload_real(client, tmp_path).json()["video_id"]
    client.post(f"/api/videos/{vid}/analyze")

    body = client.get(f"/api/videos/{vid}/results").json()
    assert body["status"] == "completed"
    assert body["analysis_status"] == "low_quality"
    assert "metrics" not in body
    assert body["quality"]["score"] == 0.2
    assert body["warnings"] == REAL["warnings"]
    assert body["diagnostics"]["stride_counts"] == {"left": 27, "right": 12}


def test_failed_job_reports_error_in_body(client, env, tmp_path):
    env.pipeline.next_result = {"status": "failed", "summary": None, "error": "boom"}
    vid = upload_real(client, tmp_path).json()["video_id"]
    client.post(f"/api/videos/{vid}/analyze")

    s = client.get(f"/api/videos/{vid}/status").json()
    assert s == {"video_id": vid, "status": "failed", "error": "boom"}
    r = client.get(f"/api/videos/{vid}/results")
    assert r.status_code == 200
    assert r.json() == {"video_id": vid, "status": "failed", "error": "boom"}


def test_pipeline_exception_becomes_failed(client, env, tmp_path):
    def explode(path):
        raise RuntimeError("kaboom")
    env.pipeline.analyze_video = explode
    vid = upload_real(client, tmp_path).json()["video_id"]
    client.post(f"/api/videos/{vid}/analyze")
    assert "RuntimeError: kaboom" in client.get(f"/api/videos/{vid}/status").json()["error"]


def test_processing_gives_202_and_blocks_second_start(client, env, tmp_path):
    vid = upload_real(client, tmp_path).json()["video_id"]
    service.begin_analysis(vid)  # job is now "processing", never finishes

    r = client.get(f"/api/videos/{vid}/results")
    assert r.status_code == 202
    assert r.json() == {"video_id": vid, "status": "processing"}

    assert client.post(f"/api/videos/{vid}/analyze").status_code == 409


def test_results_survive_restart_via_disk(client, env):
    vid = str(uuid.uuid4())
    (env.outputs / f"{vid}_summary.json").write_text(json.dumps(GOOD_SUMMARY))

    r = client.get(f"/api/videos/{vid}/results")
    assert r.status_code == 200
    assert r.json()["analysis_status"] == "ok"
    assert r.json()["files"] == ["summary"]
    assert r.json()["metrics"]["overall"]["cadence_spm"] == 191.2


def test_disk_fallback_applies_quality_gate(client, env):
    vid = str(uuid.uuid4())
    (env.outputs / f"{vid}_summary.json").write_text(json.dumps(LOW_SUMMARY))
    body = client.get(f"/api/videos/{vid}/results").json()
    assert body["analysis_status"] == "low_quality" and "metrics" not in body


def test_openapi_lists_all_routes(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/api/videos/upload", "/api/videos/{video_id}/analyze",
              "/api/videos/{video_id}/status", "/api/videos/{video_id}/results"):
        assert p in paths


# ---------- response shape on the real summary ----------

def get_ok_results(client, env, tmp_path):
    vid = upload_real(client, tmp_path).json()["video_id"]
    client.post(f"/api/videos/{vid}/analyze")
    return client.get(f"/api/videos/{vid}/results").json()


def test_clean_shape_has_no_duplicates_or_raw_keys(client, env, tmp_path):
    body = get_ok_results(client, env, tmp_path)
    assert set(body) == {"video_id", "status", "analysis_status", "quality",
                         "warnings", "analysis", "metrics", "diagnostics", "files"}
    assert "error" not in body
    dumped = json.dumps(body)
    for raw_key in ("metric_confidence", "foot_strike_pattern_approx"):
        assert raw_key not in dumped
    assert body["warnings"] == REAL["warnings"]  # once, at top level


def test_per_metric_confidence_is_flattened(client, env, tmp_path):
    left = get_ok_results(client, env, tmp_path)["metrics"]["per_side"]["left"]
    assert left["stride_time_s"]["confidence"] == "high"
    assert left["mean_trunk_lean_deg"]["confidence"] == "high"
    assert left["contact_time_s"]["confidence"] == "unvalidated"
    assert "toe-off" in left["contact_time_s"]["confidence_note"]
    assert left["knee_flexion_at_contact_deg"]["confidence"] == "medium"
    assert left["peak_stance_knee_flexion_deg"]["confidence"] == "medium"
    assert left["vertical_oscillation_pct_leg"]["confidence"] == "low"
    assert left["stride_time_s"].get("confidence_note") is None


def test_metric_stat_fields(client, env, tmp_path):
    stat = get_ok_results(client, env, tmp_path)["metrics"]["per_side"]["right"]["stride_time_s"]
    assert stat["label"] == "Stride time" and stat["unit"] == "s"
    assert (stat["median"], stat["iqr"], stat["n"]) == (0.619, 0.012, 12)


def test_symmetry_and_null_foot_strike(client, env, tmp_path):
    overall = get_ok_results(client, env, tmp_path)["metrics"]["overall"]
    assert overall["symmetry_index_pct"] == {
        "step_time": 3.6, "contact_time_s": 3.1,
        "knee_flexion_at_contact_deg": 19.7}
    # explicit null is kept: "classified as nothing", not "missing"
    assert "foot_strike_pattern" in overall and overall["foot_strike_pattern"] is None


def test_analysis_and_diagnostics_blocks(client, env, tmp_path):
    body = get_ok_results(client, env, tmp_path)
    assert body["analysis"]["direction"] == "left"
    assert body["analysis"]["fps"] == 119.96
    assert body["diagnostics"] == {
        "stride_counts": {"left": 27, "right": 12},
        "knee_valid_stride_fraction": 0.85,
        "nonpositive_flight_fraction": 0.04}
    assert body["quality"]["frames"] == 2730


def test_metric_without_data_is_omitted(client, env, tmp_path):
    summary = json.loads(json.dumps(REAL))
    summary["per_side"]["right"] = {}          # no strides at all on one side
    summary["per_side"]["left"]["flight_time_s"] = None
    env.pipeline.next_result = {"status": "ok", "summary": summary, "error": None}
    per_side = get_ok_results(client, env, tmp_path)["metrics"]["per_side"]
    assert per_side["right"] == {}
    assert "flight_time_s" not in per_side["left"]


# ---------- quality gate and report helpers in report_service ----------

def _summary(score, left_n, right_n):
    return {"quality": {"score": score},
            "per_side": {"left": {"stride_time_s": {"n": left_n}},
                         "right": {"stride_time_s": {"n": right_n}}}}


def test_classify_quality_thresholds():
    ok, low = service.AnalysisQuality.ok, service.AnalysisQuality.low_quality
    assert service.classify_quality(_summary(0.5, 3, 3)) == ok           # boundaries pass
    assert service.classify_quality(_summary(0.49, 10, 10)) == low       # score too low
    assert service.classify_quality(_summary(0.9, 2, 10)) == low         # too few left
    assert service.classify_quality(_summary(0.9, 10, 2)) == low         # too few right
    assert service.classify_quality({}) == low                           # nothing at all


# ---------- real pipeline, no mocks (pose step replaced by cached landmarks) ----------

@pytest.fixture
def real_env(tmp_path, monkeypatch):
    uploads, outputs = tmp_path / "videos", tmp_path / "output"
    uploads.mkdir(), outputs.mkdir()
    monkeypatch.setattr(service, "VIDEOS_DIR", uploads)
    monkeypatch.setattr(service, "OUTPUT_DIR", outputs)
    monkeypatch.setattr(service, "MODEL_PATH", tmp_path / "missing_model.task")
    monkeypatch.setattr(service, "_jobs", {})
    return types.SimpleNamespace(uploads=uploads, outputs=outputs)


def _upload_1000px_video(client, tmp_path):
    # pipeline reads width/height from the video; synthetic landmarks assume 1000x1000
    p = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1000, 1000))
    for _ in range(5):
        w.write(np.zeros((1000, 1000, 3), dtype=np.uint8))
    w.release()
    with open(p, "rb") as f:
        return client.post("/api/videos/upload", files={"file": ("clip.mp4", f, "video/mp4")}).json()["video_id"]


def test_real_pipeline_from_cached_landmarks(real_env, tmp_path):
    import shutil
    from tests.synthetic import make_landmarks_csv

    client = TestClient(app)
    vid = _upload_1000px_video(client, tmp_path)

    make_landmarks_csv(tmp_path / "lm.csv")
    shutil.copy(tmp_path / "lm.csv", real_env.outputs / f"{vid}_landmarks.csv")

    assert client.post(f"/api/videos/{vid}/analyze").status_code == 200

    body = client.get(f"/api/videos/{vid}/results").json()
    assert body["status"] == "completed" and body["analysis_status"] == "ok"
    assert body["metrics"]["overall"]["cadence_spm"] == pytest.approx(171.4, abs=0.5)
    assert body["quality"]["label"] == "good" and body["warnings"] == []
    assert body["diagnostics"]["stride_counts"]["left"] >= 25
    assert body["files"] == ["events", "strides", "summary", "debug_plot", "cycles_plot"]
    for name in ("_events.csv", "_strides.csv", "_summary.json", "_debug.png", "_cycles.png"):
        assert (real_env.outputs / f"{vid}{name}").stat().st_size > 0

    # the saved summary alone is enough to serve results after a "restart"
    service._jobs.clear()
    again = client.get(f"/api/videos/{vid}/results").json()
    assert again["analysis_status"] == "ok" and again["metrics"] == body["metrics"]


def test_missing_pose_model_fails_cleanly(real_env, tmp_path):
    client = TestClient(app)
    vid = _upload_1000px_video(client, tmp_path)

    client.post(f"/api/videos/{vid}/analyze")

    status = client.get(f"/api/videos/{vid}/status").json()
    assert status["status"] == "failed"
    assert "MediaPipe model not found" in status["error"]

    r = client.get(f"/api/videos/{vid}/results")
    assert r.status_code == 200 and r.json()["status"] == "failed"
    assert "metrics" not in r.json()

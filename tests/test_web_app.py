import json
import io
from pathlib import Path
import zipfile

import pytest
from fastapi.testclient import TestClient

from rheology_paper_ocr import web_app
from rheology_paper_ocr.web_app import create_app, serve_local_app
from rheology_paper_ocr.web_store import LocalRunStore


def test_local_run_store_persists_runs_and_reviewer_decisions(tmp_path: Path):
    store = LocalRunStore(tmp_path)
    run = store.create_run("July papers")

    store.update_status(run["id"], "completed")
    store.save_decision(run["id"], "curve_a", "Figure 2", "accepted", "Legend and evidence checked.")

    restored = LocalRunStore(tmp_path)
    assert restored.get_run(run["id"])["status"] == "completed"
    assert restored.decisions_for_run(run["id"])[("curve_a", "Figure 2")]["decision"] == "accepted"


def test_browser_api_uploads_papers_persists_decisions_and_exports_csv(tmp_path: Path):
    client = TestClient(create_app(tmp_path))

    home = client.get("/")
    stylesheet = client.get("/static/app.css")
    pause_icon = client.get("/static/icons/pause.svg")
    created = client.post(
        "/api/runs",
        data={"name": "Browser test"},
        files=[("files", ("paper.pdf", b"%PDF-1.4\n", "application/pdf"))],
    )

    assert home.status_code == 200
    assert "Rheology Evidence" in home.text
    assert stylesheet.status_code == 200
    assert "--accent" in stylesheet.text
    assert pause_icon.status_code == 200
    assert "lucide-pause" in pause_icon.text
    assert created.status_code == 200
    run = created.json()["run"]
    output_dir = Path(LocalRunStore(tmp_path).get_run(run["id"])["output_dir"])
    (output_dir / "results.json").write_text(
        json.dumps(
            [
                {
                    "paper_id": "paper_0001",
                    "source_pdf": "paper.pdf",
                    "figure_id": "Figure 2",
                    "curve_id": "curve_a",
                    "curve_legend_text": "Sample A",
                    "fibre_outcome": "formed fibres",
                    "decision": "needs_review",
                    "warnings": ["sample link lacks direct evidence"],
                }
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "review_queue.json").write_text("[]", encoding="utf-8")

    decision = client.post(
        f"/api/runs/{run['id']}/decisions",
        json={"curve_id": "curve_a", "figure_id": "Figure 2", "decision": "accepted", "note": "Checked."},
    )
    detail = client.get(f"/api/runs/{run['id']}")
    exported = client.get(f"/api/runs/{run['id']}/export.csv")

    assert decision.status_code == 200
    assert detail.json()["results"][0]["reviewer_decision"] == "accepted"
    assert "reviewer_decision" in exported.text
    assert "accepted" in exported.text


def test_browser_client_handles_empty_delete_responses():
    app_js = (Path(__file__).parents[1] / "rheology_paper_ocr" / "ui" / "app.js").read_text(encoding="utf-8")

    assert "response.status === 204" in app_js


def test_browser_api_extracts_pdfs_from_zip_upload(tmp_path: Path):
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("nested/paper-a.pdf", b"%PDF-1.4\nA")
        archive.writestr("nested/notes.txt", b"not a paper")
        archive.writestr("paper-b.PDF", b"%PDF-1.4\nB")

    client = TestClient(create_app(tmp_path))
    created = client.post(
        "/api/runs",
        files=[("files", ("papers.zip", archive_bytes.getvalue(), "application/zip"))],
    )

    assert created.status_code == 200
    assert created.json()["files"] == ["001-paper-a.pdf", "002-paper-b.PDF"]
    assert created.json()["run"]["paper_count"] == 2
    run = LocalRunStore(tmp_path).get_run(created.json()["run"]["id"])
    assert len([path for path in Path(run["input_dir"]).iterdir() if path.suffix.lower() == ".pdf"]) == 2


def test_browser_run_detail_exposes_current_and_queued_papers(tmp_path: Path):
    client = TestClient(create_app(tmp_path))
    created = client.post(
        "/api/runs",
        files=[
            ("files", ("first.pdf", b"%PDF-1.4\n", "application/pdf")),
            ("files", ("second.pdf", b"%PDF-1.4\n", "application/pdf")),
        ],
    )
    run_id = created.json()["run"]["id"]
    run = LocalRunStore(tmp_path).get_run(run_id)
    LocalRunStore(tmp_path).update_status(run_id, "running")
    (Path(run["output_dir"]) / "manifest.json").write_text(
        json.dumps(
            [{"paper_id": "paper_0001", "source_pdf": str(Path(run["input_dir"]) / "001-first.pdf"), "status": "started"}]
        ),
        encoding="utf-8",
    )

    detail = client.get(f"/api/runs/{run_id}").json()

    assert detail["progress"]["processing"] == 1
    assert detail["progress"]["queued"] == 1
    assert [paper["name"] for paper in detail["progress"]["papers"]] == ["first.pdf", "second.pdf"]


def test_browser_api_deletes_inactive_run_and_refuses_active_run(tmp_path: Path):
    client = TestClient(create_app(tmp_path))
    created = client.post(
        "/api/runs",
        files=[("files", ("paper.pdf", b"%PDF-1.4\n", "application/pdf"))],
    )
    run_id = created.json()["run"]["id"]
    run_dir = Path(LocalRunStore(tmp_path).get_run(run_id)["input_dir"]).parent

    deleted = client.delete(f"/api/runs/{run_id}")

    assert deleted.status_code == 204
    assert not run_dir.exists()
    assert client.get(f"/api/runs/{run_id}").status_code == 404

    active = LocalRunStore(tmp_path).create_run("Active run")
    LocalRunStore(tmp_path).update_status(active["id"], "running")
    refused = client.delete(f"/api/runs/{active['id']}")
    assert refused.status_code == 409


def test_existing_browser_run_resumes_completed_papers(tmp_path: Path, monkeypatch):
    store = LocalRunStore(tmp_path)
    run = store.create_run("Resume test")
    output_dir = Path(run["output_dir"])
    (output_dir / "manifest.json").write_text("[]", encoding="utf-8")
    seen = {}

    def fake_pipeline(input_dir, pipeline_output_dir, api_key=None, resume=False, should_pause=None):
        seen.update(input_dir=input_dir, output_dir=pipeline_output_dir, api_key=api_key, resume=resume, should_pause=should_pause)
        return []

    monkeypatch.setattr(web_app, "run_pipeline", fake_pipeline)
    web_app._run_job(store, run["id"])

    assert seen["resume"] is True
    assert store.get_run(run["id"])["status"] == "completed"


def test_browser_api_key_is_passed_only_to_the_in_memory_job(tmp_path: Path, monkeypatch):
    client = TestClient(create_app(tmp_path))
    created = client.post(
        "/api/runs",
        files=[("files", ("paper.pdf", b"%PDF-1.4\n", "application/pdf"))],
    )
    seen = {}

    def fake_job(store, run_id, api_key):
        seen.update(run_id=run_id, api_key=api_key)

    class ImmediateThread:
        def __init__(self, *, target, args, daemon):
            self.target = target
            self.args = args

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(web_app, "_run_job", fake_job)
    monkeypatch.setattr(web_app.threading, "Thread", ImmediateThread)
    key = "browser-only-test-key"
    started = client.post(f"/api/runs/{created.json()['run']['id']}/start", json={"api_key": key})

    assert started.status_code == 200
    assert seen["api_key"] == key
    assert key.encode() not in (tmp_path / "sessions.sqlite3").read_bytes()


def test_browser_pause_and_resume_run(tmp_path: Path, monkeypatch):
    client = TestClient(create_app(tmp_path))
    created = client.post(
        "/api/runs",
        files=[("files", ("paper.pdf", b"%PDF-1.4\n", "application/pdf"))],
    )
    run_id = created.json()["run"]["id"]
    LocalRunStore(tmp_path).update_status(run_id, "running")
    paused = client.post(f"/api/runs/{run_id}/pause")
    seen = {}

    def fake_job(store, resumed_run_id, api_key):
        seen.update(run_id=resumed_run_id, api_key=api_key)

    class ImmediateThread:
        def __init__(self, *, target, args, daemon):
            self.target = target
            self.args = args

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(web_app, "_run_job", fake_job)
    monkeypatch.setattr(web_app.threading, "Thread", ImmediateThread)
    resumed = client.post(f"/api/runs/{run_id}/resume", json={"api_key": "resume-key"})

    assert paused.status_code == 200
    assert paused.json()["status"] == "paused"
    assert resumed.status_code == 200
    assert seen == {"run_id": run_id, "api_key": "resume-key"}


def test_browser_startup_pauses_orphaned_runs(tmp_path: Path):
    store = LocalRunStore(tmp_path)
    run = store.create_run("Interrupted")
    store.update_status(run["id"], "running")

    create_app(tmp_path)

    restored = LocalRunStore(tmp_path).get_run(run["id"])
    assert restored["status"] == "paused"
    assert "server restarted" in restored["error"]


def test_browser_run_requires_an_explicit_api_key(tmp_path: Path):
    client = TestClient(create_app(tmp_path))
    created = client.post(
        "/api/runs",
        files=[("files", ("paper.pdf", b"%PDF-1.4\n", "application/pdf"))],
    )
    run_id = created.json()["run"]["id"]

    started = client.post(f"/api/runs/{run_id}/start", json={})

    assert started.status_code == 400
    assert "OpenRouter API key" in started.json()["detail"]
    assert LocalRunStore(tmp_path).get_run(run_id)["status"] == "ready"


def test_browser_server_rejects_non_loopback_host(tmp_path: Path):
    with pytest.raises(ValueError, match="loopback"):
        serve_local_app(tmp_path, "0.0.0.0", 8787, open_browser=False)

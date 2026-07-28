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
    created = client.post(
        "/api/runs",
        data={"name": "Browser test"},
        files=[("files", ("paper.pdf", b"%PDF-1.4\n", "application/pdf"))],
    )

    assert home.status_code == 200
    assert "Rheology Evidence" in home.text
    assert stylesheet.status_code == 200
    assert "--signal" in stylesheet.text
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


def test_existing_browser_run_resumes_completed_papers(tmp_path: Path, monkeypatch):
    store = LocalRunStore(tmp_path)
    run = store.create_run("Resume test")
    output_dir = Path(run["output_dir"])
    (output_dir / "manifest.json").write_text("[]", encoding="utf-8")
    seen = {}

    def fake_pipeline(input_dir, pipeline_output_dir, api_key=None, resume=False):
        seen.update(input_dir=input_dir, output_dir=pipeline_output_dir, api_key=api_key, resume=resume)
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


def test_browser_server_rejects_non_loopback_host(tmp_path: Path):
    with pytest.raises(ValueError, match="loopback"):
        serve_local_app(tmp_path, "0.0.0.0", 8787, open_browser=False)

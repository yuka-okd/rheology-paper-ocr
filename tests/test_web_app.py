import json
from pathlib import Path

from fastapi.testclient import TestClient

from rheology_paper_ocr import web_app
from rheology_paper_ocr.web_app import create_app
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
    created = client.post(
        "/api/runs",
        data={"name": "Browser test"},
        files=[("files", ("paper.pdf", b"%PDF-1.4\n", "application/pdf"))],
    )

    assert home.status_code == 200
    assert "Rheology Evidence" in home.text
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


def test_existing_browser_run_resumes_completed_papers(tmp_path: Path, monkeypatch):
    store = LocalRunStore(tmp_path)
    run = store.create_run("Resume test")
    output_dir = Path(run["output_dir"])
    (output_dir / "manifest.json").write_text("[]", encoding="utf-8")
    seen = {}

    def fake_pipeline(input_dir, pipeline_output_dir, resume=False):
        seen.update(input_dir=input_dir, output_dir=pipeline_output_dir, resume=resume)
        return []

    monkeypatch.setattr(web_app, "run_pipeline", fake_pipeline)
    web_app._run_job(store, run["id"])

    assert seen["resume"] is True
    assert store.get_run(run["id"])["status"] == "completed"

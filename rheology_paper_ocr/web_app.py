from __future__ import annotations

import csv
import html
import io
import shutil
import threading
import webbrowser
from pathlib import Path
from typing import Any

from rheology_paper_ocr.pipeline import run_pipeline
from rheology_paper_ocr.web_store import LocalRunStore, read_json


def create_app(data_dir: Path):
    """Create the local-only browser application without importing web extras at CLI import time."""
    try:
        from fastapi import FastAPI, File, HTTPException, UploadFile
        from fastapi.responses import FileResponse, HTMLResponse, Response
        from fastapi.staticfiles import StaticFiles
        from pydantic import BaseModel
    except ImportError as exc:
        raise RuntimeError("The local browser app requires '.[web]'. Install with: python -m pip install -e '.[web]'.") from exc
    # FastAPI resolves endpoint annotations from module globals. Keep this
    # optional dependency lazy while making the upload annotation concrete.
    globals()["UploadFile"] = UploadFile

    class DecisionRequest(BaseModel):
        curve_id: str
        figure_id: str | None = None
        decision: str
        note: str | None = None

    globals()["DecisionRequest"] = DecisionRequest

    store = LocalRunStore(data_dir)
    ui_dir = Path(__file__).with_name("ui")
    app = FastAPI(title="Rheology Paper OCR", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=ui_dir), name="static")

    @app.get("/")
    def index():
        return FileResponse(ui_dir / "index.html")

    @app.get("/api/runs")
    def list_runs():
        return [_run_summary(store.get_run(run["id"])) for run in store.list_runs()]

    @app.post("/api/runs")
    async def create_run(name: str = "", files: list[UploadFile] = File(...)):
        run = store.create_run(name)
        input_dir = Path(run["input_dir"])
        saved = []
        for index, upload in enumerate(files, start=1):
            source_name = Path(upload.filename or "paper.pdf").name
            if Path(source_name).suffix.lower() != ".pdf":
                continue
            target = input_dir / f"{index:03d}-{source_name}"
            with target.open("wb") as handle:
                shutil.copyfileobj(upload.file, handle)
            saved.append(target.name)
        if not saved:
            shutil.rmtree(input_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="Upload at least one PDF file.")
        return {"run": _run_summary(store.get_run(run["id"])), "files": saved}

    @app.post("/api/runs/{run_id}/start")
    def start_run(run_id: str):
        try:
            run = store.get_run(run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found.")
        if run["status"] == "running":
            raise HTTPException(status_code=409, detail="Run is already in progress.")
        if not list(Path(run["input_dir"]).glob("*.pdf")):
            raise HTTPException(status_code=400, detail="No PDF inputs are available for this run.")
        store.update_status(run_id, "running")
        thread = threading.Thread(target=_run_job, args=(store, run_id), daemon=True)
        thread.start()
        return _run_summary(store.get_run(run_id))

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        try:
            return _run_detail(store.get_run(run_id), store)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found.")

    @app.post("/api/runs/{run_id}/decisions")
    def save_decision(run_id: str, request: DecisionRequest):
        if request.decision not in {"accepted", "needs_review", "rejected"}:
            raise HTTPException(status_code=400, detail="Decision must be accepted, needs_review, or rejected.")
        try:
            store.get_run(run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found.")
        return store.save_decision(run_id, request.curve_id, request.figure_id, request.decision, request.note)

    @app.get("/api/runs/{run_id}/files/{artifact_path:path}")
    def artifact(run_id: str, artifact_path: str):
        try:
            run = store.get_run(run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found.")
        output_dir = Path(run["output_dir"]).resolve()
        path = (output_dir / artifact_path).resolve()
        try:
            path.relative_to(output_dir)
        except ValueError:
            raise HTTPException(status_code=403, detail="Invalid artifact path.")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found.")
        return FileResponse(path)

    @app.get("/api/runs/{run_id}/export.csv")
    def export_csv(run_id: str):
        try:
            detail = _run_detail(store.get_run(run_id), store)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found.")
        stream = io.StringIO()
        fields = [
            "paper_id", "figure_id", "curve_id", "curve_legend_text", "sample_display_name", "sample_composition",
            "rheology_class", "fibre_outcome", "confidence", "decision", "reviewer_decision", "reviewer_note", "warnings",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in detail["results"]:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})
        return Response(
            stream.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="rheology-{run_id}.csv"'},
        )

    @app.get("/api/runs/{run_id}/print")
    def print_report(run_id: str):
        try:
            detail = _run_detail(store.get_run(run_id), store)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found.")
        return HTMLResponse(_printable_report(detail))

    return app


def serve_local_app(data_dir: Path, host: str, port: int, open_browser: bool) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("The local browser app requires '.[web]'. Install with: python -m pip install -e '.[web]'.") from exc
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    uvicorn.run(create_app(data_dir), host=host, port=port, log_level="info")


def _run_job(store: LocalRunStore, run_id: str) -> None:
    run = store.get_run(run_id)
    try:
        output_dir = Path(run["output_dir"])
        run_pipeline(Path(run["input_dir"]), output_dir, resume=(output_dir / "manifest.json").exists())
        manifest = read_json(output_dir / "manifest.json", [])
        if any(item.get("status") == "blocked_insufficient_credits" for item in manifest):
            store.update_status(run_id, "blocked", "OpenRouter credit is exhausted.")
        elif any(item.get("status") == "failed" for item in manifest):
            store.update_status(run_id, "completed", "One or more papers failed; inspect the run details.")
        else:
            store.update_status(run_id, "completed")
    except Exception as exc:
        store.update_status(run_id, "failed", str(exc))


def _run_summary(run: dict) -> dict:
    manifest = read_json(Path(run["output_dir"]) / "manifest.json", [])
    return {
        "id": run["id"],
        "name": run["name"],
        "created_at": run["created_at"],
        "completed_at": run["completed_at"],
        "status": run["status"],
        "error": run["error"],
        "paper_count": len(manifest) or len(list(Path(run["input_dir"]).glob("*.pdf"))),
        "completed_papers": sum(item.get("status") in {"completed", "completed_no_findings"} for item in manifest),
    }


def _run_detail(run: dict, store: LocalRunStore) -> dict:
    output_dir = Path(run["output_dir"])
    manifest = read_json(output_dir / "manifest.json", [])
    decisions = store.decisions_for_run(run["id"])
    results = []
    for result in read_json(output_dir / "results.json", []):
        key = (result.get("curve_id", ""), result.get("figure_id"))
        reviewer = decisions.get(key, {})
        crop_path = result.get("chart_crop_path")
        result["reviewer_decision"] = reviewer.get("decision")
        result["reviewer_note"] = reviewer.get("note", "")
        result["chart_url"] = f"/api/runs/{run['id']}/files/{crop_path}" if crop_path else None
        results.append(result)
    review_rows = []
    for row in read_json(output_dir / "review_queue.json", []):
        key = (row.get("curve_id", ""), row.get("figure_id"))
        row["reviewer_decision"] = decisions.get(key, {}).get("decision")
        row["reviewer_note"] = decisions.get(key, {}).get("note", "")
        crop_path = row.get("chart_crop_path")
        row["chart_url"] = f"/api/runs/{run['id']}/files/{crop_path}" if crop_path else None
        review_rows.append(row)
    return {
        "run": _run_summary(run),
        "manifest": manifest,
        "results": results,
        "review_queue": review_rows,
        "export_csv_url": f"/api/runs/{run['id']}/export.csv",
        "print_url": f"/api/runs/{run['id']}/print",
    }


def _printable_report(detail: dict) -> str:
    rows = []
    for row in detail["results"]:
        decision = row.get("reviewer_decision") or row.get("decision")
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('paper_id', '')))}</td>"
            f"<td>{html.escape(str(row.get('figure_id', '')))}</td>"
            f"<td>{html.escape(str(row.get('curve_legend_text') or row.get('curve_id', '')))}</td>"
            f"<td>{html.escape(str(row.get('sample_display_name', '')))}</td>"
            f"<td>{html.escape(str(row.get('fibre_outcome', '')))}</td>"
            f"<td>{html.escape(str(decision))}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>Rheology extraction audit</title>
<style>body{{font-family:Arial,sans-serif;margin:32px;color:#17201d}} table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #cdd6d0;padding:8px;text-align:left}} th{{background:#edf3ef}} @media print{{button{{display:none}}}}</style>
</head><body><button onclick=\"window.print()\">Save as PDF</button><h1>{html.escape(detail['run']['name'])}</h1>
<p>{len(detail['results'])} extracted rows, {len(detail['review_queue'])} review candidates.</p>
<table><thead><tr><th>Paper</th><th>Figure</th><th>Curve</th><th>Sample</th><th>Fibre outcome</th><th>Decision</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>"""


def _csv_value(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return "" if value is None else str(value)

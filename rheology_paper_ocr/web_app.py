from __future__ import annotations

import csv
import html
import io
import shutil
import tempfile
import threading
import webbrowser
import zipfile
from pathlib import Path
from typing import Any

from rheology_paper_ocr.pipeline import run_pipeline
from rheology_paper_ocr.web_store import LocalRunStore, read_json


MAX_ARCHIVE_PDFS = 1_000
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024


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

    class StartRunRequest(BaseModel):
        api_key: str | None = None

    globals()["DecisionRequest"] = DecisionRequest
    globals()["StartRunRequest"] = StartRunRequest

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
        for upload in files:
            source_name = Path(upload.filename or "paper.pdf").name
            suffix = Path(source_name).suffix.lower()
            if suffix == ".pdf":
                saved.append(_save_pdf(upload.file, input_dir, source_name, len(saved) + 1))
            elif suffix == ".zip":
                saved.extend(_extract_pdfs_from_zip(upload.file, input_dir, len(saved) + 1))
        if not saved:
            shutil.rmtree(input_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="Upload at least one PDF file or ZIP containing PDFs.")
        return {"run": _run_summary(store.get_run(run["id"])), "files": saved}

    @app.post("/api/runs/{run_id}/start")
    def start_run(run_id: str, request: StartRunRequest):
        try:
            run = store.get_run(run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found.")
        if run["status"] == "running":
            raise HTTPException(status_code=409, detail="Run is already in progress.")
        if not _pdf_input_files(Path(run["input_dir"])):
            raise HTTPException(status_code=400, detail="No PDF inputs are available for this run.")
        api_key = request.api_key.strip() if request.api_key else None
        if not api_key:
            raise HTTPException(status_code=400, detail="Enter an OpenRouter API key before starting extraction.")
        store.update_status(run_id, "running")
        thread = threading.Thread(target=_run_job, args=(store, run_id, api_key), daemon=True)
        thread.start()
        return _run_summary(store.get_run(run_id))

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        try:
            return _run_detail(store.get_run(run_id), store)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found.")

    @app.delete("/api/runs/{run_id}", status_code=204)
    def delete_run(run_id: str):
        try:
            run = store.get_run(run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found.")
        if run["status"] == "running":
            raise HTTPException(status_code=409, detail="An active run cannot be deleted.")
        run_dir = Path(run["input_dir"]).parent
        expected_root = (store.data_dir / "runs").resolve()
        if run_dir.resolve().parent != expected_root:
            raise HTTPException(status_code=403, detail="Invalid local run path.")
        store.delete_run(run_id)
        shutil.rmtree(run_dir, ignore_errors=True)

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
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("The browser app only supports a loopback host so uploaded papers and API keys remain local.")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    uvicorn.run(create_app(data_dir), host=host, port=port, log_level="info")


def _run_job(store: LocalRunStore, run_id: str, api_key: str | None = None) -> None:
    run = store.get_run(run_id)
    try:
        output_dir = Path(run["output_dir"])
        run_pipeline(
            Path(run["input_dir"]),
            output_dir,
            api_key=api_key,
            resume=(output_dir / "manifest.json").exists(),
        )
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
        "paper_count": len(manifest) or len(_pdf_input_files(Path(run["input_dir"]))),
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
        "progress": _run_progress(run, manifest),
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


def _save_pdf(source, input_dir: Path, source_name: str, index: int) -> str:
    target = input_dir / f"{index:03d}-{Path(source_name).name}"
    with target.open("wb") as handle:
        shutil.copyfileobj(source, handle)
    return target.name


def _pdf_input_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")


def _run_progress(run: dict, manifest: list[dict]) -> dict:
    statuses_by_source = {Path(item.get("source_pdf", "")).name: item for item in manifest}
    papers = []
    for index, path in enumerate(_pdf_input_files(Path(run["input_dir"])), start=1):
        status = statuses_by_source.get(path.name, {})
        papers.append(
            {
                "paper_id": status.get("paper_id", f"paper_{index:04d}"),
                "name": _display_input_name(path.name),
                "status": status.get("status", "queued"),
            }
        )
    completed_statuses = {"completed", "completed_no_findings"}
    return {
        "total": len(papers),
        "completed": sum(item["status"] in completed_statuses for item in papers),
        "processing": sum(item["status"] == "started" for item in papers),
        "queued": sum(item["status"] == "queued" for item in papers),
        "papers": papers,
    }


def _display_input_name(name: str) -> str:
    prefix, separator, remainder = name.partition("-")
    return remainder if separator and len(prefix) == 3 and prefix.isdigit() else name


def _extract_pdfs_from_zip(source, input_dir: Path, start_index: int) -> list[str]:
    try:
        with tempfile.TemporaryFile() as archive_file:
            source.seek(0)
            shutil.copyfileobj(source, archive_file)
            archive_file.seek(0)
            with zipfile.ZipFile(archive_file) as archive:
                members = [
                    member
                    for member in archive.infolist()
                    if not member.is_dir() and Path(member.filename).suffix.lower() == ".pdf"
                ]
                total_size = sum(member.file_size for member in members)
                if not members:
                    raise ValueError("The ZIP archive does not contain any PDF files.")
                if len(members) > MAX_ARCHIVE_PDFS:
                    raise ValueError(f"The ZIP archive contains more than {MAX_ARCHIVE_PDFS} PDF files.")
                if total_size > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                    raise ValueError("The ZIP archive expands beyond the 2 GB local upload limit.")
                saved = []
                for offset, member in enumerate(members):
                    source_name = Path(member.filename).name or "paper.pdf"
                    with archive.open(member) as pdf:
                        saved.append(_save_pdf(pdf, input_dir, source_name, start_index + offset))
                return saved
    except (zipfile.BadZipFile, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc) or "The uploaded ZIP archive is invalid.") from exc

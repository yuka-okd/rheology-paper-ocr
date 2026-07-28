from __future__ import annotations

from pathlib import Path

import typer

from rheology_paper_ocr.pipeline import load_saved_results, resume_pipeline, run_pipeline
from rheology_paper_ocr.report import write_reports
from rheology_paper_ocr.web_app import serve_local_app


app = typer.Typer(help="Extract rheology chart and fibre outcome evidence from PDFs.")


@app.command()
def run(
    pdf_dir: Path,
    out: Path = typer.Option(Path("outputs/run"), "--out", help="Output run directory."),
    model: str | None = typer.Option(None, "--model", help="OpenRouter model ID."),
    max_papers: int | None = typer.Option(None, "--max-papers", help="Maximum number of PDFs to process."),
    text_only: bool = typer.Option(False, "--text-only", help="Skip vision calls and extract from text/captions only."),
    resume: bool = typer.Option(False, "--resume", help="Reuse completed papers with unchanged source hashes."),
    docling_figures: bool = typer.Option(
        True,
        "--docling-figures/--no-docling-figures",
        help="Use local Docling figure crops when the optional dependency is installed.",
    ),
    successful_fibres_only: bool = typer.Option(
        False,
        "--successful-fibres-only",
        help="Emit only formulations with explicit formed-fibre evidence.",
    ),
    screen_reviews: bool = typer.Option(
        True,
        "--screen-reviews/--include-reviews",
        help="Skip articles explicitly labelled as reviews before vision extraction.",
    ),
):
    """Run the extraction pipeline over a directory of PDFs."""
    results = run_pipeline(
        pdf_dir=pdf_dir,
        out_dir=out,
        max_papers=max_papers,
        model=model,
        text_only=text_only,
        resume=resume,
        use_docling_figures=docling_figures,
        successful_fibres_only=successful_fibres_only,
        screen_reviews=screen_reviews,
    )
    typer.echo(f"Wrote {len(results)} extracted rows to {out}")


@app.command()
def inspect(pdf_path: Path, out: Path = typer.Option(Path("debug/inspect"), "--out")):
    """Inspect one PDF with verbose artifacts."""
    results = run_pipeline(pdf_dir=pdf_path, out_dir=out, max_papers=1)
    typer.echo(f"Wrote {len(results)} extracted rows to {out}")


@app.command()
def report(run_dir: Path):
    """Regenerate reports from saved run results."""
    results = load_saved_results(run_dir)
    write_reports(run_dir, results)
    typer.echo(f"Regenerated reports for {len(results)} extracted rows in {run_dir}")


@app.command()
def serve(
    data_dir: Path = typer.Option(Path.home() / ".rheology-paper-ocr", "--data-dir", help="Local sessions, uploaded PDFs, and run artifacts."),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address. Use the default for local-only access."),
    port: int = typer.Option(8787, "--port", min=1, max=65535, help="Local browser port."),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open the local browser application automatically."),
):
    """Launch the local browser workflow for upload, review, and export."""
    serve_local_app(data_dir=data_dir, host=host, port=port, open_browser=open_browser)


@app.command()
def resume(
    run_dir: Path,
    model: str | None = typer.Option(None, "--model", help="OpenRouter model ID."),
    text_only: bool = typer.Option(False, "--text-only", help="Skip vision calls and extract from text/captions only."),
    docling_figures: bool = typer.Option(
        True,
        "--docling-figures/--no-docling-figures",
        help="Use local Docling figure crops when the optional dependency is installed.",
    ),
    successful_fibres_only: bool = typer.Option(
        False,
        "--successful-fibres-only",
        help="Emit only formulations with explicit formed-fibre evidence.",
    ),
    screen_reviews: bool = typer.Option(
        True,
        "--screen-reviews/--include-reviews",
        help="Skip articles explicitly labelled as reviews before vision extraction.",
    ),
):
    """Resume incomplete papers recorded in a run manifest."""
    results = resume_pipeline(
        run_dir,
        model=model,
        text_only=text_only,
        use_docling_figures=docling_figures,
        successful_fibres_only=successful_fibres_only,
        screen_reviews=screen_reviews,
    )
    typer.echo(f"Resumed run with {len(results)} extracted rows in {run_dir}")


def main():
    app()


if __name__ == "__main__":
    main()

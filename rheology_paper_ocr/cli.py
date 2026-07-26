from __future__ import annotations

from pathlib import Path

import typer

from rheology_paper_ocr.pipeline import load_saved_results, resume_pipeline, run_pipeline
from rheology_paper_ocr.report import write_reports


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
def resume(
    run_dir: Path,
    model: str | None = typer.Option(None, "--model", help="OpenRouter model ID."),
    text_only: bool = typer.Option(False, "--text-only", help="Skip vision calls and extract from text/captions only."),
    docling_figures: bool = typer.Option(
        True,
        "--docling-figures/--no-docling-figures",
        help="Use local Docling figure crops when the optional dependency is installed.",
    ),
):
    """Resume incomplete papers recorded in a run manifest."""
    results = resume_pipeline(
        run_dir,
        model=model,
        text_only=text_only,
        use_docling_figures=docling_figures,
    )
    typer.echo(f"Resumed run with {len(results)} extracted rows in {run_dir}")


def main():
    app()


if __name__ == "__main__":
    main()

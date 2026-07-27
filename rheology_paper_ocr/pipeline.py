from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from rheology_paper_ocr.extract_with_llm import extract_paper_with_llm
from rheology_paper_ocr.chart_geometry import analyze_chart_geometry
from rheology_paper_ocr.docling_figures import (
    DoclingFigureError,
    LocalizedFigure,
    localize_figures,
    localize_native_chart_graphics,
    page_is_explicitly_non_rheology,
    select_chart_figures,
)
from rheology_paper_ocr.openrouter_client import OpenRouterClient
from rheology_paper_ocr.native_graphics import inspect_native_graphics
from rheology_paper_ocr.pdf_extract import RHEOLOGY_KEYWORDS, discover_pdfs, extract_text_and_pages, file_sha256
from rheology_paper_ocr.report import write_reports
from rheology_paper_ocr.review import apply_decision_policy, paper_review_candidate
from rheology_paper_ocr.rheology_analysis import is_shear_rate_axis, series_quality_warnings, summarize_series
from rheology_paper_ocr.schemas import DigitizedSeries, JoinedResult, ReviewCandidate, SourcePaper


COMPLETED_STATUSES = {"completed", "completed_no_findings"}
SUCCESSFUL_FIBRE_OUTCOMES = {"formed fibres", "formed beaded fibres"}


def _paper_index_from_id(paper_id: str) -> int:
    return int(paper_id.split("_")[-1])


def completion_status(text: str, findings_count: int) -> str:
    if findings_count:
        return "completed"
    lowered = text.lower()
    if any(keyword in lowered for keyword in RHEOLOGY_KEYWORDS):
        return "completed_no_findings"
    return "completed"


def is_review_article(text: str) -> bool:
    """Detect articles labelled as reviews in the opening document matter."""
    opening = text[:6000].lower()
    return "\nreview\n" in opening or "review article" in opening or "literature review" in opening


def load_saved_results(out_dir: Path) -> list[JoinedResult]:
    results_path = out_dir / "results.json"
    if not results_path.exists():
        return []
    data = json.loads(results_path.read_text(encoding="utf-8"))
    return [JoinedResult.model_validate(item) for item in data]


def load_manifest(out_dir: Path) -> list[dict]:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_saved_reviews(out_dir: Path) -> list[ReviewCandidate]:
    path = out_dir / "review_queue.json"
    if not path.exists():
        return []
    return [ReviewCandidate.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def _write_manifest(out_dir: Path, manifest: list[dict]) -> None:
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _page_number(image_path: Path) -> int | None:
    try:
        return int(image_path.stem.rsplit("_", maxsplit=1)[-1])
    except ValueError:
        return None


def _relative_to_run(path: Path, out_dir: Path) -> str:
    try:
        return str(path.relative_to(out_dir))
    except ValueError:
        return str(path)


def _localize_chart_figures(
    pdf_path: Path,
    image_paths: list[Path],
    paper_dir: Path,
    use_docling_figures: bool,
) -> tuple[list[LocalizedFigure], str | None, str]:
    page_numbers = [page for image_path in image_paths if (page := _page_number(image_path)) is not None]
    if not use_docling_figures:
        return _localize_native_chart_graphics(pdf_path, page_numbers, paper_dir, None)
    try:
        return localize_figures(pdf_path, page_numbers, paper_dir), None, "docling"
    except DoclingFigureError as exc:
        # Figure crops improve accuracy, but a missing optional local model must
        # never block a paid extraction run.
        return _localize_native_chart_graphics(pdf_path, page_numbers, paper_dir, str(exc))


def _localize_native_chart_graphics(
    pdf_path: Path,
    page_numbers: list[int],
    paper_dir: Path,
    warning: str | None,
) -> tuple[list[LocalizedFigure], str | None, str]:
    try:
        figures = localize_native_chart_graphics(pdf_path, page_numbers, paper_dir)
        return figures, warning, "native_graphics" if figures else "none"
    except Exception as exc:
        detail = f"native graphic localization failed: {exc}"
        return [], detail if warning is None else f"{warning}; {detail}", "none"


def _extraction_requests(
    image_paths: list[Path],
    figures: list[LocalizedFigure],
    text_only: bool,
    localized_figures: list[LocalizedFigure] | None = None,
) -> list[tuple[Path | None, LocalizedFigure | None]]:
    if text_only:
        return [(None, None)]
    figures_by_page: dict[int, list[LocalizedFigure]] = defaultdict(list)
    for figure in figures:
        figures_by_page[figure.page].append(figure)
    localized_by_page: dict[int, list[LocalizedFigure]] = defaultdict(list)
    for figure in localized_figures or figures:
        localized_by_page[figure.page].append(figure)

    requests: list[tuple[Path | None, LocalizedFigure | None]] = []
    for image_path in image_paths:
        page = _page_number(image_path)
        page_figures = figures_by_page.get(page or -1, [])
        if page_figures:
            requests.extend((image_path, figure) for figure in page_figures)
        elif page_is_explicitly_non_rheology(localized_by_page.get(page or -1, [])):
            continue
        else:
            # Preserve the full-page pass when captions do not identify a chart.
            requests.append((image_path, None))
    return requests


def _request_suffix(image_path: Path | None, target_figure: LocalizedFigure | None) -> str:
    if image_path is None:
        return "text_only"
    if target_figure is None:
        return image_path.stem
    crop_name = target_figure.crop_path.stem
    return f"{image_path.stem}_{crop_name}"


def _skipped_page_numbers(
    image_paths: list[Path],
    requests: list[tuple[Path | None, LocalizedFigure | None]],
) -> list[int]:
    requested = {path for path, _ in requests if path is not None}
    skipped = []
    for path in image_paths:
        page = _page_number(path)
        if path not in requested and page is not None:
            skipped.append(page)
    return skipped


def _group_results(results: list[JoinedResult]) -> dict[str, list[JoinedResult]]:
    grouped: dict[str, list[JoinedResult]] = defaultdict(list)
    for result in results:
        grouped[result.paper_id].append(result)
    return dict(grouped)


def _group_reviews(reviews: list[ReviewCandidate]) -> dict[str, list[ReviewCandidate]]:
    grouped: dict[str, list[ReviewCandidate]] = defaultdict(list)
    for candidate in reviews:
        grouped[candidate.paper_id].append(candidate)
    return dict(grouped)


def _flatten_results(results_by_paper: dict[str, list[JoinedResult]], papers: list[SourcePaper]) -> list[JoinedResult]:
    return [result for paper in papers for result in results_by_paper.get(paper.paper_id, [])]


def _flatten_reviews(reviews_by_paper: dict[str, list[ReviewCandidate]], papers: list[SourcePaper]) -> list[ReviewCandidate]:
    return [candidate for paper in papers for candidate in reviews_by_paper.get(paper.paper_id, [])]


def _finding_results(extractions, paper: SourcePaper) -> list[JoinedResult]:
    results: list[JoinedResult] = []
    for extraction in extractions:
        for finding in extraction.findings:
            series = DigitizedSeries(
                curve_id=finding.curve_id,
                visual_label=finding.curve_visual_label,
                legend_text=finding.curve_legend_text,
                points=finding.points,
                confidence=finding.confidence,
                warnings=finding.warnings,
            )
            summary = summarize_series(series, x_axis_label=finding.x_axis_label)
            warnings = list(finding.warnings) + series_quality_warnings(series, x_axis_label=finding.x_axis_label)
            if not finding.sample.evidence_text:
                warnings.append("sample link lacks direct evidence")
            if not finding.curve_visual_label and not finding.curve_legend_text:
                warnings.append("curve lacks a visual or legend identifier")

            confidence = finding.confidence
            if confidence == "high" and warnings:
                confidence = "medium"
            if (
                summary.rheology_class == "unclear"
                and confidence == "medium"
                and (finding.x_axis_label is None or is_shear_rate_axis(finding.x_axis_label))
            ):
                confidence = "low"

            results.append(
                JoinedResult(
                    paper_id=paper.paper_id,
                    source_pdf=str(paper.path),
                    figure_id=finding.figure_id,
                    page=finding.page,
                    chart_crop_path=finding.chart_crop_path,
                    curve_id=finding.curve_id,
                    curve_visual_label=finding.curve_visual_label,
                    curve_legend_text=finding.curve_legend_text,
                    sample_id=finding.sample.sample_id,
                    sample_display_name=finding.sample.sample_display_name,
                    sample_composition=finding.sample.sample_composition,
                    x_axis_label=finding.x_axis_label,
                    x_axis_unit=finding.x_axis_unit,
                    x_axis_scale=finding.x_axis_scale,
                    y_axis_label=finding.y_axis_label,
                    y_axis_unit=finding.y_axis_unit,
                    y_axis_scale=finding.y_axis_scale,
                    start_x=summary.start_x,
                    start_y=summary.start_y,
                    end_x=summary.end_x,
                    end_y=summary.end_y,
                    fold_change=summary.fold_change,
                    loglog_slope=summary.loglog_slope,
                    rheology_class=summary.rheology_class,
                    fibre_outcome=finding.fibre_outcome.outcome,
                    fibre_evidence_text=finding.fibre_outcome.evidence_text,
                    fibre_evidence_source=finding.fibre_outcome.evidence_source,
                    confidence=confidence,
                    warnings=warnings,
                )
            )
    return results


def _is_resumable_complete(status: dict | None, sha256: str, paper_dir: Path) -> bool:
    return bool(
        status
        and status.get("status") in COMPLETED_STATUSES
        and status.get("sha256") == sha256
        and (paper_dir / "results.json").exists()
    )


def _write_paper_results(paper_dir: Path, results: list[JoinedResult]) -> None:
    (paper_dir / "results.json").write_text(
        json.dumps([result.model_dump(mode="json") for result in results], indent=2),
        encoding="utf-8",
    )


def run_pipeline(
    pdf_dir: Path,
    out_dir: Path,
    max_papers: int | None = None,
    model: str | None = None,
    text_only: bool = False,
    resume: bool = False,
    use_docling_figures: bool = True,
    successful_fibres_only: bool = False,
    screen_reviews: bool = True,
) -> list[JoinedResult]:
    papers = discover_pdfs(pdf_dir)
    if max_papers is not None:
        papers = papers[:max_papers]
    return _run_papers(
        papers,
        out_dir=out_dir,
        model=model,
        text_only=text_only,
        resume=resume,
        use_docling_figures=use_docling_figures,
        successful_fibres_only=successful_fibres_only,
        screen_reviews=screen_reviews,
    )


def resume_pipeline(
    out_dir: Path,
    model: str | None = None,
    text_only: bool = False,
    use_docling_figures: bool = True,
    successful_fibres_only: bool = False,
    screen_reviews: bool = True,
) -> list[JoinedResult]:
    manifest = load_manifest(out_dir)
    papers = [
        SourcePaper(paper_id=item["paper_id"], path=Path(item["source_pdf"]))
        for item in manifest
        if item.get("source_pdf") and Path(item["source_pdf"]).exists()
    ]
    if not papers:
        raise ValueError(f"No available source PDFs in {out_dir / 'manifest.json'}")
    return _run_papers(
        papers,
        out_dir=out_dir,
        model=model,
        text_only=text_only,
        resume=True,
        use_docling_figures=use_docling_figures,
        successful_fibres_only=successful_fibres_only,
        screen_reviews=screen_reviews,
    )


def _run_papers(
    papers: list[SourcePaper],
    out_dir: Path,
    model: str | None,
    text_only: bool,
    resume: bool,
    use_docling_figures: bool,
    successful_fibres_only: bool,
    screen_reviews: bool,
) -> list[JoinedResult]:
    load_dotenv()
    out_dir.mkdir(parents=True, exist_ok=True)
    papers_dir = out_dir / "papers"
    papers_dir.mkdir(exist_ok=True)

    previous_statuses = {item.get("paper_id"): item for item in load_manifest(out_dir)} if resume else {}
    results_by_paper = _group_results(load_saved_results(out_dir)) if resume else {}
    reviews_by_paper = _group_reviews(load_saved_reviews(out_dir)) if resume else {}
    if not papers:
        _write_manifest(out_dir, [])
        write_reports(out_dir, [], [])
        return []
    client = OpenRouterClient(model=model)
    manifest: list[dict] = []

    for paper in papers:
        paper_dir = papers_dir / paper.paper_id
        llm_dir = paper_dir / "llm"
        sha256 = file_sha256(paper.path)
        previous_status = previous_statuses.get(paper.paper_id)
        if resume and _is_resumable_complete(previous_status, sha256, paper_dir):
            manifest.append(previous_status)
            if paper.paper_id not in results_by_paper:
                results_by_paper[paper.paper_id] = [
                    JoinedResult.model_validate(item)
                    for item in json.loads((paper_dir / "results.json").read_text(encoding="utf-8"))
                ]
            continue

        llm_dir.mkdir(parents=True, exist_ok=True)
        status = {
            "paper_id": paper.paper_id,
            "source_pdf": str(paper.path),
            "sha256": sha256,
            "status": "started",
            "paper_number": _paper_index_from_id(paper.paper_id),
        }
        try:
            text, image_paths = extract_text_and_pages(paper.path, paper_dir)
            try:
                native_graphics = inspect_native_graphics(
                    paper.path,
                    [page for image_path in image_paths if (page := _page_number(image_path)) is not None],
                )
                (paper_dir / "native_graphics.json").write_text(
                    json.dumps([page.to_dict() for page in native_graphics], indent=2),
                    encoding="utf-8",
                )
            except Exception:
                # This is diagnostic evidence only; standard page extraction remains the fallback.
                native_graphics = []
            if screen_reviews and is_review_article(text):
                results_by_paper[paper.paper_id] = []
                reviews_by_paper[paper.paper_id] = []
                _write_paper_results(paper_dir, [])
                status.update(
                    {
                        "status": "completed_no_findings",
                        "screened_out_reason": "document labelled as review article",
                        "findings": 0,
                        "result_path": _relative_to_run(paper_dir / "results.json", out_dir),
                    }
                )
                manifest.append(status)
                _write_manifest(out_dir, manifest)
                write_reports(out_dir, _flatten_results(results_by_paper, papers), _flatten_reviews(reviews_by_paper, papers))
                continue
            extractions = []
            localized_figures, localization_warning, localization_source = _localize_chart_figures(
                paper.path,
                image_paths,
                paper_dir,
                use_docling_figures=use_docling_figures and not text_only,
            )
            figures = select_chart_figures(localized_figures)
            requests = _extraction_requests(
                image_paths,
                figures,
                text_only=text_only,
                localized_figures=localized_figures,
            )
            for image_path, target_figure in requests:
                page_suffix = _request_suffix(image_path, target_figure)
                attached_images = [] if image_path is None else [image_path]
                chart_crop_path = None if image_path is None else _relative_to_run(image_path, out_dir)
                if target_figure is not None:
                    attached_images.append(target_figure.crop_path)
                    chart_crop_path = _relative_to_run(target_figure.crop_path, out_dir)
                    if target_figure.native_graphic_path and target_figure.native_graphic_path != target_figure.crop_path:
                        attached_images.append(target_figure.native_graphic_path)
                chart_geometry = None
                if target_figure is not None:
                    try:
                        chart_geometry = analyze_chart_geometry(target_figure.crop_path).to_dict()
                        geometry_dir = paper_dir / "chart_geometry"
                        geometry_dir.mkdir(exist_ok=True)
                        (geometry_dir / f"{page_suffix}.json").write_text(json.dumps(chart_geometry, indent=2), encoding="utf-8")
                    except Exception:
                        pass
                extraction = extract_paper_with_llm(
                    client,
                    paper.paper_id,
                    str(paper.path),
                    text,
                    attached_images,
                    raw_response_path=llm_dir / f"{page_suffix}_raw_response.json",
                    text_only=text_only,
                    page_number=None if image_path is None else _page_number(image_path),
                    chart_crop_path=chart_crop_path,
                    target_figure_id=None if target_figure is None else target_figure.figure_id,
                    target_figure_caption=None if target_figure is None else target_figure.caption,
                    successful_fibres_only=successful_fibres_only,
                    chart_geometry=chart_geometry,
                )
                (llm_dir / f"{page_suffix}_extraction.json").write_text(
                    extraction.model_dump_json(indent=2),
                    encoding="utf-8",
                )
                (llm_dir / f"{page_suffix}_model_route.json").write_text(
                    json.dumps(
                        {
                            "primary_model": getattr(client, "model", model),
                            "fallback_model": getattr(client, "fallback_model", None),
                            "model_used": getattr(client, "last_model_used", model),
                            "attempts": getattr(client, "last_attempts", []),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                extractions.append(extraction)

            all_paper_results, paper_reviews = apply_decision_policy(_finding_results(extractions, paper))
            paper_results = all_paper_results
            excluded_non_successful = 0
            if successful_fibres_only:
                filtered_results = [
                    result for result in paper_results if result.fibre_outcome in SUCCESSFUL_FIBRE_OUTCOMES
                ]
                excluded_non_successful = len(paper_results) - len(filtered_results)
                paper_results = filtered_results
            if not all_paper_results and any(extraction.has_rheology_chart for extraction in extractions):
                warnings = [warning for extraction in extractions for warning in extraction.paper_warnings]
                paper_reviews.append(
                    paper_review_candidate(
                        paper.paper_id,
                        str(paper.path),
                        warnings or ["rheology chart detected but no series met the evidence policy"],
                    )
                )
            results_by_paper[paper.paper_id] = paper_results
            reviews_by_paper[paper.paper_id] = paper_reviews
            _write_paper_results(paper_dir, paper_results)
            status.update(
                {
                    "status": completion_status(text=text, findings_count=len(paper_results)),
                    "images_sent": [] if text_only else [str(path) for path in image_paths],
                    "docling_figures": [figure.to_dict() for figure in figures],
                    "docling_all_figures": [figure.to_dict() for figure in localized_figures],
                    "skipped_candidate_pages": _skipped_page_numbers(image_paths, requests),
                    "docling_warning": localization_warning,
                    "figure_localization_source": localization_source,
                    "text_only": text_only,
                    "successful_fibres_only": successful_fibres_only,
                    "excluded_non_successful_findings": excluded_non_successful,
                    "review_candidates": len(paper_reviews),
                    "findings": len(paper_results),
                    "result_path": _relative_to_run(paper_dir / "results.json", out_dir),
                }
            )
        except Exception as exc:
            results_by_paper.pop(paper.paper_id, None)
            status.update({"status": "failed", "error": str(exc)})
        manifest.append(status)
        _write_manifest(out_dir, manifest)
        write_reports(out_dir, _flatten_results(results_by_paper, papers), _flatten_reviews(reviews_by_paper, papers))

    all_results = _flatten_results(results_by_paper, papers)
    _write_manifest(out_dir, manifest)
    write_reports(out_dir, all_results, _flatten_reviews(reviews_by_paper, papers))
    return all_results

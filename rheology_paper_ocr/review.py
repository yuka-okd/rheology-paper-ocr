from __future__ import annotations

from rheology_paper_ocr.schemas import JoinedResult, ReviewCandidate


BLOCKING_WARNING_TERMS = (
    "sample link lacks direct evidence",
    "curve lacks a visual or legend identifier",
    "fibre outcome lacks text evidence",
    "fibre outcome is supported only by group-level evidence",
    "unjoined fibre outcome",
    "fewer than three digitized points",
    "duplicate x values",
    "non-positive digitized points",
)


def review_reasons(result: JoinedResult) -> list[str]:
    reasons = []
    if result.confidence in {"low", "unclear"}:
        reasons.append(f"result confidence is {result.confidence}")
    if result.fibre_outcome == "unclear":
        reasons.append("fibre outcome is unresolved")
    for warning in result.warnings:
        if any(term in warning.lower() for term in BLOCKING_WARNING_TERMS):
            reasons.append(warning)
    return list(dict.fromkeys(reasons))


def apply_decision_policy(results: list[JoinedResult]) -> tuple[list[JoinedResult], list[ReviewCandidate]]:
    reviewed_results = []
    candidates = []
    for result in results:
        reasons = review_reasons(result)
        updated = result.model_copy(update={"decision": "needs_review" if reasons else "accepted", "review_reasons": reasons})
        reviewed_results.append(updated)
        if reasons:
            candidates.append(
                ReviewCandidate(
                    paper_id=updated.paper_id,
                    source_pdf=updated.source_pdf,
                    figure_id=updated.figure_id,
                    page=updated.page,
                    chart_crop_path=updated.chart_crop_path,
                    curve_id=updated.curve_id,
                    sample_display_name=updated.sample_display_name,
                    fibre_outcome=updated.fibre_outcome,
                    confidence=updated.confidence,
                    reasons=reasons,
                )
            )
    return reviewed_results, candidates


def paper_review_candidate(paper_id: str, source_pdf: str, reasons: list[str]) -> ReviewCandidate:
    return ReviewCandidate(paper_id=paper_id, source_pdf=source_pdf, reasons=list(dict.fromkeys(reasons)))

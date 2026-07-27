from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from rheology_paper_ocr.openrouter_client import OpenRouterClient
from rheology_paper_ocr.schemas import PaperLLMExtraction


CONTEXT_KEYWORDS = (
    "rheolog",
    "viscos",
    "shear",
    "storage modulus",
    "loss modulus",
    "frequency sweep",
    "flow curve",
    "fig.",
    "figure",
    "electrosp",
    "fiber",
    "fibre",
    "nanofiber",
    "nanofibre",
)


def select_prompt_context(text: str, max_chars: int = 18000) -> str:
    if len(text) <= max_chars:
        return text

    windows: list[tuple[int, int]] = [(0, min(4000, len(text)))]
    lowered = text.lower()
    for keyword in CONTEXT_KEYWORDS:
        start = 0
        while len(windows) < 14:
            index = lowered.find(keyword, start)
            if index == -1:
                break
            windows.append((max(0, index - 900), min(len(text), index + 1800)))
            start = index + len(keyword)

    merged: list[tuple[int, int]] = []
    for start, end in sorted(windows):
        if not merged or start > merged[-1][1] + 200:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    chunks = []
    used = 0
    for start, end in merged:
        if used >= max_chars:
            break
        chunk = text[start:end]
        remaining = max_chars - used
        chunks.append(chunk[:remaining])
        used += min(len(chunk), remaining)
    return "\n\n[...]\n\n".join(chunks)


def build_extraction_prompt(
    paper_id: str,
    source_pdf: str,
    text: str,
    text_only: bool = False,
    attached_page: int | None = None,
    target_figure_id: str | None = None,
    target_figure_caption: str | None = None,
    successful_fibres_only: bool = False,
) -> str:
    clipped_text = select_prompt_context(text)
    mode_instruction = (
        "This is a text-only pass. Identify rheology figures and fibre outcomes from text, captions, and tables only. "
        "Do not invent chart points; leave points empty when chart data cannot be read from text."
        if text_only
        else "Use the attached page image to inspect one candidate chart page. If several images are attached, they show the same page: use the full page for context and any magnified detail crop for marker-to-legend mapping and point digitization."
    )
    target_instruction = (
        f"The target figure is {target_figure_id or 'the supplied crop'}. Its caption is: {target_figure_caption or 'not available'}. "
        "The final attached image is an exact crop of this target figure. Extract only this figure, not other charts visible in the full-page context image."
        if target_figure_id or target_figure_caption
        else ""
    )
    success_filter_instruction = (
        "Return findings only for series with explicit text evidence of formed fibres or formed beaded fibres. "
        "Omit failed, not-tested, and unclear series completely."
        if successful_fibres_only
        else ""
    )
    return f"""
You are extracting rheology and text-only fibre outcome evidence from a chemistry paper.

Paper ID: {paper_id}
Source PDF: {source_pdf}
Attached page: {attached_page if attached_page is not None else "none (text-only pass)"}

{mode_instruction} Use the text below for captions, legends,
sample definitions, formulation aliases, and fibre outcome evidence.
{target_instruction}
{success_filter_instruction}

Return ONLY valid JSON matching the provided schema.

Rules:
- Include one finding per rheology chart series/line when possible.
- When an image is attached, only report charts visible on that page and set page to the attached page number.
- Set x_axis_scale and y_axis_scale to "linear", "log", or "unclear". Keep points in ascending x order.
- Return the flat finding schema directly. Do not wrap findings in a chart object or add figure captions, chart type, or trend prose.
- A rheology chart includes viscosity/shear rate, shear stress, modulus, frequency sweep, or flow curve plots.
- This workflow is limited to bulk shear rheology. Do not report extensional/elongational viscosity, capillary-breakup, or filament-thinning plots, even when they are relevant to electrospinning.
- Treat an axis labelled "strain rate" or "extension rate" as out of scope unless it explicitly says "shear rate".
- Digitize at most three well-spaced approximate points per series: start, a turning point if present, and end.
- A concentration-viscosity plot is not a flow curve. Extract its points, axes, and series mapping, but do not call its increase "shear-thickening".
- Map each line to the sample/formulation using legend, caption, nearby text, methods, or tables.
- Fibre outcome is text-only. Do not infer fibre formation from SEM images.
- Fibre outcome must be "unclear" unless the text explicitly links that exact sample/formulation to fibre formation, beaded fibres, failure/no fibres, or not tested. Do not attach a broad molecular-weight or concentration rule to a borderline or differently named series.
- Evidence text should be a short exact or near-exact sentence/phrase from the paper text, caption, or table.
- If there are no rheology charts, return has_rheology_chart=false and findings=[].

Paper text:
{clipped_text}
""".strip()


def extract_paper_with_llm(
    client: OpenRouterClient,
    paper_id: str,
    source_pdf: str,
    text: str,
    image_paths: list[Path],
    raw_response_path: Path | None = None,
    text_only: bool = False,
    page_number: int | None = None,
    chart_crop_path: str | None = None,
    target_figure_id: str | None = None,
    target_figure_caption: str | None = None,
    successful_fibres_only: bool = False,
) -> PaperLLMExtraction:
    prompt = build_extraction_prompt(
        paper_id,
        source_pdf,
        text,
        text_only=text_only,
        attached_page=page_number,
        target_figure_id=target_figure_id,
        target_figure_caption=target_figure_caption,
        successful_fibres_only=successful_fibres_only,
    )
    raw = client.extract(prompt, image_paths, raw_response_path=raw_response_path)
    normalized = normalize_extraction_payload(
        raw,
        paper_id=paper_id,
        source_pdf=source_pdf,
        default_page=page_number,
        default_chart_crop_path=chart_crop_path,
    )
    return PaperLLMExtraction.model_validate(normalized)


def normalize_extraction_payload(
    payload: dict[str, Any],
    paper_id: str,
    source_pdf: str,
    default_page: int | None = None,
    default_chart_crop_path: str | None = None,
) -> dict[str, Any]:
    normalized = dict(payload)
    normalized.setdefault("paper_id", paper_id)
    normalized.setdefault("source_pdf", source_pdf)
    normalized.setdefault("has_rheology_chart", bool(normalized.get("findings")))
    normalized["findings"] = _normalize_findings(normalized.get("findings") or [])
    normalized["findings"].extend(_normalize_top_level_fibre_outcomes(normalized))
    normalized["findings"] = [
        _apply_finding_defaults(
            finding,
            default_page=default_page,
            default_chart_crop_path=default_chart_crop_path,
        )
        for finding in normalized["findings"]
    ]
    retained_findings = []
    dropped_non_rheology = 0
    for finding in normalized["findings"]:
        if _is_explicitly_non_rheology_finding(finding):
            dropped_non_rheology += 1
        else:
            retained_findings.append(finding)
    normalized["findings"] = retained_findings
    normalized.setdefault("paper_warnings", [])
    if dropped_non_rheology:
        normalized["paper_warnings"].append(
            f"dropped {dropped_non_rheology} explicitly non-rheology finding(s)"
        )
    return normalized


def _normalize_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for index, finding in enumerate(findings, start=1):
        if "curve_id" in finding:
            item = dict(finding)
            item["points"] = _normalize_points(item.get("points") or [])
            item.setdefault("confidence", "medium")
            item.setdefault("warnings", [])
            item.setdefault("sample", {})
            item["fibre_outcome"] = _normalize_fibre_payload(item.get("fibre_outcome") or item.get("fiber_outcome"))
            flattened.append(item)
            continue

        chart_id = finding.get("chart_id") or finding.get("figure_id") or f"chart_{index}"
        fibre_outcomes = finding.get("fibre_outcomes") or finding.get("fiber_outcomes") or []
        outcome_by_sample = {outcome.get("sample_id"): outcome for outcome in fibre_outcomes if outcome.get("sample_id")}
        series_items = finding.get("series") or []
        if not series_items and _looks_like_flat_finding(finding):
            flattened.append(_normalize_flat_finding(finding, chart_id=chart_id, index=index))
            continue
        for series_index, series in enumerate(finding.get("series") or [], start=1):
            sample_id = series.get("sample_id") or f"{chart_id}_series_{series_index}"
            fibre = outcome_by_sample.get(sample_id) or _series_fibre_payload(series)
            flattened.append(
                {
                    "figure_id": finding.get("figure_id") or chart_id,
                    "page": finding.get("page"),
                    "chart_crop_path": finding.get("chart_crop_path"),
                    "curve_id": series.get("curve_id") or sample_id,
                    "curve_visual_label": series.get("visual_label") or series.get("label"),
                    "curve_legend_text": series.get("legend_text") or series.get("label"),
                    "x_axis_label": _axis_label(finding.get("x_axis")),
                    "x_axis_unit": None,
                    "x_axis_scale": None,
                    "y_axis_label": _axis_label(finding.get("y_axis")),
                    "y_axis_unit": None,
                    "y_axis_scale": None,
                    "points": _normalize_points(series.get("points") or []),
                    "sample": {
                        "sample_id": sample_id,
                        "sample_display_name": series.get("label") or sample_id,
                        "sample_composition": series.get("composition") or series.get("notes"),
                        "evidence_text": series.get("notes"),
                        "evidence_source": "llm_text_extraction",
                        "confidence": "medium",
                    },
                    "fibre_outcome": _normalize_fibre_payload(fibre),
                    "confidence": "medium",
                    "warnings": ["normalized from chart-level LLM output"],
                }
            )
    return flattened


def _series_fibre_payload(series: dict[str, Any]) -> dict[str, Any]:
    outcome = series.get("fibre_outcome") or series.get("fiber_outcome") or series.get("outcome")
    evidence = series.get("fibre_evidence") or series.get("fiber_evidence") or series.get("evidence_text") or series.get("evidence")
    if not outcome and not evidence:
        return {}
    return {
        "outcome": outcome,
        "evidence_text": evidence,
        "evidence_source": series.get("evidence_source") or "paper_text",
        "confidence": series.get("confidence") or "medium",
    }


def _looks_like_flat_finding(finding: dict[str, Any]) -> bool:
    return any(
        key in finding
        for key in (
            "sample_id",
            "series_id",
            "rheology_points",
            "fibre_outcome",
            "fiber_outcome",
            "evidence_text",
        )
    )


def _normalize_flat_finding(finding: dict[str, Any], chart_id: str, index: int) -> dict[str, Any]:
    sample_id = finding.get("sample_id") or finding.get("series_id") or f"finding_{index}"
    fibre = finding.get("fibre_outcome") or finding.get("fiber_outcome")
    evidence_text = finding.get("evidence_text") or finding.get("evidence")
    fibre_evidence_text = (
        finding.get("fibre_evidence")
        or finding.get("fiber_evidence")
        or finding.get("fibre_evidence_text")
        or finding.get("fiber_evidence_text")
        or evidence_text
    )
    return {
        "figure_id": finding.get("figure_id") or finding.get("chart_id"),
        "page": finding.get("page"),
        "chart_crop_path": finding.get("chart_crop_path"),
        "curve_id": finding.get("curve_id") or finding.get("series_id") or sample_id,
        "curve_visual_label": finding.get("curve_visual_label") or finding.get("visual_label") or finding.get("series_id"),
        "curve_legend_text": finding.get("curve_legend_text") or finding.get("legend_text") or sample_id,
        "x_axis_label": _axis_label(finding.get("x_axis")) or finding.get("x_axis_label"),
        "x_axis_unit": finding.get("x_axis_unit"),
        "x_axis_scale": finding.get("x_axis_scale"),
        "y_axis_label": _axis_label(finding.get("y_axis")) or finding.get("y_axis_label"),
        "y_axis_unit": finding.get("y_axis_unit"),
        "y_axis_scale": finding.get("y_axis_scale"),
        "points": _normalize_points(
            finding.get("points") or finding.get("rheology_points") or finding.get("data_points") or []
        ),
        "sample": {
            "sample_id": sample_id,
            "sample_display_name": finding.get("sample_display_name") or sample_id,
            "sample_composition": (
                finding.get("sample_composition")
                or finding.get("formulation_alias")
                or finding.get("formulation")
                or finding.get("sample_description")
            ),
            "evidence_text": evidence_text,
            "evidence_source": finding.get("evidence_source") or "llm_text_extraction",
            "confidence": finding.get("confidence") or "medium",
        },
        "fibre_outcome": _normalize_fibre_payload(fibre, fibre_evidence_text),
        "confidence": finding.get("confidence") or "medium",
        "warnings": list(finding.get("warnings") or []) + ["normalized from flat LLM output"],
    }


def _normalize_top_level_fibre_outcomes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    outcomes = payload.get("fibre_outcomes") or payload.get("fiber_outcomes") or []
    normalized = []
    for index, outcome in enumerate(outcomes, start=1):
        if not isinstance(outcome, dict):
            continue
        sample_id = outcome.get("sample_id") or outcome.get("sample") or f"fibre_outcome_{index}"
        evidence_text = outcome.get("evidence_text") or outcome.get("evidence")
        normalized.append(
            {
                "figure_id": None,
                "page": outcome.get("page"),
                "chart_crop_path": None,
                "curve_id": f"fibre_outcome_{index}",
                "curve_visual_label": None,
                "curve_legend_text": None,
                "x_axis_label": None,
                "x_axis_unit": None,
                "x_axis_scale": None,
                "y_axis_label": None,
                "y_axis_unit": None,
                "y_axis_scale": None,
                "points": [],
                "sample": {
                    "sample_id": sample_id,
                    "sample_display_name": sample_id,
                    "sample_composition": outcome.get("sample_composition") or outcome.get("formulation_alias") or outcome.get("formulation"),
                    "evidence_text": evidence_text,
                    "evidence_source": outcome.get("evidence_source") or "paper_text",
                    "confidence": outcome.get("confidence") or "medium",
                },
                "fibre_outcome": _normalize_fibre_payload(
                    outcome.get("outcome") or outcome.get("fibre_outcome") or outcome.get("fiber_outcome"),
                    evidence_text,
                ),
                "confidence": outcome.get("confidence") or "medium",
                "warnings": ["unjoined fibre outcome"],
            }
        )
    return normalized


def _normalize_fibre_payload(outcome: Any, evidence_text: str | None = None) -> dict[str, Any]:
    if isinstance(outcome, dict):
        text = outcome.get("evidence_text") or outcome.get("evidence") or evidence_text
        return {
            "outcome": _normalize_fibre_outcome(
                outcome.get("outcome") or outcome.get("fibre_outcome") or outcome.get("fiber_outcome"),
                evidence_text=text,
            ),
            "evidence_text": text,
            "evidence_source": outcome.get("evidence_source") or "paper_text",
            "confidence": outcome.get("confidence") or "medium",
        }
    return {
        "outcome": _normalize_fibre_outcome(str(outcome) if outcome else None, evidence_text=evidence_text),
        "evidence_text": evidence_text,
        "evidence_source": "paper_text",
        "confidence": "medium" if outcome or evidence_text else "unclear",
    }


def _apply_finding_defaults(
    finding: dict[str, Any],
    default_page: int | None,
    default_chart_crop_path: str | None,
) -> dict[str, Any]:
    item = dict(finding)
    if item.get("page") is None and default_page is not None:
        item["page"] = default_page
    if item.get("chart_crop_path") is None and default_chart_crop_path is not None:
        item["chart_crop_path"] = default_chart_crop_path

    warnings = list(item.get("warnings") or [])
    item["points"], point_warnings = _validate_points(
        item.get("points") or [],
        x_axis_scale=item.get("x_axis_scale"),
        y_axis_scale=item.get("y_axis_scale"),
    )
    warnings.extend(point_warnings)
    fibre = item.get("fibre_outcome") or _normalize_fibre_payload(None)
    if fibre.get("outcome") not in {"unclear", "not tested"} and not fibre.get("evidence_text"):
        fibre = dict(fibre)
        fibre["outcome"] = "unclear"
        fibre["confidence"] = "unclear"
        warnings.append("fibre outcome lacks text evidence")
    item["fibre_outcome"] = fibre
    item["warnings"] = warnings
    return item


def _normalize_points(points: list[Any]) -> list[dict[str, float]]:
    normalized = []
    for point in points:
        if isinstance(point, dict) and "x" in point and "y" in point:
            if point["x"] is not None and point["y"] is not None:
                normalized.append({"x": point["x"], "y": point["y"]})
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            if point[0] is not None and point[1] is not None:
                normalized.append({"x": point[0], "y": point[1]})
    return normalized


def _validate_points(
    points: list[dict[str, Any]],
    x_axis_scale: str | None,
    y_axis_scale: str | None,
) -> tuple[list[dict[str, float]], list[str]]:
    """Keep only finite, scale-compatible coordinates and canonicalize x order."""
    validated: list[dict[str, float]] = []
    discarded = 0
    x_is_log = (x_axis_scale or "").lower() == "log"
    y_is_log = (y_axis_scale or "").lower() == "log"
    for point in points:
        try:
            x, y = float(point["x"]), float(point["y"])
        except (KeyError, TypeError, ValueError):
            discarded += 1
            continue
        if not math.isfinite(x) or not math.isfinite(y) or (x_is_log and x <= 0) or (y_is_log and y <= 0):
            discarded += 1
            continue
        validated.append({"x": x, "y": y})

    warnings = []
    if discarded:
        warnings.append(f"discarded {discarded} invalid or scale-incompatible chart point(s)")
    if len(validated) > 1 and validated != sorted(validated, key=lambda point: point["x"]):
        validated.sort(key=lambda point: point["x"])
        warnings.append("sorted chart points into ascending x order")
    if len({point["x"] for point in validated}) != len(validated):
        warnings.append("chart points contain duplicate x coordinates")
    return validated, warnings


def _axis_label(axis: Any) -> str | None:
    if axis is None:
        return None
    if isinstance(axis, str):
        return axis
    if isinstance(axis, dict):
        return axis.get("label") or axis.get("name")
    return str(axis)


def _is_explicitly_non_rheology_finding(finding: dict[str, Any]) -> bool:
    x_axis = str(finding.get("x_axis_label") or "").lower()
    axes = " ".join(
        str(finding.get(field) or "").lower()
        for field in ("x_axis_label", "y_axis_label")
    )
    if "diameter" in axes or "diam" in axes:
        return True
    if ("strain rate" in x_axis or "extension rate" in x_axis) and "shear" not in x_axis:
        return True
    return any(term in axes for term in ("extensional", "elongational", "capillary breakup", "filament thinning"))


def _normalize_fibre_outcome(outcome: str | None, evidence_text: str | None = None) -> str:
    evidence = (evidence_text or "").lower()
    if "couldn't be electrospun into nfs" in evidence or "could not be electrospun into nfs" in evidence:
        return "failed or no fibres"
    if not outcome:
        return "unclear"
    lowered = outcome.lower()
    if "bead-free" in lowered or "defect-free" in lowered:
        return "formed fibres"
    if "bead" in lowered or "defect" in lowered or "inhomogeneous" in lowered:
        return "formed beaded fibres"
    if "no" in lowered or "fail" in lowered:
        return "failed or no fibres"
    if "not tested" in lowered:
        return "not tested"
    if "fiber" in lowered or "fibre" in lowered:
        return "formed fibres"
    return "unclear"

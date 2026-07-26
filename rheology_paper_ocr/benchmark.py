from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import median
from typing import Any


def _canonical(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def _finding_keys(finding: dict[str, Any]) -> set[str]:
    sample = finding.get("sample") or {}
    fields = (
        finding.get("curve_id"),
        finding.get("curve_legend_text"),
        finding.get("curve_visual_label"),
        sample.get("sample_id"),
        sample.get("sample_display_name"),
    )
    return {_canonical(value) for value in fields if _canonical(value)}


def _find_matching_finding(reference: dict[str, Any], findings: list[dict[str, Any]]) -> dict[str, Any] | None:
    aliases = {_canonical(alias) for alias in reference["aliases"]}
    for finding in findings:
        if _finding_keys(finding) & aliases:
            return finding
    return None


def _ordered_points(finding: dict[str, Any]) -> list[dict[str, float]]:
    points = [point for point in finding.get("points") or [] if point.get("x", 0) > 0 and point.get("y", 0) > 0]
    return sorted(points, key=lambda point: point["x"])


def _log_error(observed: float, expected: float) -> float:
    return abs(math.log10(observed / expected))


def score_extraction(reference: dict[str, Any], extraction: dict[str, Any]) -> dict[str, Any]:
    findings = extraction.get("findings") or []
    expected_series = reference["series"]
    matches = [(series, _find_matching_finding(series, findings)) for series in expected_series]
    matched = [(series, finding) for series, finding in matches if finding is not None]

    x_errors: list[float] = []
    y_errors: list[float] = []
    slope_errors: list[float] = []
    comparable_anchors = 0
    axis_matches = 0
    details = []
    for series, finding in matches:
        if finding is None:
            details.append({"series": series["id"], "status": "missing"})
            continue
        expected_points = series["points"]
        observed_points = _ordered_points(finding)
        pairs = list(zip(observed_points[: len(expected_points)], expected_points))
        for observed, expected in pairs:
            x_errors.append(_log_error(observed["x"], expected["x"]))
            y_errors.append(_log_error(observed["y"], expected["y"]))
        comparable_anchors += len(pairs)
        if len(pairs) >= 2:
            observed_slope = math.log10(pairs[-1][0]["y"] / pairs[0][0]["y"]) / math.log10(
                pairs[-1][0]["x"] / pairs[0][0]["x"]
            )
            expected_slope = math.log10(pairs[-1][1]["y"] / pairs[0][1]["y"]) / math.log10(
                pairs[-1][1]["x"] / pairs[0][1]["x"]
            )
            slope_errors.append(abs(observed_slope - expected_slope))
        expected_axes = reference["axes"]
        axes_match = (
            finding.get("x_axis_scale") == expected_axes["x_scale"]
            and finding.get("y_axis_scale") == expected_axes["y_scale"]
            and expected_axes["x_label_contains"].lower() in (finding.get("x_axis_label") or "").lower()
            and expected_axes["y_label_contains"].lower() in (finding.get("y_axis_label") or "").lower()
        )
        axis_matches += int(axes_match)
        details.append(
            {
                "series": series["id"],
                "status": "matched",
                "points_compared": len(pairs),
                "axes_match": axes_match,
                "median_y_factor_error": round(10 ** median([_log_error(observed["y"], expected["y"]) for observed, expected in pairs]), 3)
                if pairs
                else None,
            }
        )

    expected_anchor_count = sum(len(series["points"]) for series in expected_series)
    return {
        "series_recall": round(len(matched) / len(expected_series), 3),
        "series_precision": round(len(matched) / len(findings), 3) if findings else 0.0,
        "axis_accuracy_on_matched_series": round(axis_matches / len(matched), 3) if matched else 0.0,
        "anchor_coverage": round(comparable_anchors / expected_anchor_count, 3),
        "median_x_factor_error": round(10 ** median(x_errors), 3) if x_errors else None,
        "median_y_factor_error": round(10 ** median(y_errors), 3) if y_errors else None,
        "anchors_within_1_5x_y": round(sum(error <= math.log10(1.5) for error in y_errors) / len(y_errors), 3) if y_errors else 0.0,
        "anchors_within_2x_y": round(sum(error <= math.log10(2) for error in y_errors) / len(y_errors), 3) if y_errors else 0.0,
        "median_loglog_slope_error": round(median(slope_errors), 3) if slope_errors else None,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a rheology extraction against a manually checked figure reference.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--extraction", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    extraction = json.loads(args.extraction.read_text(encoding="utf-8"))
    print(json.dumps(score_extraction(reference, extraction), indent=2))


if __name__ == "__main__":
    main()

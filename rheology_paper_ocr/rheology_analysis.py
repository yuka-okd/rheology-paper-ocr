from __future__ import annotations

import math

from rheology_paper_ocr.schemas import DigitizedSeries, SeriesSummary


def _ordered_points(series: DigitizedSeries) -> list[tuple[float, float]]:
    return sorted(
        ((point.x, point.y) for point in series.points if point.x > 0 and point.y > 0),
        key=lambda point: point[0],
    )


def series_quality_warnings(series: DigitizedSeries, x_axis_label: str | None = None) -> list[str]:
    warnings: list[str] = []
    valid_points = [(point.x, point.y) for point in series.points if point.x > 0 and point.y > 0]
    if len(valid_points) < 3:
        warnings.append("fewer than three digitized points; rheology class left unclear")
    if valid_points != sorted(valid_points, key=lambda point: point[0]):
        warnings.append("points were reordered by x value")
    if len({x for x, _ in valid_points}) != len(valid_points):
        warnings.append("duplicate x values in digitized points")
    if len(valid_points) != len(series.points):
        warnings.append("non-positive digitized points were excluded from rheology analysis")
    if x_axis_label and not is_shear_rate_axis(x_axis_label):
        warnings.append("x axis is not shear rate; flow-curve class left unclear")
    return warnings


def is_shear_rate_axis(x_axis_label: str | None) -> bool:
    if not x_axis_label:
        return False
    label = x_axis_label.lower()
    return "shear rate" in label or "shear-rate" in label or "shear_rate" in label


def classify_series(series: DigitizedSeries, x_axis_label: str | None = None) -> str:
    if x_axis_label and not is_shear_rate_axis(x_axis_label):
        return "unclear"
    points = _ordered_points(series)
    if len(points) < 3:
        return "unclear"

    y_values = [y for _, y in points]
    directions = []
    for previous, current in zip(y_values, y_values[1:]):
        change = (current - previous) / previous if previous else 0
        if abs(change) >= 0.15:
            directions.append(1 if change > 0 else -1)

    if directions and any(direction != directions[0] for direction in directions):
        return "non-monotonic"

    start_y = y_values[0]
    end_y = y_values[-1]
    if start_y <= 0:
        return "unclear"
    total_change = (end_y - start_y) / start_y

    if total_change <= -0.30:
        return "shear-thinning"
    if total_change >= 0.30:
        return "shear-thickening"
    if max(y_values) / min(y_values) <= 1.15:
        return "near-Newtonian or plateau"
    return "unclear"


def summarize_series(series: DigitizedSeries, x_axis_label: str | None = None) -> SeriesSummary:
    points = _ordered_points(series)
    summary = SeriesSummary(rheology_class=classify_series(series, x_axis_label=x_axis_label))
    if not points:
        return summary

    first_x, first_y = points[0]
    last_x, last_y = points[-1]
    summary.start_x = first_x
    summary.start_y = first_y
    summary.end_x = last_x
    summary.end_y = last_y
    if first_y:
        summary.fold_change = last_y / first_y
    if len(points) >= 2 and first_x != last_x and first_y > 0 and last_y > 0:
        denominator = math.log10(last_x) - math.log10(first_x)
        if denominator:
            summary.loglog_slope = (math.log10(last_y) - math.log10(first_y)) / denominator
    return summary

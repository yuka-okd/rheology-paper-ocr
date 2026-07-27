from rheology_paper_ocr.rheology_analysis import classify_series, series_quality_warnings, summarize_series
from rheology_paper_ocr.schemas import DataPoint, DigitizedSeries


def make_series(points):
    return DigitizedSeries(
        curve_id="curve_1",
        visual_label="blue",
        legend_text=None,
        points=[DataPoint(x=x, y=y) for x, y in points],
        confidence="high",
        warnings=[],
    )


def test_classifies_shear_thinning():
    series = make_series([(1, 1000), (10, 300), (100, 80)])

    assert classify_series(series) == "shear-thinning"


def test_classifies_shear_thickening():
    series = make_series([(1, 100), (10, 180), (100, 400)])

    assert classify_series(series) == "shear-thickening"


def test_classifies_plateau():
    series = make_series([(1, 100), (10, 96), (100, 104)])

    assert classify_series(series) == "near-Newtonian or plateau"


def test_classifies_non_monotonic():
    series = make_series([(1, 100), (10, 50), (100, 120)])

    assert classify_series(series) == "non-monotonic"


def test_summarizes_start_end_and_fold_change():
    series = make_series([(1, 1000), (100, 100)])

    summary = summarize_series(series)

    assert summary.start_x == 1
    assert summary.start_y == 1000
    assert summary.end_x == 100
    assert summary.end_y == 100
    assert summary.fold_change == 0.1


def test_retains_zero_start_on_a_linear_axis_for_summary_and_quality():
    series = make_series([(0, 2650), (4, 600), (80, 30)])

    summary = summarize_series(series, x_axis_label="Time")

    assert (summary.start_x, summary.start_y) == (0, 2650)
    assert (summary.end_x, summary.end_y) == (80, 30)
    assert summary.fold_change == 30 / 2650
    assert not any("excluded" in warning for warning in series_quality_warnings(series, x_axis_label="Time"))


def test_marks_two_point_series_as_unclear_for_classification():
    series = make_series([(1, 1000), (100, 100)])

    assert classify_series(series) == "unclear"
    assert any("fewer than three digitized points" in warning for warning in series_quality_warnings(series))


def test_sorts_points_by_x_before_classifying():
    series = make_series([(100, 80), (1, 1000), (10, 300)])

    assert classify_series(series) == "shear-thinning"
    assert "points were reordered by x value" in series_quality_warnings(series)


def test_leaves_concentration_viscosity_curve_unclassified():
    series = make_series([(0.1, 1), (1, 20), (10, 1000)])

    assert classify_series(series, x_axis_label="[PAA], % (w/w)") == "unclear"
    assert any(
        "x axis is not shear rate" in warning
        for warning in series_quality_warnings(series, x_axis_label="[PAA], % (w/w)")
    )

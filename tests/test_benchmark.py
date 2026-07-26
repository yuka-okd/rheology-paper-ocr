from rheology_paper_ocr.benchmark import score_extraction


def test_scores_matching_series_and_log_scale_points():
    reference = {
        "axes": {"x_label_contains": "rate", "y_label_contains": "viscosity", "x_scale": "log", "y_scale": "log"},
        "series": [
            {
                "id": "sample_a",
                "aliases": ["sample_a"],
                "points": [{"x": 1, "y": 100}, {"x": 10, "y": 10}, {"x": 100, "y": 1}],
            }
        ],
    }
    extraction = {
        "findings": [
            {
                "curve_id": "sample_a",
                "x_axis_label": "Shear rate",
                "y_axis_label": "Viscosity",
                "x_axis_scale": "log",
                "y_axis_scale": "log",
                "points": [{"x": 1, "y": 100}, {"x": 10, "y": 10}, {"x": 100, "y": 1}],
            }
        ]
    }

    score = score_extraction(reference, extraction)

    assert score["series_recall"] == 1.0
    assert score["axis_accuracy_on_matched_series"] == 1.0
    assert score["median_y_factor_error"] == 1.0


def test_matches_a_series_by_legend_when_curve_id_is_generic():
    reference = {
        "axes": {"x_label_contains": "rate", "y_label_contains": "viscosity", "x_scale": "log", "y_scale": "log"},
        "series": [{"id": "sample_a", "aliases": ["Sample A"], "points": [{"x": 1, "y": 10}]}],
    }
    extraction = {
        "findings": [
            {
                "curve_id": "series_1",
                "curve_legend_text": "Sample A",
                "x_axis_label": "Shear rate",
                "y_axis_label": "Viscosity",
                "x_axis_scale": "log",
                "y_axis_scale": "log",
                "points": [{"x": 1, "y": 10}],
            }
        ]
    }

    assert score_extraction(reference, extraction)["series_recall"] == 1.0

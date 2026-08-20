import json
from pathlib import Path

from rheology_paper_ocr.combined_plot import write_combined_plots
from rheology_paper_ocr.schemas import DataPoint, JoinedResult


def _result(**overrides) -> JoinedResult:
    defaults = dict(
        paper_id="paper_0001",
        source_pdf="paper.pdf",
        curve_id="curve_1",
        x_axis_label="Shear Rate",
        x_axis_unit="1/s",
        x_axis_scale="log",
        y_axis_label="Viscosity",
        y_axis_unit="Pa*s",
        y_axis_scale="log",
        points=[DataPoint(x=1, y=1000), DataPoint(x=10, y=300), DataPoint(x=100, y=80)],
        fibre_outcome="formed fibres",
    )
    defaults.update(overrides)
    return JoinedResult(**defaults)


def test_same_quantity_curves_are_grouped_and_unit_converted(tmp_path: Path):
    results = [
        _result(paper_id="paper_0001", curve_id="curve_a", fibre_outcome="formed fibres"),
        _result(
            paper_id="paper_0002",
            curve_id="curve_b",
            y_axis_unit="mPa*s",
            points=[DataPoint(x=1, y=1_000_000), DataPoint(x=10, y=300_000), DataPoint(x=100, y=80_000)],
            fibre_outcome="failed or no fibres",
        ),
    ]

    write_combined_plots(tmp_path, results)

    manifest = json.loads((tmp_path / "combined_plots_manifest.json").read_text())
    assert len(manifest["groups"]) == 1
    group = manifest["groups"][0]
    assert group["y_unit"] == "pascal * second"
    curves_by_id = {curve["curve_id"]: curve for curve in group["curves"]}
    assert curves_by_id["curve_a"]["y"] == [1000, 300, 80]
    assert curves_by_id["curve_a"]["category"] == "can"
    assert curves_by_id["curve_b"]["y"] == [1000, 300, 80]
    assert curves_by_id["curve_b"]["category"] == "cannot"
    assert (tmp_path / group["plot_path"]).exists()


def test_different_quantities_produce_separate_groups(tmp_path: Path):
    results = [
        _result(paper_id="paper_0001", curve_id="curve_a"),
        _result(
            paper_id="paper_0002",
            curve_id="curve_b",
            x_axis_label="Frequency",
            x_axis_unit="1/s",
            y_axis_label="Storage Modulus",
            y_axis_unit="Pa",
            points=[DataPoint(x=1, y=100), DataPoint(x=10, y=200)],
        ),
    ]

    write_combined_plots(tmp_path, results)

    manifest = json.loads((tmp_path / "combined_plots_manifest.json").read_text())
    assert len(manifest["groups"]) == 2
    plot_paths = {group["plot_path"] for group in manifest["groups"]}
    assert len(plot_paths) == 2
    for path in plot_paths:
        assert (tmp_path / path).exists()


def test_unparseable_unit_is_excluded_not_fatal(tmp_path: Path):
    results = [
        _result(paper_id="paper_0001", curve_id="curve_a"),
        _result(paper_id="paper_0002", curve_id="curve_b", y_axis_unit="not-a-real-unit"),
    ]

    write_combined_plots(tmp_path, results)

    manifest = json.loads((tmp_path / "combined_plots_manifest.json").read_text())
    assert len(manifest["groups"]) == 1
    assert len(manifest["groups"][0]["curves"]) == 1
    excluded_ids = {entry["curve_id"] for entry in manifest["excluded"]}
    assert "curve_b" in excluded_ids


def test_missing_axis_label_is_excluded(tmp_path: Path):
    results = [_result(paper_id="paper_0001", curve_id="curve_a", x_axis_label=None)]

    write_combined_plots(tmp_path, results)

    manifest = json.loads((tmp_path / "combined_plots_manifest.json").read_text())
    assert manifest["groups"] == []
    assert manifest["excluded"][0]["reason"] == "missing axis label"

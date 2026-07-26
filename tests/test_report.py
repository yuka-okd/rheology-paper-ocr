import json
from pathlib import Path

from rheology_paper_ocr.report import write_reports
from rheology_paper_ocr.schemas import JoinedResult


def test_writes_csv_json_and_html(tmp_path: Path):
    result = JoinedResult(
        paper_id="paper_0001",
        source_pdf="paper.pdf",
        figure_id="Fig. 1",
        page=1,
        chart_crop_path="charts/page_001.png",
        curve_id="curve_1",
        curve_visual_label="blue circles",
        curve_legend_text="Sample A",
        sample_id="sample_a",
        sample_display_name="Sample A",
        sample_composition="polymer in solvent",
        x_axis_label="Shear Rate",
        x_axis_unit="1/s",
        x_axis_scale="log",
        y_axis_label="Viscosity",
        y_axis_unit="mPa*s",
        y_axis_scale="log",
        start_x=1,
        start_y=1000,
        end_x=100,
        end_y=100,
        fold_change=0.1,
        loglog_slope=-0.5,
        rheology_class="shear-thinning",
        fibre_outcome="formed fibres",
        fibre_evidence_text="Sample A formed fibres.",
        fibre_evidence_source="Results",
        confidence="high",
        warnings=[],
    )

    write_reports(tmp_path, [result])

    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "report.html").exists()
    data = json.loads((tmp_path / "results.json").read_text())
    assert data[0]["sample_id"] == "sample_a"
    assert "Sample A" in (tmp_path / "report.html").read_text()

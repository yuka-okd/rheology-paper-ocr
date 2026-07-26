import json
from pathlib import Path

import rheology_paper_ocr.pipeline as pipeline
from rheology_paper_ocr.pipeline import completion_status, load_saved_results, run_pipeline
from rheology_paper_ocr.schemas import DataPoint, ExtractedFinding, FibreOutcome, JoinedResult, PaperLLMExtraction, SampleLink


def test_completion_status_flags_relevant_paper_with_no_findings():
    text = "The viscosity was measured as a function of shear rate in Fig. 2."

    assert completion_status(text=text, findings_count=0) == "completed_no_findings"


def test_completion_status_marks_extracted_rows_as_completed():
    text = "The viscosity was measured as a function of shear rate in Fig. 2."

    assert completion_status(text=text, findings_count=2) == "completed"


def test_loads_saved_results_for_report_regeneration(tmp_path):
    result = JoinedResult(paper_id="paper_0001", source_pdf="paper.pdf", curve_id="curve_a")
    (tmp_path / "results.json").write_text(json.dumps([result.model_dump(mode="json")]), encoding="utf-8")

    assert load_saved_results(tmp_path) == [result]


def test_pipeline_persists_evidence_rows_and_skips_unchanged_completed_papers(monkeypatch, tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    pdf_path = source_dir / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    page_image = tmp_path / "page_001.png"
    page_image.write_bytes(b"image")
    calls = []

    monkeypatch.setattr(pipeline, "OpenRouterClient", lambda model=None: object())
    monkeypatch.setattr(
        pipeline,
        "extract_text_and_pages",
        lambda pdf_path, paper_dir: ("Viscosity against shear rate.", [page_image]),
    )

    def fake_extract(*args, **kwargs):
        calls.append(kwargs)
        return PaperLLMExtraction(
            paper_id="paper_0001",
            source_pdf="paper.pdf",
            has_rheology_chart=True,
            findings=[
                ExtractedFinding(
                    figure_id="Fig. 1",
                    page=1,
                    chart_crop_path="papers/paper_0001/pages/page_001.png",
                    curve_id="curve_a",
                    curve_visual_label="red line",
                    curve_legend_text="Sample A",
                    x_axis_label="shear rate",
                    x_axis_scale="log",
                    y_axis_label="viscosity",
                    y_axis_scale="log",
                    points=[DataPoint(x=1, y=1000), DataPoint(x=10, y=300), DataPoint(x=100, y=80)],
                    sample=SampleLink(
                        sample_id="sample_a",
                        sample_display_name="Sample A",
                        evidence_text="Sample A is the red line.",
                        evidence_source="figure_caption",
                        confidence="high",
                    ),
                    fibre_outcome=FibreOutcome(
                        outcome="formed fibres",
                        evidence_text="Sample A formed fibres.",
                        evidence_source="results",
                        confidence="high",
                    ),
                    confidence="high",
                )
            ],
        )

    monkeypatch.setattr(pipeline, "extract_paper_with_llm", fake_extract)
    out_dir = tmp_path / "output"

    first_results = run_pipeline(source_dir, out_dir)
    second_results = run_pipeline(source_dir, out_dir, resume=True)

    assert len(calls) == 1
    assert first_results == second_results
    assert first_results[0].rheology_class == "shear-thinning"
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "papers" / "paper_0001" / "results.json").exists()
    assert (out_dir / "report.html").exists()


def test_empty_pipeline_writes_empty_reports_without_openrouter(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(pipeline, "OpenRouterClient", lambda model=None: (_ for _ in ()).throw(AssertionError("must not create a client")))

    assert run_pipeline(tmp_path, tmp_path / "output") == []
    assert (tmp_path / "output" / "results.json").exists()
    assert (tmp_path / "output" / "report.html").exists()

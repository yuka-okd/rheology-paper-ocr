import json
from pathlib import Path

import rheology_paper_ocr.pipeline as pipeline
from rheology_paper_ocr.docling_figures import LocalizedFigure
from rheology_paper_ocr.openrouter_client import OpenRouterInsufficientCreditsError
from rheology_paper_ocr.pipeline import _extraction_requests, completion_status, is_review_article, load_saved_results, run_pipeline
from rheology_paper_ocr.schemas import DataPoint, ExtractedFinding, FibreOutcome, JoinedResult, PaperLLMExtraction, SampleLink, SourcePaper


def test_completion_status_flags_relevant_paper_with_no_findings():
    text = "The viscosity was measured as a function of shear rate in Fig. 2."

    assert completion_status(text=text, findings_count=0) == "completed_no_findings"


def test_completion_status_marks_extracted_rows_as_completed():
    text = "The viscosity was measured as a function of shear rate in Fig. 2."

    assert completion_status(text=text, findings_count=2) == "completed"


def test_detects_review_label_in_opening_document_text():
    assert is_review_article("Title\nReview\nDOI: 10.1234/example") is True
    assert is_review_article("Title\nExperimental article\nResults") is False


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
        assert json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))[0]["status"] == "started"
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

    out_dir = tmp_path / "output"
    monkeypatch.setattr(pipeline, "extract_paper_with_llm", fake_extract)

    first_results = run_pipeline(source_dir, out_dir)
    second_results = run_pipeline(source_dir, out_dir, resume=True)

    assert len(calls) == 1
    assert first_results == second_results
    assert first_results[0].rheology_class == "shear-thinning"
    assert first_results[0].points == [DataPoint(x=1, y=1000), DataPoint(x=10, y=300), DataPoint(x=100, y=80)]
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "papers" / "paper_0001" / "results.json").exists()
    assert (out_dir / "report.html").exists()
    assert (out_dir / "combined_plots_manifest.json").exists()


def test_pipeline_stops_after_insufficient_openrouter_credit(monkeypatch, tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "first.pdf").write_bytes(b"%PDF-1.4\n")
    (source_dir / "second.pdf").write_bytes(b"%PDF-1.4\n")
    page_image = tmp_path / "page_001.png"
    page_image.write_bytes(b"image")

    monkeypatch.setattr(pipeline, "OpenRouterClient", lambda model=None: object())
    monkeypatch.setattr(pipeline, "extract_text_and_pages", lambda *_: ("Rheology chart", [page_image]))
    monkeypatch.setattr(
        pipeline,
        "extract_paper_with_llm",
        lambda *args, **kwargs: (_ for _ in ()).throw(OpenRouterInsufficientCreditsError("requires more credits")),
    )

    out_dir = tmp_path / "output"
    assert run_pipeline(source_dir, out_dir) == []

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 1
    assert manifest[0]["paper_id"] == "paper_0001"
    assert manifest[0]["status"] == "blocked_insufficient_credits"
    assert manifest[0]["error"] == "requires more credits"


def test_empty_pipeline_writes_empty_reports_without_openrouter(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(pipeline, "OpenRouterClient", lambda model=None: (_ for _ in ()).throw(AssertionError("must not create a client")))

    assert run_pipeline(tmp_path, tmp_path / "output") == []
    assert (tmp_path / "output" / "results.json").exists()
    assert (tmp_path / "output" / "report.html").exists()


def test_pipeline_honours_pause_before_starting_the_next_paper(monkeypatch, tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "paper.pdf").write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(pipeline, "OpenRouterClient", lambda model=None: object())

    assert run_pipeline(source_dir, tmp_path / "output", should_pause=lambda: True) == []
    assert json.loads((tmp_path / "output" / "manifest.json").read_text(encoding="utf-8")) == []


def test_single_series_chart_with_caption_evidence_does_not_require_a_legend():
    extraction = PaperLLMExtraction(
        paper_id="paper_0001",
        source_pdf="paper.pdf",
        has_rheology_chart=True,
        findings=[
            ExtractedFinding(
                curve_id="sample_a",
                sample=SampleLink(sample_id="sample_a", evidence_text="Figure 2 is the Sample A series."),
            )
        ],
    )

    result = pipeline._finding_results([extraction], SourcePaper(paper_id="paper_0001", path=Path("paper.pdf")))[0]

    assert "curve lacks a visual or legend identifier" not in result.warnings


def test_uses_a_figure_crop_when_docling_identifies_a_chart(tmp_path: Path):
    first_page = tmp_path / "page_001.png"
    second_page = tmp_path / "page_002.png"
    crop = tmp_path / "figure_2.png"
    figure = LocalizedFigure(
        figure_id="Figure 2",
        page=2,
        caption="Figure 2. Viscosity against shear rate.",
        crop_path=crop,
        bbox=(0, 1, 1, 0),
        relevance_score=2,
    )

    requests = _extraction_requests([first_page, second_page], [figure], text_only=False)

    assert requests == [(first_page, None), (second_page, figure)]


def test_skips_a_page_when_every_docling_figure_is_explicitly_non_rheology(tmp_path: Path):
    page = tmp_path / "page_003.png"
    morphology = LocalizedFigure(
        "Figure 3",
        3,
        "Figure 3. Dependence of nanofiber morphology on concentration.",
        tmp_path / "figure_3.png",
        (0, 1, 1, 0),
        0,
    )

    requests = _extraction_requests([page], [], text_only=False, localized_figures=[morphology])

    assert requests == []


def test_successful_fibres_only_filters_unlinked_and_failed_rows(monkeypatch, tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    pdf_path = source_dir / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    page_image = tmp_path / "page_001.png"
    page_image.write_bytes(b"image")

    monkeypatch.setattr(pipeline, "OpenRouterClient", lambda model=None: object())
    monkeypatch.setattr(pipeline, "extract_text_and_pages", lambda *_: ("Experimental article", [page_image]))
    monkeypatch.setattr(
        pipeline,
        "extract_paper_with_llm",
        lambda *args, **kwargs: PaperLLMExtraction(
            paper_id="paper_0001",
            source_pdf="paper.pdf",
            has_rheology_chart=True,
            findings=[
                ExtractedFinding(curve_id="formed", fibre_outcome=FibreOutcome(outcome="formed fibres", evidence_text="formed")),
                ExtractedFinding(curve_id="unclear", fibre_outcome=FibreOutcome(outcome="unclear")),
                ExtractedFinding(curve_id="failed", fibre_outcome=FibreOutcome(outcome="failed or no fibres", evidence_text="failed")),
            ],
        ),
    )

    results = run_pipeline(source_dir, tmp_path / "output", successful_fibres_only=True)

    assert [result.curve_id for result in results] == ["formed"]


def test_queues_relevant_chart_without_supported_series(monkeypatch, tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    pdf_path = source_dir / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    page_image = tmp_path / "page_001.png"
    page_image.write_bytes(b"image")

    monkeypatch.setattr(pipeline, "OpenRouterClient", lambda model=None: object())
    monkeypatch.setattr(pipeline, "extract_text_and_pages", lambda *_: ("Rheology chart", [page_image]))
    monkeypatch.setattr(
        pipeline,
        "extract_paper_with_llm",
        lambda *args, **kwargs: PaperLLMExtraction(
            paper_id="paper_0001",
            source_pdf="paper.pdf",
            has_rheology_chart=True,
            findings=[],
            paper_warnings=["A rheology chart is visible but no exact formulation link was found."],
        ),
    )

    out_dir = tmp_path / "output"
    assert run_pipeline(source_dir, out_dir, successful_fibres_only=True) == []

    review_queue = json.loads((out_dir / "review_queue.json").read_text(encoding="utf-8"))
    assert review_queue[0]["paper_id"] == "paper_0001"
    assert "no exact formulation link" in review_queue[0]["reasons"][0]

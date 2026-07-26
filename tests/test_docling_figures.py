from pathlib import Path

import fitz

from rheology_paper_ocr.docling_figures import (
    LocalizedFigure,
    _caption_for_picture,
    _figure_id,
    _localized_figures_from_document,
    page_is_explicitly_non_rheology,
    select_chart_figures,
)


def test_recovers_caption_and_figure_identifier_from_docling_references():
    texts = [{"text": "Figure 2. Specific viscosity as a function of concentration."}]
    picture = {"captions": [{"$ref": "#/texts/0"}]}

    caption = _caption_for_picture(picture, texts)

    assert caption == texts[0]["text"]
    assert _figure_id(caption) == "Figure 2"


def test_select_chart_figures_rejects_non_rheology_captions(tmp_path: Path):
    schematic = LocalizedFigure("Figure 1", 1, "Figure 1. Electrospinning setup.", tmp_path / "one.png", (0, 1, 1, 0), 0)
    viscosity = LocalizedFigure("Figure 2", 1, "Figure 2. Viscosity against shear rate.", tmp_path / "two.png", (0, 1, 1, 0), 2)

    assert select_chart_figures([schematic, viscosity]) == [viscosity]


def test_excludes_fibre_size_correlation_even_when_caption_mentions_viscosity(tmp_path: Path):
    fibre_size = LocalizedFigure(
        "Figure 6",
        1,
        "Figure 6. Dependence of electrospun fiber size on zero shear rate viscosity.",
        tmp_path / "six.png",
        (0, 1, 1, 0),
        2,
    )

    assert fibre_size.is_rheology_candidate is False
    assert select_chart_figures([fibre_size]) == []
    assert page_is_explicitly_non_rheology([fibre_size]) is True


def test_excludes_a_confidently_non_chart_picture_even_with_a_rheology_caption(tmp_path: Path):
    micrograph = LocalizedFigure(
        "Figure 3",
        1,
        "Figure 3. Viscosity response of a sample.",
        tmp_path / "three.png",
        (0, 1, 1, 0),
        1,
        picture_type="photograph",
        picture_type_confidence=0.98,
    )

    assert micrograph.is_rheology_candidate is False
    assert select_chart_figures([micrograph]) == []


def test_renders_docling_picture_bbox_on_original_page(tmp_path: Path):
    source_path = tmp_path / "source.pdf"
    source = fitz.open()
    page = source.new_page(width=200, height=300)
    page.draw_rect(fitz.Rect(60, 40, 160, 190), color=(0, 0, 0))
    page.insert_text((70, 100), "0.1 1 10 viscosity")
    source.save(source_path)
    source.close()

    document = {
        "texts": [{"text": "Figure 2. Viscosity against shear rate."}],
        "pictures": [
            {
                "captions": [{"$ref": "#/texts/0"}],
                "prov": [
                    {
                        "page_no": 1,
                        "bbox": {"l": 60, "t": 260, "r": 160, "b": 110},
                    }
                ],
                "annotations": [
                    {
                        "kind": "classification",
                        "predicted_classes": [{"class_name": "line_chart", "confidence": 0.95}],
                    }
                ],
            }
        ],
    }

    figures = _localized_figures_from_document(document, [1], source_path, tmp_path / "figures", render_scale=2)

    assert len(figures) == 1
    assert figures[0].figure_id == "Figure 2"
    assert figures[0].relevance_score > 0
    assert figures[0].crop_path.exists()
    assert figures[0].vector_text is not None
    assert figures[0].picture_type == "line_chart"

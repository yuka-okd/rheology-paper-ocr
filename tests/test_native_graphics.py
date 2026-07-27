from pathlib import Path

import fitz

from rheology_paper_ocr.docling_figures import localize_native_chart_graphics
from rheology_paper_ocr.native_graphics import export_native_graphic, inspect_native_graphics


def _write_pdf_with_native_chart_image(path: Path) -> None:
    image_path = path.with_suffix(".png")
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 400, 300), False)
    pixmap.clear_with(0x336699)
    pixmap.save(image_path)

    document = fitz.open()
    page = document.new_page(width=300, height=300)
    page.insert_image(fitz.Rect(30, 40, 130, 115), filename=image_path)
    page.insert_text((30, 135), "Figure 2. Viscosity against shear rate.")
    page.draw_line((30, 200), (140, 200), color=(0, 0, 0))
    document.save(path)
    document.close()


def test_inspects_and_exports_native_graphics(tmp_path: Path):
    pdf_path = tmp_path / "source.pdf"
    _write_pdf_with_native_chart_image(pdf_path)

    pages = inspect_native_graphics(pdf_path, [1])

    assert len(pages) == 1
    assert pages[0].vector_path_count == 1
    assert pages[0].has_vector_curve_geometry is False
    graphic = pages[0].graphics[0]
    assert graphic.width == 400
    assert graphic.height == 300
    assert graphic.effective_scale == 4
    assert "Viscosity against shear rate" in graphic.nearby_text

    target = tmp_path / "graphic.png"
    export_native_graphic(pdf_path, graphic, target)

    assert target.exists()
    assert target.stat().st_size > 0


def test_localizes_high_resolution_rheology_graphics_without_docling(tmp_path: Path):
    pdf_path = tmp_path / "source.pdf"
    _write_pdf_with_native_chart_image(pdf_path)

    figures = localize_native_chart_graphics(pdf_path, [1], tmp_path / "paper")

    assert len(figures) == 1
    assert figures[0].figure_id == "Figure 2"
    assert figures[0].is_rheology_candidate is True
    assert figures[0].crop_path.exists()
    assert figures[0].native_graphic_path == figures[0].crop_path

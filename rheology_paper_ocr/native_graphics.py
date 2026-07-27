from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import fitz


@dataclass(frozen=True)
class NativeGraphic:
    """An embedded raster graphic with its PDF placement and nearby text."""

    page: int
    xref: int
    bbox: tuple[float, float, float, float]
    width: int
    height: int
    effective_scale: float
    nearby_text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class NativePageGraphics:
    page: int
    vector_path_count: int
    vector_curve_path_count: int
    graphics: list[NativeGraphic]

    @property
    def has_vector_curve_geometry(self) -> bool:
        return self.vector_curve_path_count >= 2

    def to_dict(self) -> dict:
        return {
            "page": self.page,
            "vector_path_count": self.vector_path_count,
            "vector_curve_path_count": self.vector_curve_path_count,
            "has_vector_curve_geometry": self.has_vector_curve_geometry,
            "graphics": [graphic.to_dict() for graphic in self.graphics],
        }


def inspect_native_graphics(pdf_path: Path, pages: list[int]) -> list[NativePageGraphics]:
    """Inventory PDF vector paths and embedded graphic assets on selected pages."""
    source = fitz.open(pdf_path)
    try:
        return [_inspect_page(source[page - 1], page) for page in pages]
    finally:
        source.close()


def export_native_graphic(pdf_path: Path, graphic: NativeGraphic, target_path: Path) -> None:
    """Export an embedded PDF image as PNG for high-fidelity vision extraction."""
    source = fitz.open(pdf_path)
    try:
        pixmap = fitz.Pixmap(source, graphic.xref)
        if pixmap.colorspace and pixmap.colorspace.n not in {1, 3}:
            pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        pixmap.save(target_path)
    finally:
        source.close()


def _inspect_page(page: fitz.Page, page_number: int) -> NativePageGraphics:
    drawings = page.get_drawings()
    graphics = [
        _native_graphic(page, page_number, image)
        for image in page.get_image_info(xrefs=True)
        if isinstance(image.get("xref"), int) and image["xref"] > 0
    ]
    return NativePageGraphics(
        page=page_number,
        vector_path_count=len(drawings),
        vector_curve_path_count=sum(1 for drawing in drawings if _is_curve_path(drawing)),
        graphics=graphics,
    )


def _native_graphic(page: fitz.Page, page_number: int, image: dict) -> NativeGraphic:
    bbox = tuple(float(value) for value in image["bbox"])
    width, height = int(image["width"]), int(image["height"])
    placed_width, placed_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    effective_scale = min(width / placed_width, height / placed_height) if placed_width and placed_height else 0.0
    return NativeGraphic(
        page=page_number,
        xref=int(image["xref"]),
        bbox=bbox,
        width=width,
        height=height,
        effective_scale=effective_scale,
        nearby_text=_nearby_text(page, fitz.Rect(bbox)),
    )


def _nearby_text(page: fitz.Page, rect: fitz.Rect, margin: float = 64.0) -> str:
    region = fitz.Rect(
        max(0, rect.x0 - margin),
        max(0, rect.y0 - margin),
        min(page.rect.x1, rect.x1 + margin),
        min(page.rect.y1, rect.y1 + margin),
    )
    values = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = str(span.get("text") or "").strip()
                bbox = span.get("bbox")
                if text and bbox and fitz.Rect(bbox).intersects(region):
                    values.append(text)
    return " ".join(values)


def _is_curve_path(drawing: dict) -> bool:
    """Recognise non-axis vector paths without claiming they are data curves."""
    items = drawing.get("items") or []
    if not items or not drawing.get("color"):
        return False
    if any(item[0] == "c" for item in items):
        return True
    if len(items) < 2:
        return False
    rect = drawing.get("rect")
    if not rect:
        return False
    return rect.width > 2 and rect.height > 2

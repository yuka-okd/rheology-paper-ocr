from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import fitz


CHART_CAPTION_KEYWORDS = (
    "rheolog",
    "viscos",
    "shear",
    "modulus",
    "frequency",
    "flow curve",
    "flow behavior",
    "stress",
    "strain",
    "complex viscosity",
    "loss tangent",
    "tan delta",
)

NON_RHEOLOGY_CAPTION_RULES = (
    ("schematic", "schematic figure"),
    ("morphology", "fibre morphology figure"),
    ("fiber morphology", "fibre morphology figure"),
    ("fibre morphology", "fibre morphology figure"),
    ("fiber diameter", "fibre diameter figure"),
    ("fibre diameter", "fibre diameter figure"),
    ("fiber size", "fibre size figure"),
    ("fibre size", "fibre size figure"),
    ("diameter of electrospun", "fibre diameter figure"),
)

CHART_PICTURE_TYPES = {"line_chart", "scatter_plot", "bar_chart", "box_plot", "pie_chart"}
PICTURE_CLASSIFICATION_CONFIDENCE = 0.8


class DoclingFigureError(RuntimeError):
    """Raised when optional local figure localization cannot complete."""


@dataclass(frozen=True)
class LocalizedFigure:
    figure_id: str | None
    page: int
    caption: str | None
    crop_path: Path
    bbox: tuple[float, float, float, float]
    relevance_score: int
    vector_text: str | None = None
    picture_type: str | None = None
    picture_type_confidence: float | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["crop_path"] = str(self.crop_path)
        data["rheology_candidate"] = self.is_rheology_candidate
        data["exclusion_reason"] = self.exclusion_reason
        return data

    @property
    def exclusion_reason(self) -> str | None:
        return _caption_exclusion_reason(self.caption)

    @property
    def is_rheology_candidate(self) -> bool:
        return self.relevance_score > 0 and self.exclusion_reason is None and not self.is_confidently_non_chart

    @property
    def is_confidently_non_chart(self) -> bool:
        return bool(
            self.picture_type
            and self.picture_type not in CHART_PICTURE_TYPES
            and (self.picture_type_confidence or 0) >= PICTURE_CLASSIFICATION_CONFIDENCE
        )


def localize_figures(
    pdf_path: Path,
    candidate_pages: list[int],
    paper_dir: Path,
    render_scale: float = 4.0,
) -> list[LocalizedFigure]:
    """Find figures on selected PDF pages and render each as a high-resolution crop.

    Docling is intentionally imported here rather than at module load time. The
    core extractor remains usable without the heavy optional dependency, and a
    localization failure must not prevent a normal full-page vision pass.
    """
    if not candidate_pages:
        return []

    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.settings import settings
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise DoclingFigureError("Docling is not installed; install with '.[docling]'.") from exc

    # Avoid torch.compile on macOS installations that do not have the matching
    # C++ toolchain. Layout inference is fast enough without it for a few pages.
    settings.inference.compile_torch_models = False
    options = PdfPipelineOptions(do_ocr=False, do_table_structure=False, do_picture_classification=True)

    try:
        with TemporaryDirectory(prefix="rheology-docling-") as temp_dir:
            selected_pdf = Path(temp_dir) / "candidate-pages.pdf"
            _write_selected_pages(pdf_path, candidate_pages, selected_pdf)
            converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )
            document = converter.convert(selected_pdf).document.export_to_dict()
    except Exception as exc:
        raise DoclingFigureError(f"Docling figure localization failed: {exc}") from exc

    figures_dir = paper_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    localized = _localized_figures_from_document(document, candidate_pages, pdf_path, figures_dir, render_scale)
    (paper_dir / "docling_figures.json").write_text(
        json.dumps([figure.to_dict() for figure in localized], indent=2), encoding="utf-8"
    )
    return localized


def select_chart_figures(figures: list[LocalizedFigure], max_per_page: int = 2) -> list[LocalizedFigure]:
    """Keep only caption-supported chart candidates, preserving page order."""
    selected: list[LocalizedFigure] = []
    for page in sorted({figure.page for figure in figures}):
        candidates = [figure for figure in figures if figure.page == page and figure.is_rheology_candidate]
        selected.extend(sorted(candidates, key=lambda figure: (-figure.relevance_score, figure.figure_id or ""))[:max_per_page])
    return selected


def page_is_explicitly_non_rheology(figures: list[LocalizedFigure]) -> bool:
    """Return true only when every localized figure has an explicit exclusion."""
    return bool(figures) and all(figure.caption and figure.exclusion_reason for figure in figures)


def _write_selected_pages(pdf_path: Path, candidate_pages: list[int], target_path: Path) -> None:
    source = fitz.open(pdf_path)
    target = fitz.open()
    try:
        for page in candidate_pages:
            target.insert_pdf(source, from_page=page - 1, to_page=page - 1)
        target.save(target_path)
    finally:
        target.close()
        source.close()


def _localized_figures_from_document(
    document: dict,
    candidate_pages: list[int],
    source_pdf: Path,
    figures_dir: Path,
    render_scale: float,
) -> list[LocalizedFigure]:
    texts = document.get("texts") or []
    pictures = document.get("pictures") or []
    source = fitz.open(source_pdf)
    localized: list[LocalizedFigure] = []
    try:
        for index, picture in enumerate(pictures, start=1):
            provenance = (picture.get("prov") or [None])[0]
            if not provenance:
                continue
            selected_page = provenance.get("page_no")
            if not isinstance(selected_page, int) or not 1 <= selected_page <= len(candidate_pages):
                continue
            bbox = provenance.get("bbox") or {}
            coords = _bbox_coordinates(bbox)
            if coords is None:
                continue

            page = candidate_pages[selected_page - 1]
            caption = _caption_for_picture(picture, texts)
            figure_id = _figure_id(caption)
            picture_type, picture_type_confidence = _picture_classification(picture)
            suffix = _safe_identifier(figure_id or f"picture_{index:02d}")
            crop_path = figures_dir / f"page_{page:03d}_{suffix}.png"
            source_page = source[page - 1]
            clip = _crop_rect(source_page, coords)
            _render_crop(source_page, clip, crop_path, render_scale)
            localized.append(
                LocalizedFigure(
                    figure_id=figure_id,
                    page=page,
                    caption=caption,
                    crop_path=crop_path,
                    bbox=coords,
                    relevance_score=_caption_relevance(caption),
                    vector_text=_extract_vector_text(source_page, clip),
                    picture_type=picture_type,
                    picture_type_confidence=picture_type_confidence,
                )
            )
    finally:
        source.close()
    return localized


def _bbox_coordinates(bbox: dict) -> tuple[float, float, float, float] | None:
    try:
        left, top, right, bottom = (float(bbox[key]) for key in ("l", "t", "r", "b"))
    except (KeyError, TypeError, ValueError):
        return None
    if not left < right or not bottom < top:
        return None
    return left, top, right, bottom


def _caption_for_picture(picture: dict, texts: list[dict]) -> str | None:
    values: list[str] = []
    for reference in picture.get("captions") or []:
        ref = reference.get("$ref") if isinstance(reference, dict) else None
        if not isinstance(ref, str):
            continue
        try:
            text = texts[int(ref.rsplit("/", maxsplit=1)[-1])].get("text")
        except (IndexError, ValueError):
            continue
        if isinstance(text, str) and text.strip():
            values.append(text.strip())
    return " ".join(values) or None


def _picture_classification(picture: dict) -> tuple[str | None, float | None]:
    for annotation in picture.get("annotations") or []:
        if annotation.get("kind") != "classification":
            continue
        predictions = annotation.get("predicted_classes") or []
        if not predictions:
            continue
        prediction = predictions[0]
        picture_type = prediction.get("class_name")
        confidence = prediction.get("confidence")
        if isinstance(picture_type, str) and isinstance(confidence, (int, float)):
            return picture_type, float(confidence)
    return None, None


def _figure_id(caption: str | None) -> str | None:
    if not caption:
        return None
    match = re.search(r"\b(?:figure|fig\.)\s*([0-9]+[a-z]?)\b", caption, flags=re.IGNORECASE)
    return f"Figure {match.group(1)}" if match else None


def _safe_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "figure"


def _caption_relevance(caption: str | None) -> int:
    lowered = (caption or "").lower()
    return sum(keyword in lowered for keyword in CHART_CAPTION_KEYWORDS)


def _caption_exclusion_reason(caption: str | None) -> str | None:
    lowered = (caption or "").lower()
    for keyword, reason in NON_RHEOLOGY_CAPTION_RULES:
        if keyword in lowered:
            return reason
    return None


def _crop_rect(page: fitz.Page, bbox: tuple[float, float, float, float]) -> fitz.Rect:
    left, top, right, bottom = bbox
    width, height = right - left, top - bottom
    padding = max(6.0, 0.04 * max(width, height))
    return fitz.Rect(
        max(page.rect.x0, left - padding),
        max(page.rect.y0, page.rect.height - top - padding),
        min(page.rect.x1, right + padding),
        min(page.rect.y1, page.rect.height - bottom + padding),
    )

def _render_crop(page: fitz.Page, clip: fitz.Rect, target: Path, scale: float) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
    pixmap.save(target)


def _extract_vector_text(page: fitz.Page, clip: fitz.Rect, max_chars: int = 3000) -> str | None:
    text = re.sub(r"\s+", " ", page.get_text("text", clip=clip)).strip()
    return text[:max_chars] or None

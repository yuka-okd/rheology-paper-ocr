from __future__ import annotations

import hashlib
import re
from pathlib import Path

import fitz

from rheology_paper_ocr.schemas import SourcePaper


RHEOLOGY_KEYWORDS = (
    "rheolog",
    "viscos",
    "shear",
    "storage modulus",
    "loss modulus",
    "frequency sweep",
    "flow curve",
    "electrosp",
    "fiber",
    "fibre",
    "nanofiber",
    "nanofibre",
)

PAGE_KEYWORD_WEIGHTS = {
    "rheolog": 2,
    "viscos": 2,
    "shear": 2,
    "storage modulus": 3,
    "loss modulus": 3,
    "frequency sweep": 3,
    "flow curve": 3,
    "electrosp": 1,
    "fiber": 1,
    "fibre": 1,
    "nanofiber": 1,
    "nanofibre": 1,
}

PAGE_KEYWORD_OCCURRENCE_CAP = 3

# A caption naming a rheology quantity means the chart is on that page, which
# is far better evidence than prose mentions. Such a page also needs the help:
# it is mostly given over to the figure, so it carries little text and keyword
# counts rank it below the discussion pages that merely talk about the charts.
RHEOLOGY_FIGURE_CAPTION = re.compile(
    r"fig(?:ure)?\.?\s*\d+[.:]\s[^\n]{0,200}?"
    r"(?:viscos|shear|rheolog|modulus|flow curve|flow behaviour|flow behavior|stress|strain)",
    re.IGNORECASE,
)
RHEOLOGY_FIGURE_CAPTION_BONUS = 12

# Candidate figures can occupy only a small part of a journal page.  A low-DPI
# page render makes marker/legend association and log-scale digitization guesswork.
PAGE_RENDER_SCALE = 3.0


def discover_pdfs(pdf_dir: Path) -> list[SourcePaper]:
    if pdf_dir.is_file():
        paths = [pdf_dir] if pdf_dir.suffix.lower() == ".pdf" else []
    else:
        paths = sorted(path for path in pdf_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf")
    return [SourcePaper(paper_id=f"paper_{index:04d}", path=path) for index, path in enumerate(paths, start=1)]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_candidate_page_indexes(page_texts: list[str], max_candidate_pages: int = 3) -> list[int]:
    # Score on how often each term appears, not merely whether it appears. An
    # abstract that names every term once otherwise outranks the results page
    # that discusses viscosity throughout, and ties resolve to the earlier
    # page, so the page holding the charts is never rendered. The per-term cap
    # keeps one heavily repeated word from crowding out a page that covers
    # several of them.
    scored = []
    for index, text in enumerate(page_texts):
        lowered = text.lower()
        score = sum(
            weight * min(lowered.count(keyword), PAGE_KEYWORD_OCCURRENCE_CAP)
            for keyword, weight in PAGE_KEYWORD_WEIGHTS.items()
        )
        if RHEOLOGY_FIGURE_CAPTION.search(text):
            score += RHEOLOGY_FIGURE_CAPTION_BONUS
        scored.append((index, score))
    relevant = [(index, score) for index, score in scored if score]
    if not relevant:
        return list(range(min(len(page_texts), max_candidate_pages)))
    return [index for index, _ in sorted(relevant, key=lambda item: (-item[1], item[0]))[:max_candidate_pages]]


def extract_text_and_pages(pdf_path: Path, paper_dir: Path, max_candidate_pages: int = 3) -> tuple[str, list[Path]]:
    pages_dir = paper_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    page_texts: list[str] = []
    for index, page in enumerate(doc):
        text = page.get_text("text")
        page_texts.append(f"\n\n--- Page {index + 1} ---\n{text}")
    candidate_indexes = select_candidate_page_indexes(page_texts, max_candidate_pages=max_candidate_pages)

    image_paths: list[Path] = []
    matrix = fitz.Matrix(PAGE_RENDER_SCALE, PAGE_RENDER_SCALE)
    for index in candidate_indexes:
        page = doc[index]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image_path = pages_dir / f"page_{index + 1:03d}.png"
        pixmap.save(image_path)
        image_paths.append(image_path)

    text = "".join(page_texts)
    (paper_dir / "text.md").write_text(text, encoding="utf-8")
    return text, image_paths

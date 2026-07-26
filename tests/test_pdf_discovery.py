from pathlib import Path

from rheology_paper_ocr.pdf_extract import discover_pdfs, select_candidate_page_indexes


def test_discovers_pdfs_recursively_with_stable_ids(tmp_path: Path):
    first = tmp_path / "a.pdf"
    nested = tmp_path / "nested"
    nested.mkdir()
    second = nested / "b.PDF"
    first.write_bytes(b"%PDF-1.4\n")
    second.write_bytes(b"%PDF-1.4\n")

    papers = discover_pdfs(tmp_path)

    assert [paper.paper_id for paper in papers] == ["paper_0001", "paper_0002"]
    assert [paper.path for paper in papers] == [first, second]


def test_ranks_pages_with_multiple_rheology_signals_first():
    pages = [
        "Methods. The samples were electrospun.",
        "Figure 2. Viscosity against shear rate in the rheology measurements.",
        "Results. Storage modulus and loss modulus were measured during a frequency sweep.",
    ]

    assert select_candidate_page_indexes(pages, max_candidate_pages=2) == [2, 1]

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


def test_ranks_a_results_page_above_an_abstract_that_name_drops_each_term_once():
    """A page discussing viscosity throughout outranks one merely listing the terms.

    Scoring on keyword presence alone tied these, and the tie resolved to the
    earlier page, so the page carrying the flow curves was never rendered.
    """
    abstract = "Abstract. We report rheology, viscosity, shear, and flow curve data for electrospun fibre mats."
    filler = "References and acknowledgements."
    results = (
        "Figure 3. The viscosity decreased with shear rate. "
        "Viscosity values fell as shear rate rose, and the viscosity of each "
        "sample tracked the shear thinning seen in the shear sweep."
    )

    assert select_candidate_page_indexes([abstract, filler, results], max_candidate_pages=1) == [2]

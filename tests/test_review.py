from rheology_paper_ocr.review import apply_decision_policy
from rheology_paper_ocr.schemas import JoinedResult


def test_accepts_a_clean_high_confidence_result():
    results, reviews = apply_decision_policy(
            [
                JoinedResult(
                    paper_id="paper_0001",
                    source_pdf="paper.pdf",
                    curve_id="curve_a",
                    confidence="high",
                    fibre_outcome="formed fibres",
                )
            ]
    )

    assert results[0].decision == "accepted"
    assert results[0].review_reasons == []
    assert reviews == []


def test_queues_sparse_or_ambiguous_result_for_review():
    results, reviews = apply_decision_policy(
        [
            JoinedResult(
                paper_id="paper_0001",
                source_pdf="paper.pdf",
                curve_id="curve_a",
                confidence="medium",
                fibre_outcome="unclear",
                warnings=["fewer than three digitized points; rheology class left unclear"],
            )
        ]
    )

    assert results[0].decision == "needs_review"
    assert "fibre outcome is unresolved" in results[0].review_reasons
    assert len(reviews) == 1

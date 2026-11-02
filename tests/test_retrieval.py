from pathlib import Path

import pytest

from storeguide.retrieval import PgVectorRetriever, SearchHit, query_term_coverage


def test_retriever_validates_query_and_result_limit() -> None:
    retriever = PgVectorRetriever(database_url="postgresql://unused")

    with pytest.raises(ValueError, match="Query must not be empty"):
        retriever.search("  ")
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.search("shipping", top_k=0)


def test_retriever_requires_database_configuration() -> None:
    with pytest.raises(RuntimeError, match="DATABASE_URL is required"):
        PgVectorRetriever(database_url="").search("shipping")


def test_query_term_coverage_requires_evidence_for_query_terms() -> None:
    hit = SearchHit(
        chunk_id="gift-cards:0000",
        article_id="gift-cards",
        title="Gift cards",
        category="payments",
        content="Redeem your gift card during checkout.",
        score=0.65,
    )

    assert query_term_coverage("Do you offer gift wrapping?", [hit]) == 0.5
    assert query_term_coverage("How do I redeem a gift card?", [hit]) == 1.0
    assert query_term_coverage("Do you?", [hit]) == 0.0


def test_query_term_coverage_normalizes_common_word_forms_and_synonyms() -> None:
    hit = SearchHit(
        chunk_id="returns-refunds:0000",
        article_id="returns-refunds",
        title="Returns and refunds",
        category="returns",
        content="Hand the parcel to the carrier. Keep the purchase receipt for the item.",
        score=0.8,
    )

    assert query_term_coverage(
        "How long do I have to hand a return parcel to the carrier?",
        [hit],
    ) == 1.0
    assert query_term_coverage(
        "What proof is needed for a purchased item?",
        [hit],
    ) == 1.0


def test_retrieval_cases_have_valid_expected_article_ids() -> None:
    import json

    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / "eval" / "retrieval_cases.json").read_text(encoding="utf-8"))
    article_ids = {path.stem for path in (root / "data" / "kb").glob("*.md")}

    assert len(cases) == 48
    assert all(case["expected_article_ids"] for case in cases)
    assert all(
        expected_id in article_ids
        for case in cases
        for expected_id in case["expected_article_ids"]
    )

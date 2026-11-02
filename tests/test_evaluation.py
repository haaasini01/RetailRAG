from dataclasses import dataclass

import pytest

from storeguide.evaluation import (
    RetrievalCase,
    evaluate_retrieval,
    hit_at_k,
    load_cases,
    recall_at_k,
    reciprocal_rank,
)


@dataclass
class _Hit:
    article_id: str
    score: float = 0.8
    title: str = "Policy"
    content: str = "policy first unrelated evidence"


class _Retriever:
    def search(self, query: str, top_k: int = 5) -> list[_Hit]:
        del top_k
        ranked = {
            "first": ["a", "b", "c"],
            "missing": ["x", "y"],
            "unrelated": ["x", "y"],
        }[query]
        score = 0.4 if query == "unrelated" else 0.8
        content = "unrelated policy content" if query == "unrelated" else "first policy content"
        return [
            _Hit(article_id=article_id, score=score, content=content)
            for article_id in ranked
        ]


def test_retrieval_metrics() -> None:
    assert reciprocal_rank(["x", "a", "b"], {"a"}) == 0.5
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0
    assert hit_at_k(["x", "a"], {"a"}, 2) == 1.0
    assert hit_at_k(["x", "a"], {"a"}, 1) == 0.0
    assert recall_at_k(["a", "x"], {"a", "b"}, 2) == 0.5


def test_evaluate_retrieval_reports_expected_metrics() -> None:
    cases = [
        RetrievalCase(query="first", expected_article_ids=["b"]),
        RetrievalCase(query="missing", expected_article_ids=["a"]),
    ]

    result = evaluate_retrieval(cases, _Retriever(), top_k=2)

    assert result["hit_rate"] == 0.5
    assert result["mrr"] == 0.25
    assert result["recall_at_k"] == 0.5


def test_loads_default_labeled_questions() -> None:
    cases = load_cases()
    assert len(cases) == 48


def test_evaluate_checks_refusal_accuracy() -> None:
    cases = [
        RetrievalCase(query="first", expected_article_ids=["a"]),
        RetrievalCase(query="unrelated", expected_article_ids=[], type="out_of_corpus"),
    ]

    result = evaluate_retrieval(cases, _Retriever(), min_relevance_score=0.6)

    assert result["refusal_accuracy"] == 1.0
    assert result["in_corpus_acceptance"] == 1.0
    assert result["passed"] is True
    assert result["out_of_corpus_queries"] == 1


def test_evaluate_fails_gate_when_out_of_domain_query_is_not_refused() -> None:
    cases = [
        RetrievalCase(query="first", expected_article_ids=["a"]),
        RetrievalCase(query="unrelated", expected_article_ids=[], type="out_of_corpus"),
    ]

    result = evaluate_retrieval(cases, _Retriever(), min_relevance_score=0.2)

    assert result["refusal_accuracy"] == 0.0
    assert result["passed"] is False
    assert result["gate_failures"] == ["Refusal accuracy 0.0000 < threshold 1.0000"]


def test_evaluate_routes_account_specific_cases_without_retrieval() -> None:
    result = evaluate_retrieval(
        [
            RetrievalCase(
                query="Has my order 84721 shipped?",
                expected_article_ids=[],
                type="out_of_corpus",
            ),
            RetrievalCase(
                query="first",
                expected_article_ids=["b"],
            ),
        ],
        _Retriever(),
    )

    assert result["tool_required_queries"] == 1
    assert result["tool_required_refusal_accuracy"] == 1.0
    assert result["per_query"][0]["requires_tool"] is True


def test_rejects_empty_cases() -> None:
    with pytest.raises(ValueError, match="At least one"):
        evaluate_retrieval([], _Retriever())

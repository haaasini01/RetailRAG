from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from storeguide.config import PROJECT_ROOT
from storeguide.config import (
    EVAL_HIT_RATE_THRESHOLD,
    EVAL_IN_CORPUS_ACCEPTANCE_THRESHOLD,
    EVAL_REFUSAL_THRESHOLD,
    MIN_RELEVANCE_SCORE,
    MIN_QUERY_TERM_COVERAGE,
)
from storeguide.retrieval import PgVectorRetriever, SearchHit, query_term_coverage

DEFAULT_CASES = PROJECT_ROOT / "eval" / "retrieval_cases.json"
DEFAULT_REFUSAL_CASES = PROJECT_ROOT / "eval" / "refusal_cases.json"


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        ...


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    expected_article_ids: list[str]
    type: str = "in_corpus"


def load_cases(path: Path = DEFAULT_CASES) -> list[RetrievalCase]:
    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"Evaluation set must be a non-empty JSON list: {path}")
    cases: list[RetrievalCase] = []
    for index, row in enumerate(raw_cases):
        if not isinstance(row, dict):
            raise ValueError(f"Evaluation case {index} must be an object")
        query = row.get("query")
        expected = row.get("expected_article_ids")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Evaluation case {index} has an empty query")
        case_type = row.get("type", "in_corpus")
        if not isinstance(case_type, str) or case_type not in {
            "in_corpus",
            "out_of_corpus",
        }:
            raise ValueError(f"Evaluation case {index} has an unknown type")
        if case_type == "in_corpus":
            if (
                not isinstance(expected, list)
                or not expected
                or not all(isinstance(item, str) and item for item in expected)
            ):
                raise ValueError(f"Evaluation case {index} needs expected article IDs")
        elif expected is not None:
            raise ValueError(
                f"Out-of-corpus evaluation case {index} must not have expected article IDs"
            )
        cases.append(
            RetrievalCase(
                query=query.strip(),
                expected_article_ids=expected or [],
                type=case_type,
            )
        )
    return cases


def reciprocal_rank(ranked_ids: list[str], expected: set[str]) -> float:
    for index, article_id in enumerate(ranked_ids, start=1):
        if article_id in expected:
            return 1.0 / index
    return 0.0


def hit_at_k(ranked_ids: list[str], expected: set[str], k: int) -> float:
    return float(bool(expected & set(ranked_ids[:k])))


def recall_at_k(ranked_ids: list[str], expected: set[str], k: int) -> float:
    return len(expected & set(ranked_ids[:k])) / len(expected) if expected else 0.0


def evaluate_retrieval(
    cases: list[RetrievalCase],
    retriever: Retriever,
    top_k: int = 5,
    min_relevance_score: float = MIN_RELEVANCE_SCORE,
    min_query_term_coverage: float = MIN_QUERY_TERM_COVERAGE,
    hit_rate_threshold: float = EVAL_HIT_RATE_THRESHOLD,
    refusal_threshold: float = EVAL_REFUSAL_THRESHOLD,
    in_corpus_acceptance_threshold: float = EVAL_IN_CORPUS_ACCEPTANCE_THRESHOLD,
) -> dict[str, object]:
    if not cases:
        raise ValueError("At least one retrieval evaluation case is required")
    if top_k < 1:
        raise ValueError("top_k must be a positive integer")
    for name, value in (
        ("min_relevance_score", min_relevance_score),
        ("min_query_term_coverage", min_query_term_coverage),
        ("hit_rate_threshold", hit_rate_threshold),
        ("refusal_threshold", refusal_threshold),
        ("in_corpus_acceptance_threshold", in_corpus_acceptance_threshold),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")

    per_query: list[dict[str, object]] = []
    hits: list[float] = []
    reciprocal_ranks: list[float] = []
    recalls: list[float] = []
    accepted_in_corpus = 0
    refusal_correct = 0
    refusal_total = 0
    for case in cases:
        results = retriever.search(case.query, top_k=top_k)
        ranked_ids = [result.article_id for result in results]
        top_score = results[0].score if results else 0.0
        term_coverage = query_term_coverage(case.query, results)
        if case.type == "out_of_corpus":
            refused = (
                top_score < min_relevance_score
                or term_coverage < min_query_term_coverage
            )
            refusal_total += 1
            refusal_correct += int(refused)
            per_query.append(
                {
                    "query": case.query,
                    "type": case.type,
                    "top_score": round(top_score, 4),
                    "query_term_coverage": round(term_coverage, 4),
                    "refused": refused,
                    "correct": refused,
                }
            )
            continue

        expected = set(case.expected_article_ids)
        accepted = (
            top_score >= min_relevance_score
            and term_coverage >= min_query_term_coverage
        )
        accepted_in_corpus += int(accepted)
        hit = hit_at_k(ranked_ids, expected, top_k)
        rr = reciprocal_rank(ranked_ids, expected)
        recall = recall_at_k(ranked_ids, expected, top_k)
        hits.append(hit)
        reciprocal_ranks.append(rr)
        recalls.append(recall)
        per_query.append(
            {
                "query": case.query,
                "type": case.type,
                "expected_article_ids": case.expected_article_ids,
                "ranked_article_ids": ranked_ids,
                "top_score": round(top_score, 4),
                "query_term_coverage": round(term_coverage, 4),
                "accepted_by_gate": accepted,
                "hit": hit,
                "reciprocal_rank": round(rr, 4),
                "recall": round(recall, 4),
            }
        )

    count = len(hits)
    if count == 0:
        raise ValueError("At least one in-corpus evaluation case is required")
    hit_rate = sum(hits) / count
    refusal_accuracy = refusal_correct / refusal_total if refusal_total else 0.0
    in_corpus_acceptance = accepted_in_corpus / count
    gate_failures: list[str] = []
    if hit_rate < hit_rate_threshold:
        gate_failures.append(
            f"Hit Rate@{top_k} {hit_rate:.4f} < threshold {hit_rate_threshold:.4f}"
        )
    if refusal_accuracy < refusal_threshold:
        gate_failures.append(
            f"Refusal accuracy {refusal_accuracy:.4f} < threshold {refusal_threshold:.4f}"
        )
    if in_corpus_acceptance < in_corpus_acceptance_threshold:
        gate_failures.append(
            "In-corpus acceptance "
            f"{in_corpus_acceptance:.4f} < threshold "
            f"{in_corpus_acceptance_threshold:.4f}"
        )
    return {
        "queries": len(cases),
        "in_corpus_queries": count,
        "out_of_corpus_queries": refusal_total,
        "k": top_k,
        "hit_rate": round(hit_rate, 4),
        "mrr": round(sum(reciprocal_ranks) / count, 4),
        "recall_at_k": round(sum(recalls) / count, 4),
        "refusal_accuracy": round(refusal_accuracy, 4),
        "in_corpus_acceptance": round(in_corpus_acceptance, 4),
        "min_relevance_score": min_relevance_score,
        "min_query_term_coverage": min_query_term_coverage,
        "hit_rate_threshold": hit_rate_threshold,
        "refusal_threshold": refusal_threshold,
        "in_corpus_acceptance_threshold": in_corpus_acceptance_threshold,
        "passed": not gate_failures,
        "gate_failures": gate_failures,
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RetailRAG vector retrieval")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--refusal-cases", type=Path, default=DEFAULT_REFUSAL_CASES)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "eval" / "evaluation-results.json",
    )
    args = parser.parse_args()
    result = evaluate_retrieval(
        load_cases(args.cases) + load_cases(args.refusal_cases),
        retriever=PgVectorRetriever(),
        top_k=args.top_k,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in result.items() if key != "per_query"}, indent=2))
    print(f"Per-query results written to {args.output}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

from dataclasses import dataclass

import pytest

from storeguide.hybrid_search import HybridRetriever, reciprocal_rank_fusion
from storeguide.retrieval import SearchHit
from storeguide.reranking import BgeCrossEncoder


def _hit(chunk_id: str, score: float = 0.8) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        article_id=chunk_id.split(":")[0],
        title=chunk_id,
        category="policy",
        content=f"content for {chunk_id}",
        score=score,
    )


@dataclass
class _Retriever:
    hits: list[SearchHit]
    queries: list[tuple[str, int]]

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        self.queries.append((query, top_k))
        return self.hits[:top_k]


@dataclass
class _Reranker:
    hits: list[SearchHit]
    queries: list[tuple[str, list[str]]]

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        self.queries.append((query, [hit.chunk_id for hit in hits]))
        return self.hits

    def warmup(self) -> None:
        return None


class _FakeCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.documents: list[str] = []

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        assert query == "return window"
        self.documents = documents
        return self.scores


def test_reciprocal_rank_fusion_combines_and_deduplicates_results() -> None:
    first = _hit("shipping:0000")
    shared = _hit("returns-refunds:0000")
    lexical_only = _hit("product-warranty:0000")

    fused = reciprocal_rank_fusion(
        [[first, shared], [shared, lexical_only]],
        top_k=3,
    )

    assert [hit.chunk_id for hit in fused] == [
        "returns-refunds:0000",
        "shipping:0000",
        "product-warranty:0000",
    ]


def test_hybrid_retriever_fuses_candidates_then_reranks_to_requested_count() -> None:
    vector = _Retriever([_hit("shipping:0000"), _hit("returns-refunds:0000")], [])
    bm25 = _Retriever([_hit("returns-refunds:0000"), _hit("product-warranty:0000")], [])
    reranker = _Reranker(
        [_hit("product-warranty:0000", 0.91), _hit("returns-refunds:0000", 0.87)],
        [],
    )
    retriever = HybridRetriever(
        vector_retriever=vector,
        bm25_retriever=bm25,
        reranker=reranker,
        candidate_pool_size=3,
    )

    hits = retriever.search("return window", top_k=1)

    assert vector.queries == [("return window", 3)]
    assert bm25.queries == [("return window", 3)]
    assert reranker.queries[0][0] == "return window"
    assert set(reranker.queries[0][1]) == {
        "shipping:0000",
        "returns-refunds:0000",
        "product-warranty:0000",
    }
    assert [hit.chunk_id for hit in hits] == ["product-warranty:0000"]


def test_reranker_maps_cross_encoder_logits_to_scores_and_sorts() -> None:
    encoder = _FakeCrossEncoder([-2.0, 2.0])
    reranker = BgeCrossEncoder(encoder=encoder)
    hits = [_hit("shipping:0000"), _hit("returns-refunds:0000")]

    ranked = reranker.rerank("return window", hits)

    assert ranked[0].chunk_id == "returns-refunds:0000"
    assert ranked[0].score == pytest.approx(0.8808, abs=0.0001)
    assert ranked[1].score == pytest.approx(0.1192, abs=0.0001)
    assert encoder.documents[0].startswith("shipping:0000\n")


def test_reranker_rejects_mismatched_score_count() -> None:
    reranker = BgeCrossEncoder(encoder=_FakeCrossEncoder([0.5]))

    with pytest.raises(RuntimeError, match="different number of scores"):
        reranker.rerank("return window", [_hit("a:0000"), _hit("b:0000")])


def test_hybrid_retriever_rejects_invalid_candidate_pool() -> None:
    with pytest.raises(ValueError, match="candidate_pool_size"):
        HybridRetriever(candidate_pool_size=0)

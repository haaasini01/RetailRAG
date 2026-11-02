from __future__ import annotations

import re
from collections import defaultdict
from typing import Protocol

import psycopg
from rank_bm25 import BM25Okapi

from storeguide.config import DATABASE_URL
from storeguide.reranking import BgeCrossEncoder
from storeguide.retrieval import PgVectorRetriever, SearchHit

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class RankedRetriever(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        ...


class CandidateReranker(Protocol):
    def warmup(self) -> None:
        ...

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        ...


class BM25Retriever:
    def __init__(self, database_url: str = DATABASE_URL) -> None:
        self._database_url = database_url

    def search(self, query: str, top_k: int = 20) -> list[SearchHit]:
        if not query.strip():
            raise ValueError("Query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if not self._database_url:
            raise RuntimeError("DATABASE_URL is required for BM25 retrieval")

        with psycopg.connect(self._database_url) as connection:
            rows = connection.execute(
                """
                SELECT c.chunk_id, c.article_id, a.title, a.category, c.content
                FROM chunks AS c
                JOIN articles AS a ON a.article_id = c.article_id
                ORDER BY c.chunk_id
                """
            ).fetchall()
        if not rows:
            return []

        corpus = [
            _TOKEN_PATTERN.findall(f"{row[2]} {row[4]}".lower())
            for row in rows
        ]
        query_terms = _TOKEN_PATTERN.findall(query.lower())
        if not query_terms:
            return []
        scores = BM25Okapi(corpus).get_scores(query_terms)
        ranked_indices = sorted(
            range(len(rows)),
            key=lambda index: (float(scores[index]), -index),
            reverse=True,
        )[:top_k]
        return [
            SearchHit(
                chunk_id=rows[index][0],
                article_id=rows[index][1],
                title=rows[index][2],
                category=rows[index][3],
                content=rows[index][4],
                score=float(scores[index]),
            )
            for index in ranked_indices
            if scores[index] > 0
        ]


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchHit]],
    top_k: int,
    rank_constant: int = 60,
) -> list[SearchHit]:
    if top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if rank_constant < 1:
        raise ValueError("rank_constant must be a positive integer")

    fused_scores: dict[str, float] = defaultdict(float)
    candidates: dict[str, SearchHit] = {}
    for ranked_list in ranked_lists:
        for rank, hit in enumerate(ranked_list, start=1):
            candidates.setdefault(hit.chunk_id, hit)
            fused_scores[hit.chunk_id] += 1.0 / (rank_constant + rank)
    ranked_ids = sorted(
        fused_scores,
        key=lambda chunk_id: (-fused_scores[chunk_id], chunk_id),
    )
    return [
        candidates[chunk_id]
        for chunk_id in ranked_ids[:top_k]
    ]


class HybridRetriever:
    def __init__(
        self,
        vector_retriever: RankedRetriever | None = None,
        bm25_retriever: RankedRetriever | None = None,
        reranker: CandidateReranker | None = None,
        candidate_pool_size: int = 20,
        rank_constant: int = 60,
    ) -> None:
        if candidate_pool_size < 1:
            raise ValueError("candidate_pool_size must be a positive integer")
        if rank_constant < 1:
            raise ValueError("rank_constant must be a positive integer")
        self._vector_retriever = vector_retriever or PgVectorRetriever()
        self._bm25_retriever = bm25_retriever or BM25Retriever()
        self._reranker = reranker or BgeCrossEncoder()
        self._candidate_pool_size = candidate_pool_size
        self._rank_constant = rank_constant

    def warmup(self) -> None:
        self._reranker.warmup()

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        if not query.strip():
            raise ValueError("Query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be a positive integer")
        vector_hits = self._vector_retriever.search(
            query, top_k=self._candidate_pool_size
        )
        lexical_hits = self._bm25_retriever.search(
            query, top_k=self._candidate_pool_size
        )
        candidates = reciprocal_rank_fusion(
            [vector_hits, lexical_hits],
            top_k=self._candidate_pool_size,
            rank_constant=self._rank_constant,
        )
        return self._reranker.rerank(query, candidates)[:top_k]

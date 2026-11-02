from __future__ import annotations

import re
from dataclasses import dataclass

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from storeguide.config import DATABASE_URL
from storeguide.embeddings import OllamaEmbedder

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_QUERY_STOPWORDS = frozenset(
    "a about after all an and are as at be before by can do does for from get how "
    "have i if in is it long me my need needed of on or our please should tell that "
    "the their this to offer take up us was we what when where which who why with "
    "you your usually".split()
)
_COVERAGE_SYNONYMS = {
    "post": "appear",
    "proof": "receipt",
    "purchased": "purchase",
}


def _normalize_coverage_term(term: str) -> str:
    if term in _COVERAGE_SYNONYMS:
        return _COVERAGE_SYNONYMS[term]
    if term.endswith("ies") and len(term) > 4:
        return term[:-3] + "y"
    if term.endswith("ing") and len(term) > 5:
        stem = term[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if term.endswith("ed") and len(term) > 4:
        stem = term[:-2]
        return stem[:-1] + "e" if stem.endswith("s") else stem
    if term.endswith("s") and not term.endswith(("ss", "us")) and len(term) > 3:
        return term[:-1]
    return term


def _coverage_terms(text: str) -> set[str]:
    return {
        _normalize_coverage_term(token)
        for token in _TOKEN_PATTERN.findall(text.lower())
    }


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    article_id: str
    title: str
    category: str
    content: str
    score: float


def query_term_coverage(query: str, hits: list[SearchHit]) -> float:
    query_terms = {
        _normalize_coverage_term(term)
        for term in _TOKEN_PATTERN.findall(query.lower())
        if term not in _QUERY_STOPWORDS
    }
    if not query_terms:
        return 0.0
    evidence_terms = _coverage_terms(
        " ".join(f"{hit.title} {hit.content}" for hit in hits)
    )
    return len(query_terms & evidence_terms) / len(query_terms)


class PgVectorRetriever:
    def __init__(self, database_url: str = DATABASE_URL) -> None:
        self._database_url = database_url

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        if not query.strip():
            raise ValueError("Query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if not self._database_url:
            raise RuntimeError("DATABASE_URL is required for vector retrieval")

        with OllamaEmbedder() as embedder:
            query_vector = Vector(embedder.embed_query(query))
        with psycopg.connect(self._database_url) as connection:
            register_vector(connection)
            rows = connection.execute(
                """
                SELECT c.chunk_id, c.article_id, a.title, a.category, c.content,
                       c.embedding <=> %s AS distance
                FROM chunks AS c
                JOIN articles AS a ON a.article_id = c.article_id
                ORDER BY c.embedding <=> %s
                LIMIT %s
                """,
                (query_vector, query_vector, top_k),
            ).fetchall()
        return [
            SearchHit(
                chunk_id=row[0],
                article_id=row[1],
                title=row[2],
                category=row[3],
                content=row[4],
                score=1.0 - float(row[5]),
            )
            for row in rows
        ]

from __future__ import annotations

import sys
from pathlib import Path

import psycopg
from psycopg import Connection
from pgvector import Vector
from pgvector.psycopg import register_vector

from storeguide.config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_TARGET_WORDS,
    DATABASE_URL,
    KB_DIR,
)
from storeguide.embeddings import OllamaEmbedder
from storeguide.knowledge_base import Article, KnowledgeChunk, chunk_article, load_articles


def build_chunks(
    articles: list[Article], target_words: int, overlap_words: int
) -> list[KnowledgeChunk]:
    return [
        chunk
        for article in articles
        for chunk in chunk_article(article, target_words, overlap_words)
    ]


def sync_knowledge_base(
    connection: Connection,
    articles: list[Article],
    chunks: list[KnowledgeChunk],
    embeddings: list[list[float]],
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("Each chunk must have exactly one embedding")
    article_ids = [article.id for article in articles]
    with connection.cursor() as cursor:
        cursor.execute(
            """
            DELETE FROM articles
            WHERE NOT (article_id = ANY(%s))
            """,
            (article_ids,),
        )
        cursor.executemany(
            """
            INSERT INTO articles (
                article_id, title, category, body, source_path, content_sha256
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (article_id) DO UPDATE SET
                title = EXCLUDED.title,
                category = EXCLUDED.category,
                body = EXCLUDED.body,
                source_path = EXCLUDED.source_path,
                content_sha256 = EXCLUDED.content_sha256,
                updated_at = NOW()
            WHERE (
                articles.title,
                articles.category,
                articles.body,
                articles.source_path,
                articles.content_sha256
            ) IS DISTINCT FROM (
                EXCLUDED.title,
                EXCLUDED.category,
                EXCLUDED.body,
                EXCLUDED.source_path,
                EXCLUDED.content_sha256
            )
            """,
            [
                (
                    article.id,
                    article.title,
                    article.category,
                    article.body,
                    article.source_path,
                    article.content_sha256,
                )
                for article in articles
            ],
        )
        cursor.execute("DELETE FROM chunks WHERE article_id = ANY(%s)", (article_ids,))
        cursor.executemany(
            """
            INSERT INTO chunks (chunk_id, article_id, ordinal, content, embedding)
            VALUES (%s, %s, %s, %s, %s)
            """,
            [
                (
                    chunk.id,
                    chunk.article_id,
                    chunk.ordinal,
                    chunk.content,
                    Vector(embedding),
                )
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ],
        )
        cursor.execute("ALTER TABLE chunks ALTER COLUMN embedding SET NOT NULL")
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
            ON chunks USING hnsw (embedding vector_cosine_ops)
            """
        )


def ingest() -> int:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is required to ingest the knowledge base")

    articles = load_articles(KB_DIR)
    chunks = build_chunks(articles, CHUNK_TARGET_WORDS, CHUNK_OVERLAP_WORDS)
    with OllamaEmbedder() as embedder:
        embeddings = embedder.embed_documents([chunk.content for chunk in chunks])
    schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
    with psycopg.connect(DATABASE_URL) as connection:
        register_vector(connection)
        connection.execute(schema)
        sync_knowledge_base(connection, articles, chunks, embeddings)
    return len(chunks)


def main() -> None:
    print(f"Loading store articles from {KB_DIR} ...", flush=True)
    count = ingest()
    print(f"Ingested {count} chunks from {KB_DIR}.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, psycopg.Error) as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

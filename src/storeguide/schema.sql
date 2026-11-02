CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS articles (
    article_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    body TEXT NOT NULL,
    source_path TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    article_id TEXT NOT NULL REFERENCES articles(article_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    content TEXT NOT NULL,
    embedding vector(768),
    UNIQUE (article_id, ordinal)
);

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding vector(768);

CREATE INDEX IF NOT EXISTS chunks_article_id_idx ON chunks (article_id);

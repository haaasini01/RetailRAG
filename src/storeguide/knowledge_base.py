from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

_FRONTMATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)", re.DOTALL)
_SLUG = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"


class _ArticleMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(pattern=_SLUG)
    title: str = Field(min_length=1, max_length=160)
    category: str = Field(pattern=_SLUG)


@dataclass(frozen=True)
class Article:
    id: str
    title: str
    category: str
    body: str
    source_path: str
    content_sha256: str


@dataclass(frozen=True)
class KnowledgeChunk:
    id: str
    article_id: str
    title: str
    category: str
    content: str
    ordinal: int


def load_articles(kb_dir: Path) -> list[Article]:
    if not kb_dir.is_dir():
        raise FileNotFoundError(f"Knowledge-base directory not found: {kb_dir}")

    articles: list[Article] = []
    seen_ids: set[str] = set()
    for path in sorted(kb_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        match = _FRONTMATTER.match(raw)
        if match is None:
            raise ValueError(f"{path}: expected YAML front matter between --- lines")

        try:
            metadata: Any = yaml.safe_load(match.group(1))
            parsed = _ArticleMetadata.model_validate(metadata)
        except (yaml.YAMLError, ValidationError) as exc:
            raise ValueError(f"{path}: invalid article metadata: {exc}") from exc

        body = raw[match.end() :].strip()
        if not body:
            raise ValueError(f"{path}: article body must not be empty")
        if parsed.id in seen_ids:
            raise ValueError(f"{path}: duplicate article id {parsed.id!r}")
        seen_ids.add(parsed.id)
        articles.append(
            Article(
                id=parsed.id,
                title=parsed.title,
                category=parsed.category,
                body=body,
                source_path=path.name,
                content_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            )
        )

    if not articles:
        raise ValueError(f"No Markdown articles found in {kb_dir}")
    return articles


def chunk_article(
    article: Article, target_words: int = 180, overlap_words: int = 40
) -> list[KnowledgeChunk]:
    if target_words <= 0:
        raise ValueError("target_words must be positive")
    if overlap_words < 0 or overlap_words >= target_words:
        raise ValueError("overlap_words must be non-negative and less than target_words")

    words = article.body.split()
    if not words:
        raise ValueError(f"Article {article.id!r} has no body text")

    step = target_words - overlap_words
    chunks: list[KnowledgeChunk] = []
    for ordinal, start in enumerate(range(0, len(words), step)):
        text = " ".join(words[start : start + target_words])
        chunks.append(
            KnowledgeChunk(
                id=f"{article.id}:{ordinal:04d}",
                article_id=article.id,
                title=article.title,
                category=article.category,
                content=text,
                ordinal=ordinal,
            )
        )
    return chunks

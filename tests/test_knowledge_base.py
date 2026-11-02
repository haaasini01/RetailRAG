from pathlib import Path

import pytest

from storeguide.knowledge_base import Article, chunk_article, load_articles

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_loads_store_articles_with_source_metadata() -> None:
    articles = load_articles(PROJECT_ROOT / "data" / "kb")

    assert len(articles) == 9
    shipping = next(article for article in articles if article.id == "shipping")
    assert shipping.title == "Shipping, delivery estimates, and charges"
    assert shipping.category == "orders"
    assert shipping.source_path == "shipping.md"
    assert "four to six business days" in shipping.body
    assert len(shipping.content_sha256) == 64


def test_rejects_missing_front_matter(tmp_path: Path) -> None:
    (tmp_path / "invalid.md").write_text("No metadata here.", encoding="utf-8")

    with pytest.raises(ValueError, match="expected YAML front matter"):
        load_articles(tmp_path)


def test_rejects_invalid_article_metadata(tmp_path: Path) -> None:
    (tmp_path / "invalid.md").write_text(
        "---\ntitle: Missing id\ncategory: orders\n---\nBody",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid article metadata"):
        load_articles(tmp_path)


def test_rejects_duplicate_article_ids(tmp_path: Path) -> None:
    contents = "---\nid: same-id\ntitle: Same\ncategory: orders\n---\nBody"
    (tmp_path / "one.md").write_text(contents, encoding="utf-8")
    (tmp_path / "two.md").write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate article id"):
        load_articles(tmp_path)


def test_rejects_empty_article_body(tmp_path: Path) -> None:
    (tmp_path / "empty.md").write_text(
        "---\nid: empty\ntitle: Empty\ncategory: orders\n---\n  ",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="article body must not be empty"):
        load_articles(tmp_path)


def test_chunks_are_bounded_overlapping_and_stably_identified() -> None:
    article = Article(
        id="shipping",
        title="Shipping",
        category="orders",
        body=" ".join(f"word{i}" for i in range(25)),
        source_path="shipping.md",
        content_sha256="hash",
    )

    chunks = chunk_article(article, target_words=10, overlap_words=3)

    assert [chunk.id for chunk in chunks] == [
        "shipping:0000",
        "shipping:0001",
        "shipping:0002",
        "shipping:0003",
    ]
    assert all(len(chunk.content.split()) <= 10 for chunk in chunks)
    assert chunks[0].content.split()[-3:] == chunks[1].content.split()[:3]
    assert [chunk.ordinal for chunk in chunks] == [0, 1, 2, 3]
    assert all(chunk.title == article.title for chunk in chunks)


def test_rejects_invalid_chunk_configuration() -> None:
    article = Article(
        id="shipping",
        title="Shipping",
        category="orders",
        body="Words for a chunk.",
        source_path="shipping.md",
        content_sha256="hash",
    )

    with pytest.raises(ValueError, match="less than target_words"):
        chunk_article(article, target_words=10, overlap_words=10)

from __future__ import annotations

import argparse
import json
import sys

import psycopg
import httpx

from storeguide.retrieval import PgVectorRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the RetailRAG knowledge base")
    parser.add_argument("query", help="Customer question to retrieve support evidence for")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to return")
    args = parser.parse_args()

    hits = PgVectorRetriever().search(args.query, top_k=args.top_k)
    print(
        json.dumps(
            [
                {
                    "article_id": hit.article_id,
                    "title": hit.title,
                    "category": hit.category,
                    "chunk_id": hit.chunk_id,
                    "score": round(hit.score, 4),
                    "content": hit.content,
                }
                for hit in hits
            ],
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, psycopg.Error, httpx.HTTPError) as exc:
        print(f"Retrieval failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

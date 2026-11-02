from __future__ import annotations

import math
from collections.abc import Sequence

import httpx

from storeguide.config import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT_SECONDS,
)


class OllamaEmbedder:
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = EMBEDDING_MODEL,
        dimensions: int = EMBEDDING_DIMENSIONS,
        timeout_seconds: int = OLLAMA_TIMEOUT_SECONDS,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimensions = dimensions
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self._base_url, timeout=timeout_seconds
        )

    def __enter__(self) -> OllamaEmbedder:
        return self

    def __exit__(self, *_: object) -> None:
        if self._owns_client:
            self._client.close()

    def _embed(self, text: str, task_prefix: str) -> list[float]:
        if not text.strip():
            raise ValueError("Cannot create an embedding for empty text")

        payload = {"model": self._model, "prompt": f"{task_prefix}: {text}"}
        try:
            response = self._client.post("/api/embeddings", json=payload)
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Ollama embedding request failed: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError("Ollama returned invalid JSON for an embedding") from exc

        vector = body.get("embedding") if isinstance(body, dict) else None
        if not isinstance(vector, list) or len(vector) != self._dimensions:
            actual = len(vector) if isinstance(vector, list) else "missing"
            raise RuntimeError(
                f"Embedding model {self._model!r} returned {actual} dimensions; "
                f"expected {self._dimensions}"
            )
        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Ollama returned a non-numeric embedding value") from exc
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("Ollama returned a non-finite embedding value")
        return values

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text, "search_document") for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed(query, "search_query")

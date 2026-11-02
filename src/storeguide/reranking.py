from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from storeguide.config import RERANKER_CACHE_DIR, RERANKER_MODEL
from storeguide.retrieval import SearchHit


class CrossEncoder(Protocol):
    def rerank(self, query: str, documents: Iterable[str]) -> Iterable[float]:
        ...


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str, cache_dir: str) -> CrossEncoder:
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=model_name, cache_dir=cache_dir)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class BgeCrossEncoder:
    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        cache_dir: Path = RERANKER_CACHE_DIR,
        encoder: CrossEncoder | None = None,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._encoder = encoder

    def warmup(self) -> None:
        if self._encoder is None:
            _load_cross_encoder(self._model_name, str(self._cache_dir))

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        if not hits:
            return []
        encoder = self._encoder or _load_cross_encoder(
            self._model_name, str(self._cache_dir)
        )
        documents = [f"{hit.title}\n{hit.content}" for hit in hits]
        scores = list(encoder.rerank(query, documents))
        if len(scores) != len(hits):
            raise RuntimeError(
                "Cross-encoder returned a different number of scores than input documents"
            )
        scored_hits = [
            replace(hit, score=_sigmoid(float(score)))
            for hit, score in zip(hits, scores, strict=True)
        ]
        return sorted(scored_hits, key=lambda hit: hit.score, reverse=True)

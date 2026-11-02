from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("storeguide.observability")
_WRITE_LOCK = threading.Lock()


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def write_event(event: dict[str, Any], log_file: Path) -> None:
    line = json.dumps(event, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    print(line, file=sys.stdout, flush=True)
    try:
        with _WRITE_LOCK:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with log_file.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
    except OSError:
        logger.exception("Could not append structured event to %s", log_file)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(percentile / 100 * len(ordered)) - 1)
    return round(ordered[index], 2)


def load_metrics(log_file: Path) -> dict[str, object]:
    if not log_file.exists():
        return {
            "requests": 0,
            "errors": 0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "avg_retrieved_chunks": 0.0,
            "refused_rate": 0.0,
            "models": {},
        }

    latencies: list[float] = []
    retrieved_chunks: list[float] = []
    model_counts: dict[str, int] = {}
    refused_count = 0
    error_count = 0
    with log_file.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Ignoring malformed observability JSON at line %d", line_number)
                continue
            if not isinstance(event, dict):
                logger.warning("Ignoring non-object observability event at line %d", line_number)
                continue
            if event.get("event_type") == "chat_error":
                error_count += 1
                continue
            if event.get("event_type") != "chat":
                logger.warning("Ignoring unknown observability event at line %d", line_number)
                continue
            try:
                latency = float(event["latency_ms"])
                retrieved = float(event["retrieved_chunks"])
                model = str(event["model"])
                refused = event["refused"]
                if latency < 0 or retrieved < 0 or not isinstance(refused, bool):
                    raise ValueError("invalid metric values")
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "Ignoring invalid observability event at line %d: %s",
                    line_number,
                    exc,
                )
                continue
            latencies.append(latency)
            retrieved_chunks.append(retrieved)
            model_counts[model] = model_counts.get(model, 0) + 1
            refused_count += int(refused)

    request_count = len(latencies)
    return {
        "requests": request_count,
        "errors": error_count,
        "avg_latency_ms": round(sum(latencies) / request_count, 2) if request_count else 0.0,
        "p95_latency_ms": _percentile(latencies, 95),
        "avg_retrieved_chunks": (
            round(sum(retrieved_chunks) / request_count, 2) if request_count else 0.0
        ),
        "refused_rate": round(refused_count / request_count, 4) if request_count else 0.0,
        "models": model_counts,
    }


def chat_event(
    query: str,
    *,
    latency_ms: float,
    retrieved_chunks: int,
    top_score: float,
    query_term_coverage: float,
    model: str,
    refused: bool,
    citations: list[str],
) -> dict[str, Any]:
    return {
        "event_type": "chat",
        "timestamp": now_iso(),
        "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "query_length": len(query),
        "latency_ms": latency_ms,
        "retrieved_chunks": retrieved_chunks,
        "top_score": round(top_score, 4),
        "query_term_coverage": round(query_term_coverage, 4),
        "model": model,
        "refused": refused,
        "citations": citations,
    }

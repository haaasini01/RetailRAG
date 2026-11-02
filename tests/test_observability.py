import json
from pathlib import Path

from storeguide.observability import chat_event, load_metrics, write_event


def test_chat_events_hash_queries_instead_of_logging_text(tmp_path: Path) -> None:
    query = "Where is my order?"
    event = chat_event(
        query,
        latency_ms=125.5,
        retrieved_chunks=3,
        top_score=0.82,
        query_term_coverage=0.75,
        model="gemma4:e4b",
        refused=False,
        citations=["order-tracking"],
    )
    log_file = tmp_path / "logs" / "requests.jsonl"

    write_event(event, log_file)
    raw_log = log_file.read_text(encoding="utf-8")
    saved = json.loads(raw_log)

    assert saved["query_sha256"]
    assert query not in raw_log
    assert saved["query_length"] == len(query)
    assert saved["citations"] == ["order-tracking"]


def test_metrics_aggregate_chat_and_error_events(tmp_path: Path) -> None:
    log_file = tmp_path / "requests.jsonl"
    write_event(
        {
            "event_type": "chat",
            "latency_ms": 100,
            "retrieved_chunks": 4,
            "model": "gemma4:e4b",
            "refused": False,
        },
        log_file,
    )
    write_event(
        {
            "event_type": "chat",
            "latency_ms": 200,
            "retrieved_chunks": 0,
            "model": "none",
            "refused": True,
        },
        log_file,
    )
    write_event({"event_type": "chat_error"}, log_file)

    metrics = load_metrics(log_file)

    assert metrics["requests"] == 2
    assert metrics["errors"] == 1
    assert metrics["avg_latency_ms"] == 150
    assert metrics["p95_latency_ms"] == 200
    assert metrics["avg_retrieved_chunks"] == 2
    assert metrics["refused_rate"] == 0.5
    assert metrics["models"] == {"gemma4:e4b": 1, "none": 1}


def test_metrics_report_zeroes_when_log_is_missing(tmp_path: Path) -> None:
    metrics = load_metrics(tmp_path / "no-log.jsonl")

    assert metrics["requests"] == 0
    assert metrics["avg_latency_ms"] == 0.0

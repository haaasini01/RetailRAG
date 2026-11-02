from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

from storeguide import app as app_module
from storeguide.answering import AnswerService, ChatAnswer
from storeguide.app import create_app
from storeguide.retrieval import SearchHit


@dataclass
class _Retriever:
    hits: list[SearchHit]

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        assert query == query.strip()
        return self.hits[:top_k]


@dataclass
class _Generator:
    answer_text: str

    def generate(self, query: str, sources: list[SearchHit]) -> str:
        return self.answer_text


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def test_chat_returns_grounded_answer_with_source_metadata(tmp_path: Path) -> None:
    hit = SearchHit(
        chunk_id="shipping:0000",
        article_id="shipping",
        title="Shipping, delivery estimates, and charges",
        category="orders",
        content="Standard shipping takes four to six business days after carrier pickup.",
        score=0.82,
    )
    service = AnswerService(
        retriever=_Retriever([hit]),
        generator=_Generator(
            "Standard shipping takes four to six business days after pickup. [shipping]"
        ),
    )
    log_file = tmp_path / "requests.jsonl"
    app = create_app(service, log_file=log_file)

    async def request() -> httpx.Response:
        async with _client(app) as client:
            return await client.post(
                "/chat",
                json={"message": " How does standard shipping work? "},
            )

    response = asyncio.run(request())

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Standard shipping takes four to six business days after pickup. [shipping]"
    assert payload["citations"] == ["shipping"]
    assert payload["sources"] == [
        {
            "article_id": "shipping",
            "title": "Shipping, delivery estimates, and charges",
            "score": 0.82,
        }
    ]
    assert payload["model"] == "test-generator"
    assert payload["grounded"] is True
    assert payload["refused"] is False
    assert payload["retrieved_chunks"] == 1
    assert payload["top_score"] == 0.82
    assert payload["query_term_coverage"] == 0.6667
    assert payload["latency_ms"] >= 0
    assert log_file.exists()
    assert "How does standard shipping work?" not in log_file.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "payload",
    [{"message": " "}, {"message": ""}, {"message": "x" * 2001}, {"message": "ok", "extra": True}],
)
def test_chat_rejects_invalid_requests(payload: dict[str, object], tmp_path: Path) -> None:
    app = create_app(
        AnswerService(_Retriever([]), _Generator("unused")),
        log_file=tmp_path / "requests.jsonl",
    )

    async def request() -> httpx.Response:
        async with _client(app) as client:
            return await client.post("/chat", json=payload)

    response = asyncio.run(request())
    assert response.status_code == 422


def test_ui_and_health_routes_are_available(tmp_path: Path) -> None:
    app = create_app(
        AnswerService(_Retriever([]), _Generator("unused")),
        log_file=tmp_path / "requests.jsonl",
    )

    async def request() -> tuple[httpx.Response, httpx.Response]:
        async with _client(app) as client:
            return await client.get("/"), await client.get("/health")

    page, health = asyncio.run(request())

    assert page.status_code == 200
    assert "RetailRAG — RAG-Powered Store Support" in page.text
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}


def test_chat_returns_service_unavailable_on_dependency_failure(tmp_path: Path) -> None:
    class BrokenService(AnswerService):
        def answer(self, query: str) -> ChatAnswer:
            raise RuntimeError("internal provider error")

    log_file = tmp_path / "requests.jsonl"
    app = create_app(BrokenService(_Retriever([]), _Generator("unused")), log_file=log_file)

    async def request() -> httpx.Response:
        async with _client(app) as client:
            return await client.post("/chat", json={"message": "question"})

    response = asyncio.run(request())

    assert response.status_code == 503
    assert response.json()["detail"].startswith("The support service")
    assert '"event_type":"chat_error"' in log_file.read_text(encoding="utf-8")


def test_ready_requires_database_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("storeguide.app.DATABASE_URL", "")
    app = create_app(
        AnswerService(_Retriever([]), _Generator("unused")),
        log_file=tmp_path / "requests.jsonl",
    )

    async def request() -> httpx.Response:
        async with _client(app) as client:
            return await client.get("/ready")

    response = asyncio.run(request())

    assert response.status_code == 503
    assert response.json()["detail"] == "DATABASE_URL is not configured"


def test_ready_checks_database_chunks_and_required_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeConnection:
        def __enter__(self) -> FakeConnection:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def execute(self, query: str) -> FakeConnection:
            assert query == "SELECT count(*) FROM chunks"
            return self

        def fetchone(self) -> tuple[int]:
            return (8,)

    class FakeOllamaResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "models": [
                    {"name": "nomic-embed-text:latest"},
                    {"name": "gemma4:e4b"},
                ]
            }

    class FakeOllamaClient:
        def __enter__(self) -> FakeOllamaClient:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def get(self, url: str) -> FakeOllamaResponse:
            assert url == "http://ollama-test/api/tags"
            return FakeOllamaResponse()

    monkeypatch.setattr(app_module, "DATABASE_URL", "postgresql://test")
    monkeypatch.setattr(app_module, "OLLAMA_BASE_URL", "http://ollama-test")
    monkeypatch.setattr(app_module.psycopg, "connect", lambda *args, **kwargs: FakeConnection())
    monkeypatch.setattr(app_module.httpx, "Client", lambda **kwargs: FakeOllamaClient())
    app = create_app(
        AnswerService(_Retriever([]), _Generator("unused")),
        log_file=tmp_path / "requests.jsonl",
    )

    async def request() -> httpx.Response:
        async with _client(app) as client:
            return await client.get("/ready")

    response = asyncio.run(request())

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "indexed_chunks": 8,
        "embedding_model": "nomic-embed-text",
        "generation_model": "gemma4:e4b",
        "reranker_model": "BAAI/bge-reranker-base",
    }


def test_metrics_route_reports_empty_log(tmp_path: Path) -> None:
    app = create_app(
        AnswerService(_Retriever([]), _Generator("unused")),
        log_file=tmp_path / "missing.jsonl",
    )

    async def request() -> httpx.Response:
        async with _client(app) as client:
            return await client.get("/metrics")

    response = asyncio.run(request())

    assert response.status_code == 200
    assert response.json()["requests"] == 0

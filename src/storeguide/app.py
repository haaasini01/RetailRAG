import logging
from pathlib import Path
import time
from typing import Annotated

import httpx
import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from storeguide.answering import AnswerService
from storeguide.config import (
    DATABASE_URL,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    LOG_FILE,
    OLLAMA_BASE_URL,
    RERANKER_MODEL,
)
from storeguide.observability import chat_event, load_metrics, now_iso, write_event

STATIC_DIR = Path(__file__).parent / "static"
logger = logging.getLogger("storeguide.api")


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: Annotated[str, Field(min_length=1, max_length=2000)]

    @field_validator("message")
    @classmethod
    def strip_and_validate_message(cls, value: str) -> str:
        message = value.strip()
        if not message:
            raise ValueError("message must not be blank")
        return message


class SourceResponse(BaseModel):
    article_id: str
    title: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    citations: list[str]
    sources: list[SourceResponse]
    model: str
    grounded: bool
    refused: bool
    latency_ms: float
    retrieved_chunks: int
    top_score: float
    query_term_coverage: float


def create_app(
    answer_service: AnswerService | None = None,
    log_file: Path = LOG_FILE,
) -> FastAPI:
    application = FastAPI(
        title="RetailRAG",
        version="0.1.0",
        description="A grounded support assistant for a fictional online store",
    )
    service = answer_service or AnswerService()
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @application.get("/", response_class=FileResponse, include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post("/chat", response_model=ChatResponse)
    def chat(body: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        try:
            result = service.answer(body.message)
        except (RuntimeError, psycopg.Error) as exc:
            logger.exception("Chat request failed (%s)", type(exc).__name__)
            write_event(
                {
                    "event_type": "chat_error",
                    "timestamp": now_iso(),
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "error_type": type(exc).__name__,
                },
                log_file,
            )
            raise HTTPException(
                status_code=503,
                detail="The support service is temporarily unavailable. Please try again.",
            ) from exc
        write_event(
            chat_event(
                body.message,
                latency_ms=result.latency_ms,
                retrieved_chunks=result.retrieved_count,
                top_score=result.top_score,
                query_term_coverage=result.query_term_coverage,
                model=result.model,
                refused=result.refused,
                citations=result.citations,
            ),
            log_file,
        )
        return ChatResponse(
            answer=result.answer,
            citations=result.citations,
            sources=[
                SourceResponse(
                    article_id=source.article_id,
                    title=source.title,
                    score=round(source.score, 4),
                )
                for source in result.sources
            ],
            model=result.model,
            grounded=result.grounded,
            refused=result.refused,
            latency_ms=result.latency_ms,
            retrieved_chunks=result.retrieved_count,
            top_score=round(result.top_score, 4),
            query_term_coverage=round(result.query_term_coverage, 4),
        )

    @application.get("/ready")
    def readiness() -> dict[str, object]:
        if not DATABASE_URL:
            raise HTTPException(status_code=503, detail="DATABASE_URL is not configured")
        try:
            with psycopg.connect(DATABASE_URL, connect_timeout=3) as connection:
                indexed_chunks = int(
                    connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
                )
        except psycopg.Error as exc:
            logger.exception("Database readiness check failed")
            raise HTTPException(
                status_code=503, detail="The knowledge database is unavailable"
            ) from exc
        if indexed_chunks == 0:
            raise HTTPException(status_code=503, detail="The knowledge base is empty")

        try:
            with httpx.Client(timeout=5) as client:
                response = client.get(f"{OLLAMA_BASE_URL}/api/tags")
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.exception("Ollama readiness check failed")
            raise HTTPException(
                status_code=503, detail="The local model service is unavailable"
            ) from exc
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            raise HTTPException(
                status_code=503, detail="The local model service returned invalid status"
            )
        installed = {
            entry["name"]
            for entry in models
            if isinstance(entry, dict) and isinstance(entry.get("name"), str)
        }
        required = (EMBEDDING_MODEL, GENERATION_MODEL)
        missing = [
            model
            for model in required
            if model not in installed and f"{model}:latest" not in installed
        ]
        if missing:
            raise HTTPException(
                status_code=503,
                detail="One or more required Ollama models are not installed",
            )
        try:
            service.warmup()
        except (httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
            logger.exception("Cross-encoder readiness check failed")
            raise HTTPException(
                status_code=503, detail="The cross-encoder reranker is unavailable"
            ) from exc
        return {
            "status": "ready",
            "indexed_chunks": indexed_chunks,
            "embedding_model": EMBEDDING_MODEL,
            "generation_model": GENERATION_MODEL,
            "reranker_model": RERANKER_MODEL,
        }

    @application.get("/metrics")
    def metrics() -> dict[str, object]:
        return load_metrics(log_file)

    return application


app = create_app()

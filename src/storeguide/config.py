from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()


def read_port() -> int:
    value = os.getenv("PORT", "8000")
    try:
        port = int(value)
    except ValueError as exc:
        raise ValueError("PORT must be an integer between 1 and 65535") from exc
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be an integer between 1 and 65535")
    return port


HOST = os.getenv("HOST", "127.0.0.1")
PORT = read_port()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_positive_int(name: str, default: int) -> int:
    value = os.getenv(name, str(default))
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def read_path(name: str, default: str) -> Path:
    path = Path(os.getenv(name, default))
    return path if path.is_absolute() else PROJECT_ROOT / path


DATABASE_URL = os.getenv("DATABASE_URL", "")
KB_DIR = read_path("KB_DIR", "data/kb")
LOG_FILE = read_path("LOG_FILE", "logs/requests.jsonl")
RERANKER_CACHE_DIR = read_path("RERANKER_CACHE_DIR", "model-cache")
CHUNK_TARGET_WORDS = read_positive_int("CHUNK_TARGET_WORDS", 180)
CHUNK_OVERLAP_WORDS = read_positive_int("CHUNK_OVERLAP_WORDS", 40)
if CHUNK_OVERLAP_WORDS >= CHUNK_TARGET_WORDS:
    raise ValueError("CHUNK_OVERLAP_WORDS must be less than CHUNK_TARGET_WORDS")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIMENSIONS = read_positive_int("EMBEDDING_DIMENSIONS", 768)
OLLAMA_TIMEOUT_SECONDS = read_positive_int("OLLAMA_TIMEOUT_SECONDS", 120)
if EMBEDDING_DIMENSIONS != 768:
    raise ValueError("EMBEDDING_DIMENSIONS must be 768 for nomic-embed-text and the DB schema")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemma4:e4b")
GENERATION_TIMEOUT_SECONDS = read_positive_int("GENERATION_TIMEOUT_SECONDS", 180)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")


def read_metric_threshold(name: str, default: float) -> float:
    value = os.getenv(name, str(default))
    try:
        threshold = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be between 0 and 1") from exc
    if not 0 <= threshold <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return threshold


def read_score_threshold() -> float:
    value = os.getenv("MIN_RELEVANCE_SCORE", "0.002")
    try:
        score = float(value)
    except ValueError as exc:
        raise ValueError("MIN_RELEVANCE_SCORE must be between 0 and 1") from exc
    if not 0 <= score <= 1:
        raise ValueError("MIN_RELEVANCE_SCORE must be between 0 and 1")
    return score


MIN_RELEVANCE_SCORE = read_score_threshold()
MIN_QUERY_TERM_COVERAGE = read_metric_threshold("MIN_QUERY_TERM_COVERAGE", 0.60)
EVAL_HIT_RATE_THRESHOLD = read_metric_threshold("EVAL_HIT_RATE_THRESHOLD", 0.90)
EVAL_REFUSAL_THRESHOLD = read_metric_threshold("EVAL_REFUSAL_THRESHOLD", 1.0)
EVAL_IN_CORPUS_ACCEPTANCE_THRESHOLD = read_metric_threshold(
    "EVAL_IN_CORPUS_ACCEPTANCE_THRESHOLD", 0.95
)

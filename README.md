# RetailRAG — RAG-Powered Store Support

RetailRAG is a retrieval-augmented generation (RAG) support assistant for a
fictional online store. It retrieves original store policies and help articles as
evidence, returns cited answers, and declines when the evidence is insufficient.

See [Project stages](PROJECT_STAGES.md) for the architecture, stack, and roadmap.

## Run locally

Requirements: Python 3.11 or newer and Docker Desktop (for the local PostgreSQL
database).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
ollama pull nomic-embed-text
ollama pull gemma4:e4b
docker compose up -d db
python -m storeguide.ingest
python -m storeguide.server
```

Open `http://127.0.0.1:8000/` for the support chat. `GET /health` is the
liveness check; `GET /ready` checks PostgreSQL, indexed chunks, and both required
Ollama models. Run tests with `python -m pytest`.

The original fictional Pine & Peak Market policies are in `data/kb/`. They
intentionally include interacting timelines, conditional exceptions, shipping
thresholds, return exclusions, warranty limits, and account-specific questions;
the fictional policies are test material, not real consumer advice. Each Markdown
file has YAML front matter containing a unique slug `id`, a `title`, and a slug
`category`. Ingestion validates the metadata, splits article text into overlapping
word-bounded chunks, creates 768-dimensional `nomic-embed-text` document vectors
through Ollama, and synchronizes articles and chunks in one database transaction.
Running it again updates changed documents and removes articles whose files were
deleted. The database is available on localhost port 5433; local development
credentials are in `.env.example`. The Ollama service must be running locally.

The retrieval baseline uses query-prefixed `nomic-embed-text` embeddings and
PostgreSQL pgvector cosine search. Live retrieval now follows the architecture
shown in the project diagram:

1. Retrieve up to 20 candidates independently with pgvector cosine similarity
   and BM25 lexical ranking.
2. Combine the two ranked lists with reciprocal-rank fusion (RRF, rank constant
   60), deduplicating by chunk ID.
3. Rerank the fused candidate pool with the `BAAI/bge-reranker-base`
   cross-encoder. This model scores query/passage pairs; it does not replace
   `nomic-embed-text` or the answer model.
4. Keep the top five reranked passages. Apply the evidence gate, then ask local
   Ollama `gemma4:e4b` to produce a grounded answer with validated citations—or
   abstain.

The reranker outputs a logit which is converted through a sigmoid to a
0–1 relevance score; `MIN_RELEVANCE_SCORE` applies to that reranker score. The
current provisional score threshold is `0.002`, calibrated against this test
set after adding explicit competitor-price policy language. Do not reuse the
old cosine-similarity threshold for cross-encoder scores.
`GET /ready` initializes and verifies the cross-encoder as well as PostgreSQL
and Ollama. On first startup, FastEmbed downloads the BGE model (about 1 GB) from
Hugging Face into `RERANKER_CACHE_DIR` (`model-cache/` by default); allow time
and network access on that first readiness check. Docker Compose keeps the model
in a named volume.

The 48 supported policy questions and 16 unsupported or tool-required questions
(64 total) live in `eval/`. Run
`python -m storeguide.evaluation` to recalculate Hit Rate@5, MRR, Recall@5,
in-corpus acceptance, and out-of-corpus refusal accuracy; the command writes
per-question results to `eval/evaluation-results.json`. Seven cases require
account/order-specific lookup or action tools. RetailRAG has no such tools, so it
routes those requests to a deterministic sign-in/support response instead of
guessing from policy text. The original 18-question baseline is stored in
`eval/baseline-vector-results.json`.
Search one question with
`python -m storeguide.retrieve "How long does standard shipping take?"`.

The `POST /chat` endpoint accepts `{"message":"..."}` and returns an answer,
source citations, a refusal flag, and retrieval/latency metadata. JSON structured
events are written to stdout and `logs/requests.jsonl`; they contain a SHA-256
query fingerprint and character count, not the raw customer question.
`GET /metrics` aggregates request count, errors, average and p95 latency, average
retrieved chunks, refusal rate, and model counts from that JSONL file. The chat
page uses same-origin requests and does not send questions to any third-party
service; the typefaces use local system fonts.

The retrieval gate requires both reranker relevance of at least `0.002` and
normalized query-term coverage of at least `0.60` across the reranked passages.
These provisional thresholds pass the current 64-case regression set but are
not production-calibrated. The questions were
authored alongside the fictional policies, so a separate held-out set is needed
for an unbiased quality estimate.
CI runs the unit suite plus the hybrid retrieval/refusal regression evaluation
against PostgreSQL, Ollama embeddings, BM25, and the cross-encoder.

## Run the full stack with Docker Compose

The default Compose setup runs the FastAPI app and PostgreSQL/pgvector in
containers, and connects to Ollama running on the Windows host. Start Docker
Desktop, make sure Ollama is running with both models pulled, then:

```powershell
Copy-Item .env.example .env
ollama pull nomic-embed-text
ollama pull gemma4:e4b
docker compose up --build -d
docker compose exec app python -m storeguide.ingest
```

The app is at `http://127.0.0.1:8000/`. After ingestion,
`http://127.0.0.1:8000/ready` should report readiness. The app health check
remains unhealthy until the database has indexed chunks and Ollama has both
models; this does not prevent running the ingestion command.

To run Ollama in a container instead, set
`OLLAMA_BASE_URL_DOCKER=http://ollama:11434` in `.env`, then start the optional
profile and download the models into its persistent volume:

```powershell
docker compose --profile ollama up --build -d
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull gemma4:e4b
docker compose exec app python -m storeguide.ingest
```

The Ollama image does not automatically download the large generation model.
Default database credentials in `.env.example` are for local development only;
change them before exposing the service. Compose stores database data in a named
volume and request events in `logs/`.

To stop the database while retaining its data, run `docker compose down`.

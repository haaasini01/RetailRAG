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

The original fictional store policies are in `data/kb/`. Each Markdown file has
YAML front matter containing a unique slug `id`, a `title`, and a slug `category`.
Ingestion validates the metadata, splits article text into overlapping
word-bounded chunks, creates 768-dimensional `nomic-embed-text` document vectors
through Ollama, and synchronizes articles and chunks in one database transaction.
Running it again updates changed documents and removes articles whose files were
deleted. The database is available on localhost port 5433; local development
credentials are in `.env.example`. The Ollama service must be running locally.

The retrieval baseline uses query-prefixed `nomic-embed-text` embeddings and
PostgreSQL pgvector cosine search. The 18 labeled questions live in
`eval/retrieval_cases.json`. Run `python -m storeguide.evaluation` to recalculate
Hit Rate@5, MRR, Recall@5, in-corpus acceptance, and out-of-corpus refusal accuracy;
the command writes per-question results to `eval/evaluation-results.json`. The
baseline-only 18-question result is stored in `eval/baseline-vector-results.json`.
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

The retrieval gate requires both cosine similarity of at least `0.60` and
meaningful query-term coverage of at least `0.60` across retrieved passages. These
initial thresholds separate the current 18 supported and 12 unsupported evaluation
questions; add representative cases before treating them as production-calibrated.
CI runs the unit suite plus the retrieval/refusal regression evaluation against
PostgreSQL and the local Ollama embedding model.

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

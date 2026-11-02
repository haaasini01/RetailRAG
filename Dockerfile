FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
RUN python -m pip install --upgrade pip \
    && python -m pip install .

COPY data ./data
COPY eval ./eval
RUN mkdir -p /app/logs

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=8s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/ready || exit 1

CMD ["python", "-m", "storeguide.server"]

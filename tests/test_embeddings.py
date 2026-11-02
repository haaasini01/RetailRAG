import httpx
import pytest

from storeguide.embeddings import OllamaEmbedder


def _client(handler) -> httpx.Client:
    return httpx.Client(
        base_url="http://ollama.test",
        transport=httpx.MockTransport(handler),
    )


def test_embedder_uses_search_document_and_query_prefixes() -> None:
    requests: list[dict[str, str]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        import json

        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"embedding": [0.25, -0.5]})

    with OllamaEmbedder(
        model="test-model",
        dimensions=2,
        client=_client(respond),
    ) as embedder:
        documents = embedder.embed_documents(["shipping takes five days"])
        query = embedder.embed_query("how long does shipping take")

    assert documents == [[0.25, -0.5]]
    assert query == [0.25, -0.5]
    assert requests == [
        {"model": "test-model", "prompt": "search_document: shipping takes five days"},
        {"model": "test-model", "prompt": "search_query: how long does shipping take"},
    ]


def test_embedder_rejects_wrong_embedding_dimensions() -> None:
    client = _client(
        lambda request: httpx.Response(200, json={"embedding": [0.1]})
    )
    with OllamaEmbedder(dimensions=2, client=client) as embedder:
        with pytest.raises(RuntimeError, match="expected 2"):
            embedder.embed_query("shipping question")


def test_embedder_rejects_non_finite_values() -> None:
    client = _client(
        lambda request: httpx.Response(200, content=b'{"embedding":[NaN]}')
    )
    with OllamaEmbedder(dimensions=1, client=client) as embedder:
        with pytest.raises(RuntimeError, match="non-finite"):
            embedder.embed_query("question")


def test_embedder_surfaces_ollama_http_errors() -> None:
    client = _client(
        lambda request: httpx.Response(404, json={"error": "model not found"})
    )
    with OllamaEmbedder(client=client) as embedder:
        with pytest.raises(RuntimeError, match="embedding request failed"):
            embedder.embed_query("question")


def test_embedder_rejects_empty_text() -> None:
    client = _client(lambda request: httpx.Response(500))
    with OllamaEmbedder(client=client) as embedder:
        with pytest.raises(ValueError, match="empty text"):
            embedder.embed_query("  ")
    client.close()

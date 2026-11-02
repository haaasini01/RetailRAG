from dataclasses import dataclass
import json

import httpx
import pytest

from storeguide.answering import (
    REFUSAL_MESSAGE,
    AnswerService,
    OllamaChatGenerator,
)
from storeguide.retrieval import SearchHit


def _hit(article_id: str = "shipping", score: float = 0.8) -> SearchHit:
    return SearchHit(
        chunk_id=f"{article_id}:0000",
        article_id=article_id,
        title="Shipping",
        category="orders",
        content="Standard shipping takes three to five business days.",
        score=score,
    )


@dataclass
class _Retriever:
    hits: list[SearchHit]

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        assert query
        return self.hits[:top_k]


@dataclass
class _Generator:
    answer_text: str
    calls: int = 0

    def generate(self, query: str, sources: list[SearchHit]) -> str:
        assert query
        assert sources
        self.calls += 1
        return self.answer_text


def test_service_generates_answer_with_retrieved_citations() -> None:
    generator = _Generator("Standard shipping takes 3-5 business days. [shipping]")
    service = AnswerService(_Retriever([_hit()]), generator, min_relevance_score=0.6)

    answer = service.answer("How does standard shipping work?")

    assert answer.grounded is True
    assert answer.refused is False
    assert answer.citations == ["shipping"]
    assert answer.sources[0].article_id == "shipping"
    assert generator.calls == 1


def test_service_abstains_below_the_relevance_threshold() -> None:
    generator = _Generator("This must not be generated.")
    service = AnswerService(_Retriever([_hit(score=0.59)]), generator, 0.6)

    answer = service.answer("How does standard shipping work?")

    assert answer.answer == REFUSAL_MESSAGE
    assert answer.refused is True
    assert answer.grounded is False
    assert answer.citations == []
    assert generator.calls == 0


def test_service_abstains_when_no_sources_are_found() -> None:
    answer = AnswerService(_Retriever([]), _Generator("unused")).answer("question")

    assert answer.refused is True
    assert answer.sources == []
    assert answer.citations == []


def test_service_abstains_when_query_terms_are_missing_from_evidence() -> None:
    generator = _Generator("This must not be generated.")
    service = AnswerService(_Retriever([_hit(score=0.8)]), generator, 0.6)

    answer = service.answer("Do you offer gift wrapping?")

    assert answer.refused is True
    assert generator.calls == 0


def test_service_rejects_answer_without_a_valid_source_citation() -> None:
    service = AnswerService(
        _Retriever([_hit()]),
        _Generator("Standard shipping takes three to five business days."),
        min_relevance_score=0.6,
    )

    with pytest.raises(RuntimeError, match="did not cite any retrieved source"):
        service.answer("How does standard shipping work?")


def test_service_ignores_citations_not_in_retrieved_sources() -> None:
    generator = _Generator("Delivery takes 3-5 days [unknown] [shipping].")
    answer = AnswerService(
        _Retriever([_hit()]),
        generator,
        min_relevance_score=0.6,
    ).answer("How does standard shipping work?")

    assert answer.citations == ["shipping"]
    assert [source.article_id for source in answer.sources] == ["shipping"]


def test_generator_sends_only_retrieved_policy_context() -> None:
    requests: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "3-5 business days [shipping]"}},
        )

    client = httpx.Client(
        base_url="http://ollama.test",
        transport=httpx.MockTransport(respond),
    )
    with OllamaChatGenerator(model="gemma4:e4b", client=client) as generator:
        answer = generator.generate("How long is shipping?", [_hit()])
    client.close()

    assert answer == "3-5 business days [shipping]"
    assert requests[0]["model"] == "gemma4:e4b"
    assert requests[0]["stream"] is False
    prompt = requests[0]["messages"][1]["content"]
    assert "[shipping] Shipping" in prompt
    assert "How long is shipping?" in prompt
    assert "untrusted reference data" in requests[0]["messages"][0]["content"]


def test_generator_surfaces_empty_model_response() -> None:
    client = httpx.Client(
        base_url="http://ollama.test",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"message": {"content": ""}})
        ),
    )
    with OllamaChatGenerator(client=client) as generator:
        with pytest.raises(RuntimeError, match="empty or invalid answer"):
            generator.generate("Question?", [_hit()])
    client.close()

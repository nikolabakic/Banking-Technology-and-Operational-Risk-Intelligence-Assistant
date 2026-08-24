from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from bankscope.tools import (
    FallbackWebSearchProvider,
    OpenAIWebSearchProvider,
    TavilyWebSearchProvider,
    WebSearchAuthenticationError,
    WebSearchCitation,
    WebSearchError,
    WebSearchFallbackError,
    WebSearchNoResultError,
    WebSearchProvider,
    WebSearchRateLimitError,
    WebSearchResult,
    WebSearchSource,
    WebSearchTimeoutError,
)
from bankscope.tools.web_search import MAX_WEB_QUERY_LENGTH, TAVILY_SEARCH_URL


class FakeResponses:
    def __init__(self, *, response: object = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, responses: FakeResponses) -> None:
        self.responses = responses


class FakeRequestTimeout(RuntimeError):
    pass


class FakeHTTPResponse:
    def __init__(
        self,
        status_code: int,
        payload: object = None,
        *,
        json_error: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self.payload = payload
        self.json_error = json_error

    def json(self) -> object:
        if self.json_error is not None:
            raise self.json_error
        return self.payload


class FakeHTTPClient:
    def __init__(self, *outcomes: object) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []
        self.closed = False

    def post(self, url: str, **kwargs: object) -> object:
        self.calls.append({"url": url, **kwargs})
        if not self.outcomes:
            raise AssertionError("Unexpected Tavily HTTP request.")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self) -> None:
        self.closed = True


class StubWebSearchProvider:
    def __init__(self, *outcomes: object) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []

    def search(
        self,
        query: str,
        *,
        allowed_domains: tuple[str, ...] = (),
        blocked_domains: tuple[str, ...] = (),
    ) -> WebSearchResult:
        self.calls.append((query, tuple(allowed_domains), tuple(blocked_domains)))
        if not self.outcomes:
            raise AssertionError("Unexpected fallback-provider call.")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, WebSearchResult)
        return outcome


class ClosableStubWebSearchProvider(StubWebSearchProvider):
    def __init__(self, *outcomes: object, close_error: Exception | None = None) -> None:
        super().__init__(*outcomes)
        self.close_error = close_error
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def _provider(*, response: object = None, error: Exception | None = None):
    responses = FakeResponses(response=response, error=error)
    provider = OpenAIWebSearchProvider(
        client=FakeClient(responses),
        model="gpt-test",
        timeout_seconds=12.5,
    )
    return provider, responses


def _tavily_provider(
    *outcomes: object,
    max_results: int = 5,
) -> tuple[TavilyWebSearchProvider, FakeHTTPClient]:
    client = FakeHTTPClient(*outcomes)
    provider = TavilyWebSearchProvider(
        "tvly-secret",
        http_client=client,
        timeout_seconds=9,
        max_results=max_results,
    )
    return provider, client


def _search_result(provider: str) -> WebSearchResult:
    return WebSearchResult(
        query="Question",
        text=f"{provider} answer",
        citations=(),
        sources=(WebSearchSource(url=f"https://{provider}.example/source"),),
        used_web_search=True,
        provider=provider,
    )


def test_openai_provider_returns_typed_text_citations_and_sources() -> None:
    text = "Ally describes operational risk in its filing."
    response = {
        "id": "resp_123",
        "status": "completed",
        "output_text": text,
        "output": [
            {
                "type": "web_search_call",
                "status": "completed",
                "action": {
                    "type": "search",
                    "sources": [
                        {"type": "url", "url": "https://www.sec.gov/filing"},
                        {"type": "url", "url": "ftp://unsafe.example/file"},
                        {"type": "url", "url": "https://user:secret@example.com/file"},
                        {"type": "url", "url": "https://www.sec.gov/filing"},
                    ],
                },
            },
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://www.sec.gov/filing",
                                "title": "Ally 2025 Form 10-K",
                                "start_index": 0,
                                "end_index": 4,
                            },
                            {
                                "type": "url_citation",
                                "url": "javascript:alert(1)",
                                "title": "Unsafe",
                                "start_index": 0,
                                "end_index": 4,
                            },
                        ],
                    }
                ],
            },
        ],
    }
    provider, responses = _provider(response=response)

    result = provider.search(
        "  How does Ally define operational risk?  ",
        allowed_domains=["SEC.gov", "www.ally.com", "sec.gov"],
    )

    assert result.text == text
    assert result.query == "How does Ally define operational risk?"
    assert result.citations == (
        WebSearchCitation(
            url="https://www.sec.gov/filing",
            title="Ally 2025 Form 10-K",
            start_index=0,
            end_index=4,
        ),
    )
    assert result.sources == (WebSearchSource(url="https://www.sec.gov/filing"),)
    assert result.used_web_search is True
    assert result.provider == "openai"
    assert result.response_id == "resp_123"
    assert result.request_count == 1

    assert len(responses.calls) == 1
    call = responses.calls[0]
    assert call["model"] == "gpt-test"
    assert call["input"] == result.query
    assert call["tool_choice"] == "required"
    assert call["include"] == ["web_search_call.action.sources"]
    assert call["timeout"] == 12.5
    assert call["tools"] == [
        {
            "type": "web_search",
            "search_context_size": "medium",
            "filters": {
                "allowed_domains": ["sec.gov", "www.ally.com"],
            },
        }
    ]


def test_openai_provider_extracts_sdk_style_objects_and_combines_text_offsets() -> None:
    first_text = "First."
    second_text = "Second source."
    citation = SimpleNamespace(
        type="url_citation",
        url="https://example.com/two",
        title="Second",
        start_index=0,
        end_index=6,
    )
    response = SimpleNamespace(
        id="resp_objects",
        status="completed",
        error=None,
        output_text=first_text + second_text,
        output=[
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(type="output_text", text=first_text, annotations=[]),
                    SimpleNamespace(
                        type="output_text",
                        text=second_text,
                        annotations=[citation],
                    ),
                ],
            )
        ],
    )
    provider, _ = _provider(response=response)

    result = provider.search("A stable fact the model may answer directly")

    assert result.text == "First.Second source."
    assert result.citations[0].start_index == len(first_text)
    assert result.citations[0].end_index == len(first_text) + 6
    assert result.used_web_search is False


def test_openai_provider_accepts_nested_legacy_citation_shape() -> None:
    text = "Cited answer"
    response = {
        "output_text": text,
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url_citation": {
                                    "url": "https://example.com/source",
                                    "title": "Source",
                                    "start_index": 0,
                                    "end_index": 5,
                                },
                            }
                        ],
                    }
                ],
            }
        ],
    }
    provider, _ = _provider(response=response)

    result = provider.search("Question")

    assert result.citations == (
        WebSearchCitation(
            url="https://example.com/source",
            title="Source",
            start_index=0,
            end_index=5,
        ),
    )


def test_openai_provider_uses_content_text_when_output_text_property_is_absent() -> None:
    response = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Fallback text", "annotations": []}],
            }
        ]
    }
    provider, _ = _provider(response=response)

    assert provider.search("Question").text == "Fallback text"


@pytest.mark.parametrize("error", [TimeoutError(), FakeRequestTimeout()])
def test_openai_provider_maps_timeout_errors(error: Exception) -> None:
    provider, _ = _provider(error=error)

    with pytest.raises(WebSearchTimeoutError, match="timed out"):
        provider.search("Question")


def test_openai_provider_wraps_other_client_errors() -> None:
    provider, _ = _provider(error=ConnectionError("secret upstream detail"))

    with pytest.raises(WebSearchError, match="request failed") as captured:
        provider.search("Question")

    assert isinstance(captured.value.__cause__, ConnectionError)


@pytest.mark.parametrize(
    "response",
    [
        {"output_text": "", "output": []},
        {"output_text": "   ", "output": []},
        {"output": []},
    ],
)
def test_openai_provider_reports_no_result(response: object) -> None:
    provider, _ = _provider(response=response)

    with pytest.raises(WebSearchNoResultError, match="no answer text"):
        provider.search("Question")


@pytest.mark.parametrize(
    "response",
    [
        None,
        {"error": {"code": "upstream_error"}, "output_text": "Partial"},
        {"status": "failed", "output_text": "Partial"},
    ],
)
def test_openai_provider_reports_invalid_or_failed_responses(response: object) -> None:
    provider, _ = _provider(response=response)

    with pytest.raises(WebSearchError):
        provider.search("Question")


@pytest.mark.parametrize("query", ["", "   ", None, "x" * (MAX_WEB_QUERY_LENGTH + 1)])
def test_openai_provider_validates_query_before_calling_client(query: object) -> None:
    provider, responses = _provider(response={"output_text": "unused"})

    with pytest.raises(ValueError, match="query"):
        provider.search(query)  # type: ignore[arg-type]

    assert responses.calls == []


@pytest.mark.parametrize(
    "domains",
    [
        "sec.gov",
        ["https://sec.gov"],
        ["sec.gov/path"],
        ["bad_domain.example"],
        [""],
    ],
)
def test_openai_provider_validates_domain_filters(domains: object) -> None:
    provider, responses = _provider(response={"output_text": "unused"})

    with pytest.raises(ValueError, match="domain|allowed_domains"):
        provider.search("Question", allowed_domains=domains)  # type: ignore[arg-type]

    assert responses.calls == []


def test_openai_provider_rejects_blocked_domains_as_unsupported() -> None:
    provider, responses = _provider(response={"output_text": "unused"})

    with pytest.raises(ValueError, match="only allowed_domains.*blocked_domains is unsupported"):
        provider.search(
            "Question",
            blocked_domains=["reddit.com"],
        )

    assert responses.calls == []


def test_openai_provider_validates_configuration_and_matches_protocol() -> None:
    provider, _ = _provider(response={"output_text": "Answer"})
    assert isinstance(provider, WebSearchProvider)

    with pytest.raises(ValueError, match="model"):
        OpenAIWebSearchProvider(client=object(), model=" ")
    with pytest.raises(ValueError, match="timeout"):
        OpenAIWebSearchProvider(client=object(), model="gpt-test", timeout_seconds=0)
    with pytest.raises(ValueError, match="context size"):
        OpenAIWebSearchProvider(
            client=object(),
            model="gpt-test",
            search_context_size="huge",  # type: ignore[arg-type]
        )


def test_tavily_provider_posts_official_contract_and_returns_ranked_sources() -> None:
    response = FakeHTTPResponse(
        200,
        {
            "query": "How does Ally define operational risk?",
            "answer": "Operational risk includes losses from failed processes and systems.",
            "request_id": "tavily-request-123",
            "results": [
                {
                    "title": "Ally 2025 Form 10-K",
                    "url": "https://www.sec.gov/Archives/ally-2025-10k.htm",
                    "content": "Ally defines operational risk as the risk of loss...",
                    "score": 0.98,
                    "raw_content": "must not be used",
                },
                {
                    "title": "Unsafe result",
                    "url": "javascript:alert(1)",
                    "content": "unsafe",
                    "score": 1.0,
                },
                {
                    "title": "Duplicate",
                    "url": "https://www.sec.gov/Archives/ally-2025-10k.htm",
                    "content": "duplicate",
                    "score": 0.4,
                },
            ],
        },
    )
    provider, client = _tavily_provider(response, max_results=7)

    result = provider.search(
        "  How does Ally define operational risk?  ",
        allowed_domains=["SEC.gov", "ally.com", "sec.gov"],
        blocked_domains=["reddit.com"],
    )

    assert result.text == "Operational risk includes losses from failed processes and systems."
    assert result.query == "How does Ally define operational risk?"
    assert result.citations == ()
    assert result.sources == (
        WebSearchSource(
            url="https://www.sec.gov/Archives/ally-2025-10k.htm",
            title="Ally 2025 Form 10-K",
            snippet="Ally defines operational risk as the risk of loss...",
            score=0.98,
        ),
    )
    assert result.used_web_search is True
    assert result.provider == "tavily"
    assert result.response_id == "tavily-request-123"

    assert client.calls == [
        {
            "url": TAVILY_SEARCH_URL,
            "headers": {
                "Authorization": "Bearer tvly-secret",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            "json": {
                "query": "How does Ally define operational risk?",
                "search_depth": "basic",
                "include_answer": "basic",
                "include_raw_content": False,
                "max_results": 7,
                "include_domains": ["sec.gov", "ally.com"],
                "exclude_domains": ["reddit.com"],
            },
            "timeout": 9.0,
        }
    ]


def test_tavily_provider_uses_snippets_when_answer_is_missing() -> None:
    response = FakeHTTPResponse(
        200,
        {
            "answer": None,
            "results": [
                {
                    "title": "SEC filing",
                    "url": "https://sec.gov/filing",
                    "content": "Relevant filing excerpt.",
                    "score": 0.75,
                }
            ],
        },
    )
    provider, _ = _tavily_provider(response)

    result = provider.search("Question")

    assert result.text == (
        "Search results:\n\n1. SEC filing (https://sec.gov/filing)\nRelevant filing excerpt."
    )
    assert result.sources[0].snippet == "Relevant filing excerpt."


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [
        (401, WebSearchAuthenticationError),
        (403, WebSearchAuthenticationError),
        (429, WebSearchRateLimitError),
        (432, WebSearchRateLimitError),
        (433, WebSearchRateLimitError),
    ],
)
def test_tavily_provider_maps_authentication_and_usage_errors(
    status_code: int, error_type: type[WebSearchError]
) -> None:
    provider, _ = _tavily_provider(FakeHTTPResponse(status_code, {"detail": "secret"}))

    with pytest.raises(error_type):
        provider.search("Question")


def test_tavily_provider_maps_timeout_and_transport_errors() -> None:
    timeout_provider, _ = _tavily_provider(httpx.ReadTimeout("timed out"))
    with pytest.raises(WebSearchTimeoutError, match="timed out"):
        timeout_provider.search("Question")

    failed_provider, _ = _tavily_provider(httpx.ConnectError("upstream detail"))
    with pytest.raises(WebSearchError, match="request failed") as captured:
        failed_provider.search("Question")
    assert isinstance(captured.value.__cause__, httpx.ConnectError)


@pytest.mark.parametrize(
    "response",
    [
        FakeHTTPResponse(500, {"detail": "internal"}),
        FakeHTTPResponse(200, json_error=ValueError("invalid json")),
        FakeHTTPResponse(200, ["not", "an", "object"]),
    ],
)
def test_tavily_provider_reports_http_and_response_errors(response: object) -> None:
    provider, _ = _tavily_provider(response)

    with pytest.raises(WebSearchError):
        provider.search("Question")


@pytest.mark.parametrize(
    "payload",
    [
        {"answer": "", "results": []},
        {
            "answer": "Uncited answer",
            "results": [{"url": "ftp://unsafe.example/result", "content": "unsafe"}],
        },
    ],
)
def test_tavily_provider_reports_no_usable_results(payload: object) -> None:
    provider, _ = _tavily_provider(FakeHTTPResponse(200, payload))

    with pytest.raises(WebSearchNoResultError, match="no usable results"):
        provider.search("Question")


@pytest.mark.parametrize(
    "options",
    [
        {"api_key": ""},
        {"api_key": "key with spaces"},
        {"api_key": "key", "max_results": 0},
        {"api_key": "key", "max_results": 21},
        {"api_key": "key", "max_results": True},
        {"api_key": "key", "timeout_seconds": 0},
    ],
)
def test_tavily_provider_validates_configuration(options: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TavilyWebSearchProvider(http_client=FakeHTTPClient(), **options)  # type: ignore[arg-type]


def test_tavily_provider_validates_input_before_http_request() -> None:
    provider, client = _tavily_provider(FakeHTTPResponse(200, {}))

    with pytest.raises(ValueError, match="query"):
        provider.search(" ")
    with pytest.raises(ValueError, match="both allowed and blocked"):
        provider.search(
            "Question",
            allowed_domains=["sec.gov"],
            blocked_domains=["SEC.gov"],
        )

    assert client.calls == []


def test_tavily_provider_does_not_close_injected_client() -> None:
    provider, client = _tavily_provider()

    provider.close()

    assert client.closed is False
    assert isinstance(provider, WebSearchProvider)


def test_fallback_remembers_first_successful_provider() -> None:
    openai = StubWebSearchProvider(WebSearchError("OpenAI endpoint unavailable"))
    tavily = StubWebSearchProvider(_search_result("tavily"), _search_result("tavily"))
    provider = FallbackWebSearchProvider([openai, tavily])

    first = provider.search("Question", allowed_domains=("sec.gov",))
    second = provider.search("Question", allowed_domains=("sec.gov",))

    assert first.provider == second.provider == "tavily"
    assert first.request_count == 2
    assert second.request_count == 1
    assert len(openai.calls) == 1
    assert len(tavily.calls) == 2
    assert provider.preferred_provider_index == 1
    assert tavily.calls[0] == ("Question", ("sec.gov",), ())


def test_fallback_recovers_when_remembered_provider_later_fails() -> None:
    openai = StubWebSearchProvider(
        WebSearchError("OpenAI endpoint unavailable"),
        _search_result("openai"),
    )
    tavily = StubWebSearchProvider(
        _search_result("tavily"),
        WebSearchTimeoutError("Tavily timeout"),
    )
    provider = FallbackWebSearchProvider([openai, tavily])

    first = provider.search("Question")
    second = provider.search("Question")

    assert first.provider == "tavily"
    assert first.request_count == 2
    assert second.provider == "openai"
    assert second.request_count == 2
    assert provider.preferred_provider_index == 0
    assert len(openai.calls) == 2
    assert len(tavily.calls) == 2


def test_fallback_reports_when_every_provider_fails() -> None:
    first_error = WebSearchError("first")
    last_error = WebSearchNoResultError("last")
    provider = FallbackWebSearchProvider(
        [StubWebSearchProvider(first_error), StubWebSearchProvider(last_error)]
    )

    with pytest.raises(WebSearchFallbackError, match="All configured") as captured:
        provider.search("Question")

    assert captured.value.__cause__ is last_error
    assert captured.value.request_count == 2
    assert provider.preferred_provider_index is None


def test_fallback_validates_provider_list_and_matches_protocol() -> None:
    with pytest.raises(ValueError, match="At least one"):
        FallbackWebSearchProvider([])
    with pytest.raises(ValueError, match="implement"):
        FallbackWebSearchProvider([object()])  # type: ignore[list-item]

    provider = FallbackWebSearchProvider([StubWebSearchProvider(_search_result("test"))])
    assert isinstance(provider, WebSearchProvider)


def test_fallback_close_is_idempotent_and_closes_each_unique_child_once() -> None:
    first = ClosableStubWebSearchProvider()
    second = ClosableStubWebSearchProvider()
    provider = FallbackWebSearchProvider([first, second, first])

    provider.close()
    provider.close()

    assert first.close_calls == 1
    assert second.close_calls == 1


def test_fallback_close_attempts_every_child_and_groups_errors() -> None:
    first_error = RuntimeError("first close failed")
    last_error = ValueError("last close failed")
    first = ClosableStubWebSearchProvider(close_error=first_error)
    middle = ClosableStubWebSearchProvider()
    last = ClosableStubWebSearchProvider(close_error=last_error)
    provider = FallbackWebSearchProvider([first, middle, last])

    with pytest.raises(ExceptionGroup, match="failed to close") as captured:
        provider.close()

    assert captured.value.exceptions == (first_error, last_error)
    assert first.close_calls == middle.close_calls == last.close_calls == 1

    provider.close()
    assert first.close_calls == middle.close_calls == last.close_calls == 1


def test_fallback_close_preserves_tavily_injected_client_ownership() -> None:
    tavily, client = _tavily_provider()
    provider = FallbackWebSearchProvider([tavily])

    provider.close()
    provider.close()

    assert client.closed is False

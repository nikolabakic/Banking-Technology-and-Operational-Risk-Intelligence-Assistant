"""Provider-neutral web search with OpenAI, Tavily, and ordered fallback support."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol, runtime_checkable
from urllib.parse import urlsplit

import httpx

MAX_WEB_QUERY_LENGTH = 10_000
MAX_WEB_URL_LENGTH = 4_096
MAX_FILTER_DOMAINS = 100
MAX_WEB_ANSWER_LENGTH = 50_000
MAX_SOURCE_TITLE_LENGTH = 500
MAX_SOURCE_SNIPPET_LENGTH = 4_000
MAX_TAVILY_INCLUDE_DOMAINS = 300
MAX_TAVILY_EXCLUDE_DOMAINS = 150
MAX_TAVILY_RESULTS = 20
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
SearchContextSize = Literal["low", "medium", "high"]

_DOMAIN_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_FAILED_RESPONSE_STATUSES = frozenset({"failed", "cancelled", "canceled"})


@dataclass(frozen=True, slots=True)
class WebSearchCitation:
    """One validated inline URL citation in the returned answer text."""

    url: str
    title: str | None = None
    start_index: int | None = None
    end_index: int | None = None


@dataclass(frozen=True, slots=True)
class WebSearchSource:
    """One validated URL consulted by the search provider."""

    url: str
    title: str | None = None
    snippet: str | None = None
    score: float | None = None


@dataclass(frozen=True, slots=True)
class WebSearchResult:
    """Provider-neutral answer and source metadata for one web-search request."""

    query: str
    text: str
    citations: tuple[WebSearchCitation, ...]
    sources: tuple[WebSearchSource, ...]
    used_web_search: bool
    provider: str
    response_id: str | None = None
    request_count: int = 1


class WebSearchError(RuntimeError):
    """Raised when a web-search provider cannot return a usable response."""

    code = "web_search_error"

    def __init__(self, message: str, *, request_count: int = 1) -> None:
        super().__init__(message)
        self.request_count = request_count


class WebSearchTimeoutError(WebSearchError):
    """Raised when the provider exceeds the configured request timeout."""

    code = "web_search_timeout"


class WebSearchAuthenticationError(WebSearchError):
    """Raised when a provider rejects its configured credentials."""

    code = "web_search_authentication"


class WebSearchRateLimitError(WebSearchError):
    """Raised when a provider rejects a request due to rate or usage limits."""

    code = "web_search_rate_limit"


class WebSearchNoResultError(WebSearchError):
    """Raised when a successful provider response contains no usable result."""

    code = "web_search_no_result"


class WebSearchFallbackError(WebSearchError):
    """Raised after every configured fallback provider fails."""

    code = "web_search_all_providers_failed"


@runtime_checkable
class WebSearchProvider(Protocol):
    """Minimal contract shared by present and future search providers."""

    def search(
        self,
        query: str,
        *,
        allowed_domains: Sequence[str] = (),
        blocked_domains: Sequence[str] = (),
    ) -> WebSearchResult:
        """Search or answer directly according to the provider's model/tool policy."""


def _field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _items(value: object) -> tuple[object, ...]:
    if value is None or isinstance(value, (str, bytes, bytearray, Mapping)):
        return ()
    if not isinstance(value, Iterable):
        return ()
    return tuple(value)


def _optional_text(value: object, *, max_length: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text[:max_length] if text else None


def _validated_web_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    url = value.strip()
    contains_unsafe_character = any(
        character.isspace() or ord(character) < 32 or ord(character) == 127 or character == "\\"
        for character in url
    )
    if not url or len(url) > MAX_WEB_URL_LENGTH or contains_unsafe_character:
        return None
    try:
        parsed = urlsplit(url)
        _ = parsed.port
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return None
    if parsed.username is not None or parsed.password is not None:
        return None
    return url


def _citation_indexes(annotation: object, text_length: int) -> tuple[int | None, int | None]:
    start = _field(annotation, "start_index")
    end = _field(annotation, "end_index")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end < start
        or end > text_length
    ):
        return None, None
    return start, end


def _extract_text_and_citations(response: object) -> tuple[str, tuple[WebSearchCitation, ...]]:
    fragments: list[str] = []
    citations: list[WebSearchCitation] = []
    seen_citations: set[tuple[str, str | None, int | None, int | None]] = set()
    text_offset = 0

    for output_item in _items(_field(response, "output")):
        if _field(output_item, "type") != "message":
            continue
        for content_item in _items(_field(output_item, "content")):
            if _field(content_item, "type") != "output_text":
                continue
            fragment = _field(content_item, "text")
            if not isinstance(fragment, str):
                continue
            fragments.append(fragment)
            for annotation in _items(_field(content_item, "annotations")):
                if _field(annotation, "type") != "url_citation":
                    continue
                nested = _field(annotation, "url_citation")
                citation_value = nested if nested is not None else annotation
                url = _validated_web_url(_field(citation_value, "url"))
                if url is None:
                    continue
                local_start, local_end = _citation_indexes(citation_value, len(fragment))
                start = local_start + text_offset if local_start is not None else None
                end = local_end + text_offset if local_end is not None else None
                title = _optional_text(
                    _field(citation_value, "title"), max_length=MAX_SOURCE_TITLE_LENGTH
                )
                key = (url, title, start, end)
                if key in seen_citations:
                    continue
                seen_citations.add(key)
                citations.append(
                    WebSearchCitation(
                        url=url,
                        title=title,
                        start_index=start,
                        end_index=end,
                    )
                )
            text_offset += len(fragment)

    block_text = "".join(fragments)
    output_text = _field(response, "output_text")
    text = output_text if isinstance(output_text, str) else block_text
    return text, tuple(citations)


def _extract_sources(response: object) -> tuple[WebSearchSource, ...]:
    sources: list[WebSearchSource] = []
    source_indexes: dict[str, int] = {}

    for output_item in _items(_field(response, "output")):
        if _field(output_item, "type") != "web_search_call":
            continue
        action = _field(output_item, "action")
        for raw_source in _items(_field(action, "sources")):
            raw_url = raw_source if isinstance(raw_source, str) else _field(raw_source, "url")
            url = _validated_web_url(raw_url)
            if url is None:
                continue
            title = _optional_text(_field(raw_source, "title"), max_length=MAX_SOURCE_TITLE_LENGTH)
            existing_index = source_indexes.get(url)
            if existing_index is None:
                source_indexes[url] = len(sources)
                sources.append(WebSearchSource(url=url, title=title))
            elif sources[existing_index].title is None and title is not None:
                sources[existing_index] = WebSearchSource(url=url, title=title)
    return tuple(sources)


def _optional_score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = float(value)
    return score if math.isfinite(score) and 0 <= score <= 1 else None


def _extract_tavily_sources(payload: object) -> tuple[WebSearchSource, ...]:
    sources: list[WebSearchSource] = []
    source_indexes: dict[str, int] = {}
    for raw_result in _items(_field(payload, "results")):
        url = _validated_web_url(_field(raw_result, "url"))
        if url is None:
            continue
        title = _optional_text(_field(raw_result, "title"), max_length=MAX_SOURCE_TITLE_LENGTH)
        snippet = _optional_text(
            _field(raw_result, "content"), max_length=MAX_SOURCE_SNIPPET_LENGTH
        )
        score = _optional_score(_field(raw_result, "score"))
        existing_index = source_indexes.get(url)
        if existing_index is None:
            source_indexes[url] = len(sources)
            sources.append(WebSearchSource(url=url, title=title, snippet=snippet, score=score))
            continue
        existing = sources[existing_index]
        sources[existing_index] = WebSearchSource(
            url=url,
            title=existing.title or title,
            snippet=existing.snippet or snippet,
            score=existing.score if existing.score is not None else score,
        )
    return tuple(sources)


def _render_source_summary(sources: Sequence[WebSearchSource]) -> str:
    sections: list[str] = []
    for index, source in enumerate(sources, start=1):
        heading = source.title or source.url
        section = f"{index}. {heading} ({source.url})"
        if source.snippet:
            section = f"{section}\n{source.snippet}"
        sections.append(section)
    return "Search results:\n\n" + "\n\n".join(sections)


def _used_web_search(response: object) -> bool:
    return any(
        _field(output_item, "type") == "web_search_call"
        for output_item in _items(_field(response, "output"))
    )


def _is_timeout(error: Exception) -> bool:
    if isinstance(error, TimeoutError):
        return True
    return any("timeout" in error_type.__name__.lower() for error_type in type(error).mro())


def _normalize_domain(domain: object) -> str:
    if not isinstance(domain, str):
        raise ValueError("Web-search domains must be strings.")
    normalized = domain.strip().lower().rstrip(".")
    if not normalized or len(normalized) > 253:
        raise ValueError("Web-search domain is empty or too long.")
    if "://" in normalized or any(character in normalized for character in "/?#@:"):
        raise ValueError("Web-search filters must contain domain names without schemes or paths.")
    labels = normalized.split(".")
    if any(not _DOMAIN_LABEL.fullmatch(label) for label in labels):
        raise ValueError(f"Invalid web-search domain: {domain!r}.")
    return normalized


def _normalize_domains(
    domains: Sequence[str],
    *,
    field_name: str,
    max_domains: int = MAX_FILTER_DOMAINS,
) -> tuple[str, ...]:
    if isinstance(domains, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a sequence of domain names.")
    normalized = tuple(dict.fromkeys(_normalize_domain(domain) for domain in domains))
    if len(normalized) > max_domains:
        raise ValueError(f"{field_name} cannot contain more than {max_domains} domains.")
    return normalized


def _normalize_query(query: object) -> str:
    if not isinstance(query, str):
        raise ValueError("Web-search query must be a string.")
    normalized = query.strip()
    if not normalized:
        raise ValueError("Web-search query cannot be empty.")
    if len(normalized) > MAX_WEB_QUERY_LENGTH:
        raise ValueError(f"Web-search query cannot exceed {MAX_WEB_QUERY_LENGTH} characters.")
    return normalized


def _normalize_timeout(timeout_seconds: object) -> float:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("Web-search timeout must be a positive finite number.")
    return float(timeout_seconds)


@dataclass(slots=True)
class OpenAIWebSearchProvider:
    """OpenAI Responses provider using the model-controlled hosted web-search tool."""

    client: Any
    model: str
    timeout_seconds: float = 30.0
    search_context_size: SearchContextSize = "medium"

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("OpenAI web-search model cannot be empty.")
        self.model = self.model.strip()
        self.timeout_seconds = _normalize_timeout(self.timeout_seconds)
        if self.search_context_size not in {"low", "medium", "high"}:
            raise ValueError("Search context size must be 'low', 'medium', or 'high'.")

    def search(
        self,
        query: str,
        *,
        allowed_domains: Sequence[str] = (),
        blocked_domains: Sequence[str] = (),
    ) -> WebSearchResult:
        normalized_query = _normalize_query(query)
        allowed = _normalize_domains(allowed_domains, field_name="allowed_domains")
        blocked = _normalize_domains(blocked_domains, field_name="blocked_domains")
        if blocked:
            raise ValueError(
                "OpenAI web search supports only allowed_domains; blocked_domains is unsupported."
            )

        tool: dict[str, object] = {
            "type": "web_search",
            "search_context_size": self.search_context_size,
        }
        if allowed:
            tool["filters"] = {"allowed_domains": list(allowed)}

        try:
            response = self.client.responses.create(
                model=self.model,
                input=normalized_query,
                tools=[tool],
                tool_choice="required",
                include=["web_search_call.action.sources"],
                timeout=self.timeout_seconds,
            )
        except Exception as error:
            if _is_timeout(error):
                raise WebSearchTimeoutError("OpenAI web search timed out.") from error
            raise WebSearchError("OpenAI web search request failed.") from error

        if response is None:
            raise WebSearchError("OpenAI web search returned an invalid response.")
        if _field(response, "error") is not None:
            raise WebSearchError("OpenAI web search response reported an error.")
        response_status = _field(response, "status")
        if (
            isinstance(response_status, str)
            and response_status.lower() in _FAILED_RESPONSE_STATUSES
        ):
            raise WebSearchError("OpenAI web search response failed.")

        text, citations = _extract_text_and_citations(response)
        if not text.strip():
            raise WebSearchNoResultError("OpenAI web search returned no answer text.")
        response_id = _optional_text(_field(response, "id"), max_length=256)
        return WebSearchResult(
            query=normalized_query,
            text=text,
            citations=citations,
            sources=_extract_sources(response),
            used_web_search=_used_web_search(response),
            provider="openai",
            response_id=response_id,
        )


class TavilyWebSearchProvider:
    """Tavily Search API provider using the basic answer-and-snippets contract."""

    def __init__(
        self,
        api_key: str,
        *,
        http_client: Any | None = None,
        timeout_seconds: float = 30.0,
        max_results: int = 5,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Tavily API key cannot be empty.")
        normalized_key = api_key.strip()
        if any(character.isspace() or ord(character) < 32 for character in normalized_key):
            raise ValueError("Tavily API key contains invalid characters.")
        if isinstance(max_results, bool) or not isinstance(max_results, int):
            raise ValueError("Tavily max_results must be an integer between 1 and 20.")
        if not 1 <= max_results <= MAX_TAVILY_RESULTS:
            raise ValueError("Tavily max_results must be an integer between 1 and 20.")

        self._api_key = normalized_key
        self.timeout_seconds = _normalize_timeout(timeout_seconds)
        self.max_results = max_results
        self._owns_http_client = http_client is None
        self.http_client = http_client if http_client is not None else httpx.Client()

    def search(
        self,
        query: str,
        *,
        allowed_domains: Sequence[str] = (),
        blocked_domains: Sequence[str] = (),
    ) -> WebSearchResult:
        normalized_query = _normalize_query(query)
        included = _normalize_domains(
            allowed_domains,
            field_name="allowed_domains",
            max_domains=MAX_TAVILY_INCLUDE_DOMAINS,
        )
        excluded = _normalize_domains(
            blocked_domains,
            field_name="blocked_domains",
            max_domains=MAX_TAVILY_EXCLUDE_DOMAINS,
        )
        if set(included).intersection(excluded):
            raise ValueError("A web-search domain cannot be both allowed and blocked.")

        body: dict[str, object] = {
            "query": normalized_query,
            "search_depth": "basic",
            "include_answer": "basic",
            "include_raw_content": False,
            "max_results": self.max_results,
        }
        if included:
            body["include_domains"] = list(included)
        if excluded:
            body["exclude_domains"] = list(excluded)

        try:
            response = self.http_client.post(
                TAVILY_SEARCH_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=self.timeout_seconds,
            )
        except Exception as error:
            if _is_timeout(error):
                raise WebSearchTimeoutError("Tavily web search timed out.") from error
            raise WebSearchError("Tavily web search request failed.") from error

        status_code = _field(response, "status_code")
        if isinstance(status_code, bool) or not isinstance(status_code, int):
            raise WebSearchError("Tavily web search returned an invalid HTTP response.")
        if status_code in {401, 403}:
            raise WebSearchAuthenticationError("Tavily web search authentication failed.")
        if status_code in {429, 432, 433}:
            raise WebSearchRateLimitError("Tavily web search rate or usage limit was exceeded.")
        if not 200 <= status_code < 300:
            raise WebSearchError(f"Tavily web search failed with HTTP status {status_code}.")

        try:
            payload = response.json()
        except Exception as error:
            raise WebSearchError("Tavily web search returned invalid JSON.") from error
        if not isinstance(payload, Mapping):
            raise WebSearchError("Tavily web search returned an invalid response body.")

        sources = _extract_tavily_sources(payload)
        if not sources:
            raise WebSearchNoResultError("Tavily web search returned no usable results.")
        answer = _optional_text(_field(payload, "answer"), max_length=MAX_WEB_ANSWER_LENGTH)
        text = answer or _render_source_summary(sources)
        return WebSearchResult(
            query=normalized_query,
            text=text,
            citations=(),
            sources=sources,
            used_web_search=True,
            provider="tavily",
            response_id=_optional_text(_field(payload, "request_id"), max_length=256),
        )

    def close(self) -> None:
        """Close the internally-created HTTP client, if this provider owns it."""

        if self._owns_http_client:
            self.http_client.close()

    def __enter__(self) -> TavilyWebSearchProvider:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class FallbackWebSearchProvider:
    """Try providers in order and remember the most recent successful provider."""

    def __init__(self, providers: Sequence[WebSearchProvider]) -> None:
        if not providers:
            raise ValueError("At least one web-search provider is required.")
        if any(not isinstance(provider, WebSearchProvider) for provider in providers):
            raise ValueError("Every fallback entry must implement WebSearchProvider.")
        self._providers = tuple(providers)
        self._preferred_index: int | None = None
        self._closed = False

    @property
    def preferred_provider_index(self) -> int | None:
        """Return the remembered successful provider index, if one exists."""

        return self._preferred_index

    def close(self) -> None:
        """Close each unique child provider at most once."""

        if self._closed:
            return
        self._closed = True

        errors: list[Exception] = []
        closed_provider_ids: set[int] = set()
        for provider in self._providers:
            provider_id = id(provider)
            if provider_id in closed_provider_ids:
                continue
            closed_provider_ids.add(provider_id)
            try:
                close = getattr(provider, "close", None)
                if callable(close):
                    close()
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("One or more web-search providers failed to close.", errors)

    def search(
        self,
        query: str,
        *,
        allowed_domains: Sequence[str] = (),
        blocked_domains: Sequence[str] = (),
    ) -> WebSearchResult:
        preferred = self._preferred_index
        indexes = list(range(len(self._providers)))
        if preferred is not None:
            indexes.remove(preferred)
            indexes.insert(0, preferred)

        errors: list[WebSearchError] = []
        request_count = 0
        for index in indexes:
            try:
                result = self._providers[index].search(
                    query,
                    allowed_domains=allowed_domains,
                    blocked_domains=blocked_domains,
                )
            except WebSearchError as error:
                errors.append(error)
                request_count += error.request_count
                if self._preferred_index == index:
                    self._preferred_index = None
                continue
            request_count += result.request_count
            self._preferred_index = index
            return replace(result, request_count=request_count)

        failure = WebSearchFallbackError(
            "All configured web-search providers failed.",
            request_count=request_count,
        )
        if errors:
            raise failure from errors[-1]
        raise failure

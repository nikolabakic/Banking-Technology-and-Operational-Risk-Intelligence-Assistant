"""Deterministic tools that can be exposed to the conversational agent."""

from bankscope.tools.calculator import (
    CALCULATOR_TOOL,
    CalculatorError,
    calculate,
    evaluate_expression,
)
from bankscope.tools.web_search import (
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

__all__ = [
    "CALCULATOR_TOOL",
    "CalculatorError",
    "FallbackWebSearchProvider",
    "OpenAIWebSearchProvider",
    "TavilyWebSearchProvider",
    "WebSearchAuthenticationError",
    "WebSearchCitation",
    "WebSearchError",
    "WebSearchFallbackError",
    "WebSearchNoResultError",
    "WebSearchProvider",
    "WebSearchRateLimitError",
    "WebSearchResult",
    "WebSearchSource",
    "WebSearchTimeoutError",
    "calculate",
    "evaluate_expression",
]

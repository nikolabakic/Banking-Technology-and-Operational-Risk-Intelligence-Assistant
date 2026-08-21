import json
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from bankscope.generation.conversation import (
    CONVERSATION_TOOLS,
    ClarificationArgs,
    ConversationGraph,
    DeclineOutOfScopeArgs,
    DirectResponseArgs,
    ResearchFilingsArgs,
    RouteDecision,
    WebResearchArgs,
    is_clearly_out_of_scope,
    request_conversation_action,
)
from bankscope.generation.pipeline import BankAnswerPipeline
from bankscope.sec.bank_resolver import resolve_bank

BANK_NAMES = {
    "C": "Citigroup Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "BAC": "Bank of America Corporation",
    "COF": "Capital One Financial Corporation",
}
BANK_ALIASES = {
    "C": ("Citigroup", "Citi", "Citibank"),
    "JPM": ("JPMorgan", "JPMorgan Chase", "JP Morgan"),
    "BAC": ("Bank of America", "BofA"),
    "COF": ("Capital One",),
}


def route_arguments(action: str, **overrides) -> dict:
    defaults = {
        "action": action,
        "confidence": 0.95,
        "reason": f"Route to {action}.",
        "search_question": None,
        "response_text": None,
        "category": None,
        "missing": None,
        "citation_ids": [],
        "presentation_guidance": None,
    }
    defaults.update(overrides)
    return defaults


def tool_response(arguments: dict, *, name: str = "route_conversation"):
    function = SimpleNamespace(name=name, arguments=json.dumps(arguments))
    message = SimpleNamespace(
        content=None,
        refusal=None,
        tool_calls=[SimpleNamespace(function=function)],
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="tool_calls")])


class ToolCompletions:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def client_for(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


class FakeChatModel:
    """Minimal network-free stand-in for ChatOpenAI structured output."""

    def __init__(self, output=None, error: Exception | None = None) -> None:
        self.output = output
        self.error = error
        self.calls = []
        self.structured_options = None

    def with_structured_output(self, schema, **kwargs):
        self.structured_options = (schema, kwargs)
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return self.output


@pytest.mark.parametrize(
    ("question", "arguments", "expected_type"),
    [
        (
            "What was JPM CET1 in 2025?",
            route_arguments("filing_research", search_question="What was JPM CET1 in 2025?"),
            ResearchFilingsArgs,
        ),
        (
            "Hello",
            route_arguments("direct_response", response_text="Hello!", category="greeting"),
            DirectResponseArgs,
        ),
        (
            "Compare the ratio",
            route_arguments(
                "clarification",
                response_text="Which bank do you mean?",
                missing="bank",
            ),
            ClarificationArgs,
        ),
        (
            "Give me a recipe",
            route_arguments("out_of_scope"),
            DeclineOutOfScopeArgs,
        ),
        (
            "What is Citi's share price today?",
            route_arguments("web_research", search_question="What is Citi's share price today?"),
            WebResearchArgs,
        ),
    ],
)
def test_native_router_uses_one_strict_route_function(question, arguments, expected_type) -> None:
    completions = ToolCompletions(tool_response(arguments))
    decision = request_conversation_action(
        question,
        [],
        client=client_for(completions),
        model="test-model",
        bank_names=BANK_NAMES,
        bank_aliases=BANK_ALIASES,
        backend="legacy",
    )

    assert isinstance(decision.action, expected_type)
    assert decision.fallback is False
    request = completions.calls[0]
    assert request["tool_choice"] == "required"
    assert request["parallel_tool_calls"] is False


def test_route_tool_schema_is_strict_and_requires_every_property() -> None:
    assert len(CONVERSATION_TOOLS) == 1
    function = CONVERSATION_TOOLS[0]["function"]
    parameters = function["parameters"]

    assert function["name"] == "route_conversation"
    assert function["strict"] is True
    assert parameters["additionalProperties"] is False
    assert set(parameters["required"]) == set(parameters["properties"])


@pytest.mark.parametrize(
    "arguments",
    [
        route_arguments("filing_research"),
        route_arguments("web_research"),
        route_arguments("direct_response", response_text="Hello"),
        route_arguments("clarification", response_text="Which bank?"),
    ],
)
def test_route_decision_rejects_missing_action_fields(arguments) -> None:
    with pytest.raises(ValidationError):
        RouteDecision.model_validate(arguments)


def test_langgraph_compiles_once_and_uses_strict_structured_output() -> None:
    model = FakeChatModel(
        RouteDecision.model_validate(
            route_arguments(
                "filing_research",
                search_question="What were Citigroup's material cybersecurity risks in 2025?",
            )
        )
    )
    graph = ConversationGraph(
        client=SimpleNamespace(),
        model="test-model",
        bank_names=BANK_NAMES,
        bank_aliases=BANK_ALIASES,
        chat_model=model,
    )

    compiled_graph = graph._graph
    first = graph.route("What were Citigroup's material cybersecurity risks in 2025?", [])
    second = graph.route("What were Citigroup's material cybersecurity risks in 2024?", [])

    assert graph._graph is compiled_graph
    assert first.route_action == "filing_research"
    assert first.graph_nodes == ("prepare", "route", "validate_route")
    assert second.graph_nodes == first.graph_nodes
    assert model.structured_options == (
        RouteDecision,
        {"method": "function_calling", "strict": True},
    )


def test_citigroup_cybersecurity_regression_routes_to_filing_and_ticker_c() -> None:
    question = "What were Citigroup's material cybersecurity risks in 2025?"
    resolution = resolve_bank(question, bank_names=BANK_NAMES, bank_aliases=BANK_ALIASES)
    decision = ConversationGraph(
        client=SimpleNamespace(),
        model="test-model",
        bank_names=BANK_NAMES,
        bank_aliases=BANK_ALIASES,
        chat_model=FakeChatModel(error=TimeoutError("router unavailable")),
    ).route(question, [])

    assert isinstance(decision.action, ResearchFilingsArgs)
    assert decision.action.search_question == question
    assert decision.route_action == "filing_research"
    assert decision.fallback is True
    assert resolution.tickers == ("C",)


@pytest.mark.parametrize(
    "question",
    [
        "What cyber-security risk did Citi disclose in 2025?",
        "Summarize Citigroup cybersecurity risk.",
        "What are Citigroup's cybersecurty risks?",
        "Kako Citi opisuje sajber rizike?",
        "Šta Citibank navodi o sajber bezbednosti?",
        "Korisnik pita kako Citi opisuje operativni rizik.",
    ],
)
def test_supported_bank_variants_fail_open_to_filing_research(question) -> None:
    decision = ConversationGraph(
        client=SimpleNamespace(),
        model="test-model",
        bank_names=BANK_NAMES,
        bank_aliases=BANK_ALIASES,
        chat_model=FakeChatModel(error=TimeoutError("router unavailable")),
    ).route(question, [])

    assert isinstance(decision.action, ResearchFilingsArgs)
    assert decision.route_action == "filing_research"


def test_low_confidence_out_of_scope_becomes_clarification() -> None:
    model = FakeChatModel(
        route_arguments("out_of_scope", confidence=0.62, reason="Uncertain relevance.")
    )
    decision = ConversationGraph(
        client=SimpleNamespace(),
        model="test-model",
        bank_names=BANK_NAMES,
        bank_aliases=BANK_ALIASES,
        chat_model=model,
    ).route("Tell me more about resilience", [])

    assert isinstance(decision.action, ClarificationArgs)
    assert decision.route_action == "clarification"
    assert decision.reason == "low_confidence_out_of_scope_requires_clarification"


def test_explicit_recipe_cannot_be_answered_as_general_explanation() -> None:
    decision = ConversationGraph(
        client=SimpleNamespace(),
        model="test-model",
        bank_names=BANK_NAMES,
        bank_aliases=BANK_ALIASES,
        chat_model=FakeChatModel(
            route_arguments(
                "direct_response",
                response_text="Here is an apple pie recipe.",
                category="general_explanation",
            )
        ),
    ).route("Give me a recipe for apple pie.", [])

    assert isinstance(decision.action, DeclineOutOfScopeArgs)
    assert decision.reason == "explicit_non_banking_request_requires_scope_decline"


def test_shorter_answer_is_a_contextual_transform_with_previous_citations() -> None:
    model = FakeChatModel(
        route_arguments(
            "direct_response",
            response_text="JPMorgan manages operational risk through CCOR [E1].",
            category="contextual_transform",
            citation_ids=["E1"],
        )
    )
    pipeline = BankAnswerPipeline(
        retriever=NeverCalled(),
        query_encoder=NeverCalled(),
        client=NeverCalled(),
        generation_model="test-model",
        bank_names=BANK_NAMES,
        bank_aliases=BANK_ALIASES,
        conversation_model=model,
    )
    previous = {
        "question": "How does JPMorgan describe operational risk?",
        "answer": (
            "JPMorgan describes operational risk as inherent in all activities and manages "
            "it through its Compliance, Conduct, and Operational Risk (CCOR) framework "
            "[E1][E2]."
        ),
        "ticker": "JPM",
        "tickers": ["JPM"],
        "citations": [
            {"label": "E1", "target_chunk_id": "jpm-1"},
            {"label": "E2", "target_chunk_id": "jpm-2"},
        ],
    }

    run = pipeline.answer(
        "Give me a shorter answer",
        ticker="JPM",
        tickers=["JPM"],
        conversation_history=[
            {"role": "user", "content": previous["question"]},
            {"role": "assistant", "content": previous["answer"]},
        ],
        previous_answer=previous,
    )

    assert run.output["dialog_act"] == "contextual_transform"
    assert run.output["answer"] == "JPMorgan manages operational risk through CCOR [E1]."
    assert [item["label"] for item in run.output["citations"]] == ["E1"]
    assert run.output["ticker"] == "JPM"
    assert run.output["retrieval"]["mode"] == "none"


def test_contextual_transform_cannot_add_a_number() -> None:
    decision = ConversationGraph(
        client=SimpleNamespace(),
        model="test-model",
        bank_names=BANK_NAMES,
        bank_aliases=BANK_ALIASES,
        chat_model=FakeChatModel(
            route_arguments(
                "direct_response",
                response_text="JPMorgan's ratio was 14.2% [E1].",
                category="contextual_transform",
                citation_ids=["E1"],
            )
        ),
    ).route(
        "Make it shorter",
        [],
        previous_answer={
            "answer": "JPMorgan described the ratio as strong [E1].",
            "citations": [{"label": "E1"}],
            "ticker": "JPM",
            "tickers": ["JPM"],
        },
    )

    assert isinstance(decision.action, ClarificationArgs)
    assert decision.reason == "contextual_transform_added_number"


def test_new_bank_fact_cannot_be_smuggled_in_as_a_contextual_transform() -> None:
    decision = ConversationGraph(
        client=SimpleNamespace(),
        model="test-model",
        bank_names=BANK_NAMES,
        bank_aliases=BANK_ALIASES,
        chat_model=FakeChatModel(
            route_arguments(
                "direct_response",
                response_text="Citigroup describes the same risk [E1].",
                category="contextual_transform",
                citation_ids=["E1"],
            )
        ),
    ).route(
        "Without new research, add what Citigroup says about the same risk.",
        [],
        previous_answer={
            "answer": "JPMorgan describes the risk [E1].",
            "citations": [{"label": "E1"}],
            "ticker": "JPM",
            "tickers": ["JPM"],
        },
    )

    assert isinstance(decision.action, ResearchFilingsArgs)
    assert decision.reason == "new_bank_fact_requires_filing_research"


@pytest.mark.parametrize("proposed_action", ["out_of_scope", "direct_response", "web_research"])
def test_bank_and_filing_signals_override_unsafe_model_route(proposed_action) -> None:
    arguments = route_arguments(proposed_action, confidence=0.99)
    if proposed_action == "direct_response":
        arguments.update(response_text="A generic answer.", category="general_explanation")
    elif proposed_action == "web_research":
        arguments.update(search_question="Search the web for Citi cybersecurity risks.")
    decision = ConversationGraph(
        client=SimpleNamespace(),
        model="test-model",
        bank_names=BANK_NAMES,
        bank_aliases=BANK_ALIASES,
        chat_model=FakeChatModel(arguments),
    ).route("What cybersecurity risks did Citi disclose in 2025?", [])

    assert isinstance(decision.action, ResearchFilingsArgs)
    assert decision.route_action == "filing_research"


@pytest.mark.parametrize(
    "question",
    [
        "Compare JPMorgan and Citi operational risk disclosures.",
        "Compare JPMorgan, Citi, and Bank of America business models.",
        (
            "Compare JPMorgan, Citi, Bank of America, and Capital One "
            "cybersecurity frameworks."
        ),
    ],
)
def test_two_to_four_bank_topics_cannot_route_to_direct_response(question) -> None:
    decision = ConversationGraph(
        client=SimpleNamespace(),
        model="test-model",
        bank_names=BANK_NAMES,
        bank_aliases=BANK_ALIASES,
        chat_model=FakeChatModel(
            route_arguments(
                "direct_response",
                response_text="An ungrounded comparison.",
                category="general_explanation",
            )
        ),
    ).route(question, [])

    assert isinstance(decision.action, ResearchFilingsArgs)
    assert decision.route_action == "filing_research"
    assert decision.action.search_question == question
    assert decision.reason == "bank_specific_filing_claim_requires_filing_research"


@pytest.mark.parametrize(
    "question",
    [
        "Compare JPMorgan and Citi.",
        "How do JPMorgan, Citi, and Bank of America compare?",
        "Compare JPMorgan, Citi, Bank of America, and Capital One in 2025.",
    ],
)
def test_two_to_four_banks_without_topic_require_metric_clarification(question) -> None:
    decision = ConversationGraph(
        client=SimpleNamespace(),
        model="test-model",
        bank_names=BANK_NAMES,
        bank_aliases=BANK_ALIASES,
        chat_model=FakeChatModel(route_arguments("out_of_scope", confidence=0.99)),
    ).route(question, [])

    assert isinstance(decision.action, ClarificationArgs)
    assert decision.route_action == "clarification"
    assert decision.action.missing == "metric"
    assert decision.reason == "multi_bank_comparison_requires_topic_or_metric"


@pytest.mark.parametrize(
    "question",
    [
        "Compare JPMorgan and Citi share prices today.",
        "Compare current news about JPMorgan, Citi, and Bank of America.",
        (
            "Compare the latest market prices for JPMorgan, Citi, Bank of America, "
            "and Capital One."
        ),
    ],
)
def test_current_two_to_four_bank_comparisons_route_to_web(question) -> None:
    decision = ConversationGraph(
        client=SimpleNamespace(),
        model="test-model",
        bank_names=BANK_NAMES,
        bank_aliases=BANK_ALIASES,
        chat_model=FakeChatModel(
            route_arguments("filing_research", search_question="Use old filing data.")
        ),
    ).route(question, [])

    assert isinstance(decision.action, WebResearchArgs)
    assert decision.route_action == "web_research"
    assert decision.action.search_question == question


def test_low_confidence_web_route_becomes_clarification() -> None:
    decision = ConversationGraph(
        client=SimpleNamespace(),
        model="test-model",
        bank_names=BANK_NAMES,
        bank_aliases=BANK_ALIASES,
        chat_model=FakeChatModel(
            route_arguments(
                "web_research",
                confidence=0.3,
                search_question="Find current external information.",
            )
        ),
    ).route("Find information about resilience", [])

    assert decision.route_action == "clarification"
    assert isinstance(decision.action, ClarificationArgs)


def test_current_bank_claim_overrides_unsafe_direct_response_to_web() -> None:
    decision = ConversationGraph(
        client=SimpleNamespace(),
        model="test-model",
        bank_names=BANK_NAMES,
        bank_aliases=BANK_ALIASES,
        chat_model=FakeChatModel(
            route_arguments(
                "direct_response",
                response_text="An ungrounded current answer.",
                category="general_explanation",
            )
        ),
    ).route("Who is Citigroup's current CEO?", [])

    assert decision.route_action == "web_research"
    assert isinstance(decision.action, WebResearchArgs)


def test_router_fallback_handles_capability_without_retrieval() -> None:
    decision = ConversationGraph(
        client=SimpleNamespace(),
        model="test-model",
        bank_names=BANK_NAMES,
        bank_aliases=BANK_ALIASES,
        chat_model=FakeChatModel(error=TimeoutError("model unavailable")),
    ).route("Koje banke podržavaš?", [])

    assert isinstance(decision.action, DirectResponseArgs)
    assert decision.action.category == "capability"
    assert "JPMorgan Chase & Co." in decision.action.answer
    assert decision.error_code == "conversation_route_timeouterror"


def test_router_reports_schema_failure_separately_from_transport_failure() -> None:
    completions = ToolCompletions(
        tool_response(route_arguments("filing_research", search_question=None))
    )
    decision = request_conversation_action(
        "What was JPM's CET1 ratio?",
        [],
        client=client_for(completions),
        model="test-model",
        bank_names=BANK_NAMES,
        bank_aliases=BANK_ALIASES,
        backend="legacy",
    )

    assert isinstance(decision.action, ResearchFilingsArgs)
    assert decision.fallback is True
    assert decision.error_code == "conversation_route_invalid_schema"


def test_router_fallback_keeps_latest_user_topic_for_natural_follow_up() -> None:
    decision = ConversationGraph(
        client=SimpleNamespace(),
        model="test-model",
        bank_names=BANK_NAMES,
        bank_aliases=BANK_ALIASES,
        chat_model=FakeChatModel(error=TimeoutError("model unavailable")),
    ).route(
        "What about 2024?",
        [
            {"role": "user", "content": "What was JPM CET1 in 2025?"},
            {"role": "assistant", "content": "It was 14.6%."},
        ],
    )

    assert isinstance(decision.action, ResearchFilingsArgs)
    assert "JPM CET1" in decision.action.search_question
    assert "2024" in decision.action.search_question
    assert "2025" not in decision.action.search_question
    assert "14.6" not in decision.action.search_question


def test_recipe_is_explicitly_unrelated_but_serbian_word_pita_is_not_a_veto() -> None:
    assert is_clearly_out_of_scope("daj mi recept za pitu sa jabukama") is True
    assert is_clearly_out_of_scope("Korisnik pita kako Citi opisuje rizik") is False


class NeverCalled:
    def __getattr__(self, name):
        raise AssertionError(f"This dependency should not be called: {name}")


@pytest.mark.parametrize(
    ("route", "question", "expected_act", "expected_status"),
    [
        (
            route_arguments("direct_response", response_text="Hello!", category="greeting"),
            "Hello",
            "greeting",
            "supported",
        ),
        (
            route_arguments(
                "clarification",
                response_text="Which bank should I research?",
                missing="bank",
            ),
            "Compare the ratio",
            "clarification",
            "ambiguous",
        ),
    ],
)
def test_pipeline_returns_conversation_turn_without_retrieval(
    route, question, expected_act, expected_status
) -> None:
    pipeline = BankAnswerPipeline(
        retriever=NeverCalled(),
        query_encoder=NeverCalled(),
        client=NeverCalled(),
        generation_model="test-model",
        bank_names=BANK_NAMES,
        bank_aliases=BANK_ALIASES,
        conversation_model=FakeChatModel(route),
    )

    run = pipeline.answer(question, conversation_history=[])

    assert run.output["dialog_act"] == expected_act
    assert run.output["status"] == expected_status
    assert run.output["citations"] == []
    assert run.output["retrieval"]["mode"] == "none"
    assert run.diagnostics["route_action"] == route["action"]
    assert run.diagnostics["graph_nodes"] == ["prepare", "route", "validate_route"]


def test_pipeline_routes_current_share_price_to_stable_web_unavailable_contract() -> None:
    question = "What is Citi's share price today?"
    pipeline = BankAnswerPipeline(
        retriever=NeverCalled(),
        query_encoder=NeverCalled(),
        client=NeverCalled(),
        generation_model="test-model",
        bank_names=BANK_NAMES,
        bank_aliases=BANK_ALIASES,
        conversation_model=FakeChatModel(route_arguments("web_research", search_question=question)),
    )

    run = pipeline.answer(question, conversation_history=[])

    assert run.output["dialog_act"] == "web_research_unavailable"
    assert run.output["reason_code"] == "web_search_unavailable"
    assert run.output["retrieval"]["mode"] == "none"
    assert run.diagnostics["route_action"] == "web_research"
    assert set(pipeline._research_handlers) == {"filing_research", "web_research"}


def test_web_rewrite_scope_violation_falls_back_to_original_question() -> None:
    question = "What is Citi's share price today?"
    pipeline = BankAnswerPipeline(
        retriever=NeverCalled(),
        query_encoder=NeverCalled(),
        client=NeverCalled(),
        generation_model="test-model",
        bank_names=BANK_NAMES,
        bank_aliases=BANK_ALIASES,
        conversation_model=FakeChatModel(
            route_arguments(
                "web_research",
                search_question="What was JPM's share price in 2024?",
            )
        ),
    )

    run = pipeline.answer(question, conversation_history=[])

    assert run.output["reason_code"] == "web_search_unavailable"
    assert run.output["contextualization"]["standalone_question"] == question
    assert run.output["contextualization"]["fallback"] is True
    assert run.output["contextualization"]["error_code"] in {
        "contextualization_added_number",
        "contextualization_added_period",
        "contextualization_changed_period",
        "contextualization_added_bank_scope",
    }


def test_pipeline_recipe_does_not_retrieve_despite_stale_session_ticker() -> None:
    model = FakeChatModel(route_arguments("out_of_scope", confidence=0.99))
    pipeline = BankAnswerPipeline(
        retriever=NeverCalled(),
        query_encoder=NeverCalled(),
        client=NeverCalled(),
        generation_model="test-model",
        bank_names=BANK_NAMES,
        bank_aliases=BANK_ALIASES,
        conversation_model=model,
    )

    run = pipeline.answer(
        "daj mi recept za pitu sa jabukama",
        ticker="JPM",
        conversation_history=[
            {"role": "user", "content": "Šta JPM navodi o sajber riziku?"},
            {"role": "assistant", "content": "Prior grounded answer [E1]."},
        ],
    )

    assert run.output["dialog_act"] == "out_of_scope"
    assert run.output["retrieval"]["mode"] == "none"
    assert run.diagnostics["model_request_count"] == 1
    assert len(model.calls) == 1


def test_vague_cet1_indicator_lets_model_request_metric_clarification() -> None:
    model = FakeChatModel(
        route_arguments(
            "clarification",
            response_text="Koji CET1 pokazatelj želite da uporedim?",
            missing="metric",
        )
    )
    pipeline = BankAnswerPipeline(
        retriever=NeverCalled(),
        query_encoder=NeverCalled(),
        client=NeverCalled(),
        generation_model="test-model",
        bank_names=BANK_NAMES,
        bank_aliases=BANK_ALIASES,
        conversation_model=model,
    )

    run = pipeline.answer(
        "Uporedi JPMorgan i Bank of America CET1 pokazatelje za 2025.",
        conversation_history=[],
    )

    assert run.output["dialog_act"] == "clarification"
    assert run.output["reason_code"] == "missing_metric"
    assert run.diagnostics["model_request_count"] == 1
    assert len(model.calls) == 1


def test_orphaned_follow_up_is_resolved_by_model_without_raw_history() -> None:
    model = FakeChatModel(
        route_arguments(
            "clarification",
            response_text="Na koje pitanje želite da se nadovežem?",
            missing="intent",
        )
    )
    pipeline = BankAnswerPipeline(
        retriever=NeverCalled(),
        query_encoder=NeverCalled(),
        client=NeverCalled(),
        generation_model="test-model",
        bank_names=BANK_NAMES,
        bank_aliases=BANK_ALIASES,
        conversation_model=model,
    )

    run = pipeline.answer("Reci mi više.", ticker="JPM", conversation_history=[])

    assert run.output["dialog_act"] == "clarification"
    assert run.output["reason_code"] == "missing_intent"
    assert run.output["retrieval"]["mode"] == "none"
    assert len(model.calls) == 1


class RecordingEncoder:
    def __init__(self) -> None:
        self.calls = []

    def encode(self, text: str) -> np.ndarray:
        self.calls.append(text)
        return np.array([1.0, 0.0], dtype=np.float32)


class RecordingRetriever:
    def __init__(self) -> None:
        self.calls = []

    def search_hybrid(self, question, query_vector, **kwargs):
        self.calls.append((question, query_vector, kwargs))
        return [
            {
                "target_chunk_id": "c-cyber-1",
                "record_type": "text",
                "ticker": kwargs["ticker"],
                "evidence": "Citigroup identifies material cybersecurity and operational risks.",
                "metadata": {"report_date": "2025-12-31"},
            }
        ]


class NarrativeAnswerCompletions:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        arguments = json.dumps(
            {
                "status": "supported",
                "answer_type": "narrative",
                "answer": "Citigroup describes material cybersecurity risks [E1].",
                "facts": None,
                "citation_ids": ["E1"],
                "reason": "The filing evidence directly supports the answer.",
            }
        )
        function = SimpleNamespace(name="submit_supported_narrative_answer", arguments=arguments)
        message = SimpleNamespace(
            content=None,
            refusal=None,
            tool_calls=[SimpleNamespace(function=function)],
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls")]
        )


def test_citigroup_regression_runs_retrieval_with_ticker_c() -> None:
    question = "What were Citigroup's material cybersecurity risks in 2025?"
    retriever = RecordingRetriever()
    encoder = RecordingEncoder()
    completions = NarrativeAnswerCompletions()
    pipeline = BankAnswerPipeline(
        retriever=retriever,
        query_encoder=encoder,
        client=client_for(completions),
        generation_model="test-model",
        bank_names=BANK_NAMES,
        bank_aliases=BANK_ALIASES,
        conversation_model=FakeChatModel(
            route_arguments("filing_research", search_question=question)
        ),
    )

    run = pipeline.answer(question, conversation_history=[])

    assert run.output["dialog_act"] == "answer", run.output
    assert run.output["ticker"] == "C"
    assert run.output["status"] == "supported"
    assert retriever.calls
    assert all(call[2]["ticker"] == "C" for call in retriever.calls)
    assert run.diagnostics["route_action"] == "filing_research"

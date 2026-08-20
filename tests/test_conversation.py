import json
from types import SimpleNamespace

import pytest

from bankscope.generation.conversation import (
    CONVERSATION_TOOLS,
    ClarificationArgs,
    DeclineOutOfScopeArgs,
    DirectResponseArgs,
    ResearchFilingsArgs,
    request_conversation_action,
)
from bankscope.generation.pipeline import BankAnswerPipeline


def tool_response(name: str, arguments: dict):
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


def client_for(completions: ToolCompletions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


@pytest.mark.parametrize(
    ("name", "arguments", "expected_type"),
    [
        (
            "research_filings",
            {"search_question": "What was JPM CET1 in 2025?", "reason": "Needs filings."},
            ResearchFilingsArgs,
        ),
        (
            "respond_directly",
            {"answer": "Hello!", "category": "greeting"},
            DirectResponseArgs,
        ),
        (
            "ask_clarification",
            {"question": "Which bank do you mean?", "missing": "bank"},
            ClarificationArgs,
        ),
        (
            "decline_out_of_scope",
            {"reason": "outside_banking_research_scope"},
            DeclineOutOfScopeArgs,
        ),
    ],
)
def test_conversation_router_uses_one_native_function(name, arguments, expected_type) -> None:
    completions = ToolCompletions(tool_response(name, arguments))
    decision = request_conversation_action(
        "Current question",
        [{"role": "user", "content": "Prior question"}],
        client=client_for(completions),
        model="test-model",
        bank_names={"JPM": "JPMorgan Chase & Co."},
    )

    assert isinstance(decision.action, expected_type)
    assert decision.fallback is False
    request = completions.calls[0]
    assert request["tool_choice"] == "required"
    assert request["parallel_tool_calls"] is False


def test_conversation_tools_are_strict_and_require_every_property() -> None:
    assert {tool["function"]["name"] for tool in CONVERSATION_TOOLS} == {
        "research_filings",
        "respond_directly",
        "ask_clarification",
        "decline_out_of_scope",
    }
    for tool in CONVERSATION_TOOLS:
        function = tool["function"]
        parameters = function["parameters"]
        assert function["strict"] is True
        assert parameters["additionalProperties"] is False
        assert set(parameters["required"]) == set(parameters["properties"])


def test_conversation_router_has_safe_deterministic_fallback() -> None:
    completions = ToolCompletions(error=TimeoutError("model unavailable"))
    decision = request_conversation_action(
        "Koje banke podržavaš?",
        [],
        client=client_for(completions),
        model="test-model",
        bank_names={"JPM": "JPMorgan Chase & Co.", "BAC": "Bank of America Corporation"},
    )

    assert isinstance(decision.action, DirectResponseArgs)
    assert decision.action.category == "capability"
    assert "JPMorgan Chase & Co." in decision.action.answer
    assert decision.fallback is True
    assert decision.error_code == "conversation_action_timeouterror"


def test_router_reports_schema_failure_separately_from_transport_failure() -> None:
    completions = ToolCompletions(
        tool_response("research_filings", {"search_question": "JPM CET1 ratio"})
    )
    decision = request_conversation_action(
        "What was JPM's CET1 ratio?",
        [],
        client=client_for(completions),
        model="test-model",
        bank_names={"JPM": "JPMorgan Chase & Co."},
    )

    assert isinstance(decision.action, ResearchFilingsArgs)
    assert decision.fallback is True
    assert decision.error_code == "conversation_action_invalid_schema"


def test_router_fallback_keeps_latest_user_topic_for_natural_follow_up() -> None:
    completions = ToolCompletions(error=TimeoutError("model unavailable"))
    decision = request_conversation_action(
        "What about 2024?",
        [
            {"role": "user", "content": "What was JPM CET1 in 2025?"},
            {"role": "assistant", "content": "It was 14.6%."},
        ],
        client=client_for(completions),
        model="test-model",
        bank_names={"JPM": "JPMorgan Chase & Co."},
    )

    assert isinstance(decision.action, ResearchFilingsArgs)
    assert "JPM CET1" in decision.action.search_question
    assert "2024" in decision.action.search_question
    assert "2025" not in decision.action.search_question
    assert "14.6" not in decision.action.search_question


def test_router_fallback_declines_recipe_instead_of_searching_filings() -> None:
    completions = ToolCompletions(error=TimeoutError("model unavailable"))
    decision = request_conversation_action(
        "daj mi recept za pitu sa jabukama",
        [],
        client=client_for(completions),
        model="test-model",
        bank_names={"JPM": "JPMorgan Chase & Co."},
    )

    assert isinstance(decision.action, DeclineOutOfScopeArgs)
    assert decision.fallback is True


class NeverCalled:
    def __getattr__(self, name):
        raise AssertionError(f"Retrieval should not be called: {name}")


@pytest.mark.parametrize(
    ("response", "expected_act", "expected_status"),
    [
        (
            tool_response(
                "respond_directly",
                {"answer": "Hello — how can I help?", "category": "greeting"},
            ),
            "greeting",
            "supported",
        ),
        (
            tool_response(
                "ask_clarification",
                {"question": "Which bank should I research?", "missing": "bank"},
            ),
            "clarification",
            "ambiguous",
        ),
    ],
)
def test_pipeline_returns_conversation_turn_without_retrieval(
    response, expected_act, expected_status
) -> None:
    completions = ToolCompletions(response)
    pipeline = BankAnswerPipeline(
        retriever=NeverCalled(),
        query_encoder=NeverCalled(),
        client=client_for(completions),
        generation_model="test-model",
        bank_names={"JPM": "JPMorgan Chase & Co."},
    )

    run = pipeline.answer("Hello", conversation_history=[])

    assert run.output["dialog_act"] == expected_act
    assert run.output["status"] == expected_status
    assert run.output["citations"] == []
    assert run.output["retrieval"]["mode"] == "none"
    assert run.diagnostics["model_request_count"] == 1


def test_pipeline_declines_recipe_before_model_or_retrieval() -> None:
    pipeline = BankAnswerPipeline(
        retriever=NeverCalled(),
        query_encoder=NeverCalled(),
        client=NeverCalled(),
        generation_model="test-model",
        bank_names={"JPM": "JPMorgan Chase & Co."},
    )

    run = pipeline.answer(
        "daj mi recept za pitu sa jabukama",
        ticker="JPM",
        conversation_history=[
            {"role": "user", "content": "Šta JPM navodi o sajber riziku?"},
            {"role": "assistant", "content": "Very long prior answer with facts [E1]."},
        ],
    )

    assert run.output["dialog_act"] == "out_of_scope"
    assert run.output["status"] == "unsupported"
    assert run.output["retrieval"]["mode"] == "none"
    assert "recept" not in run.output["answer"].casefold()
    assert run.output["answer"].startswith("Mogu da pomognem")
    assert run.diagnostics["model_request_count"] == 0


def test_vague_cet1_indicator_requests_metric_clarification_without_model() -> None:
    pipeline = BankAnswerPipeline(
        retriever=NeverCalled(),
        query_encoder=NeverCalled(),
        client=NeverCalled(),
        generation_model="test-model",
        bank_names={"JPM": "JPMorgan Chase & Co.", "BAC": "Bank of America Corporation"},
    )

    run = pipeline.answer(
        "Uporedi JPMorgan i Bank of America CET1 pokazatelje za 2025.",
        conversation_history=[],
    )

    assert run.output["dialog_act"] == "clarification"
    assert run.output["reason_code"] == "missing_metric"
    assert "iznos" in run.output["answer"].casefold()
    assert "procenat" in run.output["answer"].casefold()
    assert run.diagnostics["model_request_count"] == 0


def test_orphaned_follow_up_does_not_revive_stale_session_bank() -> None:
    pipeline = BankAnswerPipeline(
        retriever=NeverCalled(),
        query_encoder=NeverCalled(),
        client=NeverCalled(),
        generation_model="test-model",
        bank_names={"JPM": "JPMorgan Chase & Co."},
    )

    run = pipeline.answer(
        "Reci mi više.",
        ticker="JPM",
        conversation_history=[],
    )

    assert run.output["dialog_act"] == "clarification"
    assert run.output["reason_code"] == "missing_intent"
    assert run.output["retrieval"]["mode"] == "none"
    assert run.diagnostics["model_request_count"] == 0

# ADR 016: Finance and technology conversation scope

- Status: Accepted
- Date: 2026-08-27

## Context

ADR 015 made BankScope a general assistant so benign non-banking requests could use direct chat or
web search. The product now needs a narrower boundary: it should remain conversational, but it
should not answer recipes, travel, weather, sports, entertainment, or similar requests unrelated to
its professional purpose.

The conversation graph already owns every threaded routing decision and already exposes a strict
`out_of_scope` action with a server-rendered response. Adding another classifier or API call would
duplicate that decision, add latency, and create a second source of routing truth.

## Decision

The central conversation-router prompt defines finance and technology as the allowed knowledge
domain. This includes banking, markets, regulation, financial metrics, programming, AI, cyber, and
enterprise technology. Greetings, acknowledgements, capability questions, clarifications,
transformations of previous allowed answers, and general arithmetic remain allowed.

Requests unrelated to finance or technology route to `out_of_scope`. That route returns the
existing unsupported response and stops before filing retrieval, web search, calculation, or answer
generation. Prior bank or technology conversation state does not make an unrelated current request
in scope. Genuinely ambiguous requests receive one clarification.

Enforcement is prompt-only: there is no keyword veto, second classifier, new API request, setting,
or public response type. The existing deterministic source policy continues to protect bank filing
and current bank claims, but it does not add a domain-scope override.

## Verification

The routing evaluation contains English and Serbian allowed and declined cases, including finance,
programming, AI, cyber, current non-bank market data, recipes, cooking, travel, weather, sports,
horoscopes, entertainment, and an off-topic turn with stale bank history. Acceptance requires at
least 95% total route accuracy, 100% supported-bank filing recall, 100% expected no-retrieval, 100%
out-of-scope recall, and no scope-preservation failures.

## Consequences

- Direct and web answers remain available across finance and technology, not only supported banks.
- Small talk, relevant follow-ups, and the deterministic calculator remain conversational.
- Off-topic requests receive a short purpose-oriented response without invoking another tool.
- Prompt-only enforcement is intentionally lightweight and must be monitored with the live routing
  evaluation because it is not a deterministic semantic guarantee.

[ADR index](README.md)

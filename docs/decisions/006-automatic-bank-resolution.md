# Automatic bank resolution

**Status: accepted on 2026-08-11.**

## Context

The answer pipeline required callers to supply `--ticker`, which would force a future chat UI to
show a bank selector. The product should instead identify one of the ten configured banks from
the user's question and ask for clarification before retrieval when the bank is missing or the
question names more than one bank.

## Decision

- Resolve banks deterministically from registry-owned legal names, aliases and tickers.
- Normalize case, punctuation, possessives, ampersands and whitespace, then match complete
  phrases only. Do not use fuzzy matching or an LLM fallback.
- Treat Citigroup ticker `C` specially: only `Citi`, `Citigroup`, `$C` or an explicit
  `ticker C` form identifies it.
- Prefer an explicit bank in the current question over the optional session/evaluation ticker.
  Use the fallback only when the current question contains no configured bank.
- Return `ambiguous` locally for missing or multiple banks, with no embedding, retrieval or
  generation request.
- Keep cross-bank answers outside the current single-bank generation scope.

## Verification

Resolver tests cover legal names, aliases, punctuation, ticker boundaries, Citigroup `C`,
session inheritance, explicit bank switching and multiple-bank questions. The frozen 30-query
set is also a deterministic resolver fixture: 26 questions resolve to their recorded ticker,
three cross-bank questions resolve as multiple, and the bankless question resolves as missing.

The existing PNC rounded extra-citation issue remains documented and deferred. It does not block
automatic bank resolution, conversation history or the local UI; GPT-5.1 remains a candidate
rather than the application default.

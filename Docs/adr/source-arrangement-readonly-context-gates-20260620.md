# ADR: Source identity and arrangement read-only context gates

## Context

Source identity and arrangement-role evidence are upstream of many Automixer
decisions: gain staging, EQ, compression, gating, panning, effects, frequency
separation, cross-channel unmasking and balance all depend on knowing what a
source is and what role it plays in the arrangement.

Before this checkpoint, stale or missing context was not consistently blocked
for these proposal families, and a fresh context payload could still carry live
authority markers. That is unsafe for downstream trust: a proposal-only review
candidate must be grounded in fresh read-only evidence before other mix
decisions can consume it.

## Options Considered

1. Allow missing context as an analysis default and rely on downstream modules
   to reject low-confidence results.
2. Require fresh read-only/no-authority context before source-identity and
   arrangement-role proposal candidates are emitted.
3. Promote source identity and arrangement role directly into queueable
   assisted actions.

## Decision

Require source-identity and arrangement-role proposal candidates to have fresh
context that is explicitly read-only, non-live-write and non-console-querying.
Missing, stale or live-authority context keeps the result analysis-only and
blocks downstream proposal trust.

## Why This Won

This moves the AI mixing engineer toward better musical reasoning without
granting console authority. The system can still analyze uncertain sources and
roles, but downstream proposal families receive a stronger evidence boundary
before they trust the labels.

## Rejected Alternatives

- Letting missing context pass keeps demos permissive but makes downstream EQ,
  balance and unmasking decisions depend on weak identity evidence.
- Promoting these labels directly into queueable assisted actions skips the
  replay/acceptance path and would widen authority before M007/M008 gates are
  ready.

## Implementation Plan

- Add no-authority checks to source-identity confidence proposal analysis.
- Add no-authority checks to arrangement-role proposal analysis.
- Surface the no-authority verdict in proposal evidence.
- Add tests for fresh read-only context, missing context and live-authority
  context.
- Add a M083 rail and evidence trail while keeping M001/M002 blocked on real
  WING/operator validation.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_source_identity_confidence_proposal.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_arrangement_role_proposal.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m083_source_arrangement_context_gates_stay_readonly -q`
- Standard backend suite.
- Memory lint and JSON/JSONL validation.

## Risks and Rollback

Risk: proposal candidates may be more conservative until fresh read-only
context is available. This is acceptable because source identity and arrangement
role feed higher-risk downstream mix decisions.

Rollback: remove the no-authority context checks, the new tests and M083
state/report updates. This does not affect live transport because the change is
offline/proposal-only.

No live transport, no OSC write, no proposal apply, no readiness closure and no
live authority are introduced by this decision.

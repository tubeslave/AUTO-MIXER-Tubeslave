# ADR: Gate/Expander Chatter-safe Proposal Layer

Date: 2026-06-15
Status: accepted

## Context

Automatic gating is an explicit project requirement. The existing
`context_aware_gate` method was transport-free and proposal-only in registry
terms, with freshness and source-identity gates. It still lacked a Product
Layer operator proposal shape and did not separately prevent hard-gate
recommendations when chatter risk was high.

## Options considered

1. Keep gate output as dry-run decision traces only.
2. Add a non-queueable Product Layer proposal wrapper and chatter guard.
3. Promote trusted gate recommendations into live gate writes.

## Decision

Use option 2. `ContextGateRecommendation` now renders a Product Layer operator
proposal, and `build_context_gate_proposals` filters out analysis-only holds.
The proposal carries bleed telemetry, false-open/false-close, chatter, tail,
hysteresis, source-identity, freshness, readback, and rollback evidence. Hard
gate mode is allowed only when chatter risk is below the configured guard.

## Why this won

It turns gate/expander work into a consistent operator-review artifact without
creating a live executor. The method remains transport-free, non-queueable, and
blocked from live apply until physical readback, rollback, replay evidence, and
hardware proof exist.

## Rejected alternatives

Dry-run-only traces were rejected because they do not give the operator a
standard proposal surface. Live promotion was rejected because gate readback,
rollback, and false-decision replay evidence are incomplete.

## Implementation plan

1. Add a proposal serializer to `ContextGateRecommendation`.
2. Add `build_context_gate_proposals` for proposal-only output filtering.
3. Add a hard-gate chatter guard and evidence.
4. Update proposal contracts, registry wording, capability matrix, backlog, and
   tests.

## Test plan

- `PYTHONPATH=backend python -m pytest -q tests/test_agentic_method_foundation.py tests/test_project_goal_capability_matrix.py tests/test_processing_registry.py tests/test_gate.py`
- Full suite with `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`.

## Risks and rollback

Risk: high-chatter drum sources may receive gentler expander suggestions instead
of hard gates. Roll back by removing the proposal wrapper, build helper,
chatter guard, evidence fields, tests, and this ADR. No live WING, OSC, or
console mutation behavior is introduced.

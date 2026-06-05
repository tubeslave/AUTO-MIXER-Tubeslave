# Brief: context-aware-gate-source-identity-freshness-20260605

Task id: context-aware-gate-source-identity-freshness-20260605

## Context

The project goal requires automatic gating and gate/expander console control.
`ContextAwareGateAgent` already emits dry-run gate/expander recommendations from
bleed telemetry, false-open/false-close risk, chatter, tail risk, and source
context. Gate decisions are source-sensitive and can damage a performance if
bleed telemetry or source identity is stale or untrusted.

## Scope

- Add freshness evidence to context-aware gate recommendations.
- Add replay-calibrated source identity downstream trust.
- Return analysis-only hold recommendations when freshness or source identity
  trust blocks the gate decision.
- Keep normal legacy recommendations compatible when freshness and calibration
  are missing.
- Keep all gate/expander output dry-run, non-queueable, and non-live-applicable.

## Non-goals

- No live gate writes.
- No WING, dLive, Mixing Station, OSC, or MIDI transport changes.
- No source identity classifier mutation.
- No memory-learning writes.

## Evidence to preserve

- `current_state_cache_freshness_status`.
- `source_identity_downstream_trust`.
- `blockers` for stale context or untrusted identity.
- `live_write=False`.
- `queue_mutation=False`.
- `model_mutation=False`.
- `memory_mutation=False`.

## Validation

- Focused agentic/gate tests.
- Source identity downstream gate tests.
- Registry/contract and capability matrix tests.
- Full backend pytest.
- `git diff --check`.
- Runtime smoke for held source identity.

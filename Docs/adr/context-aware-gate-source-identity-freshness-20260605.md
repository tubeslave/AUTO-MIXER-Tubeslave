# ADR: context-aware-gate-source-identity-freshness-20260605

## Context

Automatic gating is explicitly part of the project goal, but gate writes can
cut musical tails, close on the wrong source, or open from bleed if the context is
wrong. `ContextAwareGateAgent` already analyzes bleed telemetry and false
open/close risk without live transport. The missing guardrail was making stale
bleed context and untrusted source identity first-class blockers.

## Options considered

1. Keep context-aware gate recommendations based only on telemetry metrics.
2. Add freshness and source identity trust gates before suggesting parameters.
3. Promote trusted gate decisions into live gate writes.

## Decision

Use option 2. `BleedTelemetry` now carries optional freshness and source identity
trust evidence. `ContextAwareGateAgent.recommend` evaluates those fields before
suggesting gate/expander params. Stale context or untrusted source identity
returns an `analysis_only` hold recommendation with no suggested params. Missing
freshness and missing source identity calibration remain legacy-compatible.

## Why this won

This moves automatic gating toward Assisted Mixing and future supervised proof
while preserving the live-write boundary. It produces explainable hold records
for replay and operator review without adding any live gate executor.

## Rejected alternatives

Option 1 leaves a known safety gap. Option 3 remains unsafe because gate readback,
rollback, replay validation, and physical proof are not complete.

## Implementation plan

- Add optional freshness and source identity fields to `BleedTelemetry`.
- Normalize freshness and evaluate `source_identity_downstream_gate`.
- Return `analysis_only` holds with no suggested params when blockers exist.
- Attach no-mutation evidence to normal and held recommendations.
- Update registry, proposal contracts, capability matrix, and tests.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/context_aware_gate.py backend/source_identity_downstream_gate.py backend/processing_registry.py backend/processing_proposal_adapters.py backend/project_goal_capability_matrix.py tests/test_agentic_method_foundation.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_agentic_method_foundation.py tests/test_source_identity_downstream_gate.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`
- Runtime smoke for held source identity: `analysis_only`, no suggested gate
  params, and no live, queue, model, or memory mutation.

## Risks and rollback

Risk: upgraded callers with stale context or held source identity will see held
analysis records instead of gate proposals. Rollback is removing the telemetry
fields, gate evaluation, hold recommendation path, contract/matrix text, tests,
and this ADR. No live write behavior is affected.

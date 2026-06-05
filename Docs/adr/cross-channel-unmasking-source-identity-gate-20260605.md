# ADR: cross-channel-unmasking-source-identity-gate-20260605

## Context

Cross-channel unmasking chooses a protected source and a masking source, then
emits linked mirror-EQ cut/boost candidates. That decision depends on source
priority and source-role evidence. If a channel's source identity is held,
blocked, unknown, or based on stale replay calibration, the Automixer can cut or
boost the wrong relationship while still appearing to have fresh masking
context.

## Options considered

1. Keep cross-channel unmasking gated only by masking-context freshness.
2. Reuse the shared source identity downstream gate in the proposal wrapper.
3. Promote linked mirror EQ to live EQ writes after source identity trust.

## Decision

Use option 2. `cross_adaptive_eq.analyze_unmasking_proposals` now accepts
optional `channel_source_identity` metadata and evaluates the shared
transport-free source identity downstream gate for each channel in the masking
context. Held, blocked, unknown, or stale replay context returns no proposal
actions. Missing calibration remains legacy-compatible as `legacy_unverified`.

## Why this won

This adds trust evidence at the point where linked unmasking decisions are
formed, without changing the legacy raw correction API and without adding any
live EQ authority. It keeps unmasking aligned with the broader proposal-only
source identity guardrail now used by source/stem frequency separation and other
source-dependent proposal layers.

## Rejected alternatives

Option 1 leaves a source-role trust gap in a path that can change EQ priorities.
Option 3 remains unsafe because EQ-band readback, sum-bus verification, rollback,
and hardware proof are not complete.

## Implementation plan

- Add optional `channel_source_identity` to `analyze_unmasking_proposals`.
- Evaluate `source_identity_downstream_gate` per channel.
- Attach trust evidence to result metadata and each action's safety evidence.
- Return no proposal actions when the trust gate blocks any channel.
- Update processing registry, proposal contract, capability matrix, and tests.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/cross_adaptive_eq.py backend/source_identity_downstream_gate.py backend/processing_registry.py backend/processing_proposal_adapters.py backend/project_goal_capability_matrix.py tests/test_cross_adaptive_eq.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_cross_adaptive_eq.py tests/test_source_identity_downstream_gate.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`
- Runtime smoke for held source identity: no adjustments, no suggested actions,
  and no live, queue, model, or memory mutation flags.

## Risks and rollback

Risk: callers that pass stale or held identity evidence will produce fewer
unmasking proposals. Rollback is removing the optional wrapper argument, gate
evaluation, evidence fields, metadata, contract/matrix text, focused tests, and
this ADR. No live write behavior is affected.

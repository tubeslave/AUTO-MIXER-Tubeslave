# ADR: Cross-channel Unmasking Arrangement Priority Gate

Date: 2026-06-15
Status: accepted

## Context

Automatic cross-channel unmasking is an explicit project requirement. The
existing `cross_adaptive_eq` method was transport-free and proposal-only, with
masking-context freshness, source-identity trust, EQ-band readback, rollback,
and no live-write evidence. It still depended on caller-provided integer
priorities to identify the protected source and the masker.

Fresh arrangement-role context is a better signal for missing priorities:
lead, featured, low-anchor, and rhythmic-anchor sources should usually be
protected before harmonic support, backing, texture, or effect sources.

## Options considered

1. Keep unmasking priority opaque and caller-provided only.
2. Add optional arrangement-role priority hints and a freshness gate to the
   proposal-only unmasking wrapper.
3. Promote linked mirror EQ into queueable or live EQ writes.

## Decision

Use option 2. `CrossAdaptiveEQ.analyze_unmasking_proposals` now accepts optional
`channel_arrangement_context` and `arrangement_context_cache_freshness`.
Explicit `channel_priorities` still win. When an explicit priority is absent,
fresh trusted arrangement context can supply a priority hint. Stale, missing,
unknown, non-processable, or low-confidence provided arrangement context blocks
proposal output.

## Why this won

It moves cross-channel unmasking closer to arrangement-aware AI mixing without
widening live rights. The method still emits proposal-only actions, cannot
import to the live queue, and requires readback/rollback evidence before any
future promotion.

## Rejected alternatives

Opaque priority-only unmasking was rejected because it does not explain or
derive protected-vs-masker choice from arrangement context. Live promotion was
rejected because EQ-band physical readback, sum-bus verification, rollback, and
hardware proof are not complete.

## Implementation plan

1. Normalize optional arrangement context per channel.
2. Normalize arrangement-context freshness and block stale provided context.
3. Derive effective priorities from arrangement role only when explicit
   priorities are absent.
4. Attach arrangement priority evidence to metadata and action safety evidence.
5. Update proposal contracts, registry wording, capability matrix, backlog, and
   tests.

## Test plan

- `PYTHONPATH=backend python -m pytest -q tests/test_cross_adaptive_eq.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py`
- Related proposal-layer tests for source identity, frequency separation, and
  de-esser.
- Full suite with `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`.

## Risks and rollback

Risk: fresh but wrong arrangement context could produce a misleading proposal.
The mitigation is confidence gating, source-identity gating, and proposal-only
output with readback/rollback requirements. Roll back by removing the optional
arrangement context arguments, priority-hint derivation, evidence fields, tests,
and this ADR. No live WING, OSC, or console mutation behavior is introduced.

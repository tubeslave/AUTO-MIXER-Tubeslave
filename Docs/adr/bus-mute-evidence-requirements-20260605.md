# ADR: bus-mute-evidence-requirements-20260605

## Context

Bus/group/matrix and mute promotion targets have explicit fail-closed safety
coverage rows, but their promotion readiness rows do not yet serialize
machine-readable evidence requirement IDs.

## Options considered

1. Add evidence requirement IDs to those promotion targets.
2. Keep requirements only in safety-row prose.
3. Add a new global evidence registry before adding IDs.

## Decision

Use option 1. Add stable evidence IDs directly to the existing promotion target
definitions.

## Why this won

Promotion readiness already uses target-local evidence IDs for routing and
scene/preset. Extending the same contract keeps the gap summary useful without
creating a new abstraction or changing live authority.

## Rejected alternatives

Prose-only requirements are hard for proof runners and cockpit summaries to
consume. A registry may be useful later, but it is unnecessary for this scoped
contract update.

## Implementation plan

1. Add bus/group/matrix evidence IDs for cache freshness, fader/readback,
   matrix-send readback, headroom, physical fixture, and rollback proof.
2. Add mute evidence IDs for current mute freshness, owner/intent,
   signal-return, manual authority, physical fixture, and emergency/rollback
   proof.
3. Update readiness tests so these IDs appear in rows and gap summaries while
   live authority remains unchanged.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_autonomy_promotion_readiness.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_no_wing_release_check.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk: evidence IDs could be misread as promotion readiness. The rollback is
removing the added ID tuples; safety rows and live blockers remain intact.

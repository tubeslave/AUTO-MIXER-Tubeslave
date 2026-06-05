# promotion-readiness-evidence-requirements-20260605

## Context

Autonomy promotion readiness explains blockers in text, but it does not expose
machine-readable evidence requirement IDs per promotion target. For the long-term
AI Mixing Engineer goal, agents and UI surfaces need to know exactly which proof
artifacts are missing before an action family can move toward supervised or
autonomous operation.

## Goal

Add read-only `evidence_requirement_ids` to promotion readiness targets and rows.
Scene/preset readiness must explicitly name snapshot freshness, scoped diff,
read-only rollback evidence, and hardware proof requirements.

## Non-goals

- Do not change safety coverage statuses.
- Do not promote any target to supervised live or autonomous readiness.
- Do not add mixer transport imports, console queries, or live writes.
- Do not change frontend behavior unless backend contract tests require it.

## Safety constraints

- Full Autonomous remains blocked.
- Only fader can remain currently supervised-live allowed.
- Scene/preset and routing targets remain manual-only blocked.

## Expected evidence

- Readiness rows serialize `evidence_requirement_ids` as lists.
- `scene_preset_supervised_promotion` lists read-only scene/preset proof
  requirements.
- `routing_supervised_promotion` lists routing proof requirements.
- Tests verify the new IDs without allowing live write widening.

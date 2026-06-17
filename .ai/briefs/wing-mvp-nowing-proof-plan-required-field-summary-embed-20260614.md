# Brief: WING MVP no-WING Proof Plan Required Field Summary Embed

## Task

Expose the proof-plan required summary field contract in the aggregated no-WING `next_wing_proof_plan` summary artifact.

## Context

`next_wing_proof_plan_schema` now fails closed when `next_wing_proof_plan.summary` omits any field declared by the handoff-required proof-plan summary contract. The raw check detail carried the required and missing field lists, but the aggregated no-WING summary should also expose them so operators and Project Director can inspect the generated artifact directly.

## Constraints

- No live mixer transport.
- No live-write authority.
- No positive gain or fader movement.
- Preserve the first WING MVP blocker until real WING evidence exists.
- Keep the change scoped to report/summary propagation.

## Done When

- `next_wing_proof_plan_summary()` includes `required_supervised_proof_plan_summary_fields`.
- `next_wing_proof_plan_summary()` includes `missing_required_supervised_proof_plan_summary_fields`.
- The no-WING summary embedding test verifies both fields.
- The decision is recorded in ADR/report docs and indexed.

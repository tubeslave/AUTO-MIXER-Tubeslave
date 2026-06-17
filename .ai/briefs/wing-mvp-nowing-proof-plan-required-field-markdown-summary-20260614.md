# Brief: WING MVP no-WING Proof Plan Required Field Markdown Summary

## Task

Expose the required and missing supervised proof-plan summary field lists in the
operator-readable no-WING markdown summary.

## Context

`summary.json` already carries `next_wing_proof_plan.required_supervised_proof_plan_summary_fields`
and `next_wing_proof_plan.missing_required_supervised_proof_plan_summary_fields`.
The markdown summary should show the same contract so an operator can see why a
stale proof plan remains fail-closed without opening JSON.

## Constraints

- No WING connection.
- No live mixer transport.
- No live-write authority.
- Do not alter proof-plan authority semantics.
- Keep the WING read-only connection proof blocker open until real WING evidence exists.

## Done When

- `summary.md` renders both required and missing proof-plan summary field lists.
- Tests cover the successful empty-missing path and a stale summary with a missing required field.
- The decision is recorded in ADR/report docs.

# ADR: WING MVP no-WING Proof Plan Required Field Summary Embed

## Context

`next_wing_proof_plan_schema` reports required and missing proof-plan summary fields. Before this decision, those fields were available in raw check detail but not propagated into the aggregated no-WING `next_wing_proof_plan` summary object.

## Options considered

1. Keep the diagnostics only in raw check detail.
2. Add a separate top-level no-WING summary section.
3. Add the diagnostics to the existing `next_wing_proof_plan` summary object.

## Decision

Choose option 3.

The aggregated `next_wing_proof_plan` summary now includes:

- `required_supervised_proof_plan_summary_fields`
- `missing_required_supervised_proof_plan_summary_fields`

## Why this won

The no-WING `next_wing_proof_plan` summary is the operator-facing artifact for the next supervised WING proof boundary. It should carry the same required/missing field contract as the schema gate so stale artifacts are diagnosable without digging through raw check details.

## Rejected alternatives

- Raw detail only is too indirect for the handoff path.
- A separate top-level section would split proof-plan diagnostics across two places.

## Implementation plan

- Propagate required and missing field lists in `next_wing_proof_plan_summary()`.
- Assert the generated `summary.json` embeds both lists.
- Update WING blocker docs, report docs, and the documentation index.

## Test plan

- `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_next_wing_proof_plan`
- Run focused no-WING/handoff, rail, compile, memory-lint, and whitespace checks.

## Risks and rollback

Risk: exact-summary tests or consumers may need to accept two additional diagnostic fields.

Rollback: remove the two propagated fields and test assertions. This restores prior summary shape but makes the operator-facing no-WING artifact less complete.

# Proposal: WING MVP Proof Plan Prestart Violations Step Gate

## Short thesis

Add an explicit no-WING fail-closed gate: a supervised proof plan cannot include physical proof steps while its accepted-candidate prestart violations map is non-empty.

## What I understood

The active MVP is still the supervised WING Rack proof. Before fader/gain proof, accepted read-only candidates must have a valid no-WING prestart trace and the proof-plan summary must show no accepted-candidate prestart contract violations.

## Proposed solution

- Extend `validate_supervised_proof_report_next_wing_plan()` with a direct check for non-empty `readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source`.
- If the map is non-empty and `step_count` is non-zero, fail with `next_wing_proof_plan_steps_with_candidate_prestart_violations`.
- If the map is non-empty and the prestart blocker is absent, fail with the existing blocker-missing violation.
- Add a regression test that corrupts only the proof-plan map while steps remain planned.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`
- `Docs/reports/`

## Alternatives considered

- Rely on existing mismatch checks only.
- Add the rule only to docs.

## Risks

The explicit violation adds one more failure reason for stale proof-plan summaries. That is acceptable because it makes the safety boundary easier to diagnose.

## Test plan

- Focused pytest for the new regression.
- Existing no-WING/handoff focused suite.
- Project Director, compile, memory lint, and whitespace checks.

## Where the other agent may disagree

Kimi may argue the existing mismatch checks were enough. The direct invariant is still useful because it encodes the operator-facing rule from Current First Command without requiring readers to infer it from map comparison failures.

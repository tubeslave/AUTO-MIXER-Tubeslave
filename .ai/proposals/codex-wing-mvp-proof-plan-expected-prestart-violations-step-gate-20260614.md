# Proposal: WING MVP Proof Plan Expected Prestart Violations Step Gate

## Short thesis

Add a direct no-WING fail-closed gate for recomputed expected accepted-candidate prestart violations.

## What I understood

The active MVP is still supervised WING Rack proof. Before fader/gain proof, the accepted read-only candidate must have a valid no-WING prestart trace. The proof-plan summary may carry stale reported maps, so recomputed expected violations must independently block proof steps.

## Proposed solution

- Extend `validate_supervised_proof_report_next_wing_plan()` to check the recomputed `expected_candidate_prestart_violations_by_source`.
- If that map is non-empty and `step_count` is non-zero, fail with `next_wing_proof_plan_steps_with_expected_candidate_prestart_violations`.
- Require the existing prestart blocker when the expected map is non-empty.
- Add a regression test where the ledger's accepted-candidate prestart trace is invalid while proof steps remain planned.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`
- `Docs/reports/`

## Alternatives considered

- Rely on map mismatch failures only.
- Add the rule only to the Project Director wording.

## Risks

The explicit violation adds one more failure reason for stale proof-plan artifacts. This is acceptable because it makes the unsafe state obvious.

## Test plan

- Focused pytest for the new regression.
- Existing no-WING/handoff focused suite.
- Project Director, compile, memory lint, and whitespace checks.

## Where the other agent may disagree

Kimi may argue mismatch failures already catch stale maps. The direct gate is still safer because it prevents proof-step planning from depending on a reported map being honest.

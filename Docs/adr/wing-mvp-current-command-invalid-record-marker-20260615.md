# ADR: WING MVP Current-Command Invalid-Record Marker

## 1. Context

The supervised proof-plan summary contract now includes
`readonly_preflight_review_decision_invalid_records`, and no-WING validation
checks the field against the review-decision ledger. Project Director Current
First Command still described required proof-plan fields without naming this
new invalid-record field.

## 2. Options considered

- Leave the field implicit in the proof-plan contract.
- Add a broader generated marker list.
- Add the concrete missing marker to Current First Command and Project Director
  baseline coverage.

## 3. Decision

Add `readonly_preflight_review_decision_invalid_records` to Current First
Command, Project Director baseline coverage markers, the Codex autoloop prompt,
and the Codex autoloop README.

## 4. Why this won

The current Project Director model checks explicit operator-facing markers. A
direct marker prevents stale current-command text from passing baseline
coverage after proof-plan contract changes.

## 5. Rejected alternatives

Implicit contract coverage was rejected because it does not prove the operator
directive is current. A generated marker system was deferred to avoid widening
the patch beyond this concrete gap.

## 6. Implementation plan

- Update Current First Command field list.
- Add the marker to `CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS`.
- Update autoloop prompt and README.
- Add Project Director regression assertions.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.

## 7. Test plan

Run Project Director tests, no-WING quick, Project Director JSON, diff check,
memory lint, and standard pytest.

## 8. Risks and rollback

Risk is limited to guidance and baseline coverage. Rollback is removing the
marker from the Current First Command, baseline tuple, prompt, README, and
tests.

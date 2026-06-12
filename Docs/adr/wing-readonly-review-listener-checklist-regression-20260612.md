# ADR: WING Read-Only Review Listener Checklist Regression

## Context

The backend-port prestart check now emits `listener_confirmation_checklist`.
Manual review surfaces already render the prestart failure-detail field list,
but tests only required `listener_processes`. A future change could drop the
new checklist from operator-visible review evidence without failing tests.

## Options considered

1. Leave raw no-WING tests as the only coverage.
2. Add regression assertions to review and decision review surfaces.
3. Rewrite report rendering to special-case checklist fields.

## Decision

Use option 2. The report renderers already expose the list; tests now require
`listener_confirmation_checklist` in the relevant downstream evidence.

## Why this won

It locks the proof boundary without adding new behavior or widening authority.
The change is small and directly supports the first WING MVP blocker.

## Rejected alternatives

Raw-only coverage was rejected because operators use review reports, not only
raw JSON. Renderer rewrites were rejected because the current generic field-list
rendering is sufficient.

## Implementation plan

- Assert checklist visibility in the artifact review report.
- Assert checklist visibility in the decision review report.
- Assert checklist visibility in no-WING decision-ledger summary output.

## Test plan

- Focused review/no-WING tests.
- Adjacent WING/no-WING tests.
- Full repo validation before commit.

## Risks and rollback

The risk is limited to test brittleness if the shared field is renamed. Roll
back by updating the assertions alongside the shared contract.

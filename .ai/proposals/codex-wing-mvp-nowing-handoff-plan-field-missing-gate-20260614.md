# Codex Proposal: WING MVP no-WING Handoff Plan Field Missing Gate

## Short thesis

Missing handoff `plan_<field>` values should be a first-class no-WING failure,
not an implicit side effect.

## What I understood

The active blocker remains WING read-only connection proof. Handoff packets now
carry required-false plan fields, and no-WING validation already computes
packet-vs-plan match verdicts. The missing regression is an explicit test that
older packets without those fields fail closed.

## Proposed solution

Add a focused test that removes all backend-port required-false `plan_<field>`
values from a valid handoff packet and asserts the no-WING validator reports
failed status, `plan_<field>` as missing/invalid, packet-vs-plan match false,
and `*_plan_mismatch` violations.

## Likely files to touch

- `tests/test_no_wing_release_check.py`
- Council/report docs for this slice.

## Alternatives considered

- Rely on unsafe-value tests. Rejected because missing plan fields are a
  separate stale-artifact failure mode.
- Add production code first. Rejected unless the focused test shows the current
  validator does not fail closed.

## Risks

This is test-only unless it exposes a real validator gap. It does not change
live behavior.

## Test plan

- Focused no-WING handoff packet artifact test.
- Focused no-WING handoff/prestart subset.
- Project Director sanity and `git diff --check`.

## Where the other agent may disagree

Kimi may ask for a dedicated violation name such as `plan_field_missing`. I
would keep the current `plan_mismatch` failure unless the test shows it is too
ambiguous for operator review.

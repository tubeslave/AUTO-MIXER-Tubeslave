# ADR: WING MVP no-WING Handoff Plan Field Missing Gate

Date: 2026-06-14
Status: Accepted

## Context

Current WING read-only rails require backend-port required-false plan values to
be visible before a handoff packet can feed real hardware work. Older handoff
packets may not carry those `plan_<field>` values.

## Options considered

1. Rely on existing unsafe-value tests.
2. Add explicit no-WING regression coverage for missing `plan_<field>` values.
3. Change the validator before proving a gap.

## Decision

Use option 2. Missing handoff plan fields should be explicitly covered as a
fail-closed no-WING artifact gate.

## Why this won

It strengthens the operator-facing WING read-only release gate without adding
new behavior unless the current validator proves insufficient.

## Rejected alternatives

- Unsafe-value tests only: they do not prove stale/missing plan fields fail.
- Production-first change: unnecessary if the current validator already fails
  closed.

## Implementation plan

- Add a focused test that removes `plan_<field>` values from a valid handoff
  packet.
- Assert failed status, false packet-vs-plan verdicts, and plan-mismatch
  violations.
- Add a short validated report if the check passes.

## Test plan

- Focused no-WING handoff packet artifact test.
- Focused no-WING handoff/prestart subset.
- Project Director sanity and whitespace diff.

## Risks and rollback

No runtime behavior is changed unless a validator gap is found. Rollback removes
the regression test and report only.

# ADR: WING MVP Handoff Prestart Plan Fields

Date: 2026-06-14
Status: Accepted

## Context

The WING read-only handoff packet is the operator-facing boundary before real
hardware work. Current rails require backend-port required-false fields, their
expected false plan values, and match verdicts before supervised proof planning
can proceed.

## Options considered

1. Keep `plan_<field>` values only in no-WING prestart summaries.
2. Trust the recorded prestart match flags blindly.
3. Carry prestart-owned plan values and match verdicts through the handoff
   packet and render them directly.

## Decision

Use option 3. The handoff packet exposes the same prestart plan fields that
candidate traces and proof-plan gates already require, and computes packet-level
match verdicts from raw values versus those plan values.

## Why this won

It makes the operator packet self-auditing and prevents the handoff layer from
masking missing or stale prestart plan fields.

## Rejected alternatives

- Summary-only plan fields: too indirect for the handoff boundary.
- Blindly trusting recorded match flags: can hide raw-value or plan-value drift.

## Implementation plan

- Import shared plan/match constants into the packet builder.
- Add packet fields for `plan_<field>` and `<field>_matches_plan`.
- Render the fields in markdown next to raw denial values.
- Add focused regression tests.

## Test plan

- Focused handoff packet tests.
- Focused no-WING release checks.
- Project Director sanity, compile, and whitespace diff.

## Risks and rollback

Real prestart summaries generated before this field set may surface missing
values and remain blocked for supervised proof planning. Rollback would remove
the packet-level exposure, but the candidate prestart contract should remain
strict.

# ADR: WING MVP Prestart Required-False Plan Match Trace

Date: 2026-06-14
Status: Accepted

## Context

Accepted-candidate WING read-only prestart traces are used to block or allow
supervised proof planning. The trace already includes the raw backend-port
required-false denial fields, but it must also prove those fields match the
expected false plan recorded by the no-WING prestart summary.

## Options considered

1. Continue checking only raw required-false fields.
2. Require handoff-packet declared fields inside prestart rows.
3. Add prestart-owned `plan_<field>: false` and `<field>_matches_plan: true`
   fields and require them through evidence, review, decision, and proof-plan
   helpers.

## Decision

Use option 3. The accepted-candidate prestart contract now rejects missing or
false required-false match verdicts and missing or truthy plan values.

## Why this won

It proves the candidate prestart row is not just carrying false values, but is
also bound to the expected no-authority plan. It keeps packet-level declared
fields separate from prestart-level computed verdicts.

## Rejected alternatives

- Raw-only checks: too weak for stale prestart summaries.
- Packet declared fields in prestart rows: mixes two artifact layers and would
  blur the handoff/prestart boundary.

## Implementation plan

- Add shared plan/match field constants.
- Have `check_port_free` write plan and match verdict detail.
- Have evidence intake validate and preserve those fields.
- Have review decision rows and renderers expose those fields.
- Have proof-plan candidate helpers reject stale or missing fields.
- Add focused regression tests and update Project Director rails.

## Test plan

- Focused WING evidence/review/decision/proof-plan/no-WING tests.
- Focused Project Director and WING rails tests.
- Compile, memory lint, full suite, and whitespace diff.

## Risks and rollback

Older candidate rows fail closed until regenerated with the new prestart detail.
Rollback would remove the new plan/match helper checks, but stale prestart rows
must not be used as readiness proof.

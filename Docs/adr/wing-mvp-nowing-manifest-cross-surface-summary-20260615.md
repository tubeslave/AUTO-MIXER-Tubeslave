# ADR: no-WING Manifest Cross-Surface Summary

## Context

The supervised WING MVP remains blocked on real WING read-only proof. Offline
no-WING packets must therefore be explicit about what they prove and what they
do not prove.

The no-WING artifact manifest already checks that `summary.json` and
operator-readable `summary.md` expose the canonical manifest contract digest.
The Current First Command also requires those cross-surface digest verdicts to
be visible in the no-WING summary surface.

## Options considered

1. Keep the fields only in `artifact-manifest.json`.
2. Mirror the cross-surface digest verdicts in `summary.json` and `summary.md`.
3. Rewrite `summary.json` after `artifact-manifest.json` is generated.

## Decision

Choose option 2. The no-WING summary surface now exposes the JSON/Markdown
contract SHA match fields, the cross-surface mismatch list, and the aggregate
cross-surface match verdict.

## Why this won

It makes the operator-facing summary match the Project Director directive
without adding WING contact, live writes, readiness updates, or a new artifact
write loop.

## Rejected alternatives

Keeping the fields only in `artifact-manifest.json` leaves the summary surface
incomplete. Rewriting `summary.json` after manifest generation adds complexity
without changing the safety boundary.

## Implementation plan

- Extend manifest summary surface required fields.
- Render the new fields in no-WING `summary.json` and `summary.md`.
- Extend tests for the new fields and for missing-line failure.
- Record validation in a report.

## Test plan

- Focused no-WING manifest summary tests.
- no-WING quick check.
- Project Director check.
- Standard pytest.

## Risks and rollback

Risk is limited to offline reporting. Roll back by removing the added summary
fields and restoring the previous required surface list. This does not affect
WING transport or live authority.

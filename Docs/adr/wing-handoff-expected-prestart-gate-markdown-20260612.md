# ADR: WING Handoff Expected Prestart Gate Markdown

## Context

The WING handoff packet contains `supervised_prestart_gate`, and no-WING
validation computes `expected_supervised_prestart_gate`. The Markdown handoff
artifact did not show the expected contract beside the actual gate, so a manual
review could miss whether the packet was self-consistent.

## Options considered

1. Keep the handoff packet unchanged.
2. Add the expected gate to the packet and Markdown, then validate it.
3. Move all expected-gate visibility into no-WING release summaries only.

## Decision

Use option 2. The handoff JSON now includes
`expected_supervised_prestart_gate`, Markdown renders it, and no-WING validation
rejects declared expected-gate drift.

## Why this won

It makes the operator handoff packet self-contained while keeping the computed
validator as the authority. The packet still grants no live authority and does
not close any WING MVP blocker.

## Rejected alternatives

Keeping the packet unchanged left a review gap in the direct handoff artifact.
Summary-only visibility was rejected because the handoff packet itself is the
artifact operators use before the real read-only preflight.

## Implementation plan

- Build one expected-gate dictionary in the handoff generator.
- Store it as both actual and expected fields at generation time.
- Render expected-gate Markdown plus `actual_matches_expected`.
- Reject mismatched declared expected gates in no-WING validation.

## Test plan

- Focused handoff/no-WING packet tests.
- Adjacent WING MVP rail tests.
- Full repo validation before commit.

## Risks and rollback

The main risk is noisy Markdown. Roll back by removing the Markdown section
while preserving JSON validation, or by rendering a compact field list instead
of the JSON object.

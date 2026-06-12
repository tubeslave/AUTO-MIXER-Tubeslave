# ADR: WING Read-Only Preflight Review Prestart Fields

## Context

Real WING read-only preflight summaries now include and validate the
backend-port prestart contract. The artifact review report still showed channel,
readback, and artifact identity fields but did not surface the prestart contract
that the summary had to satisfy.

## Options considered

1. Keep prestart fields only in raw `summary.json`.
2. Propagate prestart fields into evidence records and artifact review rows.
3. Have the review tool re-open summary files separately for display.

## Decision

Use option 2. Evidence records now carry the prestart fields, and the artifact
review markdown renders them for operator review.

## Why this won

The review report is the manual gate before a decision record. It should show
the same prestart boundary that the validator enforces without adding a second
file-reading path.

## Rejected alternatives

Option 1 hides required safety evidence. Option 3 duplicates artifact loading
logic already owned by evidence intake.

## Implementation plan

- Add prestart fields to `WingReadonlyPreflightArtifactRecord`.
- Add the fields to artifact review rows and markdown rendering.
- Update review tests and run validation.

## Test plan

- Focused artifact review test.
- Adjacent WING/no-WING tests.
- Full backend suite.

## Risks and rollback

Risk: review reports become longer. Rollback: remove the display fields while
keeping underlying evidence validation. No hardware access or live write
behavior is changed.

# ADR: WING decision review prestart summary fields

## Context

The supervised WING MVP requires accepted read-only preflight source ids to have
a valid accepted-candidate prestart trace before any supervised proof. The
decision review ledger already validates the prestart summary binding, but the
Markdown report did not show the bound no-WING prestart summary fields.

## Options considered

1. Keep the prestart summary fields JSON-only.
2. Render the ledger's prestart summary fields in the decision review Markdown.
3. Recompute the prestart summary contract inside the Markdown renderer.

## Decision

Use option 2. The Markdown report should render the already-validated ledger
fields for each candidate review row.

## Why this won

It makes the operator-facing report sufficient for manual review without adding
new validation paths or contacting hardware.

## Rejected alternatives

JSON-only evidence is harder to review and can hide a missing or unsafe prestart
trace from the operator report. Recomputing the contract in the renderer would
duplicate ledger logic.

## Implementation plan

- Render the bound prestart summary path, SHA256, type, status, connection
  policy, no-contact flags, and check status in the decision review report.
- Add regression assertions for the Markdown output.
- Register the report as WING MVP evidence.

## Test plan

- Focused decision review CLI test.
- no-WING/Project Director/WING rail tests.
- no-WING quick release check.
- Full pytest before commit.

## Risks and rollback

The change is presentation-only. Rollback would remove the extra Markdown rows
and return to JSON-only review evidence.

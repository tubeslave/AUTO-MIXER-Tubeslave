# ADR: WING decision review prestart boolean rendering

## Context

The WING read-only decision review Markdown report displays candidate prestart
contract fields for operator review. Missing candidate evidence must remain
visibly missing and must not be rendered as a safe boolean value.

## Options considered

1. Keep rendering booleans with `value is True`.
2. Render candidate prestart booleans as tri-state values.
3. Remove incomplete candidate rows from the Markdown report.

## Decision

Use option 2. The report renders `true` and `false` only when the underlying
candidate value is actually boolean; otherwise it renders an empty inline value.

## Why this won

It preserves real false safety evidence while avoiding misleading output for
missing fields. It also keeps validation in the ledger instead of duplicating it
in the Markdown renderer.

## Rejected alternatives

Binary rendering can make absent no-contact fields look confirmed. Removing
incomplete rows would hide the reason the candidate remains blocked.

## Implementation plan

- Add a small `_markdown_bool()` helper.
- Use it for candidate prestart contract boolean fields.
- Add focused regression assertions for valid and incomplete rows.
- Register the report as WING MVP evidence.

## Test plan

- Focused decision review CLI test.
- Project Director/WING rail tests.
- no-WING quick release check.
- Full pytest before commit.

## Risks and rollback

The change is presentation-only. Rollback would restore binary rendering, but
that would again risk making missing evidence look safe.

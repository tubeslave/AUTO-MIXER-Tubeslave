# Brief: WING decision review prestart boolean rendering

## Context

The WING decision review Markdown report is used by the operator to inspect the
accepted-candidate prestart trace before supervised proof planning. Boolean
fields in that candidate table must not collapse missing or malformed values
into safe-looking `false`.

## Task

Render candidate prestart boolean fields as tri-state values in Markdown:
`true` and `false` only for actual booleans, and empty for missing or
non-boolean values.

## Acceptance

- Valid candidate rows still show actual `true` and `false` values.
- Blocked or incomplete candidate rows do not show missing no-contact fields as
  confirmed `false`.
- Ledger validation remains unchanged.
- No backend, WING, OSC, or live-write behavior is added.

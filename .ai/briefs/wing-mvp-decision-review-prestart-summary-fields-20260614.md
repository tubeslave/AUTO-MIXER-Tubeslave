# Brief: WING decision review prestart summary fields

## Context

The WING read-only decision review report is the operator-facing gate before a
reviewed preflight candidate can support supervised proof planning. The ledger
already carries accepted-candidate prestart summary fields, but the Markdown
report did not show the bound no-WING prestart summary path, hash, status, and
no-contact flags.

## Task

Expose the accepted-candidate `backend_port_prestart_summary_*` fields in the
decision review Markdown report.

## Acceptance

- Candidate prestart contract output shows the bound no-WING prestart summary
  path and SHA256.
- Candidate prestart contract output shows summary type, status, connection
  policy, no-contact flags, and prestart check status.
- The report remains offline/read-only and does not grant live authority.
- Tests cover the Markdown output.

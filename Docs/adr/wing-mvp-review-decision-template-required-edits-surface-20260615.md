# ADR: WING MVP Review Decision Template Required-Edits Surface

## Context

The ledger rejects accepted decision records that still carry the generated
pending `template_status` marker. The generated review report and CLI help did
not yet list that edit, leaving a stale operator-facing path.

## Options Considered

1. Leave the generated review report unchanged and rely on the runbook.
2. Add required template edits to the generated review report and CLI help.
3. Define a new accepted template status.

## Decision

Use option 2. Generated review reports expose the required edits, including
removing or replacing the pending template status marker.

## Why This Won

The physical review workflow depends on generated artifacts. The report should
not omit a condition that the ledger enforces.

## Rejected Alternatives

Runbook-only guidance was rejected because it can drift from the generated
hardware-session report. A new accepted status was rejected as unnecessary for
this fail-closed surface.

## Implementation Plan

- Add a required-edits list in `tools/review_wing_readonly_preflight_artifacts.py`.
- Render it in JSON and markdown.
- Update CLI help and tests.

## Test Plan

- Run targeted review tool tests.
- Run focused WING/no-WING tests.
- Run full standard pytest.
- Run no-WING quick, memory lint, Project Director, and `git diff --check`.

## Risks And Rollback

Risk is limited to report wording. Rollback is removing the required-edits
surface, but that would again hide a ledger-enforced operator edit.

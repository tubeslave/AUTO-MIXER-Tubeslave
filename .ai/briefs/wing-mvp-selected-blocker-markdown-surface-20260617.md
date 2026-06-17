# Brief: WING MVP Selected Blocker Markdown Surface

## Task ID

wing-mvp-selected-blocker-markdown-surface-20260617

## Goal

Render the selected WING MVP blocker and blocker status in the no-WING
`summary.md` top-level operator-facing blocker section.

## Context

The no-WING `summary.json` already mirrors the selected WING blocker from the
handoff packet. Operators also review `summary.md`, and the selected blocker
should be visible before the large nested handoff packet section.

## Scope

- Add a compact `WING Read-Only MVP Blocker Summary` markdown section.
- Include selected blocker, blocker status, anchor match, blocker-not-closed,
  readiness denial, and live authority denial.
- Keep the change artifact-only and read-only.
- Do not contact WING hardware or change live write policy.

## Validation

- Compile touched Python files.
- Run focused no-WING/WING guard tests.
- Generate a fresh no-WING quick artifact and inspect `summary.md`.
- Run artifact manifest verifier, memory lint, and full pytest before commit.

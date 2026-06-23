# ADR: WING MVP Operator Sequence List Markdown Surface

## Context

The no-WING JSON summary exposes and self-checks top-level WING read-only
handoff operator sequence lists. The operator-readable `summary.md` must also
render those fields, because the runbook asks operators to inspect no-WING
artifacts before opening a backend/WING session.

## Options considered

1. Rely on JSON-only checks.
2. Make WING-specific fields globally required in every artifact manifest.
3. Add a conditional Markdown-surface check for WING handoff summaries.

## Decision

Add a conditional fail-closed Markdown-surface check for WING handoff operator
sequence list fields.

## Why this won

The check protects the operator-readable review path without forcing unrelated
no-WING summaries to carry WING handoff fields. It runs only when the WING
handoff artifact is present and keeps readiness/live authority denied.

## Rejected alternatives

JSON-only checks leave the human review surface unguarded. Global manifest
requirements are too broad because not every no-WING summary contains a WING
handoff packet.

## Implementation plan

- Define required Markdown field lines for the promoted sequence lists and their
  summary-surface self-check verdicts.
- Add an apply function that records missing lines and fails the summary when
  any are absent.
- Call the check after the Markdown text is assembled.
- Update focused tests, blocker docs, runbook docs, generated-progress source,
  index, and validation report.

## Test plan

- Focused py_compile.
- Focused no-WING release and Project Director tests.
- Direct missing-lines regression test.
- no-WING quick summary check.
- Standard full pytest command.
- Memory lint and diff whitespace check.

## Risks and rollback

This is an offline Markdown validation change. If it blocks valid no-WING runs,
rollback by removing the conditional check, call site, tests, and docs while
keeping the JSON list self-check intact.

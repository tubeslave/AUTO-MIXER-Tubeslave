# ADR: WING direction guard skips generated guard reports

## Context

The Project Director depends on `codex_direction_guard.py` to keep Goal-mode
work focused on the supervised WING Rack MVP. Generated direction guard reports
are derived diagnostics; rescanning them as fresh evidence can create false
mixed signals from prior summaries.

## Options considered

1. Continue scanning generated guard reports.
2. Skip generated guard reports when scoring evidence.
3. Try to parse every generated report and remove nested diagnostic sections.

## Decision

Use option 2. Generated direction guard reports are excluded from evidence
scoring.

## Why this won

It removes a self-referential evidence source while preserving real WING proof
signals from git status, git history, and primary reports.

## Rejected alternatives

Scanning generated guard reports treats diagnostics as work. More parsing is
fragile because recent-report snippets can be truncated.

## Implementation plan

- Reuse generated-report detection in `analyze_evidence`.
- Add regression tests for generated Markdown and JSON guard reports.
- Confirm Project Director returns to `on_target`.
- Register the report as WING MVP evidence.

## Test plan

- Focused direction guard tests.
- Project Director/WING rail tests.
- no-WING quick release check.
- Full pytest before commit.

## Risks and rollback

The guard will no longer count its own generated reports as target evidence.
Rollback would restore the previous self-referential scoring behavior.

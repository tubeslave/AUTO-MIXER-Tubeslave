# ADR: cycle-report-worktree-evidence-20260606

## Context

Autonomous Codex cycle reports are used as durable Automixer memory. The cycle
prompt requires gathering `git status --short` and the current branch before
planning. The report validator already required broad worktree and branch
evidence, but a report could still pass with generic `git status` wording or an
empty branch line.

## Options Considered

1. Keep broad discovery evidence checks.
2. Require exact `git status --short` evidence and a non-empty branch line.
3. Require full embedded `git status --short` output in every report.

## Decision

Use option 2. The validator now requires the exact `git status --short` command
text in `## What Was Found` and requires current-branch evidence with a
non-empty value.

## Why This Won

It aligns the report artifact with the prompt while staying small and stable.
It improves audit quality without forcing reports to include a full status dump
when a summarized modified/new file list is enough.

## Rejected Alternatives

Broad checks were too weak for the current prompt. Full status-output parsing
was rejected because the worktree may contain unrelated user changes and the
report should summarize them without creating a brittle transcript format.

## Implementation Plan

1. Add a `## What Was Found` validator for exact `git status --short` command
   evidence.
2. Add a branch-line validator requiring a non-empty value.
3. Add regression tests for both failures.
4. Document the stricter contract in the autoloop README.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_codex_cycle_report_validator.py -q`
- `python3 tools/memory/memory_lint.py`
- `python3 tools/codex_autoloop/validate_cycle_report.py .paperclip/reports/<current-cycle-report>.md`

## Risks And Rollback

Risk: old or hand-written reports may fail until they include exact worktree
command evidence. Rollback is removing the new helper and tests, but the safer
fix is updating reports to include the command and branch value.

This ADR does not grant live OSC/WING write authority.

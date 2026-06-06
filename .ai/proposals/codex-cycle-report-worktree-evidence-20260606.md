# Codex Proposal: cycle-report-worktree-evidence-20260606

## Short Thesis

Make the report validator enforce the same worktree evidence the prompt asks an
autonomous cycle to gather: exact `git status --short` command text and a
non-empty current branch.

## What I Understood

Automixer reports are durable project memory. If worktree evidence is loose, a
future report can look valid while omitting the exact command or branch state
needed to reconstruct what the agent saw.

## Proposed Solution

Add a focused `## What Was Found` validator for exact `git status --short`
evidence and a branch line with a value. Add regression tests and document the
contract in `tools/codex_autoloop/README.md`.

## Likely Files To Touch

- `tools/codex_autoloop/validate_cycle_report.py`
- `tests/test_codex_cycle_report_validator.py`
- `tools/codex_autoloop/README.md`
- `Docs/adr/cycle-report-worktree-evidence-20260606.md`

## Alternatives Considered

- Keep broad `git status` and branch phrases only: rejected because it allows a
  weaker report than the prompt requires.
- Add a full parser for status output: rejected because reports can summarize
  changed/new files without embedding every status line.

## Risks

Older reports that do not name `git status --short` will fail the stricter
validator. The fix is to include the exact command evidence in future reports.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_codex_cycle_report_validator.py -q`
- `python3 tools/memory/memory_lint.py`
- Validate this cycle report with `tools/codex_autoloop/validate_cycle_report.py`.

## Where The Other Agent May Disagree

Kimi may prefer validating exact modified/untracked lists instead. That is
heavier and less stable across dirty parallel worktrees; this slice only locks
the discovery command and branch value.

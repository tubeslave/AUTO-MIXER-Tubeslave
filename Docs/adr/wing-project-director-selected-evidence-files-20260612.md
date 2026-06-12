# ADR: WING Project Director Selected Evidence Files

Date: 2026-06-12

## Context

The supervised WING Rack MVP is gated by the first open blocker in
`.ai/project-layer/wing_mvp_blockers.md`. That blocker already names the
read-only preflight tools, contracts, proof-plan logic, and historical reports
that future cycles should inspect. Project Director previously exposed a
hand-maintained `required_files` list but did not preserve the blocker-row
evidence as structured directive data.

## Options considered

1. Keep only the hand-maintained required-file list.
2. Promote every evidence report into `required_files`.
3. Expose the selected blocker evidence as its own field and derive only
   source/runbook required files from it.

## Decision

Use option 3. Project Director now emits `selected_evidence_files` and
`selected_evidence_file_count`, renders them in Markdown, and derives additional
required-file candidates only for `backend/`, `tools/`, and `Docs/runbooks/`
paths.

## Why this won

The blocker table remains the source of truth for the current proof surface.
Future cycles can see the full evidence chain without turning old reports into
mandatory source files. This reduces drift while preserving a clear separation
between runnable contract files and historical review context.

## Rejected alternatives

- Manual-only required files: too easy to miss newly added proof tools.
- Required historical reports: too noisy and blurs proof evidence with files
  that need active inspection or modification.

## Implementation plan

- Parse backtick-delimited file paths from the selected blocker evidence cell.
- Preserve deterministic order and remove duplicates.
- Add selected evidence fields to JSON and Markdown output.
- Cover parser/filter behavior and selected-blocker output in tests.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_codex_direction_guard.py`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

This change is repo-local and non-live. If the parser produces noisy output,
rollback is limited to the Project Director helper functions, tests, and
documentation added in this ADR. No WING connection, backend session, or live
write authority is affected.

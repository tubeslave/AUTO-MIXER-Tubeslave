# Proposal: WING Project Director Selected Evidence Files

Date: 2026-06-12

## Short thesis

Expose the first open blocker evidence files in Project Director output so
future Goal-mode cycles start from the authoritative proof surface already
recorded in `.ai/project-layer/wing_mvp_blockers.md`.

## What I understood

The current MVP remains supervised WING Rack proof. The first open blocker is
read-only WING connection proof. Offline/director work may improve handoff and
review tooling, but it must not claim live readiness or widen writes.

## Proposed solution

- Parse backtick-delimited file paths from the selected blocker `Evidence`
  column.
- Add `selected_evidence_files` and `selected_evidence_file_count` to Project
  Director JSON.
- Render the same evidence list in Project Director Markdown.
- Derive extra required-file candidates only from source/runbook evidence
  prefixes: `backend/`, `tools/`, and `Docs/runbooks/`.
- Keep historical `Docs/reports/` entries as evidence, not mandatory source
  files.

## Likely files to touch

- `tools/codex_autoloop/project_director_agent.py`
- `tests/test_project_director_agent.py`
- `Docs/INDEX.md`
- `Docs/adr/wing-project-director-selected-evidence-files-20260612.md`
- `Docs/reports/AUTOMIXER_WING_PROJECT_DIRECTOR_SELECTED_EVIDENCE_FILES_20260612.md`

## Alternatives considered

- Keep appending every newly missed file to the hard-coded required list. This
  preserves current behavior but allows drift from the blocker table.
- Add every report to `required_files`. This would be too noisy and would make
  historical reports look like source files that must be edited or reviewed for
  every cycle.

## Risks

- A malformed blocker evidence cell could produce a noisy evidence list.
- Over-filtering could hide useful reports from the required-files list.

## Test plan

- Focused Project Director unit tests.
- WING MVP rails tests.
- Direction guard JSON check.
- `git diff --check`.
- Memory lint.
- Full pytest before commit.

## Where the other agent may disagree

Kimi may argue for making reports required too. I would keep reports as
selected evidence because they are useful review context but not runtime or
contract files that every cycle must edit.

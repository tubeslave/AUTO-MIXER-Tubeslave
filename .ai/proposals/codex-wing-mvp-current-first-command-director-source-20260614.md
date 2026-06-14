# Codex Proposal: WING MVP Current First Command Director Source

## Short thesis

Use the explicit `## Current First Command` section as the selected-action
source for Project Director and progress reporting.

## What I understood

The WING MVP still must stay blocker-first on the read-only connection proof.
The latest detailed operator command is maintained in a dedicated markdown
section, but the Director currently selects only the first blocker row's
`Next action`. Future Goal-mode cycles can therefore miss newer diagnostic
requirements that were added to the current command section.

## Proposed solution

Add a small markdown-section reader to `update_mvp_progress.py`, expose it as
`current_first_command`, and use it in both the progress report and Project
Director. The blocker table remains the source for selected blocker status and
evidence files. When the current command section is present, selected action
text starts there and appends the table-row baseline so older safety constraints
are not dropped.

## Likely files to touch

- `tools/codex_autoloop/update_mvp_progress.py`
- `tools/codex_autoloop/project_director_agent.py`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `Docs/reports/AUTOMIXER_WING_MVP_CURRENT_FIRST_COMMAND_DIRECTOR_SOURCE_20260614.md`
- `Docs/INDEX.md`

## Alternatives considered

Keep duplicating every new command field into the first blocker row. Rejected
because the row is already large and hard to review.

Generate the selected action entirely from structured packet metadata. Rejected
for this slice because it is broader than the immediate routing bug.

## Risks

The helper is a lightweight markdown parser, and the merged action can duplicate
operator text. The mitigation is to keep parsing scoped to an exact heading and
add rail tests that fail if the section disappears, if Project Director drops
the current command diagnostics, or if the baseline safety markers disappear.

## Test plan

Run focused rail tests, regenerate the latest MVP report, run Project Director
JSON output, run the direction guard, run `git diff --check`, and run memory
lint.

## Where Kimi may disagree

Kimi may prefer removing the free-form command section and replacing it with a
structured contract. That may be better later, but this patch is intentionally a
small routing fix that keeps the current docs contract intact.

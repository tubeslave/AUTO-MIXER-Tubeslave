# Codex Proposal: WING MVP Progress Director Evidence Summary

## Short thesis

Make the generated MVP progress report show the same baseline coverage and
selected-evidence facts that Project Director uses to keep Goal-mode work on the
first WING blocker.

## What I understood

The project is still pursuing a real WING read-only connection proof. The next
offline improvement is not new live authority; it is making the operator-facing
status report auditable enough to show whether Current First Command still
covers the required no-WING handoff baseline.

## Proposed solution

Move evidence-path extraction into `tools/codex_autoloop/update_mvp_progress.py`
and have Project Director reuse it. Then make `build_progress_report` include:

- `Current First Command baseline coverage required`
- `Current First Command baseline coverage passed`
- `Current First Command baseline coverage marker count`
- `Current First Command baseline missing markers`
- `Selected evidence file count`

Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md` and add focused rail tests for
the new fields.

## Likely files to touch

- `tools/codex_autoloop/update_mvp_progress.py`
- `tools/codex_autoloop/project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- Council trail and report/index files for this task.

## Alternatives considered

- Leave these facts only in Project Director JSON.
- Duplicate evidence extraction in both tools.

## Risks

The progress report is already long because the current first command is long.
Adding compact summary fields at least keeps the audit-critical values visible
near the top.

## Test plan

Run focused WING rails/Director tests, Project Director JSON, direction guard
JSON, memory lint, diff check, no-WING quick check, and full pytest.

## Where the other agent may disagree

Kimi may prefer a separate machine-readable progress JSON. I prefer improving
the existing report first because it is already the operator-facing artifact
indexed by the WING rails.

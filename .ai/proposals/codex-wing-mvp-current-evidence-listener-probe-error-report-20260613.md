# Codex Proposal: WING MVP Current Evidence Listener Probe Error Report

## Short thesis

Add the latest listener-probe error field report to the current first blocker
evidence list so Project Director selects it as part of the active WING proof
context.

## What I understood

The `listener_process_error` report is already validated and indexed. The
remaining gap is visibility: Project Director reads report context from
`.ai/project-layer/wing_mvp_blockers.md`, not by inferring from `Docs/INDEX.md`.

## Proposed solution

- Append
  `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_LISTENER_PROBE_ERROR_FIELD_20260613.md`
  to the first blocker evidence cell.
- Add focused tests for blocker-table parsing and Project Director evidence
  exposure.
- Add an ADR and report documenting that this is an evidence visibility change,
  not a readiness closure.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `tests/test_wing_mvp_rails.py`
- `tests/test_project_director_agent.py`
- `.ai/briefs/...`
- `.ai/proposals/...`
- `Docs/adr/...`
- `Docs/reports/...`
- `Docs/INDEX.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`

## Alternatives considered

- Leave evidence unchanged and rely on `Docs/INDEX.md`.
- Teach Project Director to infer latest reports from index categories.

## Risks

The main risk is making the blocker look more complete than it is. The wording
must keep the first blocker open and still require real WING read-only proof.

## Test plan

- Focused WING rails, Project Director, and Direction Guard tests.
- Project Director JSON check.
- Direction Guard JSON check.
- `git diff --check`.
- `tools/memory/memory_lint.py`.
- Full backend pytest.

## Where the other agent may disagree

Kimi may prefer index inference to avoid growing a long blocker-table evidence
cell. I would keep evidence explicit because it is the current Project Director
source of truth and avoids hidden selection logic.

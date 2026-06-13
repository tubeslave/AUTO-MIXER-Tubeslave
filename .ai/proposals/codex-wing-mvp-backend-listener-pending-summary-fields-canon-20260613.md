# Proposal: WING MVP Backend Listener Pending Summary Fields Canon

## Short thesis

Make the Project Director rail require the pending listener-template summary
fields that the no-WING gate now emits.

## What I understood

The active blocker is still WING read-only connection proof. Runtime evidence
now shows generated listener templates are pending through explicit
`accepted_by_template: false` and placeholder-field markers. The canonical
blocker action should require those markers before any real WING preflight.

## Proposed solution

- Add the pending summary fields report to first-blocker evidence.
- Add `listener_confirmation_template_accepted_by_template: false` and
  `listener_confirmation_template_operator_placeholder_fields` to the first
  blocker action.
- Regenerate the latest MVP progress report.
- Update MVP/autoloop docs and tests that inspect Project Director output.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`
- `tests/test_wing_mvp_rails.py`
- `tests/test_project_director_agent.py`

## Alternatives considered

Relying on the runbook alone leaves Project Director stale. Adding more runtime
code is unnecessary because the no-WING summary already emits the fields.

## Risks

The first-blocker action becomes slightly longer, but this follows the existing
explicit WING prestart contract style.

## Test plan

Run focused rail/director tests, Project Director/direction guard, memory lint,
and the standard full pytest command before commit.

## Where the other agent may disagree

Kimi may prefer a structured contract file instead of the long blocker row.
That can be done later; this patch keeps the current source-of-truth flow
consistent.

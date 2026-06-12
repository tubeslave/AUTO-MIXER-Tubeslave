# Codex Proposal: wing-autoloop-backend-prestart-director-docs-20260612

## Short thesis

Automation-facing docs and tests should state both prestart gates: the first
backend-port gate and the final supervised-proof gate. Direction guard should
also ignore Project Director JSON policy fields when scoring drift.

## What I understood

Project Director already reads the selected action from
`.ai/project-layer/wing_mvp_blockers.md`, so its directive logic can stay
unchanged. The stale surface is the autoloop README and its tests, which only
mention the final supervised gate. During validation, Project Director JSON also
showed a false positive in direction guard because `forbidden_scope` contains
drift keywords as negative policy.

## Proposed solution

- Update `tools/codex_autoloop/README.md` to document
  `backend_port_prestart_gate == expected_backend_port_prestart_gate`.
- Extend `tests/test_project_director_agent.py` so selected actions, prompt,
  and README all carry the backend gate and live-authority boundary.
- Teach `tools/codex_autoloop/codex_direction_guard.py` to remove JSON policy
  fields such as `forbidden_scope` before drift scoring.
- Add regression coverage for Project Director JSON reports.

## Likely files to touch

- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_direction_guard.py`
- `tests/test_project_director_agent.py`
- `tests/test_codex_direction_guard.py`
- `Docs/adr/wing-autoloop-backend-prestart-director-docs-20260612.md`
- `Docs/reports/AUTOMIXER_WING_AUTOLOOP_BACKEND_PRESTART_DIRECTOR_DOCS_20260612.md`

## Alternatives considered

- Change Project Director code to inject the gate. Rejected: the source of
  truth is the blocker table, and the table already contains the gate.
- Leave README stale. Rejected: autoloop operators read it as the automation
  behavior contract.
- Ignore all Project Director reports in direction guard. Rejected: selected
  WING work in those reports is useful target evidence.

## Risks

Documentation could be mistaken for live proof. Mitigation: explicitly state no
WING contact, no live authority, and unchanged blocker status.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py tests/test_codex_direction_guard.py`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `/opt/homebrew/bin/git diff --check`

## Where the other agent may disagree

Kimi may prefer not touching docs-only surfaces. I would still update them
because autoloop README is the operator contract for recurring autonomous
cycles.

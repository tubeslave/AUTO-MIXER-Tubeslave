# ADR: WING MVP Backend Prestart Command Text

## Context

The WING read-only MVP rails require the failed busy-port no-WING summary and
no-WING handoff summary to show the backend prestart command exactly matching
`PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`.

The existing contract stores the command as `backend_port_prestart_check_command`
with argv-list semantics. That is good for structured comparison, but an
exact-string mirror makes operator review and downstream summary checks less
ambiguous.

## Options considered

1. Keep only the argv-list command field.
2. Replace the list with a string.
3. Add `backend_port_prestart_check_command_text` beside the existing list.

## Decision

Add `backend_port_prestart_check_command_text` as a contract-derived text mirror
while preserving the existing argv-list field.

## Why this won

It supports exact operator-facing command matching without breaking existing
structured validators. It is a report/artifact contract change only and does not
start backend, contact WING, write console state, or widen live authority.

## Rejected alternatives

Keeping only the list leaves the exact-string rail dependent on renderer
behavior. Replacing the list would be a compatibility break across existing
validators and fixtures.

## Implementation plan

- Add a shared command text constant.
- Propagate the text field through runner summaries, handoff packets, no-WING
  summaries, evidence intake, review decisions, readiness overlay, and proof
  plan summaries.
- Extend focused tests and rail docs.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.

## Test plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_autonomy_promotion_readiness.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_wing_readonly_preflight_runner.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_supervised_wing_proof_plan.py tests/test_supervised_proof_artifact_report.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk is limited to report-contract drift. Rollback removes the text field,
reverts the docs/tests, and regenerates the latest progress report. No live WING
control or readiness closure is introduced.

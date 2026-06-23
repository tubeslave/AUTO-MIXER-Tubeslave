# ADR: WING MVP Expected Artifacts Summary JSON Surface

Date: 2026-06-15

## Context

The WING read-only handoff packet already carries expected artifact globs and
plan-match verdicts. The aggregate no-WING `summary.json` did not expose those
fields at top level, even though the Project Director first command requires
them before a supervised proof can be planned.

## Options considered

1. Keep expected-artifact fields nested inside the handoff artifact.
2. Add a dedicated top-level expected-artifacts summary surface.
3. Fold the fields into an existing command-binding or packet-safety surface.

## Decision

Add a dedicated top-level expected-artifacts summary surface.

## Why this won

The dedicated surface keeps command-binding and safety-boundary surfaces focused
while making artifact-target contract evidence directly auditable from the
aggregate artifact.

## Rejected alternatives

Nested-only evidence keeps the aggregate summary incomplete. Folding artifact
targets into command binding or safety boundaries mixes separate contract
concerns.

## Implementation plan

- Add `build_wing_readonly_expected_artifacts_summary_surface()`.
- Merge the surface into `summary.json` when the handoff packet artifact is
  present.
- Assert top-level fields match the nested handoff artifact in the focused
  summary test.
- Add Council report and index the evidence.

## Test plan

- `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_wing_preflight_handoff_packet_artifact`
- `PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend`
- `PYTHONPATH=backend python tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `PYTHONPATH=backend python tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: readers could mistake expected-artifact completeness for readiness.
Mitigation: the surface explicitly denies readiness closure and live authority.
Rollback is to remove the helper, summary update, test assertions, and Council
evidence.

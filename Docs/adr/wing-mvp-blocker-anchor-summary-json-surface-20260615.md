# ADR: WING MVP Blocker Anchor Summary JSON Surface

Date: 2026-06-15

## Context

The WING read-only handoff packet already carries an MVP blocker anchor proving
which first open blocker the packet is tied to. The aggregate no-WING
`summary.json` left that anchor nested, while the Project Director current
first command requires directly auditable anchor and digest fields.

## Options considered

1. Keep blocker-anchor fields nested inside the handoff artifact.
2. Add a dedicated top-level blocker-anchor summary surface.
3. Add a broad handoff-artifact projection layer.

## Decision

Add a dedicated top-level blocker-anchor summary surface.

## Why this won

The dedicated surface is small, mirrors existing handoff values, and makes the
first-blocker identity and digest check directly auditable without changing
live WING authority.

## Rejected alternatives

Nested-only evidence keeps the aggregate summary incomplete. A broad projection
layer is larger than this current first-command gap requires.

## Implementation plan

- Add `build_wing_readonly_mvp_blocker_anchor_summary_surface()`.
- Merge the surface into `summary.json` when the handoff packet artifact is
  present.
- Assert top-level values match the nested handoff artifact in the focused
  no-WING summary test.
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

Risk: readers could mistake the projected anchor for blocker closure.
Mitigation: the surface preserves `packet_does_not_close_blocker: true` and
explicitly denies readiness closure and live authority. Rollback is to remove
the helper, summary update, test assertions, and Council evidence.

# ADR: WING MVP Operator Sequence Step Gate Summary JSON Surface

Date: 2026-06-15

## Context

The WING read-only handoff packet artifact already validates the operator
sequence anchor digest review step, supervised prestart gate step, required
markers, and gate match verdicts. The aggregated no-WING `summary.json` still
left those top-level fields empty, forcing machine consumers to parse nested
handoff data.

## Options considered

1. Keep only nested packet artifact data.
2. Add a dedicated top-level operator-sequence step-gate summary surface.
3. Add a generic nested-artifact projection framework.

## Decision

Add a dedicated top-level operator-sequence step-gate summary surface in
`tools/no_wing_release_check.py`.

## Why this won

The dedicated helper matches the existing no-WING summary-surface pattern,
limits the diff, and exposes the current first-command evidence without
changing WING proof authority.

## Rejected alternatives

Keeping only nested data leaves automation brittle. A generic projection
framework would be larger than this current reporting gap requires.

## Implementation plan

- Add `build_wing_readonly_operator_sequence_step_gate_summary_surface()`.
- Merge the surface into `summary.json` when the handoff packet artifact is
  present.
- Assert the top-level fields match the handoff artifact in the focused no-WING
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

Risk: readers could mistake the top-level step-gate surface for live authority.
Mitigation: the new surface explicitly sets readiness closure and live
authority to `false`. Rollback is to remove the helper, summary update, test
assertions, and Council evidence.

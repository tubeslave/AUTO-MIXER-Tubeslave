# ADR: WING MVP Next Proof Plan Candidate Contracts Summary JSON Surface

Date: 2026-06-15

## Context

The no-WING `next_wing_proof_plan` object already validates accepted-candidate
prestart and artifact-root contract gates before physical proof steps. The
aggregated `summary.json` still left those contract verdicts nested, which made
machine consumers parse the proof-plan object directly.

## Options considered

1. Keep only nested proof-plan object data.
2. Add a dedicated top-level candidate-contract summary surface.
3. Add a generic proof-plan projection framework.

## Decision

Add a dedicated top-level candidate-contract summary surface in
`tools/no_wing_release_check.py`.

## Why this won

The dedicated helper matches the existing no-WING summary-surface pattern,
limits the diff, and exposes the current first-command evidence without
changing WING proof authority.

## Rejected alternatives

Keeping only nested data leaves automation brittle. A generic projection
framework would be larger than this current reporting gap requires.

## Implementation plan

- Add `build_next_wing_proof_plan_candidate_contracts_summary_surface()`.
- Merge the surface into `summary.json` when `next_wing_proof_plan` is present.
- Assert the top-level fields match the nested proof-plan object in the focused
  no-WING summary test.
- Add Council report and index the evidence.

## Test plan

- `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_next_wing_proof_plan`
- `PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend`
- `PYTHONPATH=backend python tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `PYTHONPATH=backend python tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: readers could mistake the top-level candidate-contract surface for live
authority. Mitigation: the new surface explicitly sets readiness closure and
live authority to `false`. Rollback is to remove the helper, summary update,
test assertions, and Council evidence.

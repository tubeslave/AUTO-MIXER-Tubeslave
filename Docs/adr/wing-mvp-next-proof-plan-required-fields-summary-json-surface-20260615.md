# ADR: WING MVP Next Proof Plan Required Fields Summary JSON Surface

Date: 2026-06-15

## Context

The no-WING next proof plan already computes required supervised summary fields
and the corresponding missing-field list. The aggregate `summary.json` exposed
candidate contract fields at top level, but the missing required-field list
remained nested under `next_wing_proof_plan`.

## Options considered

1. Keep required/missing field evidence nested in `next_wing_proof_plan`.
2. Add a dedicated top-level next-proof-plan required-fields summary surface.
3. Fold required/missing fields into the broader candidate-contracts surface.

## Decision

Add a dedicated top-level next-proof-plan required-fields summary surface.

## Why this won

The dedicated surface keeps the candidate-contract surface focused while making
the fail-closed required-field result directly auditable from the aggregate
artifact.

## Rejected alternatives

Nested-only evidence keeps the aggregate summary incomplete for first-command
checks. Folding this into candidate contracts mixes two related but distinct
proof-plan concerns.

## Implementation plan

- Add `build_next_wing_proof_plan_required_fields_summary_surface()`.
- Merge the surface into `summary.json` when `next_wing_proof_plan` is present.
- Assert top-level fields match the nested next plan in the focused summary
  test.
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

Risk: readers could mistake required-field completeness for readiness.
Mitigation: the surface explicitly denies readiness closure and live authority.
Rollback is to remove the helper, summary update, test assertions, and Council
evidence.

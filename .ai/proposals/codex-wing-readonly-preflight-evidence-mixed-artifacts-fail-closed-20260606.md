# Codex Proposal: WING Read-Only Preflight Evidence Mixed Artifacts Fail Closed

## Short thesis

Treat the WING read-only preflight artifact set atomically for readiness-review
purposes: one invalid artifact blocks the update candidate until the operator
replaces or removes it.

## What I understood

The project focus is the supervised WING Rack MVP. This slice should reduce the
read-only connection, physical readback, operator-visible evidence, and
readiness-matrix blockers without widening live authority.

## Proposed solution

- Keep validating each `wing_readonly_preflight.v1` artifact individually.
- Compute the global readiness-review candidate only when there is at least one
  valid candidate and zero invalid records.
- Mark mixed valid+invalid sets as
  `readonly_preflight_artifact_replacement_required`.
- Keep all matrix rows `closed=false` and manual-review-required.
- Update release-check validation so `accepted_as_readonly_preflight_candidate`
  and `readiness_matrix_update_allowed` require a clean evidence set.

## Likely files to touch

- `backend/wing_readonly_preflight_evidence.py`
- `tests/test_wing_readonly_preflight_evidence.py`
- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_EVIDENCE_FAIL_CLOSED_20260606.md`

## Alternatives considered

- Accept the newest valid artifact while reporting invalid ones separately. This
  is too easy to misread as readiness-update permission.
- Delete invalid artifacts automatically. This would be destructive and would
  hide evidence the operator may need to inspect.

## Risks

- Operators may need to clean artifact roots more often before getting a
  readiness-review candidate. That is acceptable for hardware proof safety.
- This still does not close hardware proof blockers; it only prevents ambiguous
  evidence from being treated as clean.

## Test plan

- `python3 -m py_compile backend/autonomy_promotion_readiness.py backend/wing_readonly_preflight_evidence.py backend/processing_registry.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_no_wing_release_check.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Where the other agent may disagree

Kimi might prefer selecting the newest valid artifact by timestamp. For this
MVP blocker, fail-closed artifact hygiene is safer than timestamp arbitration.

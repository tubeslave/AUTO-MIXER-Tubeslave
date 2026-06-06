# ADR: WING Read-Only Preflight Evidence

## Context

The supervised WING Rack MVP needs proof for read-only connection, physical
readback, operator-visible evidence, and readiness matrix updates before any
supervised fader/gain apply proof. The read-only preflight runner now records
`wing_readonly_preflight.v1` summaries, but those summaries need a fail-closed
consumer.

## Options considered

1. Feed read-only preflight summaries into supervised proof evaluation.
2. Add a dedicated read-only evidence evaluator.
3. Leave the preflight artifact as an operator-only JSON file.

## Decision

Use option 2.

## Why this won

Read-only preflight proves a different part of the WING loop than supervised
apply. A dedicated evaluator can accept physical readback candidates while
keeping apply, rollback, emergency-stop, and live-authority proof separate.

## Rejected alternatives

- Supervised proof evaluation is for fader/gain apply and must continue to
  require apply/readback/rollback/emergency evidence.
- Standalone JSON is insufficient because it does not provide a fail-closed
  readiness matrix or invalid-artifact reasons.

## Implementation plan

- Add `backend/wing_readonly_preflight_evidence.py`.
- Add offline tests for missing, valid, invalid, and file-discovered artifacts.
- Register the evaluator as `shadow_only` in the processing registry.
- Add it to the console-read project-goal support path.
- Add a `tools/no_wing_release_check.py` schema gate and summary section so the
  release report exposes the candidate without converting it into live
  authority.

## Test plan

- `python3 -m py_compile tools/no_wing_release_check.py backend/wing_readonly_preflight_evidence.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_processing_registry.py::test_registry_snapshot_can_hide_full_method_inventory tests/test_project_goal_capability_matrix.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: a valid preflight candidate may be mistaken for completed supervised
proof. The evaluator prevents that by keeping every MVP blocker row
`closed=false`, requiring manual review, and granting no live authority.
Rollback is a normal git revert.

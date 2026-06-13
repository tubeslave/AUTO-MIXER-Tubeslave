# Automixer WING MVP Read-Only Artifact Contract Constants

Date: 2026-06-13
Status: Validated

## Summary

WING read-only proof artifact identity now has one shared contract source for:

- `wing_readonly_preflight.v1`
- `summary.json`
- `events.jsonl`
- `wing_readonly_preflight.v1 summary.json`
- `real wing_readonly_preflight.v1 summary.json`

## Why

The first WING MVP blocker must only advance from real read-only proof
artifacts. Keeping artifact names and schema IDs in one shared contract reduces
drift between the runner, handoff packet, release check, evidence intake, and
supervised proof plan.

## Changed surfaces

- `backend/wing_readonly_preflight_contract.py` defines the shared artifact
  identity constants.
- `tools/wing_readonly_preflight_runner.py` uses them for plan and run summary
  artifacts.
- `tools/build_wing_readonly_preflight_handoff_packet.py` uses them for
  expected artifact globs and next manual artifact.
- `tools/no_wing_release_check.py` uses them for WING preflight plan and
  handoff packet validation.
- `backend/wing_readonly_preflight_evidence.py` uses the shared schema and
  summary artifact name for evidence discovery.
- `backend/supervised_wing_proof_plan.py` uses the shared proof summary and
  event artifact names in the prestart gate.
- Focused tests assert the shared contract values.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_evidence.py
  tests/test_wing_readonly_preflight_runner.py
  tests/test_wing_readonly_preflight_review_tool.py
  tests/test_wing_readonly_preflight_handoff_packet.py
  tests/test_no_wing_release_check.py tests/test_supervised_wing_proof_plan.py
  tests/test_project_director_agent.py tests/test_codex_direction_guard.py`: 316
  passed in 3.82s.
- Project Director JSON check selected `WING read-only connection proof`,
  reported `guard_status: on_target`, and `guard_stop_the_line: false`.
- Direction guard JSON check reported `status: on_target` and
  `stop_the_line: false`.
- `/opt/homebrew/bin/git diff --check`: passed.
- `python3 tools/memory/memory_lint.py`: PASS.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: 3205 passed, 1 skipped, 1 warning in 246.82s (0:04:06).

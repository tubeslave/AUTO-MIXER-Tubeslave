# Automixer WING MVP Backend Listener Confirmation Artifact

Date: 2026-06-13
Status: Validated

## Summary

The no-WING quick gate now supports an optional operator confirmation JSON for a
busy backend port. If `127.0.0.1:8765` already has a listener, the gate still
fails closed by default. It only passes when the operator supplies a
`wing_backend_port_listener_confirmation.v1` artifact that confirms the listener
is the intended backend and matches the current PID and command line.

## Safety Boundary

- No WING hardware contact.
- No backend/WING listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No live authority granted.
- No MVP blocker closed from this local confirmation.

## Contract

Accepted confirmation requires:

- `type: wing_backend_port_listener_confirmation.v1`
- `status: operator_confirmed_intended_backend`
- matching `host` and `port`
- `confirms_intended_backend: true`
- matching `reviewed_listener_pid`
- matching `reviewed_listener_command_line`
- non-empty `operator_reviewed_at`
- non-empty `operator_name`
- `safe_to_stop_listener_by_gate: false`
- `live_authority_granted: false`

## Changed surfaces

- `tools/no_wing_release_check.py` validates
  `--backend-listener-confirmation`.
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md` documents the confirmation JSON
  and rerun command.
- `.ai/project-layer/wing_mvp_blockers.md` cites this report in first-blocker
  evidence.
- `tests/test_no_wing_release_check.py` covers accepted, mismatched, summary,
  and `run_release_checks` propagation paths.
- `tests/test_wing_mvp_rails.py` and `tests/test_project_director_agent.py`
  assert the report remains visible.

## Validation

- `/opt/homebrew/bin/python3 -m py_compile tools/no_wing_release_check.py`:
  passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_no_wing_release_check.py -k "backend_listener_confirmation or
  confirmed_backend_listener or confirmed_intended_listener or
  mismatched_listener_confirmation or passes_backend_listener_confirmation"`:
  4 passed, 204 deselected.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_no_wing_release_check.py`: 208 passed in 3.35s.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_mvp_rails.py tests/test_project_director_agent.py
  tests/test_codex_direction_guard.py`: 23 passed in 0.15s.
- Project Director JSON check selected `WING read-only connection proof`,
  reported `guard_status: on_target`, `guard_stop_the_line: false`, and
  exposed this report in `selected_evidence_files` with
  `selected_evidence_file_count: 28`.
- Direction guard JSON check reported `status: on_target` and
  `stop_the_line: false`.
- `/opt/homebrew/bin/git diff --check`: passed.
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py`: PASS.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: 3212 passed, 1 skipped, 1 warning in 209.52s (0:03:29).

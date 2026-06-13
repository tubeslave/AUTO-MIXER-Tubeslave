# Automixer WING MVP Read-Only Prestart Summary Binding

Date: 2026-06-13
Status: Validated

## Summary

The WING read-only preflight runner now requires a passed no-WING prestart
`summary.json` before planning or opening the backend WebSocket. The runner
records that prestart summary path, SHA256, type, status, policy, live-write
state, and `backend_port_free` status in both the offline plan and captured
read-only artifact.

## Safety Boundary

- No WING hardware contact in this implementation.
- No backend/WING listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No live authority granted.
- No MVP blocker closed from this tooling change.

## Contract

Accepted read-only candidates must now include:

- `backend_port_prestart_summary_required: true`
- `backend_port_prestart_summary_path`
- `backend_port_prestart_summary_sha256`
- `backend_port_prestart_summary_type: no_wing_release_check`
- `backend_port_prestart_summary_status: passed`
- `backend_port_prestart_summary_connection_policy:
  no_hardware_no_network_no_live_writes`
- `backend_port_prestart_summary_live_write: false`
- `backend_port_prestart_check_status: passed`

Evidence review loads the referenced prestart file and rejects the candidate if
the file is missing, hash-mismatched, unsafe, or not passed.

## Changed surfaces

- `tools/wing_readonly_preflight_runner.py` adds `--prestart-summary`.
- `backend/wing_readonly_preflight_evidence.py` verifies the referenced
  prestart file and hash.
- `backend/wing_readonly_preflight_review_decision.py` and
  `backend/supervised_wing_proof_plan.py` require the binding before
  `accepted_decision_ready` can unlock supervised proof planning.
- `backend/autonomy_promotion_readiness.py` surfaces the bound prestart summary
  fields from valid read-only candidates without closing readiness rows.
- `tools/review_wing_readonly_preflight_artifacts.py` renders the prestart
  identity for operator review.
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md` documents the new command flag.

## Validation

- `/opt/homebrew/bin/python3 -m py_compile tools/wing_readonly_preflight_runner.py
  backend/wing_readonly_preflight_evidence.py
  backend/wing_readonly_preflight_review_decision.py
  backend/supervised_wing_proof_plan.py
  tools/review_wing_readonly_preflight_artifacts.py`: passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_runner.py
  tests/test_wing_readonly_preflight_evidence.py
  tests/test_wing_readonly_preflight_review_tool.py
  tests/test_wing_readonly_preflight_review_decision.py
  tests/test_supervised_wing_proof_plan.py
  tests/test_wing_mvp_rails.py tests/test_project_director_agent.py
  tests/test_codex_direction_guard.py -x --tb=short`: 130 passed.
- Project Director JSON selected `WING read-only connection proof`, reported
  `guard_status: on_target`, `guard_stop_the_line: false`, and exposed this
  report in `selected_evidence_files` with `selected_evidence_file_count: 29`.
- Direction guard JSON reported `status: on_target` and
  `stop_the_line: false`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3
  tools/no_wing_release_check.py --quick --skip-frontend --artifact-root
  /tmp/automixer_goal_prestart_binding_nowing_20260613`: failed closed before
  WING contact on busy `127.0.0.1:8765` with
  `backend_port_listener_requires_operator_confirmation` and
  `safe_to_stop_listener_by_gate: false`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_runner.py
  tests/test_wing_readonly_preflight_evidence.py
  tests/test_wing_readonly_preflight_review_tool.py
  tests/test_wing_readonly_preflight_review_decision.py
  tests/test_wing_readonly_preflight_review_decision_tool.py
  tests/test_supervised_wing_proof_plan.py tests/test_wing_mvp_rails.py
  tests/test_autonomy_promotion_readiness.py tests/test_project_director_agent.py
  tests/test_codex_direction_guard.py -x --tb=short`: 356 passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: 3217 passed, 1 skipped, 1 warning in 161.42s (0:02:41).
- `/opt/homebrew/bin/git diff --check`: passed.
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py`: PASS.

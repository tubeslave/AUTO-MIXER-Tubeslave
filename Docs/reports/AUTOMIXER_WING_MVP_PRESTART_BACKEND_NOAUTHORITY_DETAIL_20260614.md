# Automixer WING MVP Prestart Backend No-Authority Detail - 2026-06-14

## Summary

The no-WING backend-port prestart result now carries explicit no-authority
detail flags, and the WING read-only preflight runner rejects prestart summaries
where those flags are missing or not `false`.

This moves the first WING blocker forward by making the offline prestart input
to a real read-only WING run stricter. It does not contact WING, grant live
authority, or close the blocker.

## Changed

- `backend/wing_readonly_preflight_contract.py` defines the backend-port detail
  fields that must be explicitly `false`.
- `tools/no_wing_release_check.py` emits those fields for the backend-port
  prestart check and restores preserved generated command artifacts.
- `tools/wing_readonly_preflight_runner.py` fails closed before WebSocket when a
  prestart summary lacks the required no-authority backend-port detail fields.
- `tests/test_wing_readonly_preflight_runner.py` covers stale/minimal prestart
  summaries that omit those fields.
- `tests/test_no_wing_release_check.py` covers the free-port no-authority detail
  and preserved frontend E2E `.last-run.json` behavior.

## Safety Boundary

- `read_only: true` remains the only allowed preflight posture.
- `live_write: false` remains required.
- `live_authority_granted: false` must now be explicit in backend-port detail.
- Backend prestart gate execution, readiness closure, and live authority verdicts
  must all be explicit `false`.
- Offline validation still does not prove a real WING read-only connection.

## Validation

- Focused changed-surface tests:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_runner.py
  tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short
  -x`: 319 passed in 3.80s.
- no-WING quick with frontend:
  `PYTHONPATH=backend /opt/homebrew/bin/python3
  tools/no_wing_release_check.py --quick --continue-on-failure
  --command-timeout-sec 120 --artifact-root
  /tmp/automixer_goal_frontend_preserve_nowing`: exited 1 fail-closed with
  `connection_policy: no_hardware_no_network_no_live_writes`,
  `live_write: false`, and failed checks `backend_port_free`,
  `wing_readonly_preflight_plan`, and
  `wing_readonly_preflight_handoff_packet_artifact_schema`.
- In that no-WING summary, `backend_port_free` reported
  `safe_to_stop_listener_by_gate: false`, `live_authority_granted: false`,
  `backend_port_prestart_gate_execution_allowed: false`,
  `backend_port_prestart_gate_readiness_closure_allowed: false`, and
  `backend_port_prestart_gate_live_authority_granted: false`.
- The same no-WING summary reported `frontend_e2e: passed`, and
  `git status --short -- frontend/test-results/.last-run.json` was clean after
  the run.
- Project Director selected `WING read-only connection proof`,
  `selected_blocker_status: tooling_ready_waits_real_wing`,
  `selected_action_source: Current First Command`, `guard_status: on_target`,
  `guard_score: 90`, and `selected_evidence_file_count: 100`.
- Direction guard reported `status: on_target`, `score: 90`,
  `stop_the_line: false`.
- `/opt/homebrew/bin/git diff --check`: passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`:
  PASS.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile
  backend/wing_readonly_preflight_contract.py
  tools/no_wing_release_check.py tools/wing_readonly_preflight_runner.py`:
  passed.
- Full suite:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: 3314 passed, 1 skipped, 1 warning in 168.52s.
- Standard full test suite:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: 3306 passed, 9 skipped, 1 warning in 71.88s.

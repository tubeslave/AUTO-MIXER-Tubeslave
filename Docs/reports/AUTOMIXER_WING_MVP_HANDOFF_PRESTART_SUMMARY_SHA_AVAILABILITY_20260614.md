# Automixer WING MVP Handoff Prestart Summary SHA Availability

## Summary

The no-WING handoff validator now fails closed if
`backend_port_prestart_summary_sha256_available` contradicts the packet's
prestart summary SHA256 field.

## What changed

- `tools/no_wing_release_check.py` computes digest availability from
  `backend_port_prestart_summary_sha256`.
- no-WING handoff validation adds
  `wing_readonly_preflight_handoff_packet_prestart_summary_sha256_available_mismatch`
  when the flag and digest disagree.
- no-WING summary JSON/Markdown now show
  `backend_port_prestart_summary_sha256_available` and
  `backend_port_prestart_summary_sha256_available_matches_digest`.

## Safety boundary

This remains offline and read-only. It does not contact WING, does not contact
the backend, does not mutate console state, and does not grant live authority.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short`
  - `266 passed in 3.14s`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile tools/no_wing_release_check.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --backend-port 65535 --artifact-root /tmp/automixer_goal_handoff_sha_availability_20260614`
  - `summary status: passed`
  - `handoff status: passed`
  - `sha available: True`
  - `availability matches digest: True`
  - `sha matches plan: True`
  - `violations: []`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json --output .paperclip/reports/codex_direction_guard_handoff_sha_availability_20260614.md`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --output .paperclip/reports/project_director_handoff_sha_availability_20260614.md --print`
  - `guard_status: on_target`
  - `selected_blocker: WING read-only connection proof`
  - `selected_blocker_status: tooling_ready_waits_real_wing`
  - `selected_evidence_file_count: 60`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
  - `Status: PASS`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  - `3293 passed, 1 skipped, 1 warning in 165.26s`

## Remaining blocker

The WING read-only connection proof still requires a real
`wing_readonly_preflight.v1` summary and `events.jsonl` from the intended WING
Rack session.

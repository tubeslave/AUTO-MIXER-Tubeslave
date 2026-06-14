# Automixer WING MVP Handoff Prestart Summary SHA Binding

## Summary

The WING read-only handoff packet now binds the no-WING backend prestart
summary by SHA256 when the summary file exists.

## What changed

- `tools/build_wing_readonly_preflight_handoff_packet.py` records
  `backend_port_prestart_summary_sha256`.
- Placeholder packets keep `backend_port_prestart_summary_sha256_available:
  false`.
- no-WING handoff artifact validation fails closed when packet SHA and plan SHA
  differ.
- no-WING summary Markdown renders packet SHA, plan SHA, and
  `backend_port_prestart_summary_sha256_matches_plan`.

## Safety boundary

This remains offline and read-only. It does not contact WING, does not query the
backend, does not mutate console state, and does not grant live authority.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile tools/build_wing_readonly_preflight_handoff_packet.py tools/no_wing_release_check.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --backend-port 65535 --artifact-root /tmp/automixer_goal_handoff_sha_20260614`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Remaining blocker

The WING read-only connection proof still requires a real
`wing_readonly_preflight.v1` summary and `events.jsonl` captured from the
intended WING Rack session.

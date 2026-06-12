# Automixer WING Handoff Generator Prestart Regression

## Summary

This change adds direct unit coverage that the WING read-only handoff packet
generator emits the supervised prestart gate in both JSON operator sequence and
rendered Markdown.

The generated handoff must mention:

- `accepted_decision_ready`
- `valid accepted-candidate prestart trace`
- `WING_SUPERVISED_FADER_GAIN_PROOF`

## Safety Boundary

- No backend process was started by this change.
- No WING transport was contacted.
- No OSC/WebSocket writes were added.
- This is test coverage for existing handoff behavior.

## Verification

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py -k "handoff_packet"` - PASS, 15 passed.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py tests/test_supervised_wing_proof_plan.py tests/test_wing_mvp_rails.py tests/test_project_director_agent.py` - PASS, 222 passed.
- `/opt/homebrew/bin/git diff --check` - PASS.
- `python3 tools/memory/memory_lint.py` - PASS.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` - PASS, 3187 passed, 1 skipped, 1 warning.

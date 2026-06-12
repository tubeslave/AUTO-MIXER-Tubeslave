# Automixer WING Handoff Prestart Gate Validator

## Summary

This change makes the no-WING release validator reject WING read-only handoff
packets whose operator sequence moves from `accepted_decision_ready` to
`WING_SUPERVISED_FADER_GAIN_PROOF` without requiring a valid
accepted-candidate prestart trace.

The validator now reports:

- `operator_sequence_has_supervised_prestart_gate`
- `required_supervised_prestart_gate_markers`

## Safety Boundary

- No backend process was started by this change.
- No WING transport was contacted.
- No OSC/WebSocket writes were added.
- This is stricter offline validation only.

## Verification

- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py -k "handoff_packet"` - PASS, 5 passed.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_supervised_wing_proof_plan.py tests/test_processing_registry.py tests/test_wing_mvp_rails.py tests/test_project_director_agent.py` - PASS, 257 passed.
- `/opt/homebrew/bin/git diff --check` - PASS.
- `python3 tools/memory/memory_lint.py` - PASS.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` - PASS, 3187 passed, 1 skipped, 1 warning.

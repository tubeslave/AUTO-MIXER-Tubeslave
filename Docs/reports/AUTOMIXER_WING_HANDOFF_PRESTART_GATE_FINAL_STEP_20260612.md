# Automixer WING Handoff Prestart Gate Final-Step Validation

## Summary

This change requires the WING handoff packet supervised prestart gate to be the
final operator sequence step. Release validation now fails if a packet contains
the correct gated transition and then appends a later stale instruction toward
`WING_SUPERVISED_FADER_GAIN_PROOF`.

The release summary now exposes:

- `operator_sequence_supervised_prestart_gate_step_index`
- `operator_sequence_supervised_prestart_gate_is_final`

## Safety Boundary

- No backend process was started by this change.
- No WING transport was contacted.
- No OSC/WebSocket writes were added.
- This is stricter offline validation and reporting only.

## Verification

- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py -k "handoff_packet"` - PASS, 7 passed.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_supervised_wing_proof_plan.py tests/test_wing_mvp_rails.py tests/test_project_director_agent.py` - PASS, 224 passed.
- `/opt/homebrew/bin/git diff --check` - PASS.
- `python3 tools/memory/memory_lint.py` - PASS.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` - PASS, 3189 passed, 1 skipped, 1 warning.

# Automixer WING Handoff Prestart Gate Same-Step Validation

## Summary

This change tightens the no-WING release validator for WING read-only handoff
packets. The validator now requires one operator sequence step to contain all
supervised proof transition markers:

- `accepted_decision_ready`
- `valid accepted-candidate prestart trace`
- `WING_SUPERVISED_FADER_GAIN_PROOF`

The release summary now also renders
`operator_sequence_supervised_prestart_gate_step` so reviewers can inspect the
exact accepted transition sentence.

## Safety Boundary

- No backend process was started by this change.
- No WING transport was contacted.
- No OSC/WebSocket writes were added.
- This is stricter offline validation and reporting only.

## Verification

- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py -k "handoff_packet"` - PASS, 6 passed.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_supervised_wing_proof_plan.py tests/test_wing_mvp_rails.py tests/test_project_director_agent.py` - PASS, 223 passed.
- `/opt/homebrew/bin/git diff --check` - PASS.
- `python3 tools/memory/memory_lint.py` - PASS.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` - PASS, 3188 passed, 1 skipped, 1 warning.

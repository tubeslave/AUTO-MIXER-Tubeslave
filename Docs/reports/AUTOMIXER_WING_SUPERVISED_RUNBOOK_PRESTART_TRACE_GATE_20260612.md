# Automixer WING Supervised Runbook Prestart Trace Gate

## Summary

This change aligns the operator-facing supervised fader/gain proof runbook and
read-only handoff packet with the accepted-candidate prestart trace gate.
Operators should not move from `accepted_decision_ready` to
`WING_SUPERVISED_FADER_GAIN_PROOF` unless accepted review source ids have a
valid accepted-candidate prestart trace.

The supervised proof runbook now fails closed when the proof plan carries
`accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof`.

## Safety Boundary

- No backend process was started by this change.
- No WING transport was contacted.
- No OSC/WebSocket writes were added.
- Supervised proof remains limited to `fader` and negative `gain` deltas.

## Verification

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py tests/test_no_wing_release_check.py -k "runbooks_keep_readonly or handoff"` - PASS, 8 passed.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py tests/test_no_wing_release_check.py tests/test_supervised_wing_proof_plan.py tests/test_processing_registry.py tests/test_project_director_agent.py` - PASS, 246 passed.
- `/opt/homebrew/bin/git diff --check` - PASS.
- `python3 tools/memory/memory_lint.py` - PASS.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` - PASS, 3186 passed, 1 skipped, 1 warning.

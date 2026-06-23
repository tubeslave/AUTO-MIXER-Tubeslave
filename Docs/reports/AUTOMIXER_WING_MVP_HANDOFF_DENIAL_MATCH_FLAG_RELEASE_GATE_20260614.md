# Automixer WING MVP Handoff Denial Match Flag Release Gate

Date: 2026-06-14
Status: Validated

## Summary

The no-WING release check now rejects WING read-only handoff packets whose
declared backend-port prestart denial `*_matches_plan` fields drift from the
recomputed packet-vs-plan truth.

## Changed Surfaces

- `tools/no_wing_release_check.py` compares declared and computed denial match
  flags for:
  - `safe_to_stop_listener_by_gate`
  - `live_authority_granted`
  - `backend_port_prestart_gate_execution_allowed`
  - `backend_port_prestart_gate_readiness_closure_allowed`
  - `backend_port_prestart_gate_live_authority_granted`
- The no-WING summary artifact and markdown now expose
  `declared_<field>_matches_plan` next to computed `<field>_matches_plan`.
- `tests/test_no_wing_release_check.py` covers a stale handoff packet whose
  declared flags are false while the recomputed values remain true.

## Safety Boundary

- No WING hardware contact.
- No OSC/WING writes.
- No backend or frontend server lifecycle changes.
- No readiness closure.
- No live authority granted.

## Validation

- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py --tb=short -x`
  passed: 270 tests, 8 skipped.
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short -x`
  passed: 301 tests, 8 skipped.
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
  passed: 3319 tests, 9 skipped.

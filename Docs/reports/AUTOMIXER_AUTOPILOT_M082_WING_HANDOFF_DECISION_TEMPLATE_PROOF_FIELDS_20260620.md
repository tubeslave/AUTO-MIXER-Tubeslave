# M082: WING handoff decision template proof fields

Generated: 2026-06-20T06:48:25Z

## Decision

M082 keeps the project on the WING physical-readback path without pretending the
external validation blocker is solved.

The offline handoff packet now exposes the review-decision template field
contract that M081 made mandatory:

- `decision_template_required_fields`
- `expected_decision_template_required_fields`
- `decision_template_prestart_proof_fields`
- `expected_decision_template_prestart_proof_fields`
- contract-match and preserve-before-operator-edit booleans

The runbook now tells the operator to preserve those generated proof fields
while editing the pending decision template.

## Safety Boundary

- No backend WebSocket was opened.
- No WING or OSC transport was contacted.
- No write, proposal apply, rollback, emergency-stop or readiness update was
  sent.
- No hardware evidence was created.
- `live_authority_granted` remains `false`.
- `readiness_closure_allowed` remains `false`.

## Validation Commands

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_readonly_preflight_handoff_packet.py -q`:
  `21 passed`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m082_wing_handoff_surfaces_decision_template_proof_fields -q`:
  `1 passed`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_no_wing_release_check.py -q`:
  `448 passed`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`:
  `95 passed`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`:
  `4210 passed, 1 skipped, 1 warning`
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py --root .`: `PASS`
- JSON validation for `ops/autopilot/STATE.json`: passed
- JSONL validation for `ops/autopilot/RUN_LEDGER.jsonl`: passed

## Next Required External Step

M001 and M002 remain `WAITING_FOR_EXTERNAL_VALIDATION`. The next non-simulated
step is still a human-supervised WING read-only preflight that produces a real
artifact under `artifacts/wing_readonly_preflight`, followed by artifact review
and decision review.

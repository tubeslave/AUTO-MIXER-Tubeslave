# Brief: WING Read-Only Preflight Prestart Detail Fields Contract

## Task

Keep the WING read-only preflight prestart contract aligned with the busy-port evidence now emitted by `no_wing_release_check.py`.

## Context

The no-WING prestart gate now records explicit busy-port confirmation fields, but `backend/wing_readonly_preflight_contract.py` still listed only `listener_processes` and `operator_next_step` as required failure detail fields. Runner plans, handoff packets, evidence review, and release checks should all use the same shared contract.

## Constraints

- Do not contact WING.
- Do not probe or stop a local backend listener.
- Do not widen live-write authority.
- Do not close any MVP blocker from offline evidence.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision_tool.py`
- Full backend suite before commit.

# M081 - WING Review Decision Prestart Proof Binding

Status: implemented locally; awaiting validation

## Goal

Bind every generated or accepted WING read-only preflight review decision to the
same backend-port prestart proof and artifact-manifest verifier ordering that
made the candidate clean.

## Context

M001 remains externally blocked on a real WING read-only artifact and operator
review. The existing evidence pipeline already validates `summary.json`,
`events.jsonl`, expected-plan binding, prestart summary hash and manifest
verifier ordering before a candidate becomes available. The remaining gap was
the manual review decision record: the generated decision template carried the
candidate identity and readback counts, but did not carry the prestart/verifier
proof fields that the operator decision should be bound to.

## Checkpoints

- M081-CP01: Shared contract lists the prestart/verifier proof fields required
  in review decision templates.
- M081-CP02: Review decision template generation copies those fields from the
  clean candidate and refuses mutated prestart/verifier rows.
- M081-CP03: Review decision ledger rejects accepted records whose prestart or
  verifier proof binding is missing or stale against the candidate.

## Files

- `backend/wing_readonly_preflight_contract.py`
- `tools/review_wing_readonly_preflight_artifacts.py`
- `backend/wing_readonly_preflight_review_decision.py`
- `tests/test_wing_readonly_preflight_review_tool.py`
- `tests/test_wing_readonly_preflight_review_decision.py`
- `tests/test_wing_mvp_rails.py`

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m081_wing_review_decision_binds_prestart_proof -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Safety

This milestone is offline review-contract hardening only. It does not contact
WING, open a backend WebSocket, send OSC, apply proposals, close readiness,
grant live authority or replace the required real hardware/operator validation.

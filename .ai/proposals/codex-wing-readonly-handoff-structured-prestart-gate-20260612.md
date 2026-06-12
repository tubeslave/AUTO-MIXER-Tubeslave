# Codex Proposal: WING Read-Only Handoff Structured Prestart Gate

## Short thesis

The first backend-port prestart gate should be represented as a structured
actual/expected contract, not only as individual fields.

## What I understood

The WING read-only handoff packet already has a structured
`supervised_prestart_gate` and `expected_supervised_prestart_gate`. The earlier
first-step hardening added individual backend-port prestart fields, but did not
make the first gate comparable as a complete object.

## Proposed solution

- Add `backend_port_prestart_gate`.
- Add `expected_backend_port_prestart_gate`.
- Validate both against the operator sequence and fail closed on mismatch.
- Preserve the objects in no-WING summary JSON and Markdown.
- Update tests and the runbook.

## Likely files to touch

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`

## Alternatives considered

- Keep individual fields only. Rejected because object-level equality catches
  drift between the declared gate and the derived expected gate.
- Add the gate only to Markdown. Rejected because the no-WING validator needs a
  machine-readable contract.

## Risks

Older handoff packets may fail validation until regenerated. That is acceptable
for a safety gate before real hardware work.

## Test plan

- Focused handoff/no-WING tests.
- Runbook rails tests.
- no-WING quick gate.
- Full repository pytest before commit.

## Where the other agent may disagree

Kimi may say this duplicates the individual prestart fields. The useful
distinction is that the full object can be compared to an expected contract and
surfaces drift more clearly in operator evidence.

# ADR: WING handoff decision template proof fields

## Context

M081 made WING read-only preflight review decisions stricter: generated pending
decision templates and accepted decision records must carry backend-port
prestart and artifact-manifest verifier proof fields from the clean candidate.
Without a handoff-visible list of those fields, the next real WING operator
session could capture a valid artifact but then save an accepted decision that
silently drops or edits required proof fields.

M001 and M002 still require external WING/operator validation. This ADR does not
attempt to simulate that validation or close readiness from offline evidence.

## Options Considered

1. Leave the handoff packet unchanged and rely on the ledger to reject bad
   operator edits.
2. Add the decision-template proof-field contract to the offline handoff packet
   and runbook.
3. Relax M081 ledger checks until real hardware evidence exists.

## Decision

Add machine-readable handoff fields for the full review-decision template
required field list and for the prestart/verifier proof-field subset that an
operator must preserve while editing the generated decision template.

## Why This Won

The real operator handoff should expose the exact contract that the ledger will
enforce later. This keeps the next M002 intake path actionable without adding
live authority or weakening accepted-decision validation.

## Rejected Alternatives

- Relying only on ledger rejection keeps the system safe, but it makes the real
  operator run more fragile and pushes a predictable failure into the hardware
  session.
- Relaxing the ledger would erase the M081 safety binding and allow stale
  prestart or verifier proof to support a decision.

## Implementation Plan

- Publish `decision_template_required_fields` and
  `decision_template_prestart_proof_fields` in the handoff packet.
- Render those fields in handoff Markdown beside the pending decision template
  path.
- Update the WING read-only preflight runbook to tell the operator to preserve
  those fields while editing the pending template.
- Add a M082 rail that proves the new handoff surface remains offline and does
  not change readiness.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_readonly_preflight_handoff_packet.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m082_wing_handoff_surfaces_decision_template_proof_fields -q`
- Standard backend suite.
- Memory lint and JSON/JSONL validation.

## Risks and Rollback

Risk: the handoff packet becomes more verbose. This is acceptable because the
field list is machine-readable and tied to the same contract used by the ledger.

Rollback: remove the new handoff fields, Markdown lines, runbook note and M082
state/report updates. This does not affect WING/backend transport because the
change is offline only.

No live transport, no OSC write, no proposal apply, no readiness closure and no
live authority are introduced by this decision.

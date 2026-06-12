# Codex Proposal: WING Read-Only Prestart Detail Fail-Closed Condition

## Short thesis

Make stale backend-port prestart detail contracts explicit in the read-only
preflight expected-plan fail-closed list.

## What I understood

The WING read-only preflight path now requires
`listener_confirmation_checklist` in backend-port prestart failure details. The
runner blocks stale plans, but the offline plan still listed that risk only
under the broad `expected_plan_missing_or_mismatch` condition.

## Proposed solution

- Add `stale_backend_port_prestart_detail_contract` to runner offline plans.
- Add the same condition to evidence-layer required fail-closed conditions.
- Update the read-only preflight runbook.
- Add focused assertions for runner/evidence/no-WING behavior.

## Likely files to touch

- `tools/wing_readonly_preflight_runner.py`
- `backend/wing_readonly_preflight_evidence.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tests/test_wing_readonly_preflight_runner.py`
- `tests/test_wing_readonly_preflight_evidence.py`

## Alternatives considered

- Keep the broad expected-plan mismatch only. Rejected because operator-facing
  plans should name the stale prestart-detail contract explicitly.
- Add another runtime branch. Rejected because existing expected-plan mismatch
  validation already blocks stale plans before WebSocket.

## Risks

Low. This expands a report-only/offline fail-closed list and its validation.

## Test plan

- Runner/evidence/no-WING focused tests.
- Adjacent WING/no-WING suite.
- Full repo validation before commit.

## Where the other agent may disagree

Kimi may prefer fewer fail-closed strings. I would keep the explicit condition
because this is the current prestart proof boundary before hardware contact.

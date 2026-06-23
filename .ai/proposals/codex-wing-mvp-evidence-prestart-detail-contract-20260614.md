# Codex Proposal: WING MVP Evidence Prestart Detail Contract

## Short thesis

Align post-run evidence intake with the real read-only runner by rejecting
captured artifacts when their referenced no-WING prestart summary has stale or
unsafe backend-port detail fields.

## What I understood

The project is still blocked on a real read-only WING connection proof. Offline
review must be at least as strict as the runner, because accepted review output
is the bridge toward supervised proof planning. A stale no-WING prestart detail
contract must not survive artifact review.

## Proposed solution

- Export the required-false prestart detail field set from
  `backend/wing_readonly_preflight_evidence.py`.
- During prestart summary file validation, compare
  `backend_port_prestart_check_failure_detail_fields` to the shared contract.
- Require every required-false backend-port field to be exactly `false`.
- Compare recorded summary values against the referenced prestart detail values.
- Add focused tests for stale detail-list and unsafe required-false detail.

## Likely files to touch

- `backend/wing_readonly_preflight_evidence.py`
- `tests/test_wing_readonly_preflight_evidence.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- Council trail, report, index, blocker evidence, and latest progress report.

## Alternatives considered

- Trust the runner's pre-WebSocket gate. Rejected because post-run review should
  not accept evidence if the referenced prestart file is later stale or edited.
- Only compare recorded summary fields. Rejected because the referenced file is
  authoritative for the no-WING prestart boundary.

## Risks

Older synthetic or hand-authored prestart summaries without the current detail
contract will fail review. That is intended; they must be regenerated with the
current no-WING release gate.

## Test plan

- Focused evidence pytest.
- Focused Project Director/WING rails pytest.
- memory lint and `git diff --check`.
- Full pytest when practical.

## Where the other agent may disagree

Kimi may argue this duplicates runner validation. I prefer duplication here
because runner and reviewer protect different phases of the same hardware proof.

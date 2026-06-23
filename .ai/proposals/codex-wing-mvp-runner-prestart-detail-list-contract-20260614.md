# Codex Proposal: WING MVP Runner Prestart Detail List Contract

## Short thesis

Reject stale no-WING prestart summaries in the real read-only runner when their
backend-port failure-detail field list no longer matches the shared contract.

## What I understood

The next real step is still a WING read-only connection proof. Offline plan and
handoff work must reduce the chance that a stale local summary reaches the real
backend/WING session. The runner currently validates required false detail
values, but not the exact detail-field list that declares the expected failure
contract.

## Proposed solution

- In `load_backend_port_prestart_summary`, read
  `backend_port_prestart_check_failure_detail_fields` from
  `backend_port_free.detail`.
- Compare it to
  `WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS`.
- Add a specific prestart violation and operator next step when the list is
  missing or stale.
- Update runner tests and runbook wording.

## Likely files to touch

- `tools/wing_readonly_preflight_runner.py`
- `tests/test_wing_readonly_preflight_runner.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- Council trail and report/index files.

## Alternatives considered

- Rely on the expected offline plan mismatch check. Rejected because the stale
  prestart summary should block before expected-plan validation and before any
  WebSocket attempt.
- Only document the requirement. Rejected because the real runner can enforce it
  cheaply and deterministically.

## Risks

Older synthetic no-WING summaries without the field list will fail. That is
intended; they should be regenerated from the current no-WING gate before real
WING work.

## Test plan

- Focused runner pytest.
- Project Director JSON inspection.
- no-WING quick smoke if needed.
- memory lint and `git diff --check`.
- Standard full pytest when practical.

## Where the other agent may disagree

Kimi may argue that expected-plan validation already covers this. I would keep
the runner check because the prestart summary is a separate input to the real
run and should be rejected on its own.

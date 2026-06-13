# Proposal: WING MVP Backend Prestart Listener Probe Error Field

## Short thesis

Require `listener_process_error` in the backend-port prestart failure-detail
contract so an owner-probe failure is visible before any real WING read-only
preflight session.

## What I understood

The no-WING gate already fails closed when the backend port is busy. It records
listener process details when available and records `listener_process_error`
when owner discovery fails. The handoff contract listed other failure fields but
did not require this diagnostic field, allowing packet summaries to omit a key
reason why the operator could not identify the listener.

## Proposed solution

- Add `listener_process_error` to
  `WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS`.
- Update the first blocker next action, MVP doc, read-only runbook, autoloop
  prompt, autoloop README, and generated progress.
- Add focused tests that assert the field is present in Project Director,
  blocker rails, runbook text, shared constants, and busy-port probe behavior.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- Focused no-WING/WING MVP tests and generated progress/report docs.

## Alternatives considered

- Keep `listener_process_error` as an optional detail only. Rejected because
  preflight reviewers need to know when owner detection failed.
- Add a new separate owner-probe contract. Rejected as unnecessary because this
  is part of the existing backend-port prestart failure details.

## Risks

- The exact-list change must be carried through all docs and tests or future
  cycles will see contradictory rails.
- This must not imply permission to stop a listener automatically.

## Test plan

- Run focused no-WING release, handoff packet, WING MVP rails, Project
  Director, and direction guard tests.
- Run Project Director and direction guard command checks.
- Run `git diff --check`, memory lint, and full backend pytest.

## Where the other agent may disagree

Kimi may consider the field optional because it appears only when probing
fails. The safety case is stronger if the contract explicitly allows and
surfaces that failure path.

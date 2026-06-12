# Proposal: WING Read-Only Preflight Prestart Detail Fields Contract

## Short thesis

The shared WING preflight contract should require every busy-port detail field that the no-WING gate now emits.

## What I understood

The first WING MVP blocker still waits for real read-only hardware proof. Recent offline work improved busy-port diagnostics. The shared contract now needs to catch up so expected plans and handoff packets advertise the complete failure evidence set.

## Proposed solution

- Expand `WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS`.
- Replace hardcoded test fixture lists with the shared contract constant.
- Keep behavior offline and non-mutating.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `tests/test_no_wing_release_check.py`
- `Docs/adr/wing-readonly-preflight-prestart-detail-fields-contract-20260612.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_PRESTART_DETAIL_FIELDS_CONTRACT_20260612.md`

## Alternatives considered

- Leave the contract narrow: rejected because plans and handoff packets would understate the required busy-port evidence.
- Duplicate the field list in tools: rejected because this is exactly what the shared contract avoids.

## Risks

Older generated expected-plan and handoff artifacts may fail validation until regenerated. This is acceptable because the stricter contract reflects current fail-closed evidence.

## Test plan

- Focused no-WING/WING preflight tests.
- Full backend suite before commit.

## Where the other agent may disagree

Kimi may argue that command-line diagnostics should also be required as a top-level failure detail. That field is conditional on OS permissions, so the contract requires the stable confirmation flags and keeps command-line diagnostics opportunistic.

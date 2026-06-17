# Codex Proposal: WING MVP Read-Only Event Log Run-Id Fail-Closed Contract

## 1. Short thesis

Add `event_log_run_id_mismatch` to the shared WING read-only preflight
fail-closed contract so plan, handoff, no-WING validation, and runner output all
name the same row-level event-log identity failure.

## 2. What I understood

The validator already rejects real preflight artifacts when `events.jsonl` rows
do not match `summary.json` `run_id`. The operator-facing plan still exposes a
broader event-log failure list, which makes the packet less explicit than the
validator.

## 3. Proposed solution

Update `WING_READONLY_PREFLIGHT_PLAN_FAIL_CLOSED_CONDITIONS` in the shared
contract and add focused assertions that the new marker is visible through the
runner summary, handoff markdown, and no-WING plan artifact.

## 4. Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `tests/test_wing_readonly_preflight_runner.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/reports/AUTOMIXER_WING_MVP_READONLY_EVENT_LOG_RUN_ID_FAIL_CLOSED_CONTRACT_20260614.md`

## 5. Alternatives considered

- Keep only `event_log_missing_or_invalid`: rejected because it hides the
  specific row-level identity condition from operator-facing packets.
- Add a separate validator-only violation: rejected because the shared contract
  already exists and is consumed by all relevant surfaces.

## 6. Risks

Any fixture that hard-codes the full fail-closed list must be updated. This is
acceptable because explicit mismatch visibility is the point of the change.

## 7. Test plan

Run focused WING preflight runner, handoff, evidence/review, and no-WING tests,
then standard pytest, no-WING quick runtime, `git diff --check`, memory lint,
and Project Director.

## 8. Where the other agent may disagree

Kimi may argue the broader `event_log_missing_or_invalid` condition is enough.
The counterpoint is that real operator packets should name the exact failure
that blocks manual review when an event log belongs to another run.

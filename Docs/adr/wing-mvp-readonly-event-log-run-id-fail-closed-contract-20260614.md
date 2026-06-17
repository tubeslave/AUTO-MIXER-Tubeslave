# ADR: WING MVP Read-Only Event Log Run-Id Fail-Closed Contract

## 1. Context

Real WING read-only preflight evidence now requires each `events.jsonl` row to
carry the same `run_id` as `summary.json`. A mismatch blocks the artifact
before manual readiness review. The shared expected-plan fail-closed list still
only named the broader `event_log_missing_or_invalid` condition.

## 2. Options considered

- Keep the broader fail-closed condition only.
- Add a validator-only violation and leave plan/handoff wording unchanged.
- Add `event_log_run_id_mismatch` to the shared fail-closed contract.

## 3. Decision

Add `event_log_run_id_mismatch` to
`WING_READONLY_PREFLIGHT_PLAN_FAIL_CLOSED_CONDITIONS`.

## 4. Why this won

The shared contract is already consumed by the runner, handoff packet, evidence
validator, and no-WING release check. Updating it once keeps operator-facing
packets and automated validation aligned.

## 5. Rejected alternatives

The broader condition was rejected because it does not tell the operator which
event-log identity check failed. A validator-only violation was rejected because
it would keep the plan less specific than the validator.

## 6. Implementation plan

- Add the new marker to the shared fail-closed tuple.
- Assert the marker through runner output, no-WING plan detail, and handoff
  markdown.
- Document the marker in the WING read-only preflight runbook.

## 7. Test plan

Run focused tests for runner, evidence, review, handoff, and no-WING release
checks, then standard pytest and no-WING quick runtime.

## 8. Risks and rollback

Fixtures with exact fail-closed lists may require updates. Rollback is removing
the marker from the shared tuple and its focused assertions. This ADR does not
contact WING, mutate console state, grant live authority, or close readiness.

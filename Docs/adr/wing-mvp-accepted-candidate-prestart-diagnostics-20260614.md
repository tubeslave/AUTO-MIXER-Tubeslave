# ADR: WING MVP Accepted-Candidate Prestart Diagnostics

## Context

Accepted WING read-only review source ids must have a valid accepted-candidate
prestart trace before supervised fader/gain proof can be planned. The existing
boolean gate blocks unsafe proof planning but does not identify which contract
field is missing.

## Options Considered

1. Keep the existing boolean only.
2. Add a separate diagnostic script.
3. Add shared report-only diagnostics to the existing contract helper path.

## Decision

Use option 3. Add structured per-source violation maps to the shared contract,
review decision ledger/report, supervised proof plan summary, and no-WING
summary.

## Why This Won

The existing review/proof/no-WING path is already the operator gate. Keeping the
diagnostics there avoids a second source of truth and makes blocker repair
specific without adding live authority.

## Rejected Alternatives

- Boolean only: too opaque for operator repair.
- Separate script: increases workflow surface area and can drift from the gate.

## Implementation Plan

- Add `wing_readonly_preflight_candidate_prestart_contract_violations`.
- Add `wing_readonly_preflight_accepted_candidate_prestart_violations_by_source`.
- Render `accepted_candidate_prestart_contract_violations_by_source`.
- Render `readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source`.
- Keep proof planning blocked when either map is non-empty.

## Test Plan

- Focused pytest around WING review decision, decision review CLI, supervised
  proof plan, and no-WING summary.
- Project Director and WING MVP rail tests.
- no-WING quick check and full pytest before landing.

## Risks And Rollback

The change is report-only. Rollback is to remove the diagnostic maps and keep
the existing boolean prestart gate.

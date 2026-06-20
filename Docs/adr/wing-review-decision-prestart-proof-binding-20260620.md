# M081 - WING Review Decision Prestart Proof Binding

## Context

The WING read-only preflight evidence path already fails closed on malformed
summary artifacts, event logs, expected-plan drift, backend-port prestart proof
and artifact-manifest verifier ordering. However, the generated manual review
decision template did not carry those prestart/verifier proof fields into the
operator decision JSON. That left a stale or hand-written decision record less
strongly bound to the exact proof that made the candidate eligible for review.

## Options Considered

1. Leave the binding implicit through candidate source id and summary hash.
2. Add only Markdown guidance to the review report.
3. Make the proof fields part of the shared decision-template contract and
   verify them in both the template generator and decision ledger.

## Decision

Use option 3. The shared contract now exposes
`WING_READONLY_PREFLIGHT_REVIEW_DECISION_PRESTART_PROOF_FIELDS`. Generated
review decision templates copy those fields from the clean candidate, and the
ledger rejects accepted decision records when the fields are missing or stale
against the candidate.

## Why This Won

It keeps the physical readback path fail-closed without introducing live
authority. A real WING artifact can only advance to manual decision review when
the decision record is bound to the same no-backend prestart summary, authority
denials and manifest-verifier ordering that validated the candidate.

## Rejected Alternatives

Implicit source-id binding alone was too weak for operator audit. Markdown-only
guidance was easier to miss and would not protect imported decision JSON.

## Implementation Plan

- Add shared prestart/verifier proof fields to the review decision template
  required-field contract.
- Copy those fields into generated pending decision templates.
- Reject mutated template candidates before template creation.
- Reject accepted decision records whose proof fields are missing or no longer
  match the candidate evidence.

## Test Plan

- Focused review tool and ledger tests.
- M081 WING MVP rail covering roadmap, current state, capability matrix,
  completion gate, plan, ADR and code markers.
- Full backend test suite before commit.

## Risks and Rollback

Existing hand-written review decision JSON files must include the new proof
fields. Rollback is to remove the new required fields and ledger comparison,
but that would re-open the weaker manual-decision binding. No live transport,
WING/backend contact, readiness closure or live authority is involved.

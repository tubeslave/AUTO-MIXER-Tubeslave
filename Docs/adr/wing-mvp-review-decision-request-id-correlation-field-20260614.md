# ADR: WING MVP Review Decision Request-Id Correlation Field

## 1. Context

WING read-only preflight artifacts prove physical readbacks by correlating each
response with its `query_mixer_parameter` request id. The artifact review report
exposes the request-id policy and marks correlation as required, but accepted
operator decision records did not persist those proof facts.

## 2. Options considered

- Keep request-id correlation only in artifact review output.
- Treat file hashes as sufficient durable evidence.
- Add request-id policy and correlation fields to the decision record.

## 3. Decision

Add `request_id_policy` and `request_id_correlation_required` to the shared
required fields for WING read-only preflight decision records. The decision
ledger rejects records unless the policy is canonical and the correlation flag
is true.

## 4. Why this won

The decision record is the durable operator acceptance surface. Persisting the
request-id contract there keeps the physical-readback proof tied to the exact
operator decision, without contacting WING or granting live authority.

## 5. Rejected alternatives

Keeping the fields only in review output was rejected because decisions may be
stored and audited separately. Hash-only binding was rejected because it does
not expose the machine-readable request/readback correlation rule.

## 6. Implementation plan

- Extend `WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS`.
- Add the two fields to generated decision templates.
- Reject records with a missing or non-canonical request-id policy.
- Reject records without `request_id_correlation_required: true`.
- Update runbook examples and focused tests.

## 7. Test plan

Run focused review-decision, decision-tool, review-tool, handoff, no-WING, WING
rail, and Project Director tests, then standard pytest and no-WING quick
runtime.

## 8. Risks and rollback

Older decision JSON without the fields will be rejected and must be replaced.
Rollback is removing the required fields and their validation/tests. This ADR
does not contact WING, mutate console state, grant live authority, or close
readiness.

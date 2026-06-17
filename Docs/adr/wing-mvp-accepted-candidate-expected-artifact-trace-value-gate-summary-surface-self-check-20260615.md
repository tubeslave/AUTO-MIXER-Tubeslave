# ADR: WING MVP Accepted Candidate Expected Artifact Trace Value Gate Summary Surface Self-Check

## 1. Context

The no-WING summary now exposes accepted-candidate expected artifact trace
value-gate fields in top-level `summary.json`, nested `next_wing_proof_plan`,
and operator `summary.md`. If those machine-readable surfaces diverge, an
operator or Project Director check could inspect the wrong value.

## 2. Options considered

- Trust the current builder to keep fields in sync.
- Add a markdown note only.
- Add a machine-readable self-check and fail closed on mismatches.

## 3. Decision

Add a no-WING self-check for the accepted-candidate expected artifact trace
value-gate summary surface. It compares the top-level and nested value-gate
fields, reports mismatches, renders the result in `summary.md`, and marks the
summary failed if the surfaces diverge.

## 4. Why this won

This keeps the operator handoff artifact auditable without granting readiness
or live authority. A stale top-level or nested value-gate surface becomes an
explicit no-WING failure instead of a silent inconsistency.

## 5. Rejected alternatives

Trusting the builder was rejected because future edits can still create drift.
Markdown-only reporting was rejected because automation and Project Director
checks need machine-readable evidence.

## 6. Implementation plan

- Add a canonical tuple of value-gate summary-surface fields.
- Add a self-check builder and fail-closed apply hook.
- Invoke the hook after constructing the value-gate summary surface.
- Render mismatch and match fields in the Next WING Proof Plan markdown section.
- Add passing and mismatch tests.

## 7. Test plan

- Targeted no-WING summary tests.
- no-WING quick artifact check.
- Project Director check.
- Focused WING/no-WING suite.
- Full pytest, memory lint, and whitespace diff check.

## 8. Risks and rollback

Risk: summary status can fail if generated surfaces drift unexpectedly. Rollback
is to remove the self-check hook and markdown lines while keeping the existing
value-gate validation.

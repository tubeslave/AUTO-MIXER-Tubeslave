# ADR: WING MVP Accepted Candidate Expected Artifact Trace Value Gate Summary Surface

## 1. Context

The no-WING next proof-plan validator rejects accepted-candidate expected
artifact trace values that do not match the canonical read-only preflight
review decision ledger trace. Operators also need that verdict exposed in the
summary artifacts used before a WING session.

## 2. Options considered

- Keep the mismatch list only in raw check-result internals.
- Render only markdown text.
- Mirror the value gate into both `summary.json` and `summary.md`.

## 3. Decision

Expose the trace value-gate required flag, violation ids, current violations,
pass/fail verdict, and readiness/live-authority denial fields as a no-WING
summary surface.

## 4. Why this won

The summary artifacts are the operator handoff boundary. Mirroring the
value-gate verdict makes stale accepted-candidate artifact bindings visible
without granting readiness or live authority.

## 5. Rejected alternatives

Raw check-result-only reporting was rejected because it hides the exact gate
state from top-level automation. Markdown-only reporting was rejected because
machine validation should not parse prose.

## 6. Implementation plan

- Add the canonical value-gate violation id tuple.
- Add detail fields to `next_wing_proof_plan_schema`.
- Mirror fields into nested and top-level summary surfaces.
- Render the value-gate lines in operator markdown.
- Extend targeted tests for mismatch and passing surfaces.

## 7. Test plan

- Targeted no-WING trace mismatch and summary embedding tests.
- no-WING quick artifact check.
- Project Director check.
- Focused WING/no-WING suite.
- Full pytest, memory lint, and whitespace diff check.

## 8. Risks and rollback

Risk: stale generated summaries fail because they do not render the new
value-gate lines. Rollback is to remove the top-level surface and markdown
lines while keeping the underlying value-gate validation.

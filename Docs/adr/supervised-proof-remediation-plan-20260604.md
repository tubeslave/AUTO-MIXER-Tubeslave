# ADR: Supervised Proof Remediation Plan

Date: 2026-06-04

## 1. Context

The supervised proof report now identifies dry-run-ready, physical-candidate,
blocked, invalid, and legacy ambiguous proof artifacts. Legacy executed WING
summaries without explicit schema correctly fail closed, but the report needs a
machine-readable plan for what to do next.

## 2. Options considered

- Keep remediation as manual interpretation of blocked rows.
- Add artifact-specific remediation actions to the report.
- Automatically rerun physical proof commands.

## 3. Decision

Add `supervised_proof_remediation_plan`, a transport-free report transformer
that emits remediation actions for blocked rows and invalid summaries. Legacy
ambiguous WING rows get current-schema physical rerun commands as checklist
data only.

## 4. Why this won

It makes the promotion path more operational without changing authority. The
operator can see exactly which proof artifacts are stale, invalid, or blocked,
while the system still refuses to execute commands or update safety rows.

## 5. Rejected alternatives

Manual-only interpretation was rejected because proof promotion needs
repeatable, testable evidence workflow. Automatic reruns were rejected because
physical WING writes require explicit operator approval. Folding all
remediation into the generic WING proof plan was rejected because blocked
artifact identity would be lost.

## 6. Implementation plan

1. Add a pure remediation-plan builder.
2. Generate current-schema rerun actions for legacy ambiguous artifacts.
3. Generate cleanup/review actions for invalid or otherwise blocked artifacts.
4. Embed the plan in proof report output.
5. Register the method as shadow-only in the processing registry.

## 7. Test plan

- Compile touched Python files.
- Run focused remediation/report/registry tests.
- Run broader supervised proof/readiness/no-WING tests.
- Run the standard full pytest command.

## 8. Risks and rollback

Risk: an operator mistakes a remediation command for permission. The plan marks
`execution_allowed_by_plan=false`, `live_write=false`, and
`operator_required=true`. Rollback is to remove the remediation builder and
report embedding; no live path is introduced.

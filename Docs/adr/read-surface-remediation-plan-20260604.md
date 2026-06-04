# ADR: Read Surface Remediation Plan

Date: 2026-06-04

## 1. Context

Full autonomous console control requires trustworthy read coverage before
write authority can safely expand. The project has a declarative
`read_surface_inventory`, but many surfaces remain partial or missing.

## 2. Options considered

- Leave remediation as inventory prose.
- Implement live console reads directly.
- Add a read-only remediation plan.

## 3. Decision

Add `read_surface_remediation_plan`. It converts incomplete read-surface rows
into concrete non-executing actions for Product Layer DTOs, coverage flags,
source metadata, fixtures, destructive-change markers, and parity mappings.

## 4. Why this won

It makes the full-console read gap visible and testable without contacting a
console or changing runtime behavior. This supports the long-term autonomous
goal while keeping the current safety boundary intact.

## 5. Rejected alternatives

Prose-only remediation was rejected because autonomous readiness needs
machine-readable blockers. Direct live read implementation was rejected as too
early without a complete plan across surfaces. Folding read gaps into write
proof reports was rejected because read coverage is a separate prerequisite.

## 6. Implementation plan

1. Add a pure read remediation builder.
2. Generate actions for partial, planned, and missing read surfaces.
3. Expose the plan in the processing registry snapshot.
4. Add tests for transport-free behavior, current incomplete rows, complete
   status, and registry visibility.

## 7. Test plan

- Compile touched Python files.
- Run focused read-remediation, inventory, and registry tests.
- Run broader readiness/mode/proof/no-WING tests.
- Run the standard full pytest command.

## 8. Risks and rollback

Risk: the plan is mistaken for permission to query or mutate a console. The
plan sets `read_only=true`, `live_write=false`, `mutates_console_state=false`,
and `full_autonomous_ready=false`. Rollback is to remove the plan and registry
snapshot field; no live path is introduced.

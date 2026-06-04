# routing-management-proposal-20260604

## 1. Context

The long-term project goal requires console routing control. Current live WING
deployment is limited to supervised channel fader/gain writes. Routing,
snapshot, and scene changes can rewrite broad console state, and existing
routing write paths are not approved live executors.

## 2. Options considered

- Promote existing routing write paths into the main assisted pipeline.
- Keep routing control quarantined with no proposal schema.
- Add a separate transport-free routing management proposal layer.

## 3. Decision

Add `backend/routing_management_proposal.py` as a proposal-only analyzer. It
estimates route-diff recommendations from route kind, current/proposed source
endpoints, expected source role, route readback, rollback baseline, operator
intent, route change window, active-performance state, current-route signal
activity, proposed-source verification, destructive scope, affected target
count, source confidence, route confidence, and persistence. It does not create
`requested_change` payloads for live apply.

## 4. Why this won

This gives the Product Layer a concrete routing recommendation surface while
preserving the current supervised fader/gain boundary. It separates route
reasoning and evidence from direct routing, snapshot, and scene write paths.

## 5. Rejected alternatives

Direct use of existing routing handlers would bypass current Product Layer
proof requirements. Keeping routing quarantined with no proposal schema leaves
no testable evidence packet for future review.

## 6. Implementation plan

1. Build bounded routing analysis and recommendation objects.
2. Normalize route kinds for channel input, output patch, bus, matrix, insert,
   and send source proposals.
3. Block proposals without readback, rollback, operator intent, safe change
   window, source verification, and scope limits.
4. Emit proposal-only payloads with no live apply fields.
5. Register the method and proposal contract.
6. Update the project-goal capability matrix to assisted/proposal-only for
   routing control.

## 7. Test plan

- `PYTHONPATH=backend python -m pytest tests/test_routing_management_proposal.py -q`
- `PYTHONPATH=backend python -m pytest tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- Compile/smoke checks with the available Python runtime if full test
  dependencies are unavailable.

## 8. Risks and rollback

The main risk is overstating live readiness for routing. The method is
registered as proposal-only, emits no queueable live changes, requires
readback/rollback proof before promotion, and keeps routing writes outside the
live executor. Rollback is removal of the module, registry entry, proposal
contract, matrix update, tests, brief, proposal, and ADR; no live transport code
is touched.

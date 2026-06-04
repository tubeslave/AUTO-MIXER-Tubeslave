# Codex Proposal: routing-management-proposal-20260604

## 1. Short thesis

Move routing control from quarantined capability metadata toward assisted
automation by adding a transport-free analyzer that emits non-queueable
route-diff proposals.

## 2. What I understood

The Automixer should eventually control routing, but route writes can rewrite
input patches, output patches, bus sources, matrices, inserts, and sends. The
current approved live executor remains supervised fader/gain only, so routing
logic must first become explainable evidence and operator-review proposals.

## 3. Proposed solution

Create `backend/routing_management_proposal.py` with bounded recommendation
parameters: route id/kind, current source endpoint, proposed source endpoint,
expected source role, route readback, rollback baseline, operator intent, route
change window, active-performance flag, current-route activity, proposed-source
verification, destructive scope, affected target count, confidence, blockers,
and evidence. It should block proposals without route readback, rollback
baseline, operator intent, safe change window, source verification, and scope
limits.

## 4. Likely files to touch

- `backend/routing_management_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_routing_management_proposal.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/adr/routing-management-proposal-20260604.md`

## 5. Alternatives considered

- Promote existing routing handlers: rejected because they are direct write
  paths with broad console-state impact.
- Keep routing fully quarantined with no proposal schema: rejected because the
  goal needs operator-visible progress toward safe routing control.

## 6. Risks

- Routing changes can silence or mispatch many signals. The module blocks active
  performance changes, active-route changes, broad scope changes, and missing
  rollback/readback evidence.
- A proposal-only route diff may be mistaken for live readiness. The output
  remains non-queueable and requires hardware proof before promotion.

## 7. Test plan

- Unit tests for safe soundcheck route proposal, active-performance block,
  missing readback/rollback block, unchanged-route block, proposal shape,
  registry status, and transport-free source scan.

## 8. Where the other agent may disagree

Kimi may prefer leaving routing fully quarantined until hardware proof. A
proposal-only surface is safer for progress because it exposes the evidence
needed for future review while keeping all routing writes out of live execution.

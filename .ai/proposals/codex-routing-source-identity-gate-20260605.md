# Proposal: routing-source-identity-gate-20260605

## Short thesis

Use replay-calibrated source identity trust as a routing-management blocker so
route-diff proposals cannot rely on explicitly untrusted source labels.

## What I understood

The routing layer already checks route kind, current/proposed endpoints, route
readback, rollback baseline, operator intent, performance window, active
current signal, destructive scope, source confidence, route confidence,
persistence, and current route cache freshness. It is explicitly non-queueable
and transport-free. Because routing decisions depend on which source is being
patched, source identity trust should become part of the routing safety
evidence before any route diff can be proposed.

## Proposed solution

Add optional source identity trust fields to
`RoutingManagementAnalysisInput` and evaluate them through
`source_identity_downstream_gate`. Add the resulting blockers and evidence to
`RoutingManagementRecommendation`. Keep legacy missing calibration allowed so
existing callers remain compatible, but block explicit `hold`, `blocked`,
stale, or unrecognized trust states.

Update the proposal-only contract and capability matrix support paths, then add
focused tests for legacy compatibility, held trust, stale replay context, and
contract/matrix evidence.

## Likely files to touch

- `backend/routing_management_proposal.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_routing_management_proposal.py`
- `.ai/briefs/routing-source-identity-gate-20260605.md`
- `Docs/adr/routing-source-identity-gate-20260605.md`

## Alternatives considered

- Require source identity calibration for all routing inputs immediately.
  Rejected because legacy callers do not yet pass replay artifacts.
- Ignore replay trust for routing management. Rejected because a wrong source
  identity can make a route diff look safe when it is patching the wrong
  source.
- Duplicate trust-state parsing inside routing management. Rejected because the
  shared gate keeps blocker names and no-mutation evidence consistent across
  downstream modules.

## Risks

- Conservative blocking may hold useful route proposals until replay
  calibration is consistently available.
- The gate currently relies on upstream trust-state strings; future DTO wiring
  should pass structured calibration records.

## Test plan

```bash
PYTHONPATH=backend python3 -m py_compile \
  backend/routing_management_proposal.py \
  backend/source_identity_downstream_gate.py \
  backend/processing_proposal_adapters.py \
  backend/project_goal_capability_matrix.py \
  tests/test_routing_management_proposal.py
PYTHONPATH=backend python3 -m pytest \
  tests/test_routing_management_proposal.py \
  tests/test_source_identity_downstream_gate.py \
  tests/test_processing_registry.py \
  tests/test_project_goal_capability_matrix.py -q
PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q
/opt/homebrew/bin/git diff --check
```

## Where the other agent may disagree

Kimi may argue that operator-requested routing changes should bypass source
identity trust. I would keep the gate conservative because this module is still
proposal-only; a future live routing path needs separate hardware proof,
readback, rollback, and explicit operator approval.

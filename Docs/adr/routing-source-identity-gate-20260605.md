# ADR: routing-source-identity-gate-20260605

## Context

Routing control is part of the target Automixer, but route diffs can rewrite
broad console state. The existing routing proposal layer already requires route
readback, rollback baseline, operator intent, safe routing windows, inactive
current signal, proposed-source verification, destructive-scope limits, route
confidence, source confidence, persistence, and fresh current route cache
evidence. Source identity replay calibration now exists, so routing management
should not rely on explicitly untrusted source labels.

## Options considered

1. Keep routing management using only source/route confidence and route state
   evidence.
2. Wire the shared source identity downstream gate into routing management.
3. Require fresh source identity calibration for all routing-management calls.
4. Duplicate trust-state parsing inside routing management.

## Decision

Use option 2. Add optional source identity trust fields to routing-management
analysis input and evaluate them through `source_identity_downstream_gate`.
Missing legacy calibration remains compatible, but explicit held, blocked,
unrecognized, or stale replay trust blocks routing proposals.

## Why this won

This moves replay-calibrated identity into a high-risk console-control safety
gate without adding live authority. It keeps blocker names and no-mutation
evidence consistent with existing downstream consumers while preserving the
current proposal-only routing boundary.

## Rejected alternatives

Source/route-confidence-only routing was rejected because a wrong identity can
make the wrong source patch look valid. Requiring calibration everywhere was
rejected because existing callers still need a compatibility path. Duplicating
trust parsing was rejected because downstream modules should share trust
semantics.

## Implementation plan

- Add optional source identity trust fields to
  `RoutingManagementAnalysisInput`.
- Evaluate trust through the shared downstream gate.
- Add trust blockers and evidence to routing recommendations.
- Mark source identity downstream trust as required proposal evidence.
- Add the shared trust gate as a routing capability support path.
- Add focused tests for legacy-compatible, held, and stale trust states.
- Run focused and full validation.

## Test plan

```bash
PYTHONPATH=backend python3 -m py_compile \
  backend/routing_management_proposal.py \
  backend/source_identity_downstream_gate.py \
  backend/processing_proposal_adapters.py \
  backend/project_goal_capability_matrix.py
PYTHONPATH=backend python3 -m pytest \
  tests/test_routing_management_proposal.py \
  tests/test_source_identity_downstream_gate.py \
  tests/test_processing_registry.py \
  tests/test_project_goal_capability_matrix.py -q
PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q
/opt/homebrew/bin/git diff --check
```

## Risks and rollback

Risk: explicit held source identity may block useful route-diff proposals until
operators accept labels. Rollback is removing the optional trust fields, gate
call, contract evidence entry, capability support path, and focused tests. No
live transport, queue mutation, model mutation, or memory mutation path is
added.

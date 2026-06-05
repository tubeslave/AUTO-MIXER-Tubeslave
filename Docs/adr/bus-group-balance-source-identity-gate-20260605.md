# ADR: bus-group-balance-source-identity-gate-20260605

## Context

Bus/group balance proposals use Ayaic level planes, summed child loudness,
current fader readback, assignment confidence, true-peak headroom, main-bus
headroom, feedback risk, persistence, and current bus/group/DCA/matrix
freshness. They also rely on source-role labels to choose the grouped target
plane and explain the move. Source identity replay calibration now exists, so
grouped balance should not rely on explicitly untrusted source labels.

## Options considered

1. Keep bus/group balance using only assignment confidence, loudness, headroom,
   and current-state freshness.
2. Wire the shared source identity downstream gate into bus/group balance.
3. Require fresh source identity calibration for all bus/group balance calls.
4. Duplicate trust-state parsing inside bus/group balance.

## Decision

Use option 2. Add optional source identity trust fields to bus/group balance
analysis input and evaluate them through `source_identity_downstream_gate`.
Missing legacy calibration remains compatible, but explicit held, blocked,
unrecognized, or stale replay trust blocks grouped balance proposals.

## Why this won

This moves replay-calibrated identity into the grouped-balance safety contract
without adding live authority. It keeps blocker names and no-mutation evidence
consistent with existing downstream consumers while preserving the current
proposal-only DCA/group/matrix boundary.

## Rejected alternatives

Assignment-confidence-only balance was rejected because the wrong source role
can choose the wrong Ayaic plane or imply the wrong grouped move. Requiring
calibration everywhere was rejected because existing callers still need a
compatibility path. Duplicating trust parsing was rejected because downstream
modules should share trust semantics.

## Implementation plan

- Add optional source identity trust fields to
  `BusGroupBalanceAnalysisInput`.
- Evaluate trust through the shared downstream gate.
- Add trust blockers and evidence to bus/group balance recommendations.
- Mark source identity downstream trust as required proposal evidence.
- Add the shared trust gate as a support path for mix balance and
  DCA/group/matrix control capability rows.
- Add focused tests for legacy-compatible, held, and stale trust states.
- Run focused and full validation.

## Test plan

```bash
PYTHONPATH=backend python3 -m py_compile \
  backend/bus_group_balance_proposal.py \
  backend/source_identity_downstream_gate.py \
  backend/processing_proposal_adapters.py \
  backend/project_goal_capability_matrix.py
PYTHONPATH=backend python3 -m pytest \
  tests/test_bus_group_balance_proposal.py \
  tests/test_source_identity_downstream_gate.py \
  tests/test_processing_registry.py \
  tests/test_project_goal_capability_matrix.py -q
PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q
/opt/homebrew/bin/git diff --check
```

## Risks and rollback

Risk: explicit held source identity may block useful grouped-balance proposals
until operators accept labels. Rollback is removing the optional trust fields,
gate call, contract evidence entry, capability support paths, and focused
tests. No live transport, queue mutation, model mutation, or memory mutation
path is added.

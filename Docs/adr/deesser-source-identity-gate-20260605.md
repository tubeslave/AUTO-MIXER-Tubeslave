# ADR: deesser-source-identity-gate-20260605

## Context

Automatic de-essing is part of the target Automixer. It is a source-specific
processing decision: the proposal is only meaningful when the source is truly a
vocal or speech channel. Source identity replay calibration now exists, so the
de-esser proposal layer should not rely on explicitly untrusted source labels.

## Options considered

1. Keep de-esser using only `source_role` and sibilance evidence.
2. Wire the shared source identity downstream gate into de-esser.
3. Require fresh source identity calibration for all de-esser calls.
4. Duplicate trust-state parsing inside de-esser.

## Decision

Use option 2. Add optional source identity trust fields to de-esser analysis
input and evaluate them through `source_identity_downstream_gate`. Missing
legacy calibration remains compatible, but explicit held, blocked, unknown, or
stale replay trust blocks de-esser proposals.

## Why this won

This moves replay-calibrated identity from a generic evidence layer into a real
vocal-processing safety gate without adding live authority. It keeps blocker
names and no-mutation evidence consistent with arrangement-role and
frequency-separation consumers.

## Rejected alternatives

Source-role-only de-essing was rejected because a wrong vocal label can mislead
dynamics or dynamic-EQ recommendations. Requiring calibration everywhere was
rejected because existing callers still need a compatibility path. Duplicating
trust parsing was rejected because downstream modules should share trust
semantics.

## Implementation plan

- Add optional source identity trust fields to `DeesserAnalysisInput`.
- Evaluate trust through the shared downstream gate.
- Add trust blockers and evidence to de-esser recommendations.
- Mark source identity downstream trust as required proposal evidence.
- Add focused tests for legacy-compatible, held, and stale trust states.
- Run focused and full validation.

## Test plan

```bash
PYTHONPATH=backend python -m py_compile \
  backend/deesser_vocal_sibilance.py \
  backend/source_identity_downstream_gate.py \
  backend/processing_proposal_adapters.py \
  backend/project_goal_capability_matrix.py
PYTHONPATH=backend python -m pytest \
  tests/test_deesser_vocal_sibilance.py \
  tests/test_source_identity_downstream_gate.py \
  tests/test_processing_registry.py \
  tests/test_project_goal_capability_matrix.py -q
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
/opt/homebrew/bin/git diff --check
```

## Risks and rollback

Risk: explicit held source identity may block useful de-esser proposals until
operators accept labels. Rollback is removing the optional trust fields, gate
call, contract evidence entry, capability support path, and focused tests. No
live transport, queue mutation, model mutation, or memory mutation path is
added.

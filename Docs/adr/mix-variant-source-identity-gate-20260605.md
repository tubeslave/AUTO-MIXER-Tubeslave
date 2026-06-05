# ADR: mix-variant-source-identity-gate-20260605

## Context

Mix-variant proposals use genre, task, reference metrics, arrangement context,
Ayaic level planes, headroom, feedback risk, style-clone guards, and current
context freshness. They can fan out into bus/group balance, FX send,
master-bus, and scene/preset proposal layers. Source identity replay
calibration now exists, so mix variants should not fan out from explicitly
untrusted source-role context.

## Options considered

1. Keep mix variants using only arrangement confidence, reference metrics,
   headroom, feedback, and freshness evidence.
2. Wire the shared source identity downstream gate into mix variants.
3. Require fresh source identity calibration for all mix-variant calls.
4. Duplicate trust-state parsing inside mix variants.

## Decision

Use option 2. Add optional source identity trust fields to mix-variant analysis
input and evaluate them through `source_identity_downstream_gate`. Missing
legacy calibration remains compatible, but explicit held, blocked,
unrecognized, or stale replay trust blocks mix-variant profile proposals.

## Why this won

This moves replay-calibrated identity into a profile fan-out safety gate
without adding live authority. It keeps blocker names and no-mutation evidence
consistent with existing downstream consumers while preserving the current
proposal-only mix-variant boundary.

## Rejected alternatives

Arrangement-confidence-only variants were rejected because a wrong source-role
context can shape every downstream proposal. Requiring calibration everywhere
was rejected because existing callers still need a compatibility path.
Duplicating trust parsing was rejected because downstream modules should share
trust semantics.

## Implementation plan

- Add optional source identity trust fields to
  `MixVariantProfileAnalysisInput`.
- Evaluate trust through the shared downstream gate.
- Add trust blockers and evidence to mix-variant recommendations.
- Mark source identity downstream trust as required proposal evidence.
- Add the shared trust gate as a genre/task variant capability support path.
- Add focused tests for legacy-compatible, held, and stale trust states.
- Run focused and full validation.

## Test plan

```bash
PYTHONPATH=backend python3 -m py_compile \
  backend/mix_variant_profile_proposal.py \
  backend/source_identity_downstream_gate.py \
  backend/processing_proposal_adapters.py \
  backend/project_goal_capability_matrix.py
PYTHONPATH=backend python3 -m pytest \
  tests/test_mix_variant_profile_proposal.py \
  tests/test_source_identity_downstream_gate.py \
  tests/test_processing_registry.py \
  tests/test_project_goal_capability_matrix.py -q
PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q
/opt/homebrew/bin/git diff --check
```

## Risks and rollback

Risk: explicit held source identity may block useful genre/task profile
proposals until operators accept labels. Rollback is removing the optional
trust fields, gate call, contract evidence entry, capability support path, and
focused tests. No live transport, queue mutation, model mutation, or memory
mutation path is added.

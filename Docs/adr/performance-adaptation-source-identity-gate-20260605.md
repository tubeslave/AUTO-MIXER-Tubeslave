# ADR: performance-adaptation-source-identity-gate-20260605

## Context

Musician-level adaptation, concert-time adaptation, and real-time corrections
are part of the target Automixer. These are source-specific decisions: lead
sources, backing sources, high-bleed sources, and critical anchors need
different risk handling. Source identity replay calibration now exists, so the
performance-adaptation proposal layer should not rely on explicitly untrusted
source labels.

## Options considered

1. Keep performance adaptation using only `source_role` and level evidence.
2. Wire the shared source identity downstream gate into performance adaptation.
3. Require fresh source identity calibration for all adaptation calls.
4. Duplicate trust-state parsing inside performance adaptation.

## Decision

Use option 2. Add optional source identity trust fields to performance
adaptation analysis input and evaluate them through
`source_identity_downstream_gate`. Missing legacy calibration remains
compatible, but explicit held, blocked, unknown, or stale replay trust blocks
adaptation proposals.

## Why this won

This moves replay-calibrated identity into a real live-adaptation safety gate
without adding live authority. It keeps blocker names and no-mutation evidence
consistent with arrangement-role, frequency-separation, de-esser, FX send, pan,
and compressor consumers.

## Rejected alternatives

Source-role-only adaptation was rejected because a wrong label can mislead
high-bleed handling, critical-source handling, and future ride priority.
Requiring calibration everywhere was rejected because existing callers still
need a compatibility path. Duplicating trust parsing was rejected because
downstream modules should share trust semantics.

## Implementation plan

- Add optional source identity trust fields to
  `PerformanceAdaptationAnalysisInput`.
- Evaluate trust through the shared downstream gate.
- Add trust blockers and evidence to adaptation recommendations.
- Mark source identity downstream trust as required proposal evidence.
- Add focused tests for legacy-compatible, held, and stale trust states.
- Run focused and full validation.

## Test plan

```bash
PYTHONPATH=backend python3 -m py_compile \
  backend/performance_adaptation_proposal.py \
  backend/source_identity_downstream_gate.py \
  backend/processing_proposal_adapters.py \
  backend/project_goal_capability_matrix.py
PYTHONPATH=backend python3 -m pytest \
  tests/test_performance_adaptation_proposal.py \
  tests/test_source_identity_downstream_gate.py \
  tests/test_processing_registry.py \
  tests/test_project_goal_capability_matrix.py -q
PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q
/opt/homebrew/bin/git diff --check
```

## Risks and rollback

Risk: explicit held source identity may block useful adaptation proposals until
operators accept labels. Rollback is removing the optional trust fields, gate
call, contract evidence entry, capability support paths, and focused tests. No
live transport, queue mutation, model mutation, or memory mutation path is
added.

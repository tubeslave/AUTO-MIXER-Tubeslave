# ADR: pan-source-identity-gate-20260605

## Context

Automatic panning and console pan control are part of the target Automixer.
They are source-specific decisions: lead vocals, kick, bass, stereo guitars,
overheads, playback, and support sources need different placement policies.
Source identity replay calibration now exists, so the pan proposal layer should
not rely on explicitly untrusted source labels.

## Options considered

1. Keep pan using only `source_role`, stereo, stage, and mono evidence.
2. Wire the shared source identity downstream gate into pan.
3. Require fresh source identity calibration for all pan calls.
4. Duplicate trust-state parsing inside pan.

## Decision

Use option 2. Add optional source identity trust fields to pan analysis input
and evaluate them through `source_identity_downstream_gate`. Missing legacy
calibration remains compatible, but explicit held, blocked, unknown, or stale
replay trust blocks pan proposals.

## Why this won

This moves replay-calibrated identity into a real spatial-decision safety gate
without adding live authority. It keeps blocker names and no-mutation evidence
consistent with arrangement-role, frequency-separation, de-esser, and FX send
consumers.

## Rejected alternatives

Source-role-only pan was rejected because a wrong label can mislead center
anchors, stereo-pair placement, and stage-position moves. Requiring calibration
everywhere was rejected because existing callers still need a compatibility
path. Duplicating trust parsing was rejected because downstream modules should
share trust semantics.

## Implementation plan

- Add optional source identity trust fields to `PanAnalysisInput`.
- Evaluate trust through the shared downstream gate.
- Add trust blockers and evidence to pan recommendations.
- Mark source identity downstream trust as required proposal evidence.
- Add focused tests for legacy-compatible, held, and stale trust states.
- Run focused and full validation.

## Test plan

```bash
PYTHONPATH=backend python3 -m py_compile \
  backend/pan_spatial_proposal.py \
  backend/source_identity_downstream_gate.py \
  backend/processing_proposal_adapters.py \
  backend/project_goal_capability_matrix.py
PYTHONPATH=backend python3 -m pytest \
  tests/test_pan_spatial_proposal.py \
  tests/test_source_identity_downstream_gate.py \
  tests/test_processing_registry.py \
  tests/test_project_goal_capability_matrix.py -q
PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q
/opt/homebrew/bin/git diff --check
```

## Risks and rollback

Risk: explicit held source identity may block useful pan proposals until
operators accept labels. Rollback is removing the optional trust fields, gate
call, contract evidence entry, capability support paths, and focused tests. No
live transport, queue mutation, model mutation, or memory mutation path is
added.

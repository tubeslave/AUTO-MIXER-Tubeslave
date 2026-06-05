# ADR: mute-management-source-identity-gate-20260605

## Context

Mute control is part of the target Automixer, but mute/unmute decisions are
high-risk console actions. The proposal layer distinguishes critical sources,
safe-idle sources, auto-muted sources, manual mutes, and emergency feedback
intent from `source_role` and channel state. Source identity replay calibration
now exists, so mute management should not rely on explicitly untrusted source
labels.

## Options considered

1. Keep mute management using only `source_role` and mute/signal evidence.
2. Wire the shared source identity downstream gate into mute management.
3. Require fresh source identity calibration for all mute-management calls.
4. Duplicate trust-state parsing inside mute management.

## Decision

Use option 2. Add optional source identity trust fields to mute-management
analysis input and evaluate them through `source_identity_downstream_gate`.
Missing legacy calibration remains compatible, but explicit held, blocked,
unknown, or stale replay trust blocks mute-state proposals.

## Why this won

This moves replay-calibrated identity into a real console-control safety gate
without adding live authority. It keeps blocker names and no-mutation evidence
consistent with existing downstream consumers while preserving manual mute
authority and the current proposal-only boundary.

## Rejected alternatives

Source-role-only mute management was rejected because a wrong label can mislead
critical-source, safe-idle, and upcoming-cue handling. Requiring calibration
everywhere was rejected because existing callers still need a compatibility
path. Duplicating trust parsing was rejected because downstream modules should
share trust semantics.

## Implementation plan

- Add optional source identity trust fields to `MuteManagementAnalysisInput`.
- Evaluate trust through the shared downstream gate.
- Add trust blockers and evidence to mute recommendations.
- Mark source identity downstream trust as required proposal evidence.
- Add focused tests for legacy-compatible, held, and stale trust states.
- Run focused and full validation.

## Test plan

```bash
PYTHONPATH=backend python3 -m py_compile \
  backend/mute_management_proposal.py \
  backend/source_identity_downstream_gate.py \
  backend/processing_proposal_adapters.py \
  backend/project_goal_capability_matrix.py
PYTHONPATH=backend python3 -m pytest \
  tests/test_mute_management_proposal.py \
  tests/test_source_identity_downstream_gate.py \
  tests/test_processing_registry.py \
  tests/test_project_goal_capability_matrix.py -q
PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q
/opt/homebrew/bin/git diff --check
```

## Risks and rollback

Risk: explicit held source identity may block useful mute-state proposals until
operators accept labels. Rollback is removing the optional trust fields, gate
call, contract evidence entry, capability support path, and focused tests. No
live transport, queue mutation, model mutation, or memory mutation path is
added.

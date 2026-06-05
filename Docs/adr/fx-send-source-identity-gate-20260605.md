# ADR: fx-send-source-identity-gate-20260605

## Context

Automatic FX management and effect-send tuning are part of the target
Automixer. They are source-specific processing decisions: dry anchors, vocals,
playback, drums, and harmonic sources use different send and return strategies.
Source identity replay calibration now exists, so the FX send proposal layer
should not rely on explicitly untrusted source labels.

## Options considered

1. Keep FX sends using only `source_role`, headroom, and risk evidence.
2. Wire the shared source identity downstream gate into FX sends.
3. Require fresh source identity calibration for all FX send calls.
4. Duplicate trust-state parsing inside FX send.

## Decision

Use option 2. Add optional source identity trust fields to FX send analysis
input and evaluate them through `source_identity_downstream_gate`. Missing
legacy calibration remains compatible, but explicit held, blocked, unknown, or
stale replay trust blocks FX send proposals.

## Why this won

This moves replay-calibrated identity into a real FX-management safety gate
without adding live authority. It keeps blocker names and no-mutation evidence
consistent with arrangement-role, frequency-separation, and de-esser consumers.

## Rejected alternatives

Source-role-only FX sends were rejected because a wrong label can mislead send
choice, dry-anchor handling, and intelligibility risk. Requiring calibration
everywhere was rejected because existing callers still need a compatibility
path. Duplicating trust parsing was rejected because downstream modules should
share trust semantics.

## Implementation plan

- Add optional source identity trust fields to `FXSendAnalysisInput`.
- Evaluate trust through the shared downstream gate.
- Add trust blockers and evidence to FX send recommendations.
- Mark source identity downstream trust as required proposal evidence.
- Add focused tests for legacy-compatible, held, and stale trust states.
- Run focused and full validation.

## Test plan

```bash
PYTHONPATH=backend python3 -m py_compile \
  backend/fx_send_proposal.py \
  backend/source_identity_downstream_gate.py \
  backend/processing_proposal_adapters.py \
  backend/project_goal_capability_matrix.py
PYTHONPATH=backend python3 -m pytest \
  tests/test_fx_send_proposal.py \
  tests/test_source_identity_downstream_gate.py \
  tests/test_processing_registry.py \
  tests/test_project_goal_capability_matrix.py -q
PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q
/opt/homebrew/bin/git diff --check
```

## Risks and rollback

Risk: explicit held source identity may block useful FX send proposals until
operators accept labels. Rollback is removing the optional trust fields, gate
call, contract evidence entry, capability support paths, and focused tests. No
live transport, queue mutation, model mutation, or memory mutation path is
added.

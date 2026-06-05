# ADR: source-identity-downstream-gate-20260605

## Context

The project goal requires decisions to use previous outcomes and project
history. Source identity replay calibration now summarizes accepted/rejected
label outcomes, but arrangement-role and frequency-separation proposals did not
consume that trust evidence before making source-specific recommendations.

## Options considered

1. Keep replay calibration as report-only evidence.
2. Add a shared downstream trust gate and wire early consumers to it.
3. Duplicate trust-state checks inside each downstream module.
4. Require fresh calibration for every source immediately.

## Decision

Use option 2. Add `source_identity_downstream_gate` as a transport-free helper
and wire it into arrangement-role and frequency-separation proposals. Missing
legacy calibration remains compatible, but explicit hold, block, unknown, or
stale replay-calibration states prevent downstream proposals.

## Why this won

This moves source identity memory from passive evidence into actual proposal
safety logic without adding live authority. It keeps the rule names and evidence
shape shared across downstream modules, making future EQ, dynamics, FX, and
balance integrations easier to audit.

## Rejected alternatives

Report-only calibration was rejected because it does not change downstream
decision quality. Per-module checks were rejected because they would drift.
Requiring calibration for all callers was rejected because existing offline and
legacy proposal paths do not yet pass replay artifacts everywhere.

## Implementation plan

- Add a shared source identity downstream trust decision helper.
- Add optional source identity trust fields to arrangement-role and
  frequency-separation inputs.
- Block arrangement-role recommendations when explicit source identity trust is
  held, blocked, unknown, or stale.
- Block frequency-separation lane proposals when a source has explicit untrusted
  source identity evidence.
- Expose trust evidence in proposal contracts and goal support paths.
- Add focused tests and run the standard validation.

## Test plan

```bash
PYTHONPATH=backend python -m py_compile \
  backend/source_identity_downstream_gate.py \
  backend/arrangement_role_proposal.py \
  backend/frequency_separation_proposal.py \
  backend/processing_proposal_adapters.py \
  backend/project_goal_capability_matrix.py
PYTHONPATH=backend python -m pytest \
  tests/test_source_identity_downstream_gate.py \
  tests/test_arrangement_role_proposal.py \
  tests/test_frequency_separation_proposal.py \
  tests/test_processing_registry.py \
  tests/test_project_goal_capability_matrix.py -q
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
/opt/homebrew/bin/git diff --check
```

## Risks and rollback

Risk: the first frequency-separation integration blocks all lane proposals when
any provided source identity is explicitly untrusted. Rollback is removing the
helper, optional trust fields, source identity blockers/evidence, contract
entries, and focused tests. No live transport, model mutation, or memory
mutation path is added.

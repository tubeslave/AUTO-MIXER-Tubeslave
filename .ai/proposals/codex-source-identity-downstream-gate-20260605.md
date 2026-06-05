# Proposal: source-identity-downstream-gate-20260605

## Short thesis

Wire replay-calibrated source identity trust into downstream proposal families
through a shared read-only gate, starting with arrangement role and frequency
separation.

## What I understood

The previous source identity layers reviewed weak classifier output and
summarized accepted/rejected replay outcomes. That evidence only helps the
autonomous mixing goal if downstream decisions consume it before making
source-specific proposals.

## Proposed solution

Create `backend/source_identity_downstream_gate.py` with
`evaluate_source_identity_downstream_trust`. The gate should:

- treat explicit `trusted_for_downstream_priority` as usable evidence;
- preserve compatibility when calibration is not yet provided;
- block `hold_for_operator_review`, `needs_more_replay_outcomes`,
  `blocked_from_downstream_trust`, unknown states, or stale replay context;
- report no live, queue, model, or memory mutation.

Then add optional trust fields to `ArrangementRoleAnalysisInput` and
`FrequencySeparationSource`. Arrangement role should add trust blockers to the
existing recommendation blockers. Frequency separation should prevent untrusted
sources from entering lane proposals.

## Likely files to touch

- `backend/source_identity_downstream_gate.py`
- `backend/arrangement_role_proposal.py`
- `backend/frequency_separation_proposal.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_source_identity_downstream_gate.py`
- `tests/test_arrangement_role_proposal.py`
- `tests/test_frequency_separation_proposal.py`
- `.ai/briefs/source-identity-downstream-gate-20260605.md`
- `Docs/adr/source-identity-downstream-gate-20260605.md`

## Alternatives considered

- Let every downstream module parse calibration strings independently. Rejected
  because blocker names and evidence would drift.
- Require calibration for all legacy calls immediately. Rejected because many
  existing offline/proposal callers do not yet pass calibration artifacts.
- Let untrusted sources participate with lower priority. Rejected for now
  because wrong source labels can steer EQ, role, and lane ownership.

## Risks

- Conservative blocking may hold valid proposals until replay calibration is
  consistently available.
- Frequency separation currently blocks the whole lane analysis if any provided
  source identity is explicitly untrusted; this is safer but may be refined to
  per-pair filtering later.

## Test plan

```bash
PYTHONPATH=backend python -m py_compile \
  backend/source_identity_downstream_gate.py \
  backend/arrangement_role_proposal.py \
  backend/frequency_separation_proposal.py \
  backend/processing_proposal_adapters.py \
  backend/project_goal_capability_matrix.py \
  tests/test_source_identity_downstream_gate.py \
  tests/test_arrangement_role_proposal.py \
  tests/test_frequency_separation_proposal.py
PYTHONPATH=backend python -m pytest \
  tests/test_source_identity_downstream_gate.py \
  tests/test_arrangement_role_proposal.py \
  tests/test_frequency_separation_proposal.py \
  tests/test_processing_registry.py \
  tests/test_project_goal_capability_matrix.py -q
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
/opt/homebrew/bin/git diff --check
```

## Where the other agent may disagree

Kimi may argue that untrusted sources should simply be ignored per pair in
frequency separation. I would start fail-closed because the first explicit
calibration consumers should prove that untrusted labels cannot quietly steer
source-specific proposals.

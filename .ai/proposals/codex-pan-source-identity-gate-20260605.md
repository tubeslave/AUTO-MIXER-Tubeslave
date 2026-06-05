# Proposal: pan-source-identity-gate-20260605

## Short thesis

Use replay-calibrated source identity trust as a pan proposal blocker so spatial
placement decisions cannot rely on explicitly untrusted source labels.

## What I understood

The pan layer already checks source family, stereo pair, stage position,
low-frequency share, mono-compatibility risk, persistence, and current pan state
freshness. It still trusts `source_role` directly. After source identity replay
calibration was added, pan decisions should consume that trust evidence before
making recommendations.

## Proposed solution

Add optional source identity trust fields to `PanAnalysisInput` and evaluate
them through `source_identity_downstream_gate`. Add the resulting blockers and
evidence to `PanRecommendation`. Keep legacy missing calibration allowed so
existing callers remain compatible, but block explicit `hold`, `blocked`,
stale, or unknown trust states.

Update the proposal-only contract and capability matrix support paths, then add
focused tests.

## Likely files to touch

- `backend/pan_spatial_proposal.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_pan_spatial_proposal.py`
- `.ai/briefs/pan-source-identity-gate-20260605.md`
- `Docs/adr/pan-source-identity-gate-20260605.md`

## Alternatives considered

- Require source identity calibration for all pan inputs immediately. Rejected
  because legacy offline callers do not yet pass replay artifacts.
- Ignore replay trust for pan. Rejected because a wrong label can steer center
  anchors, stereo-pair placement, and stage-position moves.
- Duplicate trust-state parsing inside pan. Rejected because the shared gate
  keeps blocker names and evidence consistent across downstream modules.

## Risks

- Conservative blocking may hold valid pan proposals until replay calibration is
  consistently available.
- The gate currently relies on upstream trust-state strings; future DTO wiring
  should pass structured calibration records.

## Test plan

```bash
PYTHONPATH=backend python3 -m py_compile \
  backend/pan_spatial_proposal.py \
  backend/source_identity_downstream_gate.py \
  backend/processing_proposal_adapters.py \
  backend/project_goal_capability_matrix.py \
  tests/test_pan_spatial_proposal.py
PYTHONPATH=backend python3 -m pytest \
  tests/test_pan_spatial_proposal.py \
  tests/test_source_identity_downstream_gate.py \
  tests/test_processing_registry.py \
  tests/test_project_goal_capability_matrix.py -q
PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q
/opt/homebrew/bin/git diff --check
```

## Where the other agent may disagree

Kimi may argue that pan proposals should still emit low-priority suggestions
when source identity is held. I would block explicitly untrusted source labels
first: pan can become a console write later, and wrong spatial placement is
hard to treat as harmless in live sound.

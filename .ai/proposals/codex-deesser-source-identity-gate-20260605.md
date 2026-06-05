# Proposal: deesser-source-identity-gate-20260605

## Short thesis

Use replay-calibrated source identity trust as a de-esser proposal blocker so
vocal-specific processing cannot rely on explicitly untrusted source labels.

## What I understood

The de-esser layer already checks vocal role, sibilance ratio, persistence,
true-peak headroom, and current-state freshness. It still trusts `source_role`
directly. After source identity replay calibration was added, vocal-specific
processing should consume that trust evidence before making recommendations.

## Proposed solution

Add optional source identity trust fields to `DeesserAnalysisInput` and evaluate
them through `source_identity_downstream_gate`. Add the resulting blockers and
evidence to `DeesserRecommendation`. Keep legacy missing calibration allowed so
existing callers remain compatible, but block explicit `hold`, `blocked`, stale,
or unknown trust states.

Update the proposal-only contract and capability matrix support paths, then add
focused tests.

## Likely files to touch

- `backend/deesser_vocal_sibilance.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_deesser_vocal_sibilance.py`
- `.ai/briefs/deesser-source-identity-gate-20260605.md`
- `Docs/adr/deesser-source-identity-gate-20260605.md`

## Alternatives considered

- Require source identity calibration for all de-esser inputs immediately.
  Rejected because legacy offline callers do not yet pass replay artifacts.
- Ignore replay trust for de-essing. Rejected because a wrong vocal label can
  steer source-specific dynamics/EQ decisions.
- Duplicate trust-state parsing inside de-esser. Rejected because the shared
  gate keeps blocker names and evidence consistent across downstream modules.

## Risks

- Conservative blocking may hold valid de-esser proposals until replay
  calibration is consistently available.
- The gate currently relies on upstream trust-state strings; future DTO wiring
  should pass structured calibration records.

## Test plan

```bash
PYTHONPATH=backend python -m py_compile \
  backend/deesser_vocal_sibilance.py \
  backend/source_identity_downstream_gate.py \
  backend/processing_proposal_adapters.py \
  backend/project_goal_capability_matrix.py \
  tests/test_deesser_vocal_sibilance.py
PYTHONPATH=backend python -m pytest \
  tests/test_deesser_vocal_sibilance.py \
  tests/test_source_identity_downstream_gate.py \
  tests/test_processing_registry.py \
  tests/test_project_goal_capability_matrix.py -q
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
/opt/homebrew/bin/git diff --check
```

## Where the other agent may disagree

Kimi may argue that de-esser should still emit low-priority proposals when
source identity is held. I would block explicitly untrusted vocal labels first:
de-essing can become EQ or dynamics automation later, so wrong identity should
not silently steer downstream action.

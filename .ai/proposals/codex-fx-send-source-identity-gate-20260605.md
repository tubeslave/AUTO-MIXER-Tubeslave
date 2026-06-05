# Proposal: fx-send-source-identity-gate-20260605

## Short thesis

Use replay-calibrated source identity trust as an FX send proposal blocker so
source-specific ambience and delay decisions cannot rely on explicitly
untrusted source labels.

## What I understood

The FX send layer already checks source family, return headroom, feedback risk,
intelligibility risk, persistence, and current FX state freshness. It still
trusts `source_role` directly. After source identity replay calibration was
added, FX decisions should consume that trust evidence before making
recommendations.

## Proposed solution

Add optional source identity trust fields to `FXSendAnalysisInput` and evaluate
them through `source_identity_downstream_gate`. Add the resulting blockers and
evidence to `FXSendRecommendation`. Keep legacy missing calibration allowed so
existing callers remain compatible, but block explicit `hold`, `blocked`,
stale, or unknown trust states.

Update the proposal-only contract and capability matrix support paths, then add
focused tests.

## Likely files to touch

- `backend/fx_send_proposal.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_fx_send_proposal.py`
- `.ai/briefs/fx-send-source-identity-gate-20260605.md`
- `Docs/adr/fx-send-source-identity-gate-20260605.md`

## Alternatives considered

- Require source identity calibration for all FX send inputs immediately.
  Rejected because legacy offline callers do not yet pass replay artifacts.
- Ignore replay trust for FX sends. Rejected because a wrong label can steer
  send selection and dry-anchor handling.
- Duplicate trust-state parsing inside FX send. Rejected because the shared
  gate keeps blocker names and evidence consistent across downstream modules.

## Risks

- Conservative blocking may hold valid FX proposals until replay calibration is
  consistently available.
- The gate currently relies on upstream trust-state strings; future DTO wiring
  should pass structured calibration records.

## Test plan

```bash
PYTHONPATH=backend python3 -m py_compile \
  backend/fx_send_proposal.py \
  backend/source_identity_downstream_gate.py \
  backend/processing_proposal_adapters.py \
  backend/project_goal_capability_matrix.py \
  tests/test_fx_send_proposal.py
PYTHONPATH=backend python3 -m pytest \
  tests/test_fx_send_proposal.py \
  tests/test_source_identity_downstream_gate.py \
  tests/test_processing_registry.py \
  tests/test_project_goal_capability_matrix.py -q
PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q
/opt/homebrew/bin/git diff --check
```

## Where the other agent may disagree

Kimi may argue that FX sends should still emit low-priority proposals when
source identity is held. I would block explicitly untrusted source labels first:
FX sends can become console writes later, so wrong identity should not silently
steer downstream action.

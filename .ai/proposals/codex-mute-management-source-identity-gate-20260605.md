# Proposal: mute-management-source-identity-gate-20260605

## Short thesis

Use replay-calibrated source identity trust as a mute-management blocker so
mute/unmute proposals cannot rely on explicitly untrusted source labels.

## What I understood

The mute layer already checks signal activity, source family, manual authority,
dropout risk, feedback risk, upcoming cue timing, persistence, and current mute
state freshness. It still trusts `source_role` directly when deciding whether a
source is safe-idle or critical. After source identity replay calibration was
added, mute decisions should consume that trust evidence before making
recommendations.

## Proposed solution

Add optional source identity trust fields to `MuteManagementAnalysisInput` and
evaluate them through `source_identity_downstream_gate`. Add the resulting
blockers and evidence to `MuteManagementRecommendation`. Keep legacy missing
calibration allowed so existing callers remain compatible, but block explicit
`hold`, `blocked`, stale, or unknown trust states.

Update the proposal-only contract and capability matrix support paths, then add
focused tests.

## Likely files to touch

- `backend/mute_management_proposal.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_mute_management_proposal.py`
- `.ai/briefs/mute-management-source-identity-gate-20260605.md`
- `Docs/adr/mute-management-source-identity-gate-20260605.md`

## Alternatives considered

- Require source identity calibration for all mute inputs immediately. Rejected
  because legacy callers do not yet pass replay artifacts.
- Ignore replay trust for mute management. Rejected because a wrong label can
  affect critical-source and safe-idle mute decisions.
- Duplicate trust-state parsing inside mute management. Rejected because the
  shared gate keeps blocker names and evidence consistent across downstream
  modules.

## Risks

- Conservative blocking may hold useful mute-state proposals until replay
  calibration is consistently available.
- The gate currently relies on upstream trust-state strings; future DTO wiring
  should pass structured calibration records.

## Test plan

```bash
PYTHONPATH=backend python3 -m py_compile \
  backend/mute_management_proposal.py \
  backend/source_identity_downstream_gate.py \
  backend/processing_proposal_adapters.py \
  backend/project_goal_capability_matrix.py \
  tests/test_mute_management_proposal.py
PYTHONPATH=backend python3 -m pytest \
  tests/test_mute_management_proposal.py \
  tests/test_source_identity_downstream_gate.py \
  tests/test_processing_registry.py \
  tests/test_project_goal_capability_matrix.py -q
PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q
/opt/homebrew/bin/git diff --check
```

## Where the other agent may disagree

Kimi may argue that emergency mute proposals should bypass source identity
trust. I would keep the gate conservative here because this module is still
proposal-only; a future emergency live path should have its own explicit
operator-safety and hardware-proof contract.

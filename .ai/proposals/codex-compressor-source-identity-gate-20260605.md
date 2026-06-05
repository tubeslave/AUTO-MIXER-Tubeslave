# Proposal: compressor-source-identity-gate-20260605

## Short thesis

Use replay-calibrated source identity trust as a compressor proposal blocker so
source-specific dynamics curves cannot rely on explicitly untrusted source
labels.

## What I understood

The compressor layer already checks crest factor, loudness range, true-peak
headroom, existing gain reduction, bounded makeup, persistence, and current
dynamics state freshness. It still trusts `source_role` directly. After source
identity replay calibration was added, compressor decisions should consume that
trust evidence before making recommendations.

## Proposed solution

Add optional source identity trust fields to `CompressorAnalysisInput` and
evaluate them through `source_identity_downstream_gate`. Add the resulting
blockers and evidence to `CompressorRecommendation`. Keep legacy missing
calibration allowed so existing callers remain compatible, but block explicit
`hold`, `blocked`, stale, or unknown trust states.

Update the proposal-only contract and capability matrix support paths, then add
focused tests.

## Likely files to touch

- `backend/compressor_dynamics_proposal.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_compressor_dynamics_proposal.py`
- `.ai/briefs/compressor-source-identity-gate-20260605.md`
- `Docs/adr/compressor-source-identity-gate-20260605.md`

## Alternatives considered

- Require source identity calibration for all compressor inputs immediately.
  Rejected because legacy offline callers do not yet pass replay artifacts.
- Ignore replay trust for compression. Rejected because a wrong label can steer
  ratio, attack, release, and makeup decisions.
- Duplicate trust-state parsing inside compressor. Rejected because the shared
  gate keeps blocker names and evidence consistent across downstream modules.

## Risks

- Conservative blocking may hold valid compressor proposals until replay
  calibration is consistently available.
- The gate currently relies on upstream trust-state strings; future DTO wiring
  should pass structured calibration records.

## Test plan

```bash
PYTHONPATH=backend python3 -m py_compile \
  backend/compressor_dynamics_proposal.py \
  backend/source_identity_downstream_gate.py \
  backend/processing_proposal_adapters.py \
  backend/project_goal_capability_matrix.py \
  tests/test_compressor_dynamics_proposal.py
PYTHONPATH=backend python3 -m pytest \
  tests/test_compressor_dynamics_proposal.py \
  tests/test_source_identity_downstream_gate.py \
  tests/test_processing_registry.py \
  tests/test_project_goal_capability_matrix.py -q
PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q
/opt/homebrew/bin/git diff --check
```

## Where the other agent may disagree

Kimi may argue that compressor proposals should still emit low-priority
suggestions when source identity is held. I would block explicitly untrusted
source labels first: compressor settings can become console writes later, and
wrong dynamics behavior can quickly damage intelligibility or headroom.

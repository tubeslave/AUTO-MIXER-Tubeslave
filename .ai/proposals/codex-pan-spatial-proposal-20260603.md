# Codex Proposal: pan-spatial-proposal-20260603

## 1. Short thesis

Move panning from lab-only toward assisted automation by adding a transport-free
spatial analyzer that emits pan proposals without becoming a console executor.

## 2. What I understood

The Automixer should eventually pan sources automatically, but pan moves can
hurt mono compatibility, stage realism, and source anchoring. Current live
writes are limited to supervised fader/gain, so panning must first become
evidence and proposal output.

## 3. Proposed solution

Create `backend/pan_spatial_proposal.py` with bounded pan recommendations:
current pan, target pan, delta, confidence, blockers, and evidence. It should
center lead/core/low-frequency sources and only widen non-anchor sources when
stereo-pair side or stage-position evidence exists and mono risk is acceptable.
Register it as `proposal_only`, add a proposal-only contract, and update the
project-goal matrix from lab-only to assisted/proposal-only.

## 4. Likely files to touch

- `backend/pan_spatial_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_pan_spatial_proposal.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/adr/pan-spatial-proposal-20260603.md`

## 5. Alternatives considered

- Promote existing `auto_panner.py`: rejected because it includes apply
  behavior and is still documented as lab/offline.
- Keep panning lab-only: rejected because the goal needs product-visible
  progress toward assisted panning.

## 6. Risks

- Pan moves can damage mono compatibility. The proposal layer blocks widening
  when mono risk is high.
- Without stage context, pan guesses can feel arbitrary. The module requires
  stage-position or stereo-pair evidence except for center-anchor correction.

## 7. Test plan

- Unit tests for center-anchor correction, stereo-pair proposals, high mono-risk
  block, missing-context block, proposal shape, registry status, and
  transport-free source scan.

## 8. Where the other agent may disagree

Kimi may prefer using the existing panner directly. A separate proposal-only
surface is safer because it keeps direct pan writes quarantined and gives the
Product Layer explainable spatial evidence first.

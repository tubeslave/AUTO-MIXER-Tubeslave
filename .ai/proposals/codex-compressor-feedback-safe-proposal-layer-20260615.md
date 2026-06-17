# Codex Proposal: Compressor Feedback-safe Proposal Layer

## Short Thesis

Keep compressor automation proposal-only, but require explicit feedback safety
before emitting makeup-bearing dynamics proposals.

## What I Understood

The long-term Automixer needs automatic compression and console compressor
control. Compression itself can reduce peaks, but makeup gain and changed
dynamics can raise perceived level and feedback risk, so this must stay blocked
from live apply until physical proof exists.

## Proposed Solution

- Add `feedback_risk` to `CompressorAnalysisInput`.
- Block proposals with `feedback_risk_blocks_compressor_makeup` when feedback
  risk is high and the recommendation would add makeup gain.
- Add `feedback_safety` evidence to compressor recommendations.
- Add explicit `old_value` and `rollback_value` sources to operator proposals.
- Update proposal contracts, registry, capability matrix, and backlog.

## Likely Files to Touch

- `backend/compressor_dynamics_proposal.py`
- `backend/processing_proposal_adapters.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_compressor_dynamics_proposal.py`
- `tests/test_project_goal_capability_matrix.py`
- `.ai/project-layer/correction_methods_backlog.md`

## Alternatives Considered

- Leave compressor as-is and only update backlog: rejected because feedback
  safety was named in the backlog and not yet represented as a blocker.
- Promote compressor writes: rejected because dynamics readback, rollback,
  feedback proof, and hardware proof are missing.

## Risks

- Conservative feedback blocking can reduce proposal volume for loud vocal or
  open-mic sources.
- Some callers may need to start passing feedback-risk evidence for richer
  proposals.

## Test Plan

- Compile touched Python files.
- Run focused compressor, registry, and capability matrix tests.
- Run full project pytest and no-WING release checks before calling the
  increment validated.

## Where The Other Agent May Disagree

Kimi may prefer feedback risk to block every compressor proposal, even without
makeup. This pass blocks the direct level-increase risk while keeping no-live
authority and leaving stricter promotion policy for future physical proof.

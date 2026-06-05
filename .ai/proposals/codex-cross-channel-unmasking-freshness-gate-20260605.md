# Proposal: cross-channel-unmasking-freshness-gate-20260605

## Short thesis

Wrap cross-adaptive mirror EQ in a proposal-only freshness gate so cross-channel
unmasking can move toward Assisted Mixing without implying live EQ authority.

## What I understood

The target Automixer needs to solve masking between channels, not only analyze
single sources. The existing `CrossAdaptiveEQ.calculate_corrections` can already
produce mirror-EQ cuts and boosts, but those corrections are too raw to promote:
they need current-state freshness, linked-pair evidence, readback, and rollback
requirements before operator review or any future supervised proof.

## Proposed solution

Keep `calculate_corrections` backward-compatible. Add
`analyze_unmasking_proposals`, which normalizes optional current-state freshness,
blocks stale context, and emits non-queueable proposal records for fresh or
legacy missing freshness. Each action should include masking pair, proposed EQ
value, old-value source, rollback source, bounded gain evidence, sum-bus check
requirement, EQ-band readback requirement, and `live_write_allowed=False`.

Update `processing_registry`, `processing_proposal_adapters`, and
`project_goal_capability_matrix` so the source-of-truth contract matches the new
behavior.

## Likely files to touch

- `backend/cross_adaptive_eq.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_cross_adaptive_eq.py`
- `.ai/briefs/cross-channel-unmasking-freshness-gate-20260605.md`
- `Docs/adr/cross-channel-unmasking-freshness-gate-20260605.md`

## Alternatives considered

- Change `calculate_corrections` to return proposal records directly. Rejected
  because it would risk breaking existing live/shared/offline consumers.
- Require freshness for every legacy call immediately. Rejected because upstream
  call sites do not all pass snapshot freshness yet.
- Allow stale masking context but mark lower confidence. Rejected because linked
  EQ moves can damage balance when role or band-energy evidence is stale.

## Risks

- Missing freshness remains a legacy compatibility gap until upstream integration
  is complete.
- Proposal output can grow quickly in dense mixes; consumers should still apply
  their own display limits and operator-review sorting.

## Test plan

- `PYTHONPATH=backend python -m py_compile backend/cross_adaptive_eq.py backend/processing_registry.py backend/processing_proposal_adapters.py backend/project_goal_capability_matrix.py tests/test_cross_adaptive_eq.py`
- `PYTHONPATH=backend python -m pytest tests/test_cross_adaptive_eq.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`

## Where the other agent may disagree

Kimi may argue that legacy missing freshness should block all mirror-EQ output.
I would keep legacy missing freshness allowed for now because this is a
compatibility wrapper, not a live executor. Freshness should become mandatory
when every upstream caller can pass current band-energy and role evidence.

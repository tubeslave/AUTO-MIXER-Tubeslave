# Codex Proposal: Corrective EQ Protected-band Proposal Layer

## Short Thesis

Add a transport-free corrective EQ proposal layer that can explain safe channel
EQ intent to an operator without creating live EQ writes or queue imports.

## What I Understood

The long-term Automixer needs automatic equalization, but EQ writes are still
unsafe for live promotion. They can damage feedback margins, overwrite an
operator's manual work, or mutate the wrong EQ band if the current console state
is stale or misidentified.

## Proposed Solution

- Add `backend/corrective_eq_proposal.py` as a Product Layer proposal source.
- Require fresh current EQ state, fresh phase context, band id, and matching
  band frequency before a proposal can be emitted.
- Preserve feedback-notch and manual bands by turning those cases into
  `analysis_only`.
- Bound boost/cut deltas with true-peak headroom and feedback-risk evidence.
- Include explicit `old_value` and `rollback_value` readback sources in every
  operator proposal.
- Register the method in processing contracts, registry, capability matrix, and
  correction backlog.

## Likely Files to Touch

- `backend/corrective_eq_proposal.py`
- `backend/processing_proposal_adapters.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_corrective_eq_proposal.py`
- `tests/test_project_goal_capability_matrix.py`
- `.ai/project-layer/correction_methods_backlog.md`

## Alternatives Considered

- Extend `auto_eq` directly: rejected because its direct apply/reset surfaces
  are intentionally fail-closed and should stay quarantined evidence.
- Promote EQ into the live executor: rejected because EQ-band readback,
  rollback, feedback proof, and physical WING proof are missing.

## Risks

- Strict band identity and freshness gates can reduce proposal volume until
  readback evidence exists.
- Legacy EQ analysis may need adapter work before it can feed the new proposal
  layer.

## Test Plan

- Compile touched Python files.
- Run focused corrective-EQ, AutoEQ fail-closed, registry, and capability matrix
  tests.
- Run full project pytest and no-WING release checks before calling the
  increment validated.

## Where The Other Agent May Disagree

Kimi may prefer cross-channel EQ and corrective EQ to share one planner. The
narrower layer is better for this pass because it closes single-channel
protected-band safety without widening the already existing cross-adaptive EQ
contract.

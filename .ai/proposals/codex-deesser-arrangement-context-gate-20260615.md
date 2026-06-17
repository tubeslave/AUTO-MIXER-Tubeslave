# Codex Proposal: De-esser Arrangement Context Gate

## Short thesis

The vocal de-esser should be arrangement-aware before it emits proposal-only
processing recommendations.

## What I understood

The current de-esser recognizes vocal sibilance and blocks unsafe live apply,
but the long-term goal requires decisions based on arrangement context. A
backing or tucked vocal should not receive the same aggressive processing as a
lead vocal, and stale arrangement context should prevent confident downstream
recommendations.

## Proposed solution

Add `arrangement_role` and `arrangement_context_cache_freshness` to
`DeesserAnalysisInput`. Derive a de-esser profile from arrangement/source role:
lead uses the existing behavior, general vocal is mildly conservative, and
tucked/backing/double/harmony vocals use a less aggressive threshold/ratio
profile. If provided arrangement context freshness is not fresh, demote the
recommendation to `analysis_only`.

## Likely files to touch

- `backend/deesser_vocal_sibilance.py`
- `tests/test_deesser_vocal_sibilance.py`
- `backend/processing_proposal_adapters.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `.ai/project-layer/correction_methods_backlog.md`
- `Docs/adr/`
- `Docs/reports/`

## Alternatives considered

Keeping source-role-only de-essing was rejected because it ignores available
arrangement evidence. Making de-esser output queueable was rejected because
hardware readback, rollback, and live authority are not proven.

## Risks

Tucked/backing vocal proposals may be less aggressive than an operator expects.
That is acceptable for proposal-only output and can be tuned after replayed
operator decisions.

## Test plan

Run focused de-esser/capability tests, related proposal contract tests, then
the full test suite.

## Where the other agent may disagree

Kimi may argue the first WING read-only proof is still the higher operational
blocker. That is true, but hardware is unavailable in this environment and this
work stays explicitly offline/proposal-only.

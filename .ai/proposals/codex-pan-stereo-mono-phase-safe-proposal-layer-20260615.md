# Codex Proposal: Pan Stereo Mono/Phase-safe Proposal Layer

## Short Thesis

Use the existing transport-free pan analyzer as the implementation surface, but
make non-center pan moves require fresh mono/phase safety evidence and render
operator proposals with explicit readback and rollback blockers.

## What I Understood

The long-term Automixer needs autonomous panning, but live pan writes are not
approved. The current useful move is to produce better operator-facing pan
proposals while keeping the WING MVP live boundary unchanged.

## Proposed Solution

- Extend `PanAnalysisInput` with mono/phase evidence fields.
- Block non-center pan moves when mono/phase evidence is missing, stale, or
  unsafe.
- Add `old_value` and `rollback_value` sources to pan proposals.
- Update processing contracts, registry, capability matrix, and backlog.

## Likely Files to Touch

- `backend/pan_spatial_proposal.py`
- `backend/processing_proposal_adapters.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_pan_spatial_proposal.py`
- `tests/test_project_goal_capability_matrix.py`
- `.ai/project-layer/correction_methods_backlog.md`

## Alternatives Considered

- Build a separate panner adapter: rejected because `pan_spatial_proposal`
  already owns the transport-free proposal surface.
- Allow non-center proposals with missing phase evidence: rejected because it
  weakens mono compatibility and live PA translation safety.

## Risks

- Stricter phase evidence can reduce proposal count until telemetry is present.
- Existing legacy callers may still produce analysis-only results until they add
  mono/phase evidence.

## Test Plan

- Compile touched Python files.
- Run `tests/test_pan_spatial_proposal.py`,
  `tests/test_project_goal_capability_matrix.py`, and registry tests.
- Run the full project pytest before treating the increment as validated.

## Where The Other Agent May Disagree

Kimi may prefer a broader stereo-field optimizer. The narrower proposal is
better for this turn because it closes a safety contract without live-write
scope creep.

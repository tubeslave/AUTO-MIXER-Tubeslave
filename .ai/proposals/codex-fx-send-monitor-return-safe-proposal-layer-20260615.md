# Codex Proposal: FX Send Monitor/Return-safe Proposal Layer

## Short Thesis

Use the existing transport-free FX send analyzer, but require explicit monitor
safety and return-level safety evidence before any operator-facing FX/send
proposal is emitted.

## What I Understood

The long-term Automixer needs automatic FX management and send tuning. Those
actions can affect front-of-house depth, lyric intelligibility, feedback risk,
and monitor mixes, so they must stay proposal-only until physical readback and
rollback proof exist.

## Proposed Solution

- Extend `FXSendAnalysisInput` with monitor exposure, monitor risk, and monitor
  freshness evidence.
- Add `monitor_safety` and `return_level_safety` evidence objects.
- Add explicit `old_value` and `rollback_value` sources to FX proposals.
- Keep high monitor risk or stale monitor context as `analysis_only`.
- Update processing contracts, registry, capability matrix, and backlog.

## Likely Files to Touch

- `backend/fx_send_proposal.py`
- `backend/processing_proposal_adapters.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_fx_send_proposal.py`
- `tests/test_project_goal_capability_matrix.py`
- `.ai/project-layer/correction_methods_backlog.md`

## Alternatives Considered

- Build a new FX planner: rejected because `fx_send_proposal` already owns the
  transport-free proposal surface.
- Allow monitor exposure without freshness evidence: rejected because live
  monitor changes can surprise performers and increase feedback risk.

## Risks

- Stricter monitor evidence can reduce FX proposal volume until telemetry is
  available.
- Some legacy callers may remain analysis-only until they pass monitor context.

## Test Plan

- Compile touched Python files.
- Run focused FX, registry, and capability matrix tests.
- Run full project pytest before calling the increment validated.

## Where The Other Agent May Disagree

Kimi may prefer a broader FX scene/return planner. The narrower change is better
for this pass because it closes the safety contract without broad live scope.

# Codex Proposal: Master / Output PA-feed-safe Proposal Layer

## Short Thesis

Strengthen the existing master bus proposal layer so it can close the
Master / Output backlog gap without granting live authority: true-peak and
PA-feed safety become required evidence for proposal-only recommendations.

## What I Understood

The Automixer goal requires automatic master-bus and output optimization, but
master and matrix/output moves affect the entire PA. The current safe increment
is not live output control; it is a non-queueable recommendation that refuses to
act without fresh output-feed evidence and reference-feed exclusion.

## Proposed Solution

- Extend `MasterBusAnalysisInput` with output-feed role, PA-feed readback,
  PA-feed safety, PA-feed freshness, and reference-feed flags.
- Block proposal-only output when PA-feed evidence is missing, stale, unsafe, or
  marked as reference/measurement/capture.
- Keep all proposals `can_import_to_queue: false` and `can_apply_live: false`.
- Add PA-feed safety and reference-feed exclusion to the proposal contract.
- Update registry, capability matrix, backlog, docs, and tests.

## Likely Files To Touch

- `backend/master_bus_optimization_proposal.py`
- `backend/processing_proposal_adapters.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_master_bus_optimization_proposal.py`
- `tests/test_project_goal_capability_matrix.py`
- `.ai/project-layer/correction_methods_backlog.md`

## Alternatives Considered

- Add a separate output optimizer module: rejected because the existing master
  bus proposal layer already owns loudness, true-peak, tonal, and glue dynamics
  recommendations.
- Promote master/output writes: rejected because WING read-only proof,
  physical readback, rollback, emergency stop, and operator approval remain
  incomplete.

## Risks

- Existing callers that omit PA-feed safety now receive analysis-only output.
- Physical PA-feed evidence still needs adapters from Product Layer snapshots
  or operator-provided readback artifacts.

## Test Plan

- Compile touched Python files.
- Run focused master-output proposal, proposal-contract, registry, and
  capability matrix tests.
- Run full project pytest and no-WING release checks before calling the
  increment validated.

## Where The Other Agent May Disagree

Kimi may prefer a separate output-feed read contract first. This change is still
useful because it makes the current master proposal layer fail closed until that
contract evidence exists.

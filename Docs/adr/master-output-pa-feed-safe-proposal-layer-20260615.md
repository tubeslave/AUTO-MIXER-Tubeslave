# ADR: Master / Output PA-feed-safe Proposal Layer

Date: 2026-06-15
Status: accepted

## Context

The Automixer goal requires automatic master-bus and output optimization. The
correction backlog still listed Master / Output as `offline_method_needed`, even
though a proposal-only master bus analyzer already existed. The missing safety
boundary was explicit PA/output-feed proof and reference-feed exclusion.

## Options Considered

1. Leave Master / Output as `offline_method_needed`.
2. Strengthen the existing master proposal layer with PA-feed safety blockers.
3. Add live master/output writes or query WING output paths.

## Decision

Use option 2. `master_bus_optimization_proposal` now requires PA-feed readback,
PA-feed safety, fresh PA-feed cache status, and reference-feed exclusion before
it can emit proposal-only output.

## Why This Won

It advances output optimization while preserving the live boundary. The method
can now express loudness, true-peak, tonal, glue-compression, and output-feed
safety evidence in one non-queueable proposal artifact.

## Rejected Alternatives

- A separate output optimizer would duplicate master-bus proposal logic.
- Querying hardware or promoting output writes is premature before WING
  read-only proof, physical fixtures, rollback, emergency stop, and manual
  review are complete.

## Implementation Plan

- Extend `MasterBusAnalysisInput` with PA/output-feed evidence fields.
- Block missing/stale/unsafe PA feed evidence and reference/measurement feeds.
- Add PA-feed and reference-feed evidence to proposal contracts.
- Update registry, capability matrix, backlog, and focused tests.

## Test Plan

- `PYTHONPATH=backend python -m py_compile ...`
- `PYTHONPATH=backend python -m pytest -q tests/test_master_bus_optimization_proposal.py tests/test_project_goal_capability_matrix.py tests/test_processing_registry.py tests/test_agentic_method_foundation.py`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend`

## Risks And Rollback

Some callers may now get analysis-only results until PA-feed safety evidence is
provided. Rollback is removing the new evidence fields, blockers, contract
requirements, registry/capability text, and tests. No live transport code is
touched.

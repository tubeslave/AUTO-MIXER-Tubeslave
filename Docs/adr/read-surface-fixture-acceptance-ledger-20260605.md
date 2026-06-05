# ADR: read-surface-fixture-acceptance-ledger-20260605

## Context

The read-surface physical readback artifact review queue aggregates candidate
artifacts from per-surface intake gates. A valid candidate still needs operator
review. The repository needs a safe place to record accepted and rejected
candidate decisions without implying that fixtures have been closed or that live
authority has changed.

## Options Considered

1. Keep acceptance decisions outside the repository evidence model.
2. Add acceptance fields directly to queue items.
3. Add a separate read-only fixture acceptance ledger.

## Decision

Use option 3. Add a separate read-only fixture acceptance ledger that validates
operator acceptance records against review queue candidates and publishes its
status in no-WING release evidence.

## Why This Won

A separate ledger keeps the lifecycle clear: intake finds candidates, the queue
aggregates candidates, the ledger records operator accept/reject decisions, and
a future manual fixture-closure step must still decide whether to update the
physical fixture baseline.

## Rejected Alternatives

- External-only acceptance was rejected because release artifacts would not show
  review progress.
- Queue-embedded acceptance was rejected because it mixes artifact aggregation
  with operator decision records.
- Automatic fixture closure was rejected because acceptance records alone do not
  prove a safe fixture baseline update.

## Implementation Plan

1. Add a transport-free backend ledger module.
2. Validate surface IDs, artifact source IDs, contract method IDs, decisions,
   reviewer metadata, and unsafe authority flags.
3. Register the ledger as `shadow_only` read-only evidence.
4. Add no-WING release-check validation and JSON/Markdown summaries.
5. Cover accepted, rejected, invalid, and unreviewed candidate paths in tests.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_read_surface_physical_readback_fixture_acceptance_ledger.py tests/test_no_wing_release_check.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/read_surface_physical_readback_fixture_acceptance_ledger.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks And Rollback

Risk: accepted records may be mistaken for fixture closure or live readiness.
Mitigation: the ledger and release summaries force
`fixture_closure_allowed_by_acceptance=false`,
`physical_fixture_gap_remains=true`, `live_authority_granted=false`, and
`full_autonomous_ready=false`.

Rollback: remove the ledger validator from `run_release_checks()` and remove
the registry/capability row. The artifact review queue and per-surface intake
gates can remain unchanged.

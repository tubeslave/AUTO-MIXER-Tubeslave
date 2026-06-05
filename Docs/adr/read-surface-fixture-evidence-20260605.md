# ADR: read-surface-fixture-evidence-20260605

## Context

Full autonomous console control requires reliable read coverage before write
promotion. The repository has read-surface inventory rows, remediation actions,
and cached Product Layer DTO fixture tests. It lacks an explicit snapshot that
shows which read surfaces have fixture evidence and which proof gaps remain.

## Options considered

1. Keep fixture evidence implicit in pytest names.
2. Add a declarative read-only fixture evidence snapshot.
3. Treat cached DTO fixtures as full readback readiness.

## Decision

Use option 2. Add `read_surface_fixture_evidence` as a shadow/evaluation method
and include its snapshot in the processing registry. It maps read-surface IDs to
fixture paths, test names, covered contracts, and remaining proof gaps. It never
executes tests, queries hardware, or mutates console state.

## Why this won

This makes read coverage progress visible while preserving the safety boundary.
The snapshot can say "cached DTO fixture exists" and "physical readback proof is
still missing" at the same time, which is the truthful state needed before
claiming full console awareness.

## Rejected alternatives

Keeping fixture evidence implicit was rejected because it hides readiness gaps
from Product Layer snapshots. Treating cached DTO fixtures as readiness proof was
rejected because autonomous console control still needs physical readback,
stale-cache proof, scoped scene/preset diffs, and dLive/WING parity mapping.

## Implementation plan

- Add a declarative fixture evidence map for current read surfaces.
- Build a read-only snapshot with status counts and physical fixture blockers.
- Register the method as `shadow_only`, outside main live executors.
- Include the snapshot in `build_processing_registry_snapshot`.
- Reference fixture evidence from the `console_read_parameters` goal row.
- Add focused tests for safety flags, known fixture rows, registry visibility,
  and transport-free behavior.

## Test plan

```bash
PYTHONPATH=backend python -m py_compile \
  backend/read_surface_fixture_evidence.py \
  backend/processing_registry.py \
  backend/project_goal_capability_matrix.py \
  tests/test_read_surface_fixture_evidence.py
PYTHONPATH=backend python -m pytest \
  tests/test_read_surface_fixture_evidence.py \
  tests/test_processing_registry.py \
  tests/test_project_goal_capability_matrix.py \
  tests/test_read_surface_inventory.py \
  tests/test_read_surface_remediation_plan.py -q
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
/opt/homebrew/bin/git diff --check
```

## Risks and rollback

Risk: the declarative map can drift if tests are renamed. Rollback is removing
the evidence module, registry entry, goal-row references, and focused tests. No
live transport path is added.

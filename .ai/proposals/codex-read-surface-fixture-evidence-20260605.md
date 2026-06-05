# Proposal: read-surface-fixture-evidence-20260605

## Short thesis

Expose read-surface fixture evidence as a read-only registry snapshot so the
Automixer can track DTO proof without overstating physical console readiness.

## What I understood

The target system must read and analyze full console state before broad
autonomous control is credible. Current Product Layer DTOs already have cached
fixture tests, but the registry does not show which surfaces have fixture
coverage and which still need physical readback, stale-cache, scoped diff, or
dLive/WING parity proof.

## Proposed solution

Add `backend/read_surface_fixture_evidence.py` with a declarative evidence map.
For each read surface, expose fixture paths, test names, covered contracts, and
remaining gaps. Build a snapshot that is explicitly read-only, does not execute
tests, does not query hardware, and sets `full_autonomous_ready=False`.

Register the module as a `shadow_only` evaluation method and include the snapshot
inside `build_processing_registry_snapshot`. Update the `console_read_parameters`
capability row to reference this evidence layer.

## Likely files to touch

- `backend/read_surface_fixture_evidence.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_read_surface_fixture_evidence.py`
- `.ai/briefs/read-surface-fixture-evidence-20260605.md`
- `Docs/adr/read-surface-fixture-evidence-20260605.md`

## Alternatives considered

- Fold this into `read_surface_remediation_plan`. Rejected because remediation
  asks what to do next, while fixture evidence answers what is already covered.
- Parse pytest dynamically. Rejected because Product Layer snapshots must be
  deterministic and should not execute or inspect test runtime state.
- Treat cached DTO fixtures as readiness proof. Rejected because physical
  readback and parity proof are still missing.

## Risks

- The evidence map is declarative and can drift if tests are renamed. Focused
  tests should catch important rows, and future work can automate consistency.
- This does not reduce the need for actual physical fixture artifacts.

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

## Where the other agent may disagree

Kimi may prefer implementing physical readback fixtures immediately. That is the
right next proof step, but this evidence layer makes the remaining proof gap
explicit and prevents cached DTO tests from being misread as hardware readiness.

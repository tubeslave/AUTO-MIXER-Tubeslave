# project-goal-capability-matrix-20260603

## 1. Context

The active project goal names a full AI Mixing Engineer: it must analyze
multichannel audio, identify sources and roles, perform the major processing
families, adapt during concerts, control the digital console, support four
runtime modes, and learn from long-term memory. Current repo surfaces cover
some parts as canonical, many as proposal/shadow/offline, and some not at all.

## 2. Options considered

- Keep tracking progress only through scattered docs and registry entries.
- Add product-goal rows directly to the processing registry.
- Add a standalone read-only project-goal capability matrix.

## 3. Decision

Add `backend/project_goal_capability_matrix.py` with tested rows for every
explicit project-goal capability. Each row records status, mode ceiling,
registered method IDs, target console surfaces, current live write kinds,
blocker, and next step.

Expose the matrix inside `build_processing_registry_snapshot()` so Product
Layer consumers can judge processing methods, read/write inventories, safety
coverage, and long-term goal coverage from one read-only snapshot.

## 4. Why this won

The matrix turns the long-term objective into a machine-checkable roadmap
without widening live writes. It keeps target autonomy visible while still
showing that current live writes are only narrow supervised fader/gain paths.

## 5. Rejected alternatives

Scattered docs are too easy to drift. Folding this into `processing_registry.py`
would mix method inventory with product-goal accountability and would touch an
already dirty file.

## 6. Implementation plan

1. Create the read-only capability matrix module.
2. Reference registered processing methods where a capability has an existing
   implementation surface.
3. Mark missing capabilities explicitly and promote them only through
   proposal-only surfaces first.
4. Expose the matrix through the processing registry snapshot without changing
   live write authority.
5. Add tests for coverage, method references, fail-closed safety, and blocked
   full autonomous/broad console writes.

## 7. Test plan

- `PYTHONPATH=backend python -m pytest tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python -m pytest tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- Compile/smoke checks with the available Python runtime if full test
  dependencies are unavailable.

## 8. Risks and rollback

The main risk is stale roadmap metadata. The method-reference test reduces typo
drift, and future feature work should update this matrix when status changes.
Rollback is a clean removal of the matrix, test, brief, proposal, and ADR; no
transport or live-write code is touched.

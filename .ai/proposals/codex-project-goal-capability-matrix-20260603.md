# Codex Proposal: project-goal-capability-matrix-20260603

## 1. Short thesis

Make the long-term Automixer goal measurable by adding a read-only capability
matrix that maps each required product capability to current methods, gaps,
blockers, and next evidence.

## 2. What I understood

The project is no longer just a gain-staging or cockpit task. Every future
algorithm and interface should be judged by whether it moves toward a fully
autonomous AI Mixing Engineer while preserving safety limits.

## 3. Proposed solution

Create `backend/project_goal_capability_matrix.py` as a declarative inventory.
It should reference existing `processing_registry` method IDs where possible,
mark missing capabilities explicitly, and report that the matrix is read-only
with no live-write widening.

## 4. Likely files to touch

- `backend/project_goal_capability_matrix.py`
- `tests/test_project_goal_capability_matrix.py`
- `.ai/briefs/project-goal-capability-matrix-20260603.md`
- `Docs/adr/project-goal-capability-matrix-20260603.md`

## 5. Alternatives considered

- Put the mapping directly into `processing_registry.py`: rejected because that
  file is already dirty and tracks methods/pipelines, not product-goal coverage.
- Only document the gaps in prose: rejected because tests cannot enforce prose.

## 6. Risks

- The matrix may lag behind implementation. Mitigate by testing method-ID
  references and using the matrix as a required roadmap surface.
- A target console surface may be mistaken as currently allowed. Mitigate by
  separating `target_console_surfaces` from `current_live_write_kinds`.

## 7. Test plan

- Run focused pytest for the new matrix.
- Run compile/smoke checks when full project dependencies are unavailable.

## 8. Where the other agent may disagree

Kimi may prefer integrating this into the Product Layer UI immediately. The
standalone read-only module is a safer first step because it avoids dirty UI and
write-surface files while making the roadmap testable.

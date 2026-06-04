# Codex Proposal: read-surface-inventory-20260603

## 1. Short thesis

Add a declarative read-surface inventory so the project can measure progress
toward full console-state awareness without touching mixer transport.

## 2. What I understood

The system must eventually read all console parameters before it can safely
control them. Today, cached state and helper methods exist, but there is no
single contract that says what is complete, partial, or missing.

## 3. Proposed solution

Create `backend/read_surface_inventory.py` with rows for raw state, channels,
processing, buses/DCA/main, matrices, FX/inserts, routing/sends, snapshots,
presets, and dLive. Keep it read-only and connect it to the project-goal matrix.

## 4. Likely files to touch

- `backend/read_surface_inventory.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_read_surface_inventory.py`
- `Docs/adr/read-surface-inventory-20260603.md`

## 5. Alternatives considered

- Put the read inventory into `write_surface_inventory.py`: rejected because
  read completeness and write authority are different safety questions.
- Query the mixer live to discover coverage: rejected because this turn should
  remain transport-free and deterministic.

## 6. Risks

- Inventory rows can become stale. Tests ensure rows stay unique, read-only, and
  tied to the project-goal capability.
- Partial reads may be overclaimed. Snapshot summary explicitly says coverage is
  inventoried but incomplete.

## 7. Test plan

- Unit tests for inventory uniqueness, read-only safety, goal coverage, partial
  status, and transport-free source scan.

## 8. Where the other agent may disagree

Kimi may prefer generating the inventory dynamically from WING docs. A
declarative first pass is safer because it avoids live transport and creates a
stable contract for future dynamic proof.

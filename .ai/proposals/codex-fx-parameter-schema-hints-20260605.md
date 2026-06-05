# Proposal: fx-parameter-schema-hints-20260605

## 1. Short thesis

Add read-only FX parameter schema hints to the Product Layer so cached FX
parameters become explainable without creating any FX apply path.

## 2. What I understood

The autonomous mixing goal needs FX control, but FX rack, insert, and send writes
are still unsafe as direct operations. Current snapshots can read cached numeric
parameter slots, but they do not say whether parameter 1 means decay, delay
time, feedback, or something unknown.

## 3. Proposed solution

Add a small generic schema map for the same FX families already used by the
AutoFX planner: reverb, delay, and modulation. For mapped families, attach
`semantic_name`, `unit`, and conservative `safe_range` metadata to each cached
parameter row. For unknown models, expose `generic_numeric` with raw parameter
names so the UI and proposal layer can explain uncertainty.

The snapshot must continue to report `physical_model_schema_verified=False` and
`live_write_authority=False`.

## 4. Likely files to touch

- `backend/operator_product_state.py`
- `backend/read_surface_inventory.py`
- `backend/read_surface_fixture_evidence.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_product_state.py`
- `tests/test_read_surface_inventory.py`
- `tests/test_read_surface_fixture_evidence.py`
- `tests/test_project_goal_capability_matrix.py`
- `.ai/briefs/fx-parameter-schema-hints-20260605.md`
- `Docs/adr/fx-parameter-schema-hints-20260605.md`

## 5. Alternatives considered

Implement physical WING model schemas first. Better final state, but it needs
hardware fixtures and protocol validation. Generic hints are a low-risk
intermediate layer and make the remaining proof gap explicit.

Leave parameters raw. Rejected because raw numeric slots block explainable
assisted FX proposals and increase the chance of unsafe assumptions later.

## 6. Risks

Generic schemas can be mistaken for verified WING model schemas. Mitigation:
name them schema hints, keep physical verification false, and preserve the
inventory blocker for complete model-specific schemas.

## 7. Test plan

- Compile changed Python files.
- Focused tests for FX snapshot schema hints, read-surface inventory, fixture
  evidence, and project-goal capability matrix.
- Broader Product Layer/read-surface/registry tests.
- Full suite: `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`.

## 8. Where the other agent may disagree

Kimi may prefer skipping generic hints until physical WING FX schemas exist. My
counterpoint is that the hint layer is explicitly non-authoritative and improves
assisted-mode explainability without widening live writes.

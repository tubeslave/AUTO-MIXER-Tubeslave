# ADR: fx-parameter-schema-hints-20260605

## 1. Context

`fx_inserts_snapshot` exposes cached FX slot metadata, numeric FX parameters,
insert assignments, and cache freshness without querying or mutating the console.
For autonomous FX management, raw numeric parameter slots are not explainable
enough: proposal logic needs to distinguish decay, predelay, feedback, width, and
generic unknown controls.

FX and insert apply paths remain quarantined because they can rewrite live
processing chains.

## 2. Options considered

1. Add read-only generic FX parameter schema hints.
2. Build complete physical WING model-specific FX schemas first.
3. Keep parameters as raw numeric slots only.

## 3. Decision

Use option 1. Add generic schema hints for reverb, delay, and modulation
families. Unknown models fall back to `generic_numeric` instead of being
mislabelled.

## 4. Why this won

This makes cached FX parameters explainable for assisted proposals and UI review
without claiming physical proof. It also matches the existing AutoFX planner's
generic parameter semantics while keeping direct FX writes blocked.

## 5. Rejected alternatives

Complete physical WING schemas are still required, but they need hardware
fixtures and model validation. Raw numeric slots are safer than false labels, but
they are not good enough for explainable FX recommendations.

## 6. Implementation plan

1. Add generic FX family schema helpers to `backend/operator_product_state.py`.
2. Attach schema hints to cached parameter rows and slot/snapshot summaries.
3. Update read-surface inventory, fixture evidence, and project-goal metadata.
4. Add tests for mapped and unknown FX models.

## 7. Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/operator_product_state.py backend/read_surface_inventory.py backend/read_surface_fixture_evidence.py backend/project_goal_capability_matrix.py tests/test_operator_product_state.py tests/test_read_surface_inventory.py tests/test_read_surface_fixture_evidence.py tests/test_project_goal_capability_matrix.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_fx_inserts_snapshot_publishes_cached_fx_and_insert_state_without_queries tests/test_operator_product_state.py::test_fx_inserts_snapshot_uses_generic_schema_for_unknown_fx_models tests/test_operator_product_state.py::test_fx_inserts_snapshot_marks_missing_cache_without_queries tests/test_read_surface_inventory.py::test_fx_insert_read_surface_tracks_cache_freshness_without_write_authority tests/test_read_surface_fixture_evidence.py::test_read_surface_fixture_evidence_snapshot_tracks_cached_dto_fixtures_only tests/test_project_goal_capability_matrix.py::test_fx_is_tracked_as_transport_free_proposal_layer -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_read_surface_inventory.py tests/test_read_surface_fixture_evidence.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## 8. Risks and rollback

Risk: generic hints may be mistaken for verified model schemas. The snapshot
therefore reports `physical_model_schema_verified=False`,
`live_write_authority=False`, `fx_apply_allowed=False`, and
`insert_assignment_allowed=False`.

Rollback is removing the schema helper, parameter metadata, inventory/evidence
updates, tests, and this ADR. No live write path is introduced.

# ADR: fx-send-snapshot-schema-bridge-20260605

## 1. Context

`fx_send_proposal` can block recommendations when FX parameter schema evidence is
generic, missing, or mismatched. The canonical read-only source for that evidence
is `fx_inserts_snapshot`, which exposes cached FX slot schema hints, cache
freshness, and safety flags.

## 2. Options considered

1. Add a transport-free bridge from `fx_inserts_snapshot` to FX-send schema
   evidence.
2. Require all callers to manually normalize schema evidence.
3. Import Product Layer snapshot builders into `fx_send_proposal`.

## 3. Decision

Use option 1. `FXSendAnalysisInput` accepts an optional
`current_fx_inserts_snapshot`, and `fx_send_proposal` derives scoped schema
evidence from it.

## 4. Why this won

The bridge connects proposal decisions to the same read-only state shown to the
operator, while keeping `fx_send_proposal.py` transport-free and write-free. It
also avoids duplicating schema selection logic across callers.

## 5. Rejected alternatives

Manual caller normalization is fragile. Importing Product Layer builders would
blur module boundaries and make an analysis/proposal module responsible for
snapshot construction.

## 6. Implementation plan

1. Add `current_fx_inserts_snapshot` to `FXSendAnalysisInput`.
2. Add `fx_parameter_schema_from_snapshot(snapshot, fx_type)`.
3. Derive schema evidence from the snapshot when explicit schema evidence is not
   supplied.
4. Keep generic/unknown snapshot schemas blocked by the existing schema gate.
5. Update registry metadata and tests.

## 7. Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/fx_send_proposal.py backend/processing_registry.py tests/test_fx_send_proposal.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_fx_send_proposal.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_fx_send_proposal.py tests/test_processing_registry.py tests/test_operator_product_state.py tests/test_read_surface_inventory.py tests/test_read_surface_fixture_evidence.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## 8. Risks and rollback

Risk: cached snapshot evidence may be mistaken for hardware proof. The bridge
records snapshot freshness and safety flags, and FX proposals remain
`can_apply_live=False` and `can_import_to_queue=False`.

Rollback is removing the snapshot field, helper, tests, registry note, and this
ADR. No live write path is introduced.

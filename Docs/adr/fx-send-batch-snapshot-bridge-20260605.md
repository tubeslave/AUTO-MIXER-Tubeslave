# ADR: fx-send-batch-snapshot-bridge-20260605

## 1. Context

The FX-send proposal layer is proposal-only and write-free. It can derive schema
evidence from a read-only `fx_inserts_snapshot` for a single input, but real
multi-channel recommendation flows need to process many channel analyses against
one current FX snapshot.

## 2. Options considered

1. Add a shared snapshot argument to `build_fx_send_proposals()`.
2. Require callers to attach snapshot evidence to every input.
3. Mutate analysis inputs in place before analysis.

## 3. Decision

Use option 1. Add keyword-only `current_fx_inserts_snapshot` and propagate it to
inputs that do not already carry explicit schema or snapshot evidence.

## 4. Why this won

It keeps the interface backward-compatible, avoids repeated caller boilerplate,
and preserves deterministic frozen input objects by using `dataclasses.replace`.

## 5. Rejected alternatives

Manual per-input enrichment is easy to wire inconsistently. In-place mutation is
not compatible with frozen dataclasses and would weaken evidence traceability.

## 6. Implementation plan

1. Add optional `current_fx_inserts_snapshot` to `build_fx_send_proposals()`.
2. Use `replace()` to enrich only inputs that lack explicit schema/snapshot
   evidence.
3. Add mapped and generic shared-snapshot tests.
4. Update registry metadata and council artifacts.

## 7. Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/fx_send_proposal.py backend/processing_registry.py tests/test_fx_send_proposal.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_fx_send_proposal.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_fx_send_proposal.py tests/test_processing_registry.py tests/test_operator_product_state.py tests/test_read_surface_inventory.py tests/test_read_surface_fixture_evidence.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## 8. Risks and rollback

Risk: a stale shared snapshot could affect many recommendations. Existing
freshness/schema blockers still run per recommendation, and all FX proposals
remain `can_apply_live=False` and `can_import_to_queue=False`.

Rollback is removing the keyword argument, `replace()` enrichment, tests,
registry note, and this ADR. No live write path is introduced.

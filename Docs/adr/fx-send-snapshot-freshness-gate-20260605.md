# ADR: fx-send-snapshot-freshness-gate-20260605

## 1. Context

`fx_send_proposal` can derive FX parameter schema evidence from the read-only
`fx_inserts_snapshot`. A mapped schema from that snapshot is useful only if the
snapshot is fresh. Stale cached FX state can describe a slot model or insert
assignment that no longer exists on the console.

## 2. Options considered

1. Block proposal-only promotion when snapshot schema evidence is stale.
2. Trust mapped schemas regardless of snapshot freshness.
3. Require all callers to pass explicit freshness and ignore snapshot freshness.

## 3. Decision

Use option 1. If normalized schema evidence comes from `fx_inserts_snapshot` and
`snapshot_cache_freshness_status` is not `fresh`, add a blocker and keep the
recommendation `analysis_only`.

## 4. Why this won

It preserves the distinction between cached read DTOs and current console proof,
while still allowing fresh snapshots to support assisted FX-send proposals.

## 5. Rejected alternatives

Trusting stale mapped schemas is unsafe. Requiring all callers to pass separate
freshness evidence would duplicate state already carried by Product Layer
snapshots.

## 6. Implementation plan

1. Add `fx_inserts_snapshot_cache_not_fresh` in `analyze_fx_send_need`.
2. Treat the blocker as medium severity.
3. Add mapped-but-stale single and batch tests.
4. Record the gate in processing registry metadata.

## 7. Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/fx_send_proposal.py backend/processing_registry.py tests/test_fx_send_proposal.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_fx_send_proposal.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_fx_send_proposal.py tests/test_processing_registry.py tests/test_operator_product_state.py tests/test_read_surface_inventory.py tests/test_read_surface_fixture_evidence.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## 8. Risks and rollback

Risk: more FX-send recommendations remain `analysis_only` when snapshot
freshness is missing or stale. This is the safer failure mode for live sound.

Rollback is removing the blocker, registry note, tests, and this ADR. No live
write path is introduced.

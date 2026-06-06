# ADR: source-identity-memory-trust-review-freshness-safety-20260606

## 1. Context

Source identity calibration memory trust review consumes read-only memory index
evidence and fresh replay context before surfacing operator-review candidates.
The previous review contract checked freshness status but did not require the
freshness payload to prove read-only/no-live/no-query safety flags.

## 2. Options considered

1. Continue trusting `status=fresh` alone.
2. Validate freshness safety only in Product Layer preview.
3. Validate freshness safety and memory-index status inside the trust-review
   builder.

## 3. Decision

Use option 3. The trust-review builder now rejects unsafe `fresh` context and
inconsistent memory-index statuses before exposing candidates.

## 4. Why this won

The trust-review builder is the contract boundary reused by registry and Product
Layer surfaces. Validating there keeps all consumers fail-closed without adding
transport code, queue mutation, model mutation, runtime memory mutation, or live
console authority.

## 5. Rejected alternatives

Trusting `status=fresh` alone was rejected because forged context could still
look candidate-ready. Product Layer-only validation was rejected because a
direct builder caller could bypass it.

## 6. Implementation plan

1. Add freshness-context validation for `read_only=true`, `live_write=false`,
   and `queries_console=false` when status is `fresh`.
2. Add memory-index status consistency checks for replacement-required,
   not-ready-with-records, and available-without-records cases.
3. Cover both cases with focused unit tests.
4. Record the durable safety lesson in project memory.

## 7. Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_review.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_index.py tests/test_source_identity_calibration_memory_trust_review.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_trust_review.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## 8. Risks and rollback

Risk: older fixtures that provide `status=fresh` without safety flags now fail
candidate creation. This is intentional for trust review because missing
read-only/no-live/no-query proof is not enough for downstream trust decisions.

Rollback: remove the freshness and status consistency helper checks and keep
trust-review candidate creation limited to the earlier record/envelope
validation.

# ADR: source-identity-promotion-acceptance-record-safety-20260606

## Context

Source identity memory-trust promotion acceptance records can feed a read-only
promotion execution plan. The acceptance ledger creates safe records, but the
execution plan also accepts an explicit ledger payload. That boundary must fail
closed if a supplied record claims live write, console query, queue, model,
runtime memory, downstream priority, or live-widening authority.

## Options considered

1. Trust the acceptance ledger builder output.
2. Keep safety only in ledger-level `safety` and `summary`.
3. Require self-contained record-level safety and validate it at the execution
   plan boundary.

## Decision

Use option 3. Acceptance records now carry explicit read-only/no-live/no-query
and no-mutation flags, and the execution plan rejects accepted records unless
those fields are present with safe values.

## Why this won

The execution plan is the next boundary in the source-identity chain. It should
not trust a supplied ledger record just because parent metadata looks safe or
because `unsafe_authority_flags` is empty. Self-contained records are safer for
future Product Layer, UI, and report consumers.

## Rejected alternatives

- Builder-only trust was rejected because forged or stale payloads can bypass
  assumptions made inside the builder.
- Parent-only safety was rejected because individual records may be rendered or
  reviewed without the full parent object.
- Executor work was rejected because this cycle is safety hardening only; real
  downstream priority mutation remains blocked.

## Implementation plan

1. Add record-level safety envelope fields to promotion acceptance ledger rows.
2. Require the envelope in promotion execution plan accepted-record validation.
3. Add regression coverage for forged direct authority flags.
4. Assert Product Layer preview preserves the record-level envelope.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_promotion_acceptance_ledger.py tests/test_source_identity_calibration_memory_trust_promotion_execution_plan.py tests/test_operator_product_state.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_trust_promotion_acceptance_ledger.py backend/source_identity_calibration_memory_trust_promotion_execution_plan.py backend/processing_pipeline_preview.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: external acceptance-ledger fixtures without the new record-level envelope
will now fail execution-plan validation. This is acceptable because missing
safety evidence should not be promoted.

Rollback: remove the added record-level fields and execution-plan validation,
then keep promotion acceptance records as builder-trusted evidence only.

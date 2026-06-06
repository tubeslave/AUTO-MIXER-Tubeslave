# ADR: decision-queue-apply-preflight-visibility-20260606

## Context

The system can now dry-run preflight a single operator proposal apply. The
decision queue should expose which pending proposals are candidates for that
preflight without performing validation or changing runtime state.

## Options Considered

1. Run preflight automatically during `build_decision_queue`.
2. Add a separate queue preflight endpoint immediately.
3. Add read-only apply-preflight candidate visibility to the decision queue.

## Decision

Use option 3. `build_decision_queue` will publish an `apply_preflight` summary
derived from pending proposal dictionaries.

## Why This Won

It improves operator guidance and UI ergonomics without adding side effects to
a snapshot builder. It also keeps the later batch-preflight step straightforward
because the candidate contract is explicit.

## Rejected Alternatives

- Automatic preflight in the snapshot builder was rejected because it could
  query pre-read state and make a read-only queue snapshot expensive or
  surprising.
- A batch endpoint was deferred because candidate visibility is the smaller
  contract needed first.
- Treating `can_apply=true` as final readiness was rejected because pre-read,
  approval, idempotency, mode, and delta gates still need dry-run validation.

## Implementation Plan

1. Add a helper that splits pending actions into apply-preflight candidates and
   analysis-only entries.
2. Attach the summary to `decision_queue.summary`.
3. Add stable top-level count aliases.
4. Test normal proposals and source-identity imported proposals.
5. Record the boundary in project memory.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_decision_queue_summarizes_apply_preflight_candidates tests/test_operator_product_state.py::test_decision_queue_summarizes_source_identity_queue_import_visibility -q`
- `PYTHONPATH=backend python3 -m py_compile backend/operator_product_state.py`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`

## Risks And Rollback

Risk: candidate counts may be interpreted as live-write readiness. Mitigation:
the summary uses candidate wording and keeps `live_write=false`,
`preflight_required=true`, and `live_apply_entrypoint=apply_operator_proposal`.

Rollback: remove the `apply_preflight` summary fields and tests. Existing
proposal queue and single-proposal preflight behavior remain intact.

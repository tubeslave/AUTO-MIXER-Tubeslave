# Proposal: source-identity-promotion-acceptance-record-safety-20260606

## Short Thesis

Harden the source-identity promotion acceptance boundary by making each accepted
record carry its own safety envelope and requiring the execution plan to verify
that envelope before surfacing any non-executing candidate.

## What I Understood

The current chain intentionally stops before real memory-trust promotion. An
acceptance ledger can say an operator accepted a promotion contract record, but
that must remain evidence only. A forged ledger row must not be able to claim
live write, console query, queue import, model mutation, runtime memory
mutation, or downstream priority mutation authority.

## Proposed Solution

- Add record-level safety fields to promotion acceptance ledger records:
  `read_only`, `live_write`, `mutates_console_state`, `queries_console`,
  direct mutation flags, queue/apply flags, and `live_widening_allowed`.
- Extend promotion execution plan validation so accepted records must carry the
  expected true/false safety envelope.
- Add a regression test where a forged acceptance ledger sets direct authority
  flags while leaving `unsafe_authority_flags` empty.
- Assert the new record envelope survives through Product Layer preview.

## Likely Files To Touch

- `backend/source_identity_calibration_memory_trust_promotion_acceptance_ledger.py`
- `backend/source_identity_calibration_memory_trust_promotion_execution_plan.py`
- `tests/test_source_identity_calibration_memory_trust_promotion_acceptance_ledger.py`
- `tests/test_source_identity_calibration_memory_trust_promotion_execution_plan.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/source-identity-promotion-acceptance-record-safety-20260606.md`

## Alternatives Considered

- Trust the acceptance ledger builder only. Rejected because execution plan
  accepts a supplied ledger payload and should fail closed on forged records.
- Add an executor now. Rejected because executor semantics, rollback apply, and
  downstream priority mutation are outside this safe cycle.
- Keep safety only in parent `safety` and `summary`. Rejected because future UI
  or report consumers may render individual records independently.

## Risks

- Older external ledger fixtures that omit record-level safety fields will now
  be rejected by the execution plan. This is intentional because missing safety
  is not proof.
- The new fields could be mistaken for authority if read carelessly. Mitigation:
  all authority and mutation flags are explicit false values.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_promotion_acceptance_ledger.py tests/test_source_identity_calibration_memory_trust_promotion_execution_plan.py tests/test_operator_product_state.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_trust_promotion_acceptance_ledger.py backend/source_identity_calibration_memory_trust_promotion_execution_plan.py backend/processing_pipeline_preview.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where The Other Agent May Disagree

Kimi might argue this is redundant with acceptance-ledger validation. The
counterpoint is that a downstream consumer must distrust supplied evidence at
its own boundary, especially for safety flags that could later influence
operator-facing priority decisions.

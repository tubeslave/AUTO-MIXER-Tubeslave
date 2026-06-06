# Proposal: source-identity-apply-result-safety-envelope-20260606

## Short Thesis

Harden the source identity promotion apply-result row contract so rollback
audit evidence cannot ignore forged row-level console-query, priority, or live
widening claims.

## What I Understood

The promotion executor can mutate only an explicit caller-provided trust store
after `apply=true` and `operator_ack=true`. The apply audit turns that output
into read-only rollback evidence. Because future gates may inspect individual
apply-result rows, those rows need their own negative authority flags instead
of relying only on executor-level summary fields.

## Proposed Solution

- Add explicit false safety fields to executor apply-result rows:
  `queries_console`, `downstream_priority_change_allowed_by_executor`, and
  `live_widening_allowed`.
- Extend the apply-audit row validator to require those fields to be false.
- Add a regression test where the executor summary remains consistent but the
  apply-result row forges those authority flags.

## Likely Files To Touch

- `backend/source_identity_calibration_memory_trust_promotion_executor.py`
- `backend/source_identity_calibration_memory_trust_promotion_apply_audit.py`
- `tests/test_source_identity_calibration_memory_trust_promotion_executor.py`
- `tests/test_source_identity_calibration_memory_trust_promotion_apply_audit.py`
- `.ai/memory/project.md`
- `Docs/adr/source-identity-apply-result-safety-envelope-20260606.md`

## Alternatives Considered

- Trust executor-level safety only. Rejected because row-level consumers could
  later miss a forged field.
- Add a new downstream-priority gate now. Rejected because this cycle should
  only harden existing evidence and not create policy authority.

## Risks

- External fixtures that build apply-result rows by hand must include the new
  false fields. That is acceptable because missing safety evidence should fail
  closed.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_promotion_apply_audit.py tests/test_source_identity_calibration_memory_trust_promotion_executor.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_trust_promotion_apply_audit.py backend/source_identity_calibration_memory_trust_promotion_executor.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where The Other Agent May Disagree

Kimi might prefer waiting until a downstream-priority gate exists. I would
still harden this now because it is a small fail-closed validation improvement
on the current evidence contract.

# Automixer WING MVP Review Decision Strict Scope Type Gate

Date: 2026-06-14

## Summary

The WING read-only review decision ledger now rejects coerced channel scope
values in manual decision records. String and float channel ids no longer become
valid integers through `int(...)`.

## Changed

- `_int_list()` in the review decision ledger accepts only real `int` values and
  still excludes `bool`.
- A regression test verifies that `"3"` and `3.0` scope values make an otherwise
  accepted decision record invalid.

## Safety

This is read-only/offline ledger validation. It does not contact WING, mutate
console state, update readiness, or grant live authority.

## Validation

Focused validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py
```

Result:

```text
398 passed, 8 skipped
```

Standard validation:

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

Result:

```text
3364 passed, 9 skipped
```

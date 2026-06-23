# Automixer WING MVP Evidence Boolean Numeric Scope Gate

Date: 2026-06-14

## Summary

The WING read-only preflight evidence validator now rejects boolean values in
numeric scope and readback-count positions. This prevents malformed artifacts
from relying on Python's `true == 1` behavior before manual review.

## Changed

- `_safe_int()` returns `0` for boolean inputs.
- `_int_tuple()` skips boolean list entries.
- A regression test corrupts channel scope, operator-confirmed channel scope,
  readback count, physical readback count, and one readback channel with boolean
  values and verifies the artifact is not accepted as a read-only preflight
  candidate.

## Safety

This is read-only/offline validation. It does not contact WING, does not mutate
console state, does not update readiness, and does not grant live authority.

## Validation

Focused validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py
```

Result:

```text
395 passed, 8 skipped
```

Standard validation:

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

Result:

```text
3361 passed, 9 skipped
```

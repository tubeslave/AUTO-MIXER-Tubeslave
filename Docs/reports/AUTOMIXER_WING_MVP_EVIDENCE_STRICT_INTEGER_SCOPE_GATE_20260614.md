# Automixer WING MVP Evidence Strict Integer Scope Gate

Date: 2026-06-14

## Summary

The WING read-only preflight evidence validator now requires declared channel
scope fields to contain real integers. Numeric strings and floats are rejected
instead of being coerced through `int(...)`.

## Changed

- `_int_tuple()` accepts only `int` values and still excludes `bool`.
- A regression test proves string and float channel scope values prevent an
  artifact from becoming an accepted read-only preflight candidate.

## Safety

This is read-only/offline validation. It does not contact WING, mutate console
state, update readiness, or grant live authority.

## Validation

Focused validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py
```

Result:

```text
396 passed, 8 skipped
```

Standard validation:

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

Result:

```text
3362 passed, 9 skipped
```

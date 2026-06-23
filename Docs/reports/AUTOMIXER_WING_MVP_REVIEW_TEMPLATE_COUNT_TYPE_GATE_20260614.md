# Automixer WING MVP Review Template Count Type Gate

Date: 2026-06-14

## Summary

The WING read-only review decision template now rejects coerced readback count
values. Boolean or string counts no longer become valid-looking integers through
`int(...)`.

## Changed

- Added non-coercing count normalization to the review tool.
- Applied it to review rows, decision template creation, and markdown rendering.
- Added a regression test that mutates a clean report with malformed count
  values and expects template generation to fail closed.

## Safety

This is read-only/offline review tooling. It does not contact WING, mutate
console state, update readiness, or grant live authority.

## Validation

Focused validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py
```

Result:

```text
397 passed, 8 skipped
```

Standard validation:

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

Result:

```text
3363 passed, 9 skipped
```

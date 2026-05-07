# Task <task-id>: <title>

## Problem

Describe the concrete problem or refactor target.

## Constraints

- Preserve safety-critical behavior unless the brief explicitly changes it.
- Keep the public API stable unless the brief says otherwise.
- Avoid new production dependencies without a strong written reason.

## Definition of Done

- Tests pass
- Behavior matches the brief
- ADR is updated when architecture or behavior changes

## Test Command

`PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

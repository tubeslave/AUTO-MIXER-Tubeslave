# ADR: WING Read-Only Preflight Busy-Port Markdown Evidence

## Context

The prestart no-WING gate fails closed when the backend port already has a listener. The JSON summary includes machine-readable confirmation fields, but the markdown summary previously showed only listener process details and the operator next step.

## Options considered

1. Keep busy-port confirmation fields only in JSON.
2. Render the confirmation fields in the existing markdown summary.
3. Create a separate busy-port report artifact.

## Decision

Use option 2. The existing markdown summary now renders the busy-port confirmation fields alongside listener process details.

## Why this won

It keeps the operator-visible report aligned with the machine-readable artifact without adding another report surface or contacting the backend.

## Rejected alternatives

Option 1 hides critical safety state from the most readable artifact. Option 3 adds unnecessary report churn for a small diagnostic improvement.

## Implementation plan

- Update `_backend_port_markdown_details`.
- Extend the existing markdown summary test.

## Test plan

- `python3 -m py_compile tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py -k "summary_surfaces_backend_port_owner or backend_port or listener_processes"`
- Broader no-WING/WING focused tests.

## Risks and rollback

Risk: markdown snapshots or exact-text consumers may need updates. Rollback: remove the added markdown lines. No live write or hardware behavior is changed.

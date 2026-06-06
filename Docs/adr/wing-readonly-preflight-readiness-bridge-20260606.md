# ADR: WING Read-Only Preflight Readiness Bridge

## Context

The supervised WING Rack MVP tracks concrete blockers for read-only connection,
physical console readback, operator-visible evidence, and readiness matrix
updates. `wing_readonly_preflight_evidence` can evaluate real preflight
artifacts, but the autonomy promotion readiness snapshot did not expose that
state.

## Options Considered

1. Automatically update promotion or safety rows from valid preflight evidence.
2. Add a read-only overlay to `autonomy_promotion_readiness`.
3. Leave preflight evidence only in the processing registry and no-WING report.

## Decision

Use option 2.

## Why This Won

The readiness snapshot is the right audit surface for goal progress, but
read-only preflight is not supervised apply proof. A read-only overlay makes the
candidate visible while keeping safety rows and live policy unchanged.

## Rejected Alternatives

- Automatic updates were rejected because they could blur readback proof with
  apply, rollback, and emergency-stop proof.
- Leaving evidence outside readiness was rejected because the project would not
  have a single canonical view of whether the WING read-only blocker is missing
  or ready for manual review.

## Implementation Plan

- Add a `wing_readonly_preflight_evidence` summary overlay inside
  `backend/autonomy_promotion_readiness.py`.
- Keep all authority flags false in the overlay.
- Allow tests to pass an already-built preflight evidence snapshot.
- Add default-missing and valid-candidate tests.

## Test Plan

- `python3 -m py_compile backend/autonomy_promotion_readiness.py backend/wing_readonly_preflight_evidence.py backend/processing_registry.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_autonomy_promotion_readiness.py tests/test_wing_readonly_preflight_evidence.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_processing_registry.py::test_registry_snapshot_can_hide_full_method_inventory tests/test_no_wing_release_check.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks And Rollback

Risk: a valid read-only candidate may be overstated as completed hardware proof.
The overlay prevents this by keeping readiness closure, live authority, and Full
Autonomous authority false. Rollback is a normal git revert.

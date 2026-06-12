# ADR: WING no-WING operator next-step summary

## Context

WING read-only preflight artifacts can be blocked before backend/WING contact
and include `operator_next_step`. The no-WING summary previously carried the
WING evidence status and blockers, but not the artifact-level repair guidance.

## Options Considered

1. Keep next-step guidance only in artifact review reports.
2. Add the guidance only to raw check detail.
3. Carry unique artifact next steps into no-WING JSON and Markdown summaries.

## Decision

Carry unique artifact next steps into the WING evidence schema check detail and
the no-WING release summary.

## Why This Won

The no-WING summary is the first prestart gate operators run before the intended
WING session. If an artifact is blocked because an expected plan is stale or
invalid, the summary should say exactly what to regenerate.

## Rejected Alternatives

Review-only guidance was rejected because the prestart release summary would
remain less actionable. Raw-detail-only guidance was rejected because it would
not be visible in the Markdown report.

## Implementation Plan

- Collect de-duplicated `operator_next_step` values from evidence artifacts.
- Add `artifact_operator_next_steps` to the WING evidence summary.
- Render it in the Markdown release summary.
- Add regression tests for de-duplication and Markdown visibility.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py -k "wing_readonly_preflight_evidence or wing_preflight_artifact_operator_next_steps"`
- Adjacent WING/no-WING tests.
- Full repository pytest.

## Risks And Rollback

The field is operator guidance only. Rollback is to remove the field from the
check detail and summary; evidence acceptance rules are unchanged.

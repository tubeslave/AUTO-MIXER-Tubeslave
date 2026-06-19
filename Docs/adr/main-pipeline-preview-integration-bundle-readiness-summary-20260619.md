# ADR: Main Pipeline Preview Integration Bundle Readiness Summary

## Context

M026 added an `offline_integration_bundle` overlay to autonomy promotion
readiness, and M027 surfaced that overlay in frontend module status rows.

`main_pipeline_preview` already exposes aggregate autonomy readiness fields in
its top-level `summary`, but the integration bundle overlay was only available
inside the nested registry snapshot. Preview and Director consumers should be
able to see the same blocker without walking registry internals.

M001 still waits for a real WING read-only artifact and operator review. This
decision is backend/read-only preview only.

## Options Considered

1. Leave integration bundle readiness only in the nested registry snapshot.
2. Copy integration bundle readiness fields into `main_pipeline_preview.summary`.
3. Add a preview action or queue task for integration bundle blockers.

## Decision

Copy read-only integration bundle summary fields from
`registry.autonomy_promotion_readiness.summary.offline_integration_bundle` into
`main_pipeline_preview.summary`.

The fields report readiness, status, blockers, missing integration checks,
source authority denial and authority-denial flags. They do not queue work,
contact hardware, open transport, mutate policy or grant live authority.

## Why This Won

`main_pipeline_preview.summary` is already the aggregate preview contract for
runtime and Director consumers. Adding read-only fields there avoids a second
interpretation of the nested registry and keeps the M026/M027 blocker visible
at the same level as other autonomy readiness summary fields.

## Rejected Alternatives

Leaving the fields nested was rejected because preview consumers could miss the
integration bundle blocker while still reading other autonomy readiness fields.

Adding an action/task row was rejected because the integration bundle state is
evidence status, not an apply command.

## Implementation Plan

- Read `offline_integration_bundle` from autonomy promotion readiness summary
  inside `backend/processing_pipeline_preview.py`.
- Add `autonomy_promotion_offline_integration_bundle_*` fields to
  `main_pipeline_preview.summary`.
- Cover the behavior in
  `tests/test_processing_pipeline_preview_integration_bundle_summary.py`.
- Update roadmap, completion gate, capability matrix, current state and
  autopilot evidence.

## Test Plan

- `PYTHONPATH=backend python -m pytest tests/test_processing_pipeline_preview_integration_bundle_summary.py -q`
- `PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py::test_m028_main_pipeline_preview_surfaces_integration_bundle_readiness_summary -q`
- `PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk: preview summaries grow larger and duplicate nested registry fields. This
is acceptable because the fields are compact, read-only and promotion-critical.

Rollback: revert the M028 preview summary/test/docs/state changes. The M026
backend overlay and M027 frontend adapter summary remain available.

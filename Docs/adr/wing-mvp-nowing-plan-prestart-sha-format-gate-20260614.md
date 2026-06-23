# ADR: WING MVP no-WING Plan Prestart SHA Format Gate

Date: 2026-06-14
Status: Accepted

## Context

The WING read-only offline plan is the source of truth for the no-WING
prestart `summary.json` path and SHA256 used by the handoff packet. A malformed
non-empty digest should not be accepted as plan truth.

## Options considered

1. Keep checking only that the plan digest is non-empty.
2. Reuse the shared SHA-256 hex helper in the no-WING plan artifact validator.
3. Check digest format only in handoff packet validation.

## Decision

Use option 2. The plan artifact validator now rejects malformed
`backend_port_prestart_summary_sha256` values before handoff validation.

## Why this won

The plan is upstream of handoff packets and accepted-candidate traces. Failing
at the plan boundary prevents a malformed digest from becoming expected state.

## Rejected alternatives

- Non-empty only: allows placeholder strings.
- Handoff-only validation: still lets the upstream plan artifact look valid.

## Implementation plan

- Rename the shared contract helper to `is_sha256_hexdigest()`.
- Use it in `validate_wing_readonly_preflight_plan_artifact_snapshot()`.
- Add plan and handoff regression tests for malformed prestart summary SHA.

## Test plan

- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py
  -k 'preflight_plan or prestart_summary_sha256' --tb=short -x`
- Full no-WING module.
- Project Director rails, memory lint, and `git diff --check`.

## Risks and rollback

Rollback restores the non-empty-only plan digest check and removes the focused
tests/report. The change is offline-only and does not contact hardware, mutate
console state, grant live authority, or close readiness.

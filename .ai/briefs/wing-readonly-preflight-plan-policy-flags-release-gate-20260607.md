# Brief: WING Read-Only Preflight Plan Policy Flags Release Gate

## Task ID

wing-readonly-preflight-plan-policy-flags-release-gate-20260607

## Context

The WING read-only preflight runner and evidence validator now rely on two
policy flags:

- `expected_plan_required_for_real_run: true`
- `physical_preflight_requires_existing_backend_connection: true`

The no-WING release checker must also validate these fields in the offline plan
artifact so operator handoff packets cannot silently drop them before a real
WING session.

## Scope

- Keep the work offline and read-only.
- Validate the plan artifact policy flags in `tools/no_wing_release_check.py`.
- Expose the flags in JSON and Markdown release summaries.
- Keep all live writes and readiness closure blocked.

## Deliverables

- Release gate validation for both plan policy flags.
- Updated no-WING and runner tests.
- Runbook, ADR, report, and documentation index update.

## Validation

- Focused no-WING and runner tests.
- No-WING quick release check.
- Full pytest suite before commit.

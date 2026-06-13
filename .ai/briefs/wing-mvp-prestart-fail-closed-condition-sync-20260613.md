# WING MVP Prestart Fail-Closed Condition Sync

Task id: `wing-mvp-prestart-fail-closed-condition-sync-20260613`

## Context

The first open MVP blocker remains `WING read-only connection proof`.
The WING readonly preflight runner and evidence validator require expected plans
to fail closed on missing or mismatched no-WING prestart summaries:

- `prestart_summary_missing_or_invalid`
- `prestart_summary_sha256_mismatch`

`tools/no_wing_release_check.py` had an older required fail-closed list without
those two entries, so a generated offline WING readonly plan could fail the
release check on `wing_readonly_preflight_plan_fail_closed_conditions_mismatch`.

## Constraints

- Do not contact WING hardware.
- Do not start, stop, or reconfigure the backend listener.
- Do not weaken prestart summary validation.
- Preserve the supervised WING Rack MVP first-blocker rail.

## Required Outcome

The no-WING release-check validator must accept the same WING readonly
preflight fail-closed conditions emitted by the runner and required by evidence
validation, including the two prestart summary conditions above.

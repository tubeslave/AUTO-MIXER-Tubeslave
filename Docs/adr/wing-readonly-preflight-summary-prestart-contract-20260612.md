# ADR: WING Read-Only Preflight Summary Prestart Contract

## Context

The WING read-only preflight expected plan already carries the backend-port
prestart contract. The real-run `wing_readonly_preflight.v1` summary did not
echo that contract, and evidence intake did not require it. A future real
hardware artifact could therefore prove read-only WebSocket behavior while
omitting the required prestart gate metadata.

## Options considered

1. Keep prestart metadata only in the expected plan.
2. Add the prestart contract to real-run summaries and validate it in evidence
   intake.
3. Add prestart metadata only to reports, not the summary artifact.

## Decision

Use option 2. The real-run summary now includes the shared prestart fields, and
evidence intake rejects summaries whose fields are missing or mismatched.

## Why this won

The real summary is the durable proof artifact for the first hardware blocker.
It should independently show the prestart boundary that was required before the
backend/WING session.

## Rejected alternatives

Option 1 makes the summary dependent on a separate plan for prestart context.
Option 3 keeps human reports informative but leaves artifact validation weak.

## Implementation plan

- Add shared prestart fields to `tools/wing_readonly_preflight_runner.py`
  summary initialization.
- Validate those fields in `backend/wing_readonly_preflight_evidence.py`.
- Update runner and evidence tests.
- Run focused and full validation.

## Test plan

- Focused runner/evidence tests.
- Adjacent WING/no-WING tests.
- Full backend suite.

## Risks and rollback

Risk: old preflight summaries without the new fields become invalid. Rollback:
remove the summary fields and validation checks. No hardware access or live
write behavior is changed.

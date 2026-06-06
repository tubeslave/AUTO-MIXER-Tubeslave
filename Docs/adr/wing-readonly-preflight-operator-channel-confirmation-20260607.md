# ADR: WING Read-Only Preflight Operator Channel Confirmation

## Context

The first supervised WING MVP blocker is a real read-only connection proof. The
preflight runner can inspect one or more WING input channels, but the selected
channel must be safe for the operator's current console/session.

The runbook already required operator awareness, but the artifact schema did
not include a machine-checkable confirmation that the channels passed to the
runner were the channels the operator approved.

## Options considered

1. Keep operator confirmation as prose in the runbook.
2. Require an interactive prompt before every preflight query.
3. Require an explicit `--operator-confirm-channels` CLI flag that must exactly
   match `--channels` when `--require-physical-readback` is used.

## Decision

Use option 3. Physical readback preflight runs now require
`--operator-confirm-channels`. The runner writes both
`operator_channel_confirmation_required` and `operator_confirmed_channels` into
`summary.json`. Evidence intake rejects artifacts where the confirmation is
missing or does not exactly match `channels`.

## Why this won

The WING proof must be scriptable and artifact-driven, but the selected channel
is still an operator safety decision. A required flag keeps the command
non-interactive for controlled sessions while making the approval visible in the
captured evidence.

## Rejected alternatives

Runbook-only confirmation is easy to skip or lose when artifacts are copied.
Interactive prompts are safer for ad hoc use, but they make automation and
repeatable evidence capture harder.

## Implementation plan

1. Add `--operator-confirm-channels` to `tools/wing_readonly_preflight_runner.py`.
2. Require it to exactly match `--channels` when physical readback is required.
3. Write confirmation metadata into the preflight summary.
4. Reject summaries without matching operator confirmation in
   `backend/wing_readonly_preflight_evidence.py`.
5. Update runbook, Project Director rails, tests, and reports.

## Test plan

- Runner argument validation rejects missing, malformed, or mismatched
  confirmation.
- Runner success summary includes `operator_confirmed_channels`.
- Evidence validation rejects missing or mismatched confirmation.
- Review tool tests use the stronger valid artifact.
- MVP runbook tests require the confirmation flag.
- Focused and full pytest suites pass.

## Risks and rollback

Risk: old preflight summaries captured before this change no longer pass as
valid physical readback candidates. This is acceptable because no real WING
MVP blocker is closed yet.

Rollback is a small revert of the runner flag, evidence checks, tests, and
runbook wording. No mixer state is touched.

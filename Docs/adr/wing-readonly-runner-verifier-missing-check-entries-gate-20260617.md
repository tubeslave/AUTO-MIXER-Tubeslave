# ADR: WING read-only runner blocks missing verifier check entries explicitly

## Context

The supervised WING MVP cannot advance to any WebSocket preflight unless the
no-WING artifact manifest verifier passes first. The verifier already reports
`failed_checks_missing_check_entries` when a failed check has no matching
diagnostic entry. That list must remain empty before a real WING read-only
candidate is captured.

## Options considered

1. Keep only the generic verifier failure plus
   `prestart_summary_artifact_manifest_verifier_failed_check_entry_missing`.
2. Add an explicit violation for a non-empty
   `failed_checks_missing_check_entries` list.

## Decision

Use option 2. The runner now reports
`prestart_summary_artifact_manifest_verifier_failed_checks_missing_check_entries_not_empty`
before opening a backend/WING WebSocket whenever the verifier report contains
missing failed-check entries.

## Why this won

The no-WING handoff contract requires
`artifact_manifest_verifier_failed_checks_missing_check_entries: []`. A
dedicated violation makes that exact contract failure visible in the preflight
summary and prevents it from being hidden behind a broader verifier failure.

## Rejected alternatives

Keeping only the generic violation was too indirect for operator review. The
operator needs to know whether the verifier failed because the artifact was
bad, because a failed check lacked a diagnostic entry, or because the report
shape drifted.

## Implementation plan

Add the dedicated violation in
`tools/wing_readonly_preflight_runner.py` when
`failed_checks_missing_check_entries` is non-empty. Extend the existing
pre-WebSocket regression test for missing verifier failed-check entries.

## Test plan

Run the targeted missing-check-entry regression, the focused WING/no-WING test
set, `tools/no_wing_release_check.py --quick`, `git diff --check`, and the
standard pytest gate.

## Risks and rollback

Risk is low because this only adds a more specific fail-closed reason. Rollback
is to remove the added violation and its assertion; the broader verifier failure
would still block the preflight.

# ADR: WING Read-Only Preflight Readback Identity Gate

## Context

The WING read-only preflight runner captures the first real hardware artifact
needed for the supervised WING Rack MVP. Artifact review already validates that
`events.jsonl` query/readback pairs match by channel, value type, and address.
However, the runner itself could still mark a run as `passed` if the response
looked physical but belonged to a different channel or address.

## Options considered

1. Leave identity mismatch detection only in artifact review.
2. Warn on mismatched readback identity but keep the runner pass/fail status.
3. Fail the runner at capture time when readback identity mismatches the query.

## Decision

Use option 3. The runner now adds `readback_identity_mismatch` to blockers when
a `mixer_parameter_value` response does not match the requested channel, value
type, and address.

## Why this won

The capture artifact is the operator's first proof object. It should not say
`passed` when it contains the wrong readback target. Failing at capture time
keeps the operator handoff clearer and reduces the chance that a bad artifact
is treated as a clean candidate.

## Rejected alternatives

- Review-only detection still works, but it leaves a contradictory `passed`
  runner summary.
- Warning-only behavior is too weak for a live-sound proof gate.
- Rejecting `mixer_parameter_read_blocked` without preserving backend reasons
  would lose useful diagnostic context.

## Implementation plan

1. Add `readback_identity_matches()` to the runner.
2. Compare each `mixer_parameter_value` response to its request.
3. Block the run with `readback_identity_mismatch:chN:<value_type>` on mismatch.
4. Add `readback_identity_mismatch` to the offline plan and no-WING plan schema
   fail-closed conditions.
5. Add focused tests and document the safety boundary.

## Test plan

- `python3 -m py_compile tools/wing_readonly_preflight_runner.py tools/no_wing_release_check.py`
- Focused runner and no-WING plan-artifact pytest.
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer-no-wing-release-check-wing-readback-identity --command-timeout-sec 300`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

This change does not contact WING by itself, mutate console state, update
readiness, approve supervised proof, or widen live writes. Rollback is to remove
the identity helper, blocker, tests, and docs row; artifact review would still
reject mismatched event logs.

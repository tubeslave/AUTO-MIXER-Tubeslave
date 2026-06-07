# ADR: Supervised Proof Evaluator Physical Status Gate - 2026-06-07

## Context

The supervised WING proof path now requires physical readback inside the runner.
The evaluator and report tool are the downstream local proof boundary used by
promotion evidence and no-WING checks.

## Options considered

1. Trust source labels such as `physical_readback`.
2. Require both physical source labels and `verified_from_physical_readback`
   status for physical proof artifacts.
3. Reject dry-run artifacts unless they also use physical status.

## Decision

Use option 2. Physical proof rows must have physical status on pre-read, apply,
and rollback. Dry-run remains non-promotable when physical proof is required,
but it may still use cache/fake-mixer evidence for rehearsal.

## Why this won

Promotion evidence should not pass because a summary label says the source was
physical while the status says the proof came from cache. Checking both fields
keeps the downstream report path aligned with the live runner boundary.

## Rejected alternatives

Source-only checking was too easy to spoof or misread. Requiring physical status
for dry-run would weaken offline rehearsal without improving live safety.

## Implementation plan

1. Add evaluator-side `PHYSICAL_READBACK_STATUSES`.
2. Add blockers for non-physical pre-read/apply/rollback statuses in physical
   proof mode.
3. Add evaluator and report regression tests for cache-status physical rows.
4. Add a no-WING static policy check for the evaluator source.

## Test plan

- `python3 -m py_compile backend/supervised_proof_artifact_evaluator.py tools/no_wing_release_check.py`
- Focused evaluator/report/no-WING pytest.
- Quick no-WING release check.
- Full pytest before landing.

## Risks and rollback

This intentionally blocks older physical-looking summaries whose readback status
is cache-backed. Rollback would reallow weak promotion evidence and should only
be considered if the backend cannot emit physical status.

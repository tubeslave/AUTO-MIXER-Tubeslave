# Soundcheck Workflow 95: Proposal-First Runtime Boundary

## Context

Canonical web soundcheck already forced observation-only operation, but the
engine derived most recommendations only inside its apply phase. The active
frontend also had no canonical soundcheck workflow, while CLI/config surfaces
could imply broader direct-write authority than the current M001 gate permits.

## Options considered

1. Enable the existing apply phase for web soundcheck.
2. Keep analysis-only behavior and improve labels.
3. Separate no-write recommendation derivation from supervised execution,
   fail-close alternate live writers, and expose the result in the active UI.

## Decision

Use option 3. Measurement and absolute recommendation derivation are always
available in proposal-only mode. Mixer mutation remains exclusively behind the
canonical supervised server authority gate and is not enabled by this change.

## Why this won

It makes soundcheck useful without claiming unproven live authority, preserves
the M005/M006 boundary, and produces machine-checkable zero-write evidence.

## Rejected alternatives

- Calling apply from the handler was rejected because naming and lifecycle
  would conceal the no-write guarantee.
- Keeping empty proposals was rejected because it is not a working soundcheck.
- Restoring a legacy UI was rejected because it is outside the active entrypoint.

## Implementation plan

- Introduce an explicit derivation phase and absolute recommendation contract.
- Reject direct CLI live apply and configuration-based authority expansion.
- Inject shared runtime services and respect ownership on stop.
- Add an active TypeScript Soundcheck cockpit.
- Add zero-write, lifecycle, idempotence, UI, build and full-suite gates.

## Test plan

- Focused backend and frontend tests.
- Long-run 24-cycle drift proof.
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`.
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`.
- frontend test/build and `git diff --check`.

## Risks and rollback

Risk is accidental mutation through a legacy correction builder or lifecycle
regression in shared services. Tests must bind mixer call counts and ownership.
Rollback restores analysis-only web behavior and removes the new UI; no live
authority migration is required.


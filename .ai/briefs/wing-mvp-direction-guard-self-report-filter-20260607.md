# Brief: WING MVP Direction Guard Self-Report Filter - 2026-06-07

## Task

Fix the Codex direction guard so generated policy warnings and stale guard
diagnostic sections do not create recursive `mixed_needs_review` drift signals.

## Context

The current MVP rail remains the supervised WING Rack proof. The first open
blocker is still the real WING read-only connection proof. The direction guard
is part of the stop-the-line workflow that keeps Codex on that blocker.

The guard was reporting `mixed_needs_review` even when `drift_item_count` was
zero, because it classified:

- `Forbidden Next Actions` and `Forbidden Scope` policy sections;
- generated `Mixed Evidence` from earlier `codex_direction_guard_*.md` reports.

Those sections mention memory-trust, queue-import, and result-consumer as
forbidden work, not as work performed in the current cycle.

## Safety Boundary

- No WING, OSC, WebSocket, or mixer transport is contacted.
- No live authority is granted.
- No MVP blocker is closed by this change.
- Real drift outside policy/diagnostic sections must still fail the guard.

## Expected Output

- Direction guard drift scoring ignores policy-only forbidden sections.
- Generated guard evidence sections do not recursively feed future drift.
- Tests prove both the ignore path and the real-drift path.
- Current repo guard status becomes `on_target` only for direction, not for live
  readiness.

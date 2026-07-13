# Independent critique of the primary Codex proposal

## 1. Strongest part

The primary proposal correctly refuses to weaken the coordinator, distinguishes
global from local blockers, treats substitutions as evidence-bound replanning,
and defines readiness over logical corrections rather than physical writes.

## 2. Possible mistakes or blind spots

A new stateful stage orchestrator is too broad as the first write-path change.
Preflight can become stale during 51 operations, a musical recalculation
invalidates the original approval, and a crash after a partially committed
stage needs batch-level durable resume that does not exist today.

## 3. Where it is too complex, fragile or costly

Automatic split execution and substitution inside an orchestrator would create
a second policy engine near the transport. It could launder cumulative pan,
trim, threshold or EQ limits through repeated small steps and could substitute
across risk classes without full mix evidence.

## 4. What should be kept

- Typed `scope` and `disposition` vocabulary.
- Parent run/revision identity.
- Logical readiness counts and required coverage.
- No constructive authority promotion.
- Fresh analysis and a new approval after a residual.

## 5. Revised recommendation

First emit an immutable parent manifest and one coordinator child per semantic
operation. Use stages only for order and readiness. Make split/replacement a
new planner revision. Add the stateful scheduler only after batch WAL,
idempotent resume and approval invalidation are proven.

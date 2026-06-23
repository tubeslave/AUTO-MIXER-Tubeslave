# ADR: WING MVP Review Decision Ledger Artifact Root Contract Gate

## 1. Context

The accepted read-only WING preflight review decision ledger already exposes
prestart contract validity and blocks manual readiness candidate status when
prestart metadata is invalid. Artifact-root contract checks existed later in
supervised proof planning and no-WING validation, which meant the ledger could
temporarily report an accepted candidate as manually available even when
artifact-root metadata would later fail closed.

## 2. Options considered

- Keep artifact-root checks only downstream.
- Make the shared artifact-root helper tolerate synthetic inline source ids.
- Add ledger-level artifact-root summary fields with a local inline-source
  tolerance wrapper.

## 3. Decision

Add ledger-level accepted-candidate artifact-root contract fields and require
them before `manual_readiness_update_candidate_available` can become true.

## 4. Why this won

The ledger becomes self-consistent while downstream proof-plan and no-WING
validation keep using the strict shared helper. Inline tests stay valid without
weakening file-backed artifact path checks.

## 5. Rejected alternatives

Downstream-only validation leaves a stale intermediate signal. Changing the
shared helper would weaken the stricter proof-plan/no-WING source-id path
contract. Rewriting all inline tests around file-backed source ids is larger
than needed.

## 6. Implementation plan

- Import the artifact-root blocker and shared helper in the ledger module.
- Add a ledger-local wrapper that tolerates synthetic inline source ids.
- Add artifact-root contract fields to the ledger summary.
- Require the artifact-root contract for manual readiness candidate status.
- Add regressions for valid and invalid accepted candidates.

## 7. Test plan

Run focused tests, WING-focused tests, standard pytest, no-WING release check,
diff check, memory lint, and Project Director.

## 8. Risks and rollback

Risk is limited to stricter offline ledger status. Rollback is reverting the
ledger wrapper, summary fields, and regressions.

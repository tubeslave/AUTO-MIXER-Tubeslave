# ADR: full-project readiness safety rebaseline

Status: accepted for implementation during the 2026-07-11 audit

## 1. Context

The dirty M006 checkout has green backend/frontend tests and real-WING proof
artifacts, but runtime authority, proof execution, test persistence, rollback
semantics, and release truth disagree. Static family metadata permits normal
supervised writes while the owner decision remains pending and all current
authority surfaces report zero standing live kinds.

## 2. Options considered

1. Keep static `live_allowed` metadata and treat proof output as promotion.
2. Remove the broad transaction kernel and return to fader/gain-only code.
3. Preserve the transaction kernel, fail closed by default, and make proof,
   promotion, execution, rollback, test evidence, and release scoring separate
   explicit gates.

## 3. Decision

Choose option 3.

- Hardware proof may produce `hardware_proven` evidence, never standing live
  authority by itself.
- Normal execution must intersect the family contract, operator-mode allowlist,
  WING deployment boundary, and observation-only state.
- The local proof runner is plan-only by default and must validate typed
  operator confirmation before constructing `WingClient`.
- A successful proof is reported as `proof_passed`; `accepted` remains false
  until an independent owner/readiness decision exists.
- A partially touched operation belongs to the rollback set immediately.
  Apply/verify/audio exceptions must restore current and prior operations;
  unverified restoration activates the system emergency stop.
- Pytest manual-write artifacts use per-test temporary roots.
- Public release readiness stays false below the 95 threshold and while any
  required external gate is open.

## 4. Why this won

It preserves useful semantic adapters and real proof evidence while removing
the four paths that could falsely convert code, tests, or proof output into live
authority. It also makes failure behavior reviewable without discarding the
user's current WIP.

## 5. Rejected alternatives

- Static promotion from module constants: contradicts the accepted M006 ADR and
  current owner-decision artifact.
- Report-text promotion: documents are not executable gates.
- Disabling audit persistence in tests: would hide bugs in the persistence path.
- Catching exceptions without rollback: leaves unknown physical state.

## 6. Implementation plan

1. Demote unpromoted family contracts and add runtime operator/deployment gates.
2. Guard the proof runner and correct proof-versus-acceptance semantics.
3. Isolate test audit roots.
4. Harden coordinator/adapter partial-write and exception recovery.
5. Correct score/truth surfaces.
6. Fix frontend truth and connection-state defects.
7. Re-run focused, full, no-WING, frontend, and browser checks without touching
   the real console.

## 7. Test plan

- Registry and queue tests assert no normal family execution without promotion.
- Server tests assert observation-only and empty operator allowlists block before
  snapshot/query/send.
- Proof-runner tests assert no `WingClient` construction without every typed
  confirmation and supplied approval id.
- Adapter/coordinator tests cover partial send failure, apply exception, verify
  exception, audio exception, successful recovery, failed recovery, and
  emergency-stop activation.
- Test runs compare the canonical manual-write log metadata before/after.
- Full backend/frontend/no-WING and browser checks run only after focused tests.

## 8. Risks and rollback

The immediate behavior change intentionally blocks current broad normal
supervised M006 transactions. Rollback is a source revert of this checkpoint;
it must not rewrite historical hardware artifacts. If a valid owner promotion
source is later provided, it can enable exact kinds through a separately tested
loader without changing the fail-closed defaults.

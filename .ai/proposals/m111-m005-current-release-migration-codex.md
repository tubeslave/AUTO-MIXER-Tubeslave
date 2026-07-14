# Codex proposal: M111 truthful proof migration

## 1. Short thesis

Add an explicit two-evidence migration route rather than retrofitting current
provenance onto the historical M005 artifact. Historical hardware facts and
current-code compatibility evidence remain separately named, separately
hashed, and jointly required.

## 2. What I understood

The owner closed M005 behavior and authorized narrow M006. The verifier was
subsequently hardened, so the old run has valid physical semantics but cannot
meet runtime-identity-v2 retrospectively. The repository needs a truthful way
to carry those physical facts forward while proving that current safety code
still blocks the same dangerous transition.

## 3. Proposed solution

Introduce `automixer.m005_current_release_migration.v1` as a report-only
candidate with four independent roots of trust:

1. `historical_evidence`: exact paths, byte hashes, run id and facts extracted
   directly from the immutable legacy summary/events using a schema-specific
   state machine.
2. `current_source`: clean HEAD/tree/status/diff, current runtime-source
   component manifest and file identities for the proof runner, backend and
   M006 verifiers.
3. `compatibility_replay`: an artifact from the canonical current proposal
   apply path under an injected stale/disconnected read surface. It must prove
   the apply terminal is blocked and all console-writer call counts are zero.
4. `console_attestation`: a separately stored owner challenge for WING Rack
   3.0.5, non-empty serial and `same_console_as_historical_m005=true`.

The migration verifier recomputes all four roots. The M006 packet verifier gets
an optional migration input and returns distinct booleans for native current
provenance, historical behavioral readiness, migration readiness and the
selected acceptance route. The decision verifier recomputes the packet with
the same migration input and preserves the exact fader/gain-down scope.

The generator may emit a blocked candidate when the owner challenge is absent;
it must not emit an accepted-looking template. Native current M005 remains the
preferred direct route and is unchanged.

## 4. Likely files to touch

- new migration generator/verifier tool and tests;
- M006 packet/decision verifier CLI plumbing and tests;
- a console-attestation template/schema fixture if one does not already exist;
- M111 ADR and finite truth updates.

No production backend behavior or live registry authority needs to change.

## 5. Alternatives considered

- Mutate the old summary: rejected as fabricated provenance.
- Waive current provenance because the owner said M005 is closed: rejected;
  owner intent cannot substitute for current code compatibility.
- Repeat cable pull with the current runner: technically strongest, but the
  owner explicitly treats M005 as closed and did not request another pull.
- Accept unit tests alone: rejected; they do not bind the historical physical
  evidence or current source identity.

## 6. Risks

- A permissive legacy parser could manufacture a flow not present in events.
- A replay could accidentally use mocked functions below the real safety gate.
- A sidecar could be mistaken for historical provenance.
- Exact-current-source self-reference can recurse when generated artifacts or
  verifier files are part of their own digest.
- An operator attestation can identify a console class but not prove historical
  runtime execution.

Mitigations: narrow run/schema allowlist, ordered event state machine, replay at
the server handler boundary with writer spies below it, explicit evidence-class
labels, generated artifacts outside the tracked source digest, and fail-closed
CLI recomputation from a clean checkout.

## 7. Test plan

- Golden old artifact extracts the five physical facts without rewriting it.
- Every path/hash/order/target/value/status mutation blocks.
- Replay observes `operator_proposal_apply_blocked` and zero writer calls.
- Dirty source, changed runtime component or verifier identity blocks.
- Missing/wrong model, firmware, serial or same-console flag blocks.
- Packet/decision accept only a fully ready migration, report the selected
  route, keep all authority flags false and reject widened rows/kinds.
- Run focused M005/M006/server/no-WING tests, full suite and independent review.

## 8. Where Kimi may disagree

Kimi may argue that any migration weakens runtime-identity-v2 and that only a
new physical run is defensible. It may also require the compatibility replay to
start a proof-owned backend process rather than exercise the handler in-process.
The synthesis should adopt the stronger boundary if it is implementable without
console contact and without turning synthetic replay into claimed hardware
evidence.

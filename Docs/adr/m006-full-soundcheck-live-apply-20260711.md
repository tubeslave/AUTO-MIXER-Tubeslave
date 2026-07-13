# ADR: M006-FULL supervised soundcheck transaction kernel

Status: accepted for implementation by explicit owner request on 2026-07-11;
live promotion remains evidence-gated per write kind.

Implementation result: the operation-level registry permits only exact
hardware-proven write kinds. A real WING soundcheck applied 51/51 generated
channel operations through `apply_operator_proposal`. WING logarithmic and
stepped parameter quantization is represented in adapter tolerances; verified
physical values are used as rollback CAS anchors. Inactive DCA targets near
negative infinity are excluded instead of recorded as fake successful writes.

## 1. Context

The owner rejected closing M006 as a permanent fader/gain-only outcome. The
required product behavior is a real soundcheck in which Automixer can regulate
and send all relevant channel, processing, spatial, FX, bus and output
corrections to WING.

An older M005 run passed on the real WING, but it does not satisfy the current
runtime-identity/provenance contract and is not accepted for the current
release. A clean-checkout, operator-attended current-contract M005 remains a
hard prerequisite. The repository already has broad DSP methods,
WING write wrappers and typed correction actions, but the canonical Product
Layer transaction path supports only a narrow set of scalar writes. Broad
direct or legacy executors cannot become a second live writer.

The worktree contains uncommitted M005 correctness fixes and generated evidence.
Those changes are prerequisites and must be preserved separately from unrelated
artifacts.

## 2. Options considered

### A. Add every family to the existing allowed-kind tuple

Small diff, but unsafe. It grants authority before composite readback,
rollback, model and audio semantics exist.

### B. Promote all actions in `GatedCorrectionActionExecutor`

Reuses typed actions, but creates a parallel live writer outside Product Layer
approval/idempotency/audit.

### C. Generic raw-parameter transaction registry

Centralizes mechanics but can become a hidden raw OSC escape hatch. Data-only
method names and argument lists are difficult to statically review.

### D. Transaction coordinator plus explicit family adapters

One coordinator owns approval, sequencing, emergency generation, artifacts and
rollback orchestration. Code-reviewed adapters own WING-specific validation,
mapping, physical snapshot, apply, verification, rollback and audio policy.

## 3. Decision

Choose option D.

`apply_operator_proposal` remains the only live entrypoint. It delegates an
approved proposal or approved soundcheck plan to a supervised transaction
coordinator. The coordinator resolves an explicit adapter by family and refuses
unknown or unproven families.

Adapter lifecycle:

- `proposal_only`: visible and reviewable, never executable;
- `proof_candidate`: executable only in an explicitly named hardware proof;
- `hardware_proven`: accepted physical apply/readback/rollback/emergency proof;
- `live_allowed`: owner/readiness decision permits normal supervised soundcheck use.

No lifecycle change is inferred from unit tests, code presence or transport
method availability.

## 4. Why this won

- Preserves one writer and one audit surface.
- Avoids a generic raw OSC API.
- Supports multi-parameter families and reverse rollback.
- Makes model/address/tolerance rules locally reviewable.
- Lets the user approve one reviewed low/medium-risk soundcheck plan rather than
  clicking every EQ band, while keeping high-risk routing/scenes/master actions
  separately approved.
- Allows complete product coverage without falsely marking all families proven.

## 5. Rejected alternatives

- Direct AutoSoundcheck writes.
- Voice or frontend direct setters.
- Scene-based bulk replacement as the normal correction mechanism.
- Widening `WING_DEPLOYMENT_LIVE_WRITE_KINDS` before accepted family artifacts.
- Treating offline/replay proof as live authority.

## 6. Implementation plan

### CP1: Core contracts, no authority widening

- Add family lifecycle registry and adapter protocol.
- Add transaction coordinator with immutable plan, idempotency, emergency
  generation and reverse rollback.
- Add fader/gain compatibility adapter using existing proven manual kernel.
- Add tests proving every unproven family stays blocked.

### CP2: Soundcheck plan generation

- Convert typed corrections into absolute Product Layer proposals.
- Build dependency-ordered plans.
- Separate high-risk mute/routing/scene/master actions.
- Keep AutoSoundcheck derivation-only.

### CP3: Channel control adapters

- mute, pan, polarity and delay.

### CP4: Spectral and dynamics adapters

- HPF, cut-only EQ, gate threshold, compressor threshold/ratio with
  makeup-neutral first proofs.

### CP5: FX and mix-structure adapters

- sends/returns, FX model parameters, inserts, buses, groups, DCA and matrix.

### CP6: Output and destructive-scope adapters

- master processing, routing and scene/preset with separate approval and full
  diff preview.

### CP7: Frontend and runtime visibility

- family lifecycle, blockers, exact transaction steps, readback, rollback and
  audio-recheck state.

### CP8: Hardware promotion

- one real-WING artifact per family;
- accepted verifier and owner/readiness decision;
- normal supervised soundcheck permission only after acceptance.

Before CP8, generated M006 packet/decision verifier outputs must remain ignored
runtime artifacts so they cannot invalidate the clean-checkout provenance they
recompute. The packet verifier must also bind its
`accepted_readonly_source_id` exactly to the source accepted by the M005
preflight decision gate; a packet from a different read-only run fails closed.

## 7. Test plan

- Registry integrity and lifecycle tests.
- Coordinator success, partial failure, timeout, stale/cache-only readback,
  mismatch, emergency-generation race, idempotency and reverse rollback tests.
- Adapter bounds/model/quantization tests.
- Existing fader/gain and M005 regression tests.
- AutoSoundcheck no-direct-write and absolute-target tests.
- Replay/fault/no-WING gates.
- Frontend contract/build/E2E.
- Real WING apply/readback/rollback/emergency proof per family.
- Main LR audio recheck where observable; explicit deferred/not-observable
  outcome otherwise.

## 8. Risks and rollback

### Risks

- Partial multi-address application.
- WING model/address drift and quantization.
- Incorrect readback freshness.
- Audio taps that cannot observe a correction.
- Feedback or output interruption from high-risk surfaces.

### Rollback

- Roll back successful operations in reverse order using the immutable physical
  snapshot and transaction id.
- Refuse rollback when an intervening physical change is detected.
- Activate system emergency stop after an unverified write or rollback.
- Demote the failed family to `proposal_only` and preserve the artifact.
- Revert readiness metadata independently of code if hardware proof is rejected.

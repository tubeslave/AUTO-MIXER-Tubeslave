# M110 — attended WING reset and staged soundcheck

## Problem

The current tree can analyze the real WING and construct staged proposals, but
it cannot safely perform the operator's terminal workflow: prove the Main
47/48 capture route, reset prior DSP corrections, and commit a new soundcheck.

Four independent gaps block execution:

1. The existing route-proof loader trusts a hand-authored JSON file plus SHA.
2. The canonical server rejects every hardware `proof_run`, while the legacy
   family runner contains an intentionally blocked direct-WING path.
3. There is no typed DSP-only maintenance reset with complete preimage, WAL,
   physical verification and rollback.
4. Resetting DSP bypasses EQ/gate/compressor, but the soundcheck path cannot
   separately authorize the processor enable-state changes needed to form a
   complete mix.

## Required outcome

Create an attended, backend-owned path that:

- issues an immutable authenticated Main → USB 47/48 proof;
- proves each required write kind with apply/readback/audio/restore evidence;
- resets only channel DSP for channels 1–16, physically proves a finite
  gain/balance/input-route/Main identity manifest unchanged, and proves no
  mutation attempt outside the reset set for unmeasured state;
- commits exact reviewed soundcheck child transactions through one-time
  capabilities without creating standing live authority;
- remeasures and reports `safe_core_ready`, `show_mix_ready`, or
  `soundcheck_incomplete` truthfully.

## Safety invariants

- Normal live allowlist remains unchanged and may stay empty.
- No client supplies an OSC address, target value, transaction, proof flag or
  rollback policy that becomes authority.
- Every capability is backend-issued only to the authenticated proof-owned
  process/session, opaque, digest-stored, short-lived, single-use and bound to
  one runtime, console, route proof, proposal and exact operation set.
- Route capture, arm and capability issuance require a fresh nonce plus HMAC
  from the random secret delivered with the exact prestart manifest over a
  one-shot inherited FD; no launch authority comes from environment metadata.
- Every possible write is preceded by durable WAL prepare and complete physical
  preimage capture.
- A restore-only proof is never reported as an applied mix change.
- A reset is a maintenance transaction, not a soundcheck correction.
- Processor enable is separate, explicitly reviewed, and applied last after
  conservative parameters are already physically verified.
- Disconnect, stale readback, emergency, writer conflict, unresolved WAL,
  route drift, headroom failure or unknown rollback stops the affected lane.
- Local correction failure does not erase safe sibling progress; global hazards
  remain fatal.

## Contract decisions required before code

1. Fixed route-proof request, same-process authority and audit-artifact format.
2. One-shot capability lifecycle and exact bindings.
3. `proof_passed` terminal and `proof_restored` WAL semantics.
4. Fixed DSP reset scope and preserve set.
5. Separate processor-enable and attended soundcheck-commit authority.
6. Runner provenance, token handling and no-direct-OSC rule.
7. Hardware checkpoint order and honest readiness rules.

The issuance contract must include the exact v1 message, fixed confirmation
literal per capability kind, `m110_attended_runner` process-start provenance,
one-token-per-owner/eight-global limits, 30-second latest-start TTL and revoke
on disconnect/reconnect/route/emergency/proposal terminal state.

## Acceptance criteria

1. A hand-authored legacy route JSON is rejected even when its SHA matches.
2. The route endpoint reads only the fixed six addresses twice around fresh
   real 48-channel/48 kHz audio and proves safe Main 47/48 headroom.
   A restart always revokes its authority and requires a new proof.
3. Generic client `proof_run`, target/address injection and token replay all
   block before a console send.
4. A valid exact canonical bundle proof applies, verifies, audio-checks,
   restores and ends as
   `proof_passed` / WAL `proof_restored` with no normal authority change.
5. Reset touches at most `/ch/1..16/{eq/on,flt/lc,gate/on,dyn/on}`, proves every
   finite-manifest surface physically unchanged, proves no outside mutation
   send, and reports any outside transition event actually delivered by WING
   for sends/FX/buses/DCA/matrix/other Main/scenes/names without treating event
   silence as their unmeasured physical equality.
6. Processor enable and permanent soundcheck commits require distinct exact
   one-time capabilities and a recent `proof_passed` for the identical
   canonical operation-bundle digest. Only the fixed reset and atomic processor
   bundles may be composite; staged children are decomposed to one logical
   operation before proof/review/commit. Family-level sibling proof is
   insufficient.
7. Cancellation or failure after a possible send either proves restoration or
   latches emergency/manual review.
8. Targeted adversarial tests, standard full suite, no-WING gate and independent
   review pass before real hardware contact.
9. The attended run produces immutable route, proof, reset, apply, rollback,
   headroom and final readiness artifacts.

## Out of scope

- Standing autonomous live authority.
- Generic client-authored transactions or arbitrary OSC query/write endpoints.
- Routing, scenes, names, fader/gain/mute/pan/delay/sends/FX/bus/master reset.
- Automatic registry promotion solely because a proof passed.
- Claiming room/PA release readiness without operator listening evidence.

## Dirty-state note

HEAD is `bdf6bf8cf`. Tracked source is clean. Untracked
`reports/master_mix_eval_20260713T*.json` are generated test evidence and stay
outside M110 commits.

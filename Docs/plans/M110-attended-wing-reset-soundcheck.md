# M110 — attended WING reset and staged soundcheck

## Safety boundary

- Planning and implementation checkpoints do not contact WING.
- Normal live allowlist and registry standing authority remain unchanged.
- Only committed, fully validated source may enter the attended checkpoint.
- STOP, writer lease, WAL recovery, emergency and headroom rails remain fatal.
- Arming M110 acknowledges no console-setting changes and no other OSC
  controller from before each Apply/Proof executor invocation until its
  durable terminal response is visible to the operator.
- Generated `reports/master_mix_eval_*` stay outside source commits.

| Checkpoint | Goal | Scope | Required evidence | Hardware |
|---|---|---|---|---|
| M110-CP00 | Council contract | brief, proposals, critiques, ADR, plan | exact messages, grants, terminals, reset preserve set and rollback | none |
| M110-CP01 | Main 47/48 provenance | fixed route module/handler/persistent runner/verifier, CoreAudio+USB binding, backend ingress gate | inherited-FD manifest/HMAC, stream/USB drift, semantic artifact, teardown, raw-ingress and no-write tests | none during implementation |
| M110-CP02 | One-shot proof authority | token store, grants, queue/result/wire/coordinator | replay/concurrency/binding/cancellation and `proof_restored` tests | none during implementation |
| M110-CP03 | DSP reset, M006 preimage restore and processor enable | maintenance reset, finite physical preserve manifest, three exact M006 stages, atomic params+enable lane | exact 64-address cap, M006 23/14/14 operation stages, full preimage, no-outside-send, rollback, restart and manifest tests | none during implementation |
| M110-CP04 | Staged child commit | exact-bundle proof-to-commit capability, single-operation or declared atomic-group execution and replan | identical complete bundle digest, no sibling inference, local-vs-global blocker, atomic parent journal and honest readiness tests | none during implementation |
| M110-CP05 | Final software gate | frontend, no-WING, full suite, static scan, independent review, commit, full Git-tree binding | clean committed source, released development lease, passed canonical marker and source digest | none |
| M110-HW01 | Route proof | proof-owned backend in analyze/observation mode | issued artifact/ledger, exact console/capture/headroom | attended real WING |
| M110-HW02 | Exact bundle proofs | one canonical permanent bundle at a time, forced restore | `proof_passed`, WAL `proof_restored`, physical restoration and identical bundle digest | attended real WING |
| M110-HW03 | DSP reset and known-correction removal | exact channels 1–16 maintenance proposal plus three fixed M006 preimage stages | complete preimage, applied reset, exact M006 CAS/commit, finite-manifest equality, no-outside-send and delivered-observer-event evidence | attended real WING |
| M110-HW04 | Staged soundcheck | foundation → tonal/dynamics → spatial/FX | exact reviewed child receipts, replans, headroom, no unresolved important corrections | attended real WING |
| M110-HW05 | Final audit | two-pass ledger/readiness/audio audit, SHA-bound listening, repeated audit and teardown | identical verified-state/ledger hashes, `show_mix_ready` or explicit incomplete blockers, score | attended real WING |

CP03 defines and tests the executor contract only; it grants no hardware
authority. Before CP05 can pass, the production server must own construction of
the M110 hooks and its durable append must compute the receipt SHA-256 from the
exact canonical bytes it writes and fsyncs. A caller-supplied/self-attested
digest is insufficient, and no HW checkpoint may run while that binding is
absent.

## Hardware checkpoint order

1. Confirm the backend-owned CP05 marker binds the entire committed Git tree;
   confirm clean HEAD, no STOP/development lease, one backend, healthy WAL and
   operator presence.
2. Connect observation-only and issue Main 47/48 route proof.
3. Keep the same proof-owned process and use exact
   `arm_m110_attended_control` to enter supervised capability-only mode without
   reconnecting. Any restart/reconnect repeats step 2.
4. With every audible output physically contained and Main 1 verified muted,
   prepare and review the fixed reset bundle, run its exact restore-only proof,
   recapture, issue its commit-once grant, apply and verify preserved state.
5. Still muted, prove and commit the immutable M006 preimage restore in fixed
   `foundation_level` (23 operations / 33 scoped / 19 changed),
   `tonal_dynamics` (14 / 63 / 52), `spatial_balance` (14 / 14 / 6)
   order: 77 unique changed addresses total. A CAS mismatch blocks before send;
   state outside M006 stays untouched.
6. Verify Main 1 unmuted, run fresh canonical analysis, then generate processor
   and soundcheck bundles. Musical children are single operations unless the
   planner marks an exact indivisible `soundcheck_atomic_group`.
7. For each resulting bundle in stage order: review exact bytes, run the
   matching restore-only proof, issue the exact commit-once grant, and apply.
   Fixed reset, M006 restore, complete processor blocks and declared atomic
   groups are the only composite cases; family-level proof is insufficient.
8. Bound every local replan, regenerate and re-prove any changed bundle, and
   preserve already verified safe sibling progress unless a global hazard stops
   the run.
9. Re-read every committed surface twice around audio/Main/route checks, obtain
   the verified-state SHA, require five-minute SHA-bound physical listening,
   repeat the full audit, verify clean teardown and record honest readiness.
   Historical verification-cut receipts alone cannot satisfy HW05 freshness.

## Mandatory software commands

```bash
PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest <M110 focused tests> -q
cd frontend && npm test && npm run build && npm run test:e2e
PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick
PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q
git diff --check
```

## Rollback

Revert the active checkpoint commit. Never use broad legacy reset or direct
runner OSC as fallback. During hardware work, any unknown physical state stops
forward progress, activates emergency/manual review and requires physical
readback plus WAL recovery before resumption.

## Current status

- M110-CP00: complete; Council contract and ADR are accepted.
- M110-CP01: implementation is present and offline-validated; hardware remains
  gated by CP05.
- M110-CP02: implemented and offline-validated, including typed single-use
  grants, strict durable WAL projection, result capacity, receipt minting and
  same-session proof authority.
- M110-CP03: implemented and offline-validated. The fixed 64-address reset and
  three exact M006 restore stages (23/14/14 operations) and processor-enable
  bundles are server-owned, preimage/CAS-bound, physical-readback-gated and
  recover through durable child and parent journals without widening ordinary
  live authority.
- M110-CP04: implemented and offline-validated. Processor and musical children
  use exact proof-to-commit bundles, local blockers preserve safe sibling
  progress, bounded replans are run/revision/content bound, and an audio-safety
  failure never blindly rolls a verified attenuation target upward.
- M110-CP05 is complete under the two canonical committed markers. Commit
  `5c054eb9e28d29601d974850b8874858b3fe0881` passed the clean detached
  launch-only probe with loopback backend contact, no WING/OSC/UDP/audio stream
  and no live write. Any later tracked change still invalidates the source
  binding until the markers are recomputed in one marker-only commit.
- Passive CoreAudio/macOS USB enumeration found no physical WING device. HW01
  is blocked before start on `m110_wing_usb_coreaudio_device_missing`;
  HW02-HW05 remain pending and must run in order in one attended session. Dante
  Virtual Soundcard is not a substitute for the serial-bound WING USB capture
  with 48 inputs/48 outputs at 48 kHz.
- A pending/stale CP05 marker blocks connect, USB capture, route proof and every
  discovery/alternate-mixer ingress before resource construction; cleanup and
  emergency messages remain available.

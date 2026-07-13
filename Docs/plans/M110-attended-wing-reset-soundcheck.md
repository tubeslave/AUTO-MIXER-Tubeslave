# M110 — attended WING reset and staged soundcheck

## Safety boundary

- Planning and implementation checkpoints do not contact WING.
- Normal live allowlist and registry standing authority remain unchanged.
- Only committed, fully validated source may enter the attended checkpoint.
- STOP, writer lease, WAL recovery, emergency and headroom rails remain fatal.
- Generated `reports/master_mix_eval_*` stay outside source commits.

| Checkpoint | Goal | Scope | Required evidence | Hardware |
|---|---|---|---|---|
| M110-CP00 | Council contract | brief, proposals, critiques, ADR, plan | exact messages, grants, terminals, reset preserve set and rollback | none |
| M110-CP01 | Main 47/48 provenance | fixed route module/handler/persistent runner/verifier, CoreAudio+USB binding, backend ingress gate | inherited-FD manifest/HMAC, stream/USB drift, semantic artifact, teardown, raw-ingress and no-write tests | none during implementation |
| M110-CP02 | One-shot proof authority | token store, grants, queue/result/wire/coordinator | replay/concurrency/binding/cancellation and `proof_restored` tests | none during implementation |
| M110-CP03 | DSP reset and processor enable | maintenance reset, finite physical preserve manifest, atomic params+enable lane | exact 64-address cap, full preimage, no-outside-send, delivered-observer-event, rollback and manifest tests | none during implementation |
| M110-CP04 | Staged child commit | exact-bundle proof-to-commit capability, single-operation decomposition and replan | identical bundle digest, no sibling inference, local-vs-global blocker and honest readiness tests | none during implementation |
| M110-CP05 | Final software gate | frontend, no-WING, full suite, static scan, independent review, commit, full Git-tree binding | clean committed source, released development lease, passed canonical marker and source digest | none |
| M110-HW01 | Route proof | proof-owned backend in analyze/observation mode | issued artifact/ledger, exact console/capture/headroom | attended real WING |
| M110-HW02 | Exact bundle proofs | one canonical permanent bundle at a time, forced restore | `proof_passed`, WAL `proof_restored`, physical restoration and identical bundle digest | attended real WING |
| M110-HW03 | DSP reset | exact channels 1–16 maintenance proposal | complete preimage, applied reset, finite-manifest equality, no-outside-send and delivered-observer-event evidence | attended real WING |
| M110-HW04 | Staged soundcheck | foundation → tonal/dynamics → spatial/FX | exact reviewed child receipts, replans, headroom, no unresolved important corrections | attended real WING |
| M110-HW05 | Final audit | full readback/reanalysis and operator truth | `safe_core_ready`/`show_mix_ready` or explicit incomplete blockers, score | attended real WING |

## Hardware checkpoint order

1. Confirm the backend-owned CP05 marker binds the entire committed Git tree;
   confirm clean HEAD, no STOP/development lease, one backend, healthy WAL and
   operator presence.
2. Connect observation-only and issue Main 47/48 route proof.
3. Keep the same proof-owned process and use exact
   `arm_m110_attended_control` to enter supervised capability-only mode without
   reconnecting. Any restart/reconnect repeats step 2.
4. Prepare and review the fixed reset bundle, run its exact restore-only proof,
   then issue its commit-once grant, apply it and verify preserved state.
5. Run canonical analysis, generate processor and soundcheck bundles, and
   decompose every staged child into separately reviewed single operations.
6. For each resulting bundle in stage order: review exact bytes, run the
   matching restore-only proof, issue the exact commit-once grant, and apply.
   Only fixed reset and atomic processor bundles may be composite; family-level
   proof is insufficient.
7. Bound every local replan, regenerate and re-prove any changed bundle, and
   preserve already verified safe sibling progress unless a global hazard stops
   the run.
8. Re-read route and affected surfaces twice, remeasure audio and record the
   honest terminal readiness.

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

- M110-CP00: accepted for implementation.
- M110-CP01: implementation is present and offline-validated; hardware remains
  gated by CP05.
- M110-CP02..CP05: pending.
- M110-HW01..HW05: forbidden until CP05 is committed and clean.
- A pending/stale CP05 marker blocks connect, USB capture, route proof and every
  discovery/alternate-mixer ingress before resource construction; cleanup and
  emergency messages remain available.

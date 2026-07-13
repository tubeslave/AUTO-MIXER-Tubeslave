# Real WING soundcheck — 2026-07-13

## Outcome

The real WING Rack was connected, measured twice, and analyzed through the
canonical AutoSoundcheck path. The run produced proposals only. No console
correction was sent or committed.

Final result: `soundcheck_incomplete`.

The requested correction reset was **not performed**. The current repository
has no approved maintenance transaction that can reset only DSP, preserve
preamp/fader/routing state, physically verify every write, and roll the complete
operation back. Using the broad legacy reset scripts would also change trim,
faders, or pan and would bypass the current supervised safety boundary.

## Console and capture

- Console: WING Rack `ATMOSPHERA-WNG`, `192.168.1.102`, firmware
  `3.0.5-4-ga4ea833d:release`.
- Capture: real `WING` CoreAudio device, 48 inputs, 48 kHz.
- Sources: channels 1–16; the console returned all expected channel names.
- The 60-second observation captured 564 real meter frames with Main 47/48
  present in every frame.
- In that observation, Main L/R sampled peaks were `-11.2 dBFS`, median RMS was
  about `-30.9 dBFS`, and the hottest input was VOCAL at `-2.7 dBFS`.
- Channels 3, 14, and 15 had no signal; channel 13 was almost silent during the
  continuous observation.

Primary observation evidence:
`artifacts/soundcheck_runtime_observation/20260713T052931Z_soundcheck_runtime_observation/summary.json`.

## Proposal-only result

The second assist run completed recommendation derivation and populated the
operator queue:

- 105 logical requirements;
- 59 require operator review;
- 46 staged supervised transactions remain pending;
- 0 physical steps committed;
- 101 pending queue records in total;
- normal live family allowlist: 0;
- global blocker: `master_capture_route_proof_required`.

The final Main verdict was `ISSUES`: `-32.65 LUFS`, `-13.92 dBTP`, stereo
correlation `0.997`, and lead-vocal presence `+7.47 dB`. This is a very quiet,
vocal-forward measurement, not a completed show mix.

Machine-readable summary:
`artifacts/soundcheck_runtime_observation/20260713T053749Z_real_wing_assist_soundcheck_summary.json`.

## No-write and unchanged proof

Before and after the soundcheck, the official read-only runner performed two
physical round trips for each of 240 parameters: 16 channels × 15 supported
parameter types. Both runs passed and all 240 values matched exactly.

- Before:
  `artifacts/wing_readonly_preflight/20260713T052747Z_wing_readonly_preflight/summary.json`.
- After:
  `artifacts/wing_readonly_preflight/20260713T054228Z_wing_readonly_preflight/summary.json`.
- Comparison:
  `artifacts/wing_readonly_preflight/20260713T054228Z_wing_readonly_preflight/comparison_to_20260713T052747Z.json`.

This proves no observed change on that measured surface. It is not a full WING
snapshot: routing, mute, sends, inserts, FX, buses, DCA, Main, scenes, channels
17–48, and complete EQ/dynamics state remain outside this comparison.

The unchanged after-state also proves that reset did not happen: EQ remains on
for 16/16 channels, dynamics for 16/16, gates for 6/16, and non-zero EQ gains
remain, including channel 16 high gain at `+3 dB`.

## Defects found and changes made

1. `query_mixer_parameter` was missing from the unauthenticated read-only
   allowlist even though the official physical preflight runner requires it.
2. The runner sent a 240-query plan faster than the backend ingress bucket and
   silently timed out after the first 30 frames. Large plans are now paced.
3. The automatic proposal-import broadcast exceeded 64 MiB because it repeated
   full raw evidence inside every result. Automatic import events are now
   compacted and hard-limited to 128 KiB; full proposals remain in the queue.
4. The full decision-queue response is still not bounded by serialized bytes.
5. During the recorded run, eight legacy fader records exposed
   `item.can_apply=true` while the effective queue/operator capability was
   `can_apply=false`. Apply was not attempted. Post-run hardening now keeps
   unmatched soundcheck corrections operator-review-only with
   `soundcheck_correction_requires_supervised_transaction`, and repeats that
   gate in backend preflight/apply before idempotency or console send. The
   machine summary above intentionally remains the immutable historical result.
6. M006 owner intent is recorded, but no correction family has current normal
   live apply authority and the immutable Main 47/48 route proof is absent.

## Verification

- `tests/test_soundcheck_handlers.py`: 15 passed.
- Combined focused server/preflight/soundcheck tests: 91 passed.
- Post-run P1 regression: 56 soundcheck/transaction tests and 30 apply-preflight
  tests passed; an independent read-only patch review found no findings.
- Physical readback: 240/240 before and 240/240 after, zero differences.
- Backend shutdown was graceful and disconnected the WING client.
- The final no-WING quick check remained failed only because the artifact root
  intentionally retains seven earlier failed preflight attempts; it selected
  `20260713T054228Z_wing_readonly_preflight` as the latest valid candidate and
  confirmed the manual-write audit, decision traces, decision memory, WAL, and
  soundcheck parent journal were unchanged.

## Required next work

1. Add a backend-integrated DSP-only maintenance reset transaction with durable
   WAL, complete physical snapshot, CAS readback, and reverse rollback.
2. Prove the immutable Main 47/48 capture route and fresh headroom evidence.
3. Bound full decision-queue responses by serialized byte size and produce a
   fresh live import artifact demonstrating the corrected apply truth surface.
4. Promote correction families only through isolated physical apply/readback/
   rollback proofs, then execute the staged plan and re-measure.

Until those are complete, the project must not report `safe_core_ready`,
`show_mix_ready`, or a completed reset.

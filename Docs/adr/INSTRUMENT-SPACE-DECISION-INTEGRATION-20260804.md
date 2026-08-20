# ADR: Instrument Space decision integration

Status: accepted and implemented in software; final validation in progress

Date: 2026-08-04

## 1. Context

Automixer has a read-only Canvas-inspired Instrument Space that plots measured
pan, spectral centroid and level. The project needs this evidence to participate
in correction decisions for WING and dLive without adding a console-specific or
visual-gesture write path.

## 2. Options considered

1. Keep Instrument Space visual-only.
2. Generate direct pan/EQ/fader corrections from scene coordinates.
3. Build deterministic decision support that annotates the existing proposal
   and safety pipeline.

## 3. Decision

Choose option 3. Instrument Space relations are coarse evidence used to select
and prioritize existing analyzers. They never become `TypedCorrectionAction`,
never contain mixer commands and never grant queue or live authority.

The initial relation model uses only measured values:

- pan proximity;
- log-frequency/centroid proximity;
- level competition;
- recognition confidence and evidence freshness.

It may classify a pair for review and recommend a next analyzer family. It may
not choose a final EQ band, pan target, fader target or gain target. Final
corrections remain owned by the existing frequency-separation/ERB, pan/phase,
balance, proposal, Safety Kernel and supervised transaction paths.

## 4. Why this won

It creates a real end-to-end product path while preserving the strongest
existing safety boundary. The decision evidence is console-neutral; WING and
dLive differences remain confined to their transport, readback and promotion
contracts.

## 5. Rejected alternatives

- Visual-only work does not influence Automixer decisions.
- Direct coordinate-to-write control invents authority and treats coarse
  spectral evidence as a correction diagnosis.
- A separate scene proposal queue would duplicate the established review and
  apply control plane.

## 6. Implementation plan

1. Add versioned relation, conflict and drift DTOs.
2. Keep a bounded in-memory snapshot history, bound to mixer/audio source
   identity and reset on source changes.
3. Embed relations, history and runtime drift in
   `automixer.instrument_space.v1` as decision support.
4. Add a proposal annotation bridge that preserves proposed values and all
   existing blockers, carries measurement provenance and rejects stale or
   undated Instrument Space evidence.
5. Render operator-readable conflicts, drift, measurements, sources and next
   analyzers.
6. Add deterministic, dependency, unsafe-contract, source-reset and bounded
   performance tests.

## 7. Test plan

Run focused unit/integration/frontend contract tests, the production frontend
build, diff validation and the safe no-console validation rails. A dirty-tree
CP05 failure must be reported as a source-state blocker, not bypassed.

The local performance contract evaluates all 2,016 channel pairs at the 64
object ceiling but emits at most 128 non-clear relations. The timing assertion
uses a generous 250 ms ceiling; measured local evidence is recorded separately
and is not a production latency guarantee.

## 8. Risks and rollback

Centroid proximity can over-report masking. Therefore the result is triage,
not correction proof, and requires an existing analyzer before action.
Rollback is removal of the decision-support field and annotation call; the
visualizer and existing proposal behavior remain independently functional.
The in-memory history requires no mixer, queue or console cleanup.

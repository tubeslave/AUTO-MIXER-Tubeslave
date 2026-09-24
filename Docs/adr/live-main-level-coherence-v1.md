# Live Main Level Coherence PATCH_VERIFY v1

Date: 2026-09-24
Status: accepted for R3 software gate; physical meter provider/HIL pending

## Context

Routing identity alone is not sufficient proof that the reserved USB return is a trustworthy post-console Main observation. A correctly labelled route may still be silent, delayed, stale or otherwise inconsistent with the physical Main bus. Conversely, matching values during silence are not useful evidence of signal identity.

The canonical live runtime already measures RMS/peak/crest from an explicitly routed post-console Main USB tap. The missing second proof layer is an independent physical Main-meter observation covering the same short time window.

Repository audit did not find a validated physical WING Main-meter OSC endpoint that can safely be promoted to production authority. This ADR therefore defines the evidence and comparison contract without guessing a meter address.

## Decision

`backend/live_runtime/patch_verify.py` owns a read-only two-layer PATCH_VERIFY gate:

1. `MainTapPatchVerifier` performs fresh physical WING routing readback for every reserved USB slot and requires the exact declared `MAIN N` source.
2. `MainTapLevelCoherenceVerifier` compares the resulting post-console USB evidence with typed `PhysicalMainMeterEvidence` from an independent meter provider.
3. `MainTapPatchGateVerifier` succeeds only when both route identity and level coherence succeed.

The level gate is fail-closed. It requires finite evidence, a bounded timestamp skew, an active signal above a minimum level and a bounded peak difference. Independent RMS is checked whenever provided and can be made mandatory by policy.

The initial software safety defaults are:

- maximum timestamp skew: 100 ms;
- maximum peak difference: 1.5 dB;
- maximum RMS difference: 1.5 dB;
- minimum active peak: -50 dBFS;
- independent RMS optional until a validated physical provider exposes equivalent RMS ballistics.

These values are startup safety bounds, not asserted WING calibration constants. Hardware HIL must measure meter ballistics/latency and may tighten the policy before production authorization.

## Safety properties

- PATCH_VERIFY never writes or repairs mixer routing.
- Main is never synthesized from input stems.
- Near-silence cannot establish level coherence merely because two readings are similarly low.
- Stale or time-misaligned evidence is rejected.
- A route mismatch short-circuits the level proof.
- No WING physical meter OSC path is hard-coded until its endpoint and semantics are validated on hardware.
- Passing this software gate alone does not yet authorize autonomous Main mutation. The concrete physical meter provider plus WING HIL are still required.

## Migration classification

**KEEP_CORE**
- WING OSC query/callback transport and fresh physical routing readback.

**ADAPT**
- level/RMS/peak evidence primitives behind typed `live_runtime` contracts.
- the legacy AutoFOH observability warning remains useful: raw input channels cannot prove post-console processing.

**ARCHIVE**
- legacy `PendingActionEvaluation`, proxy acoustic evaluation policy, heuristic improvement thresholds and rollback orchestration in `backend/autofoh_evaluation.py` after runtime references are severed. They do not become Main PATCH_VERIFY authority.

**DELETE_AFTER_PROOF**
- none in this pass.

## Validation

Focused tests cover:

- close peak evidence inside the time window;
- stale physical evidence;
- false coherence during silence;
- peak mismatch;
- optional and required RMS behavior;
- non-finite physical evidence;
- aggregate route + level success;
- route failure short-circuiting level verification;
- no WING writes anywhere in PATCH_VERIFY.

## Remaining HIL gate

Implement and validate a concrete independent physical Main-meter provider. It must prove the exact WING meter endpoint, units, tap point, channel semantics, update cadence and latency. Then run hardware PATCH_VERIFY against the simultaneously captured post-console USB Main return. Until that evidence exists, autonomous Main actions remain blocked even though the software coherence verifier is complete.

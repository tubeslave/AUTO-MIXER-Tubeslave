# Live EQ Evidence -> Service Composition v1

Date: 2026-09-23
Status: implemented in software, physical WING HIL still required

## Decision

The canonical live service now composes the realtime EQ path instead of forcing callers to manually stitch together physical WING band discovery and the Director.

The path is:

`EqTargetEvidence -> fresh WING F/Q readback -> RealtimeEqLocatorSelector -> EqBandLocator -> live_runtime Director -> ProposedAction -> LiveControlPlane -> WING write -> fresh readback -> VerifiedAction`

`LiveSoundcheckService.propose_hypothesis()` accepts explicit realtime EQ evidence keyed by `(channel, intent)`. It resolves only the evidence supplied by the realtime analysis layer. It does not infer a PEQ band from instrument presets or import the old AutoEQ decision policy.

`LiveSoundcheckService.select_eq_locator()` is read-only. Low-confidence or unmatched evidence returns no locator, which leaves the corresponding EQ hypothesis non-actionable.

The service reuses one `WingWriteAdapter` for read-only band selection and control writes. The adapter still requires a fresh inbound OSC callback for every query, so sharing the adapter does not turn optimistic local `WingClient.state` into authority.

## Why this seam exists

Before this pass the pieces were individually safe but composition lived in tests/callers:

- `EqTargetEvidence` represented spectral evidence;
- `RealtimeEqLocatorSelector` selected an existing physical band;
- `decision_engine.propose_one()` accepted explicit locators;
- `LiveControlPlane` verified writes.

That left room for future callers to bypass one of the gates or to provide a manually invented locator. The service now owns the canonical composition seam for an active WING session.

## Migration classification

### KEEP_CORE
- `backend/wing_client.py`
- `backend/wing_addresses.py`
- `backend/osc/*`
- fresh callback/readback transport behavior

### ADAPT
- `backend/live_runtime/service.py`: now owns realtime EQ evidence composition as well as lifecycle/control composition.
- `backend/live_runtime/wing_adapter.py`: shared read-only/write adapter for the same physical WING transport.
- `backend/auto_soundcheck_engine.py`: remains a temporary bridge for discovery/audio/connection plumbing only.

### ARCHIVE
- `backend/auto_eq.py`: old heuristic EQ authority remains a runtime reference in `backend/server.py`, so it is not moved yet.
- legacy AutoFOH EQ decision paths: retain only until references are severed and HIL proves the replacement.
- `backend/cross_adaptive_eq.py`: reference material for DSP/evidence ideas only; its hard-coded policy is not imported into `live_runtime`.

### DELETE_AFTER_PROOF
None in this pass.

## Automated evidence

`tests/test_live_runtime_eq_composition.py` covers:

1. explicit harshness evidence;
2. fresh enumeration of the physical WING PEQ bands;
3. selection of the existing matching band;
4. Director creation of a relative EQ-gain action;
5. BENCH_TEST execution through `LiveControlPlane`;
6. relative gain resolution against fresh current gain;
7. physical-style readback verification and rollback value capture;
8. low-confidence evidence failing closed without even querying the console.

The focused live CI workflow includes this composition test.

## HIL gate before legacy write authority is severed

On a real WING in explicitly declared BENCH_TEST:

1. capture band F/Q/gain before;
2. supply one narrow high-confidence spectral target near that existing band;
3. confirm the selector chooses the visible band;
4. apply a small gain delta (target: <= 0.5 dB for the first HIL run);
5. confirm the physical console moves;
6. require fresh post-write readback;
7. rollback to the captured gain and verify it again.

Until this passes, legacy AutoEQ write paths remain ARCHIVE candidates rather than deleted code.

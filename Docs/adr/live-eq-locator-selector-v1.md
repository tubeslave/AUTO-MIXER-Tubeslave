# Live EQ Locator Selector v1

Date: 2026-09-23
Status: simulator-tested migration slice; physical WING HIL pending

## Decision

`backend/live_runtime/eq_locator.py` is the canonical selector that turns explicit realtime spectral evidence into an `EqBandLocator` for an already-existing WING PEQ band.

The selector does not invent a PEQ slot and does not move frequency or Q. It receives:

- evidence center frequency from the realtime analysis layer;
- evidence confidence;
- an explicit maximum frequency distance in octaves;
- optional acceptable/preferred Q constraints;
- fresh physical WING frequency/Q fingerprints for bands 1..4.

It returns one existing `EqBandLocator` only when the evidence is confident enough and an existing band falls inside the declared constraints. Otherwise it returns `None`, leaving the Director's EQ idea non-actionable.

## Why

The first channel-EQ cutover intentionally required an explicit `EqBandLocator`, but did not yet define how that locator is chosen. Choosing a fixed band number or a hard-coded instrument frequency would simply move the old heuristic ambiguity one layer upstream.

The selector therefore separates three concerns:

1. realtime analysis identifies a concrete spectral target;
2. fresh WING state tells us where the four physical PEQ bands actually are now;
3. the selector decides whether any existing band is suitable for a gain-only correction.

No suitable existing band means no write. Frequency/Q movement can be migrated later under its own bounded contract and HIL gate.

## Runtime invariants

1. Low-confidence evidence does not query the console and cannot produce a locator.
2. Band frequency distance is measured in octaves, not raw Hz.
3. Optional Q bounds are eligibility gates; preferred Q is only a secondary ranking term.
4. Selection returns the fresh physical band's actual frequency/Q fingerprint.
5. The selector never changes mixer state.
6. `WingWriteAdapter.read_eq_locators()` obtains fresh inbound OSC readback for all four band frequency/Q pairs and never trusts the optimistic local `WingClient.state` cache.
7. Missing or distant bands fail closed.
8. The selected locator must still pass the existing pre-write fresh F/Q fingerprint check immediately before gain mutation.

## Renovation classification

### KEEP_CORE

- `backend/wing_client.py`
- `backend/wing_addresses.py`
- OSC callback/query transport
- fresh physical readback semantics

### ADAPT

- `backend/live_runtime/wing_adapter.py`: adds read-only fresh PEQ fingerprint enumeration for the selector.
- `backend/live_runtime/eq_locator.py`: new canonical live selection policy built on explicit evidence and current console state.
- `backend/cross_adaptive_eq.py`: remains an ADAPT candidate only for evidence/DSP ideas. Its current `CrossAdaptiveEQ` policy hard-codes band centers, channel priorities, overlap tolerance and mirror boost/cut behavior, so it is not imported as a live decision authority.

### ARCHIVE

- `backend/auto_eq.py` after runtime imports are severed and replacement HIL passes.
- legacy AutoFOH channel-EQ decision policy after the new Director/selector/control-plane chain owns the required behavior.

### DELETE_AFTER_PROOF

None in this slice. Legacy EQ runtime references still exist and physical WING HIL is still pending.

## Replacement evidence

Focused tests cover:

- fresh readback of all four WING band frequency/Q fingerprints;
- nearest eligible band selection from explicit spectral evidence;
- fail-closed behavior when no existing band is within the evidence corridor;
- Q eligibility constraints without moving Q;
- low-confidence suppression before any console query;
- selected physical locator feeding the new harshness Director hypothesis;
- invalid evidence/channel validation.

## Next gate

The software chain can now produce a deliberate locator without importing legacy AutoEQ policy. The remaining channel-EQ migration gate is physical WING HIL: select a band from measured evidence, apply one small `BENCH_TEST` `eq_gain_delta_db`, verify fresh F/Q/gain readback, then rollback and record the evidence. Only after that should direct legacy AutoEQ write authority begin to be severed.

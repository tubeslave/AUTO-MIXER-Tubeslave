# ADR: System Inspector Phase 0 and Snapshot Replay

## 1. Context

Automixer needs an independent read-only Inspector for macOS + WING Rack +
Dante. The repository already has a capable WING client, but that class mixes
zero-argument reads with value-bearing OSC writes and many setters. The current
live milestone is externally gated, and this work must not contact or change
hardware.

The first vertical must work entirely from snapshots, distinguish imported
Dante routes from live/audio proof, emit evidence-backed findings and keep
unknown data visible.

## 2. Options considered

1. Reuse `WingClient` and call only getters.
2. Add Inspector logic directly to `server.py`.
3. Create an isolated offline package with an explicit immutable model.
4. Build a generic graph/rules framework before the first detector.
5. Start with UI and fill the backend contract later.

For Dante:

1. Assume a live Dante Controller API.
2. Automate Controller GUI.
3. Use validated offline imports first and capability-gate any official
   licensed API later.

## 3. Decision

Create `backend/system_inspector/` as a pure offline analysis package.

- It does not import WING/network/audio/write transports.
- The first source adapters are versioned JSON snapshot loaders.
- Show profile uses safe YAML; baseline is a separate immutable JSON input.
- The model is explicit typed dataclasses, not a generic graph framework.
- Phase 0 includes one detector module with the initial MVP safety checks.
- `UNKNOWN` is a first-class status.
- Readiness is numeric only above minimum coverage and always includes
  coverage, confidence, unknowns and freshness.
- The v1 read-only invariant cannot be disabled.
- A runtime block/audit function exists because the product explicitly
  requires attempted writes to raise and be recorded; AST tests independently
  prove there is no write transport to reach.
- JSON/CLI ship first. UI, HTML/PDF and live collectors remain later phases.
- Dante is import-only until a concrete supported adapter and tests exist.

## 4. Why this won

It is the smallest design that covers the complete Mac → Dante → WING route
vertical without touching live code. It keeps replay deterministic, makes the
trust boundary testable, avoids presenting ping as Dante health and leaves a
clear adapter seam for future consoles/protocols.

The explicit model is justified because even the first detector must join
devices, endpoints, subscriptions and WING inputs. A fully generic graph engine
would add indirection without a second implementation proving its value.

## 5. Rejected alternatives

- `WingClient` import: rejected because write methods remain in the dependency
  graph.
- `server.py` integration: rejected because it couples offline analysis to the
  live coordinator.
- Controller GUI automation: rejected as fragile and unsafe.
- Assumed public Dante API: rejected; only configured official/licensed APIs
  may be implemented.
- UI-first: rejected because it would visualize an unstable contract.
- No runtime guard: rejected because it would miss an explicit product
  acceptance criterion.
- Score without coverage: rejected as false precision.

## 6. Implementation plan

1. Bounded profile/bundle parsing.
2. Immutable normalized model.
3. Expected/observed route construction.
4. Initial detectors.
5. Transparent readiness/coverage.
6. Deterministic reports and CLI.
7. Security/static/determinism tests.
8. Complete Phase 0 documentation and iteration report.

No existing transport or frontend file is modified.

## 7. Test plan

- identical inputs produce identical JSON bytes;
- healthy fixture has no findings;
- duplicate Playback R emits the stable evidence-backed finding;
- missing route is `UNKNOWN`;
- read-only cannot be disabled;
- write attempts raise and are redacted in the security log;
- unsafe YAML and duplicate IDs fail closed;
- AST proves no live/write dependency;
- input profile/snapshots remain unchanged;
- CLI exit code distinguishes clean, finding and input-failure outcomes;
- focused tests, compile, diff check and standard project suite.

## 8. Risks and rollback

Risks:

- future real exports may require schema migration;
- fixture confidence can be mistaken for hardware confidence;
- readiness weights are provisional;
- a later collector can accidentally widen dependencies.

Mitigations:

- schema versions and raw references;
- simulator/import labels and evidence limitations;
- visible coverage/unknowns;
- permanent AST boundary test and manual live checkpoints.

Rollback is removal of the new Inspector package, CLI wrapper, fixtures,
tests and documents. No database, baseline, WING, Dante, macOS or frontend
state is migrated or mutated by this decision.


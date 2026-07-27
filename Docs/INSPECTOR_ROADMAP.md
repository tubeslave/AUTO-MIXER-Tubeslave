# System Inspector Roadmap

Status: Phase 0 vertical implemented; live checkpoints not entered

## Phase 0 — Audit, capability research and replay vertical

Implemented:

- repository audit for WING client, OSC, write gate, frontend, connection model
  and audio capture;
- product, architecture, capability matrix, threat model and ADR;
- read-only boundary and security audit;
- immutable normalized model;
- versioned snapshot bundle and safe show profile;
- known-good baseline comparison;
- healthy and duplicated-right fixtures;
- deterministic CLI/JSON report;
- initial detectors and focused tests.

Still open in Phase 0:

- fixture catalog for every acceptance scenario;
- HTML report;
- complete schema documentation/JSON Schema files;
- baseline create/version/export/import commands.

## Phase 1 — Complete system model

- add ClockDomain, AudioSignal, SoftwareProcess, Computer, Bus, MainOutput and
  CableOrLogicalLink;
- capability and confidence propagation rules;
- snapshot manifest digests and migration policy;
- serialization and adversarial schema tests.

Hardware/network: none.

## Phase 2 — macOS collector

- allowlisted commands/APIs with `shell=False` and timeouts;
- interfaces, service order, routes, DNS, USB adapter identity and power state;
- processes and CoreAudio metadata;
- raw snapshot persistence;
- cross-version fixtures.

Manual checkpoint before the first local live collection or permission prompt.

## Phase 3 — WING read adapter

- narrow `WingReadClient` protocol;
- typed/allowlisted official endpoints;
- firmware, patch, channels, buses, Main, clock and meter snapshots;
- timeout/latency/freshness evidence;
- unknown endpoint denial and no-write proof.

Manual checkpoint before the first real WING connection.

## Phase 4 — Dante import and topology

- Dante Controller preset XML parser after format fixtures are provided;
- CSV and manual topology providers;
- baseline workflows;
- expected/observed subscription and clock model;
- optional Dante Managed API research only when a real licensed DDM deployment
  is declared.

Manual checkpoint before reading real Dante Controller files or using any
licensed SDK/API.

## Phase 5 — Rules engine

- complete connectivity, routing, clocking, configuration drift and playback
  safety scenario matrix;
- finding lifecycle, deduplication, acknowledgement and recovery proof;
- false-positive suppression that cannot hide critical findings.

Hardware/network: none using replay.

## Phase 6 — Signal analysis

- explicit device selection and permission UX;
- CoreAudio pre-fader capture;
- RMS/peak/true peak, clipping, correlation, polarity indication, duplicate
  channels, dropouts and noise floor;
- metrics-only storage by default.

Manual checkpoint before first capture or permission request.

## Phase 7 — Risk engine and reports

- owner-reviewed category weights;
- coverage/confidence/freshness policy;
- JSON, HTML and PDF;
- baseline diff and incident timeline;
- no false precision and no live-authority inference.

## Phase 8 — UI

- backend read-only API/WebSocket surface;
- System Inspector tab;
- overview, devices, network, Dante, WING routing, signal flow, findings,
  baseline and reports;
- non-color status labels and accessible topology.

## Phase 9 — live validation

Only after snapshot gates are green:

1. approved local macOS read-only collection;
2. approved WING observation-only connection;
3. imported Dante topology comparison;
4. no-write verification;
5. operator review of findings.

No actuation is part of this roadmap.


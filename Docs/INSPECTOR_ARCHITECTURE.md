# System Inspector Architecture

Status: Phase 0 implemented boundary

## Logical pipeline

```text
Validated profile + offline snapshot files
                    |
                    v
            Bounded input loading
                    |
                    v
              Normalization
                    |
                    v
        Immutable explicit system graph
                    |
                    v
       Route comparison and detectors
                    |
                    v
      Findings + readiness/coverage model
                    |
                    v
         Deterministic CLI / JSON report
```

No edge from this graph reaches Decision Engine, Actuation Layer, Supervised
Write Gate, `WingClient`, OSC transport, CoreAudio or network scanners.

## Physical placement

The repository uses a flat `backend/` architecture with many top-level
production modules. The Inspector is isolated as
`backend/system_inspector/` rather than adding more concerns to `server.py`.

```text
backend/system_inspector/
  model.py       immutable entities, graph, findings, readiness
  io.py          bounded JSON bundle loader
  profile.py     safe YAML show-profile loader
  normalize.py   snapshot-to-graph normalization
  detectors.py   Phase 0 evidence-backed detectors
  readiness.py   transparent score, coverage and unknowns
  reports.py     deterministic JSON and concise text
  safety.py      fail-closed v1 guard and redacted security audit
  engine.py      pure offline orchestration
  cli.py         command contract
```

The placement decision and alternatives are recorded in
`Docs/adr/system-inspector-phase0-replay-20260727.md`.

## Dependency boundary

Phase 0 package imports are limited to:

- Python standard library;
- PyYAML safe loader;
- other modules inside `backend.system_inspector`.

An AST regression test rejects imports of:

- `wing_client`;
- `pythonosc` or `osc`;
- `socket`;
- `subprocess`;
- `sounddevice`;
- `audio_capture` or CoreAudio bindings.

It also rejects `.send()` and `set_*()` calls in the Inspector package.
Local report/security-log file writes are allowed; device/config/network writes
are not.

## Read-only guard

`INSPECTOR_READ_ONLY` is a compile-time `True` invariant in v1.

- unset environment: read only;
- `INSPECTOR_READ_ONLY=true`: read only;
- any explicit false value: startup fails closed;
- no CLI flag can disable it;
- `prohibit_write()` records only a redacted action class and raises
  `InspectorWriteProhibited`;
- targets, addresses, values, credentials and payloads are not logged.

The runtime guard is defence in depth. The structural absence of write
dependencies remains the primary control.

## Snapshot bundle

A replay directory contains:

- `macos.json` — `automixer.inspector.macos.v1`;
- `wing.json` — `automixer.inspector.wing.v1`;
- `dante.json` — `automixer.inspector.dante-import.v1`;
- `baseline.json` — `automixer.inspector.baseline.v1`.

The profile is an explicit YAML path. Each input is limited to 1 MiB and the
four-file bundle to 4 MiB. Symlinks, missing files, wrong top-level types,
unknown schema versions, duplicate stable IDs and invalid confidence values
fail closed.

## Normalized model

Implemented Phase 0 entities:

- `Device`;
- `NetworkInterface`;
- `AudioEndpoint`;
- `TransmitChannel`;
- `ReceiveChannel`;
- `Subscription`;
- `ConsoleInput`;
- `ConsoleChannel`;
- `Bus`;
- `MainOutput`;
- `ClockDomain`;
- `AudioSignal`;
- `SoftwareProcess`;
- `Computer`;
- `CableOrLogicalLink`;
- `ExpectedRoute`;
- `ObservedRoute`;
- `RouteComparison`;
- `Evidence`;
- `Finding`.

Each entity with physical/logical identity carries `EntityMetadata`: stable ID,
display name, vendor/model/address/protocol where available, availability,
last seen, source, confidence, capabilities and a raw-source reference. Phase 0
defines the complete model but only populates entities supported by current
snapshots; empty collections are not a claim of observed healthy state.

## Determinism

- inputs are never modified;
- all graph collections are sorted by stable ID or role;
- JSON keys are sorted;
- timestamps come from input snapshots, never implicit `now()`;
- equal inputs produce equal report bytes and SHA-256.

The security audit uses a clock because it records a runtime attempted action;
it is outside the canonical replay result.

## WING adapter direction

Existing `backend/wing_client.py` contains both reads and writes. Inspector does
not import it. Phase 3 must introduce a narrow `WingReadClient` protocol and
verified allowlisted query adapter. The adapter must:

- expose typed zero-argument queries only;
- reject unknown endpoints;
- record query/result/timeout evidence;
- never expose `send(address, value)` or setters;
- stop before first live connection until the operator approves the hardware
  checkpoint.

## Dante provider direction

The provider interface remains a roadmap contract:

```python
class DanteTopologyProvider:
    async def get_devices(self): ...
    async def get_channels(self): ...
    async def get_subscriptions(self): ...
    async def get_clock_status(self): ...
    async def get_health(self): ...
```

Only JSON snapshot import is implemented now. Dante Controller preset XML/CSV,
manual topology and any official API adapter remain unimplemented capabilities
and must not be advertised as working.

# System Inspector Iteration 001

Date: 2026-07-27  
Scope: Phase 0 research and offline Snapshot Replay vertical  
Hardware/network contact: none  
Actuation authority: none  
Live writes: none

## Outcome

The first evidence-backed System Inspector vertical is implemented entirely
offline:

```text
show profile + simulated macOS + simulated WING + imported Dante + baseline
  -> validated normalization
  -> explicit system graph
  -> route comparison
  -> evidence-backed findings
  -> readiness/coverage
  -> CLI and deterministic JSON
```

The duplicate Playback R fixture emits the stable critical finding
`routing.playback-right-duplicate` with source paths and observed values. The
healthy fixture emits no findings, but its overall result remains `UNKNOWN` at
a coverage-adjusted score of 71 because unsupported checks are not silently
promoted to OK. Partial evidence remains `UNKNOWN`.

This is not a live WING/Dante validation and is not the complete Inspector MVP.
HTML/PDF, frontend UI, live macOS/WING collectors, broader fixture coverage and
baseline lifecycle commands remain roadmap work.

## Delivered

- product, architecture, capability matrix, threat model, roadmap and ADR;
- Council brief, two proposals, mutual critiques and synthesized decision;
- fail-closed read-only invariant and redacted security audit;
- bounded, versioned, symlink-rejecting snapshot/profile input;
- immutable normalized system model and topology;
- imported Dante provider contract without a fabricated live API;
- initial connectivity, clock, routing and playback-safety detectors;
- coverage-aware readiness result with visible unknowns and freshness;
- deterministic CLI/text/JSON replay;
- healthy and duplicated-right fixtures;
- unit, CLI, determinism, security and static dependency-boundary tests.

## Safety evidence

- `backend/system_inspector` has no import path to `WingClient`, OSC transport,
  sockets, subprocess, CoreAudio/sounddevice or the actuation layer.
- The Inspector has no CLI option that disables read-only mode.
- `INSPECTOR_READ_ONLY=false` fails closed.
- `prohibit_write()` raises and records only a redacted action class.
- An AST test rejects live/write dependencies, `.send()` and `set_*()` calls.
- No WING connection, active network scan, Dante Controller read, permission
  request or audio capture occurred.

## Validation

Passed:

- Python compilation for the Inspector package and CLI wrapper;
- focused Inspector suite: `12 passed`;
- Inspector plus existing write-surface static suite: `26 passed`;
- standard suite with the pre-existing CP05 current-tree test deselected:
  `7284 passed, 3 skipped, 1 deselected`;
- `git diff --check`;
- healthy and faulted CLI replay.

The unmodified standard command reached:

```text
1 failed, 6761 passed, 3 skipped
```

The sole failure was
`tests/test_wing_mvp_rails.py::test_m120_current_truth_tracks_accepted_route_and_teardown_repeat_blocker`.
It reports `m110_cp05_tracked_tree_not_clean`. The tracked dirt was already
present before this iteration:

- `frontend/package.json`;
- `reports/loop_soundcheck_pipeline.jsonl`.

Those user-owned changes were preserved. The Inspector does not modify either
file, and this iteration does not rebind or claim the existing hardware gate.
Black, Ruff and a configured Python type checker are not installed in the
current environment; no new development dependency was added.

## Manual checkpoints preserved

Approval is still required before:

- first local live macOS collection or permission prompt;
- first real WING connection;
- any active network scan;
- any real Dante Controller file/configuration read;
- any licensed Dante SDK/API use;
- first live audio capture;
- any device, route, network or mixer write.

## Next offline iteration

Expand the fixture/scenario catalog and schemas, then implement and test the
allowlisted macOS collector behind an explicit live-collection checkpoint.
The existing M001 external WING checkpoint and zero-authority state remain
unchanged.

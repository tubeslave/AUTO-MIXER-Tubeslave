# Codex Proposal: System Inspector Phase 0 and Snapshot Replay

## 1. Short thesis

Build the first Inspector slice as a small pure-analysis package under
`backend/system_inspector/`, fed only by validated fixture/import files. Keep
live collectors behind later manual checkpoints and make the no-write boundary
an executable contract plus a static dependency test.

## 2. What I understood

The product must explain the real MacBook → Dante → WING route with evidence,
but the first implementation must work without hardware and must never infer
Dante health from network reachability. The current `WingClient` mixes query
and mutation methods, so importing it directly would violate the desired
dependency boundary even if only query methods were called.

## 3. Proposed solution

- Define immutable dataclasses and enums for devices, interfaces, endpoints,
  subscriptions, console inputs, expected/observed routes, evidence, findings,
  coverage and readiness.
- Validate one versioned snapshot-bundle schema with bounded file size,
  strict required keys, unique stable IDs and confidence in `[0, 1]`.
- Parse a safe YAML show profile with explicit roles and forbidden
  destinations.
- Normalize imported snapshots into a deterministic system graph.
- Run a small detector registry whose first detector catches missing,
  duplicated and swapped Playback L/R routes.
- Calculate an intentionally coarse readiness result with coverage,
  confidence and unknown checks; do not claim Dante audio flow.
- Provide `python -m backend.system_inspector.cli inspector scan` and a thin
  `tools/automixer.py inspector ...` wrapper.
- Add JSON report generation first. HTML is a later bounded iteration unless
  the vertical slice remains small enough to include safely.
- Add a security audit logger and write guard, then static-test that Inspector
  source has no `WingClient`, `.send(`, setter, socket, subprocess or audio
  capture imports.

## 4. Likely files to touch

- `backend/system_inspector/**`
- `tools/automixer.py`
- `config/show_profile.yaml`
- `tests/fixtures/system_inspector/**`
- `tests/test_system_inspector_*.py`
- `Docs/INSPECTOR_*.md`
- `Docs/adr/system-inspector-phase0-replay-20260727.md`
- `.paperclip/reports/inspector-iteration-001.md`
- Council brief/proposal/review/decision files under `.ai/`

No existing production transport or frontend file is required for the first
vertical slice.

## 5. Alternatives considered

- Reuse `WingClient` now: rejected because the class exposes writes and
  connection itself is a manual checkpoint.
- Extend `server.py` immediately: rejected because Phase 0 can remain a pure
  offline package and `server.py` is an oversized live coordinator.
- Treat Dante Controller as a live API: rejected; current supported path is
  import-only unless a licensed official API is configured later.
- Build the UI first: rejected because it would visualize an unproven data
  contract.

## 6. Risks

- Overclaiming route confidence when one leg is imported rather than observed.
- Schema drift between profile, fixtures and future live collectors.
- Read-only guard being treated as sufficient while an unsafe dependency is
  still importable.
- Risk score suggesting false precision.
- Repository-wide static scans treating guard test tokens as real writes.

## 7. Test plan

- Healthy replay is deterministic and has no critical finding.
- Duplicate Playback R fixture emits the exact finding and evidence paths.
- Missing data produces `UNKNOWN`.
- Invalid/oversized/duplicate-ID inputs fail closed.
- YAML safe loader rejects object construction.
- Write guard raises and records a redacted security event.
- Baseline/profile input bytes remain unchanged.
- Inspector package static dependency scan proves no live/write imports.
- CLI and JSON output are deterministic apart from explicitly controlled time.
- Run focused tests, compile/type checks available in the repo, diff check, then
  the standard suite if the focused gate is green.

## 8. Where the other agent may disagree

An independent reviewer may prefer a generic graph abstraction immediately or
may argue for direct reuse of `WingClient`. My recommendation is to keep the
first model explicit and typed; introduce generic graph machinery only after
two real adapters prove the shared shape.


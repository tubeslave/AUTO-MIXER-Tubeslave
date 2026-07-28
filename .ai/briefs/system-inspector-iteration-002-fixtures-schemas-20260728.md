# System Inspector Iteration 002 Brief — Fixture Catalog, Schemas, Baseline Lifecycle

Issued by the process lead (Claude) on 2026-07-28 after independently
verifying and committing iteration 001 (`3e5ce5eef` source, `ea5f5d737`
final marker rebind; rails `138 passed`; full repository suite
`7303 passed, 3 skipped, 0 failed`).

## Goal

Close the remaining offline Phase 0 items from `Docs/INSPECTOR_ROADMAP.md`
without any hardware, network, Dante Controller or audio contact:

1. A scenario fixture catalog covering every shipped detector family.
2. JSON Schema files and schema documentation for all snapshot inputs and
   the show profile.
3. Baseline lifecycle CLI commands (`create`, `show`, `validate`, `export`,
   `import`).

HTML report, Phase 1 model classes and the macOS collector stay out of this
iteration; the macOS collector remains behind its manual first-live-collection
checkpoint.

## Scope: fixture catalog

One fixture directory per scenario under `tests/fixtures/system_inspector/`,
each with tests asserting exact finding IDs, severities, CLI exit code and
byte-identical JSON across repeated runs:

- `discovery.required-device-missing.<id>`
- `connectivity.required-device-offline.<id>`
- `network.interface-changed.<role>` and `network.interface-role-drift.<role>`
- `network.wifi-used-for-dante`
- `clocking.sample-rate-unknown` and `clocking.sample-rate-mismatch`
- `routing.required-route-missing.<role>`
- `playback.<role>-routed-to-main`
- existing `healthy_system` and `playback_right_duplicate` remain unchanged.

Readiness semantics per fixture stay per `Docs/INSPECTOR_PRODUCT.md`:
`UNKNOWN` is never promoted, coverage below 50% keeps `score: null`.

## Scope: schemas

- JSON Schema (draft 2020-12) files for `macos.json`, `wing.json`,
  `dante.json`, `baseline.json` and the show profile, shipped inside the
  Inspector package and documented in Docs.
- Loader behavior stays bounded and fail-closed; adversarial fixtures
  (wrong `schema_version`, duplicate stable IDs, oversized file, symlink,
  non-object top level) are covered by tests.
- No new runtime dependency: if a JSON Schema validator library is not
  already available, implement the narrow checks directly and keep the
  schema files as the documented contract.

## Scope: baseline lifecycle

- `automixer inspector baseline create --snapshot <dir> --output <file>`
  derives a versioned baseline from a validated snapshot bundle.
- `show`/`validate` are read-only; `import` validates schema and content
  digest of an explicit file path.
- File output goes ONLY to an explicitly provided `--output` path, rejects
  symlinks, refuses to overwrite an existing file without `--force`, and
  never mutates fixtures, `config/`, or any input in place.
- Baseline files carry `schema_version` and a content digest.

## Safety boundary (unchanged from iteration 001)

No WING, OSC, CoreAudio, Dante or network contact. No write-client imports.
`INSPECTOR_READ_ONLY` stays fail-closed and cannot be disabled. YAML uses
`yaml.safe_load`. Missing evidence stays `UNKNOWN`, never `OK`. The AST
dependency-boundary test remains mandatory and must cover every new module.
Replay evidence grants no live authority and does not alter M001/CP05.

## Process discipline (mandatory; learned from iteration 001)

1. Run validation AFTER the final writes to `ops/autopilot/STATE.json`,
   `HEARTBEAT.md`, `LAST_REPORT.md`, `RUN_LEDGER.jsonl` and all operator
   handoff surfaces. Validation that predates a truth edit is void.
2. The new `RUN_LEDGER.jsonl` row must use the complete handoff schema:
   `authority`, `decision_ledger`, `current_blocker`, `next_operator_action`,
   `external_validation_requirements`, `hardware_contacted`, `live_writes` —
   and the "Latest ledger handoff row" line plus heartbeat timestamp must be
   updated on every surface the rails enumerate.
3. Never deselect failing rails to make a run pass. A red dynamic CP05 test
   on a dirty mid-iteration tree is expected and is resolved only by the
   source commit plus exactly one marker-only CP05 rebind
   (`backend/m110_cp05_hardware_gate.py`).
4. Bare `N passed, M skipped` result strings in ledger `results` are bound to
   the committed M115 baseline contract; prefix any new full-suite result
   (for example "full repository suite: ...").
5. Before reporting, verify every declared deliverable is tracked by git
   (iteration 001 left `tools/automixer.py` untracked).

## Definition of done

- Focused inspector suites and the write-surface static scan pass.
- Determinism proven for every scenario fixture.
- Full repository suite green after the truth-surface sync.
- Council flow recorded as for iteration 001: proposals, mutual critiques,
  decision, ADR update if architecture changes.
- The process lead independently verifies and lands the source checkpoint
  plus the marker-only CP05 rebind.

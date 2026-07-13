# ADR: No-WING Release Check And Local Package Timestamp Policy

## Context

Automixer needs a repeatable handoff check that can run without a real WING Rack.
The check must prove repo health, frontend health, offline supervised proposal
behavior, and macOS package structure without connecting to hardware or sending
live write commands.

A full local package check initially failed during Electron `codesign` because
the timestamp service was unavailable. The app bundle and backend executable
were otherwise produced.

## Options considered

- Keep separate manual commands only.
- Add a no-WING release-check runner that orchestrates the existing commands.
- Require timestamped signing for every local package smoke.
- Disable timestamping for local package smoke and keep public
  distribution/notarization separate.

## Decision

Add `tools/no_wing_release_check.py` as the local no-WING readiness runner and
disable Electron mac timestamping in `frontend/package.json` for local package
smoke.

## Why this won

The runner turns the current safe validation path into one reproducible command
with logs and a machine-readable summary. Disabling timestamping removes an
external service dependency from local package smoke while keeping package
structure, backend resources, and config resources verifiable.

## Rejected alternatives

- Manual commands only: too easy to skip checks or lose evidence.
- Timestamp required for local smoke: fragile because an unavailable timestamp
  service blocks local validation even when the app bundle is structurally
  correct.
- Silent package failure ignore: unsafe because package readiness must be a
  visible pass/fail result.

## Implementation plan

- Add no-WING release-check runner.
- Add tests for command safety, summary output, and blocked-port behavior.
- Update runbook and implementation plan.
- Preserve public signing/notarization as a separate release step.

## Test plan

- `PYTHONPATH=backend python -m pytest tests/test_no_wing_release_check.py -q`
- `python3 tools/no_wing_release_check.py --include-package --artifact-root artifacts/no_wing_release_check/current`
- Verify `frontend/dist/mac-arm64/AUTO MIXER Tubeslave.app/Contents/Resources/backend/server`.
- Verify required config files under `Contents/Resources/config/`.

## Risks and rollback

Risk: local package smoke is not the same as public notarized distribution.
Mitigation: docs explicitly preserve notarization as a separate check.

Rollback: remove the runner and restore Electron mac timestamping if release
signing infrastructure becomes stable enough for every local run.

## 2026-07-12 safety addendum: isolate fake proof runtime state

### Context

The full no-WING gate exposed that `gain_proof_dry_run_runner.py` used a
`FakeProofMixer` but still let `AutoMixerServer` select the production per-user
console runtime directory. The runner therefore appended committed synthetic
apply/rollback rows to `~/.automixer/runtime/manual_write_wal.jsonl`. It did not
contact WING or open a WING/OSC transport, but a dry-run gate must not mutate the
operator's recovery WAL.

The same gate also exposed an Electron E2E fixture that omitted the newly
required `packagedOfflineLock` bootstrap field. Production code correctly
failed closed; the fixture was stale.

### Decision

- Patch `AutoMixerServer._resolve_console_runtime_dir` only inside the
  in-process fake gain-proof runner so its writer lock and WAL live below that
  proof's artifact run directory.
- Do not add a production environment override that could split the real
  console writer lock or WAL.
- Snapshot the canonical manual-write evidence log and the per-user console WAL
  before every no-WING run. Append explicit final checks that fail the gate if
  either file's existence, size or SHA-256 changes.
- Keep the synthetic WAL rows produced by the failed 2026-07-12 gate for
  provenance; do not truncate or rewrite recovery history.
- Update the Electron E2E fixture to provide the complete secure bootstrap
  schema and continue asserting authentication precedes Product Layer traffic.

### Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_gain_proof_dry_run_runner.py tests/test_no_wing_release_check.py -q`
- `npm --prefix frontend run test:e2e`
- Standalone fake gain proof with before/after per-user WAL size and SHA-256
- Full `PYTHONPATH=backend python3 tools/no_wing_release_check.py` after writer
  lease release

### Risks and rollback

The fake proof intentionally exercises the real proposal/WAL code against a
fake mixer, so its run-local WAL must remain present in the artifact directory.
Removing WAL writes entirely would weaken the proof. Rollback is to revert the
runner-local resolver patch only if an equivalent non-production runtime
injection is implemented without widening real runtime configuration.

## 2026-07-12 superseding addendum: four-store integrity and offline trace quarantine

### Context

The broader audit found that the affected fake runs had appended `32/32`
synthetic committed rows to the per-user console WAL, not only the eight rows
from the final reproducer. It also found `25,730` dry-run trace rows in the
canonical decision-trace store and related unbound decision-memory rows. The
append-only evidence files must not be rewritten, but fake/offline rows must
not render as verified physical readback or contribute to applied/verified
operator counts.

### Decision

- Preserve all four canonical files byte-for-byte: manual-write events,
  decision traces, decision memory and the per-user console WAL.
- Make the no-WING and packaged-runtime gates snapshot existence, byte size and
  SHA-256 for all four stores, and fail if any changes.
- Normalize dry-run, replay, fake-mixer and offline decision traces at read
  time to `proof_status=offline` and `applied=offline_simulated`; nested fake
  readbacks become `offline_unverified`.
- Keep the original values in normalization metadata for auditability. Do not
  rewrite or truncate historical JSONL.
- Apply the same classification defensively in the frontend so retained fake
  rows cannot become a green hardware-proof state even if a stale backend
  payload is displayed.
- Use UUID-bearing, exclusively created artifact directories for fake gain
  proof runs so same-second executions cannot share a WAL or overwrite a
  summary.

Current preserved baselines after the contaminated historical runs:

- manual-write events: `115380009` bytes,
  SHA-256 `aaaba5d3b0dd7e9fb9f54b12affdf54ff444a16bba31c1d92297afda39941fc4`;
- decision traces: `246109600` bytes,
  SHA-256 `2392e248399f7c789b76f75af34bdb781df395a3924513c288567bc703d70556`;
- decision memory: `20845559` bytes,
  SHA-256 `47a38efe999e037ea3b8b3226498680135f9ea3e76447e6e98f6bf5d791eb724`;
- console WAL: `11329` bytes,
  SHA-256 `58d548ecd28593d1791ff7784b72c394ae644fa7b101331528eec2454dfe7bca`.

### Verification

The source-bound local arm64 package smoke at
`artifacts/packaged_runtime_smoke/20260712T073254Z_af976647-7612-419d-a5a5-9922930b203e_packaged_runtime_smoke/result.json`
started its packaged backend in `5038 ms`, verified all 18 challenged identity
checks against the embedded 15-file manifest, exact backend, launcher-token
digest and observed PID, opened no external or WING connection, blocked
hardware operations and left all four baselines unchanged. It is an
ad-hoc-signed dirty-tree local artifact with `releaseEligible=false`, not
public-release evidence. The separate final four-store gate at
`artifacts/no_wing_release_check/20260712T070641Z_no_wing_release_check/summary.json`
passed all `65` checks and detached manifest verification with all four
canonical stores unchanged.

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

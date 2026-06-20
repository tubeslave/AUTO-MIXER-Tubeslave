# ADR: WING Read-Only Reserved-Root Intake Guard

## Context

M001 waits for a real WING Rack read-only artifact and operator review. The
repo also contains no-WING artifacts and autopilot evidence under reserved
control-plane roots. Those roots are useful for prestart and validation, but
they must never be accepted as the real WING external-validation artifact.

## Options Considered

1. Trust the claimed `wing_readonly_preflight.v1` summary fields only.
2. Reject candidate artifact roots and candidate summary/event paths that live
   under reserved no-WING or autopilot evidence roots.
3. Disable all file-based WING read-only intake until hardware is connected.

## Decision

Use option 2. WING read-only evidence intake now rejects candidate
`artifact_root`, `summary_path`, `events_path` or source file paths under
`artifacts/no_wing_release_check/...` and `ops/autopilot/evidence/...`.

## Why This Won

It preserves the separation between control-plane/no-WING proof and real WING
external validation. A future operator can still provide a real artifact, but
it cannot be laundered through reserved no-WING or autopilot evidence
directories.

## Rejected Alternatives

- Trusting summary fields alone was rejected because a copied or spoofed
  summary could live under a no-WING evidence root.
- Disabling file intake was rejected because operator-supervised hardware runs
  still need artifact review tooling.

## Implementation Plan

1. Add reserved-root detection for candidate WING artifact root, summary path,
   event path and source path.
2. Keep prestart no-WING summary references allowed; those prove the backend
   prestart no-contact gate and are not candidate WING artifacts.
3. Record M095 rail coverage without changing WING readiness.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_readonly_preflight_evidence.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m095_wing_readonly_reserved_root_intake_guard_stays_readonly -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

No live transport is introduced. Rollback is to revert the M095 commit, which
restores the previous WING read-only intake behavior and M095 state/report
rows.

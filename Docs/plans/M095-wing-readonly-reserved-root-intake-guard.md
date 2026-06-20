# Plan M095: WING Read-Only Reserved-Root Intake Guard

## Checkpoints

| Checkpoint | Objective | Files | Expected behavior | Boundary | Tests | Evidence | Rollback point | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M095-CP01 | Reject spoofed WING candidates from no-WING roots | `backend/wing_readonly_preflight_evidence.py`, `tests/test_wing_readonly_preflight_evidence.py` | `artifacts/no_wing_release_check/...` cannot be accepted as a real WING read-only candidate even if a summary claims `wing_readonly_preflight.v1` | Read-only artifact intake; no backend/WING/OSC contact | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_readonly_preflight_evidence.py -q` | focused evidence run | revert reserved-root checks | completed |
| M095-CP02 | Reject spoofed WING candidates from autopilot control-plane evidence roots | `backend/wing_readonly_preflight_evidence.py`, `tests/test_wing_readonly_preflight_evidence.py` | `ops/autopilot/evidence/...` cannot be accepted as a real WING read-only candidate; it remains control-plane evidence only | Read-only artifact intake; no hardware contact | focused evidence run | focused run | revert reserved-root checks | completed |
| M095-CP03 | Record intake guard without changing WING readiness | `Docs/ROADMAP.md`, `Docs/CURRENT_STATE.md`, `ops/autopilot/LAST_REPORT.md`, `ops/autopilot/STATE.json`, `ops/autopilot/RUN_LEDGER.jsonl`, `Docs/reports/AUTOMIXER_AUTOPILOT_M095_WING_READONLY_RESERVED_ROOT_INTAKE_GUARD_20260620.md` | M095 is recorded as external-validation intake hardening while M001/M002 remain waiting for real hardware/operator validation | Report/control-plane only; no readiness closure | WING rail, full suite, memory lint | M095 rail and full-suite evidence recorded after validation | revert state/report updates | completed |

## Required validation at checkpoint close

- WING read-only preflight evidence focused tests pass.
- M095 rail proves reserved no-WING/autopilot evidence roots cannot become
  accepted real WING read-only candidates.
- Standard backend test command passes in a clean staged worktree.
- Memory lint passes.
- `ops/autopilot/STATE.json` and `ops/autopilot/RUN_LEDGER.jsonl` remain valid.
- No backend/WING/OSC transport, live write, proposal apply, queue import,
  readiness closure, hardware evidence or Full Autonomous readiness is created.

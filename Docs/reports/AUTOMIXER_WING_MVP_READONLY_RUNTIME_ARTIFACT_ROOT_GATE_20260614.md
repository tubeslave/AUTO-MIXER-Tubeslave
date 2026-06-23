# Automixer WING MVP read-only Runtime Artifact Root Gate

Date: 2026-06-14
Status: Validated

## Summary

Real WING read-only preflight summaries now record the runtime `artifact_root`.
Evidence intake rejects read-only candidates when the summary omits
`artifact_root` or when the referenced offline expected plan declares a
different root.

## Changed Surfaces

- `tools/wing_readonly_preflight_runner.py` records `artifact_root` in real-run
  summaries before any WebSocket contact.
- `backend/wing_readonly_preflight_evidence.py` treats missing
  `artifact_root` as invalid evidence and compares it against the offline plan.
- `tests/test_wing_readonly_preflight_runner.py` checks successful read-only
  summaries include the runtime artifact root.
- `tests/test_wing_readonly_preflight_evidence.py` covers missing summary root
  and mismatched plan/root candidates.
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md` documents the new acceptance and
  fail-closed criteria.

## Safety Boundary

- No WING hardware contact.
- No backend/WING listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No live authority granted.
- No readiness closure from offline evidence.

## Validation

- Focused read-only runner/evidence tests: `93 passed`.
- Full no-WING release-check module: `273 passed, 8 skipped`.
- Project Director/WING rails: `11 passed`.
- `PYTHONPATH=backend python -m py_compile
  tools/wing_readonly_preflight_runner.py
  backend/wing_readonly_preflight_evidence.py
  tests/test_wing_readonly_preflight_runner.py
  tests/test_wing_readonly_preflight_evidence.py` passed.
- `PYTHONPATH=backend python tools/memory/memory_lint.py --root
  /Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main` passed.
- `/opt/homebrew/bin/git diff --check` passed.
- Project Director JSON selected `WING read-only connection proof`, reported
  `selected_blocker_status: tooling_ready_waits_real_wing`, evidence count
  `118`, baseline coverage passed with no missing markers, and guard
  `on_target 90 False`.

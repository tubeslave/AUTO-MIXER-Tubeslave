# Automixer WING MVP Evidence SHA Format Gate

Date: 2026-06-14
Status: Validated

## Summary

Read-only WING evidence intake now rejects malformed
`expected_plan_sha256` and `backend_port_prestart_summary_sha256` values. A
candidate can no longer satisfy the digest presence checks with an arbitrary
non-empty string.

## Changed Surfaces

- `backend/wing_readonly_preflight_evidence.py` reuses the shared
  `is_sha256_hexdigest()` helper for expected-plan and prestart-summary digest
  validation.
- `tests/test_wing_readonly_preflight_evidence.py` covers malformed expected
  plan and no-WING prestart summary digests.

## Safety Boundary

- No WING hardware contact.
- No backend/WING listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No live authority granted.
- No readiness closure from offline evidence.

## Validation

- Focused read-only evidence tests: `58 passed`.
- Full no-WING release-check module: `273 passed, 8 skipped`.
- Project Director/WING rails: `11 passed`.
- `PYTHONPATH=backend python -m py_compile
  backend/wing_readonly_preflight_evidence.py
  tests/test_wing_readonly_preflight_evidence.py` passed.
- `PYTHONPATH=backend python tools/memory/memory_lint.py --root
  /Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main` passed.
- `/opt/homebrew/bin/git diff --check` passed.
- Project Director JSON selected `WING read-only connection proof`, reported
  `selected_blocker_status: tooling_ready_waits_real_wing`, evidence count
  `119`, baseline coverage passed with no missing markers, and guard
  `on_target 90 False`.

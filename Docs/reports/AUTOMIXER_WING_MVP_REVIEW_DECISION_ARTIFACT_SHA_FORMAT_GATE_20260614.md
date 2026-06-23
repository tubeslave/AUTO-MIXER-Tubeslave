# Automixer WING MVP Review Decision Artifact SHA Format Gate

Date: 2026-06-14
Status: Validated

## Summary

WING read-only review-decision records now reject malformed `summary_sha256`
and `events_sha256` values. Operator decisions can no longer carry arbitrary
non-empty artifact digest strings at the manual acceptance boundary.

## Changed Surfaces

- `backend/wing_readonly_preflight_review_decision.py` reuses the shared
  `is_sha256_hexdigest()` helper for manual summary/event-log digest fields.
- `tests/test_wing_readonly_preflight_review_decision.py` covers malformed
  review-decision artifact digests and updates its valid candidate fixture with
  the required `artifact_root`.

## Safety Boundary

- No WING hardware contact.
- No backend/WING listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No live authority granted.
- No readiness closure from offline evidence.

## Validation

- Focused review-decision ledger tests: `23 passed`.
- Full no-WING release-check module: `273 passed, 8 skipped`.
- Project Director/WING rails: `11 passed`.
- `PYTHONPATH=backend python -m py_compile
  backend/wing_readonly_preflight_review_decision.py
  tests/test_wing_readonly_preflight_review_decision.py` passed.
- `PYTHONPATH=backend python tools/memory/memory_lint.py --root
  /Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main` passed.
- `/opt/homebrew/bin/git diff --check` passed.
- Project Director JSON selected `WING read-only connection proof`, reported
  `selected_blocker_status: tooling_ready_waits_real_wing`, evidence count
  `120`, baseline coverage passed with no missing markers, and guard
  `on_target 90 False`.

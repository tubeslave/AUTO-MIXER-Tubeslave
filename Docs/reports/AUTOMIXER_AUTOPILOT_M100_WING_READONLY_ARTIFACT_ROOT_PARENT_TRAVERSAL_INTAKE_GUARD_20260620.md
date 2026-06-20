# Automixer Autopilot M100: WING Read-Only Artifact Root Parent Traversal Intake Guard

## Summary

M100 closes a WING read-only artifact-intake parent-traversal gap. Candidate
summaries now fail before manual review when `artifact_root` contains `..` as a
path segment while still ending in the canonical WING read-only artifact root.

The concrete violation is `artifact_root_contains_parent_traversal`.

## Boundary

This is read-only intake hardening only. It contacts no WING/backend transport,
opens no WebSocket/OSC write path, imports no proposal into a queue, applies no
proposal, grants no live authority, closes no readiness row, creates no
hardware evidence and does not replace real WING/operator validation.

## Validation

- Focused WING read-only evidence tests:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_readonly_preflight_evidence.py -q`:
  `72 passed`.
- M100 rail:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m100_wing_readonly_artifact_root_parent_traversal_intake_guard_stays_readonly -q`:
  `1 passed`.
- Full WING MVP rail:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`:
  `113 passed`.
- Full staged suite:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`:
  `4031 passed, 1 skipped, 1 warning`.
- Memory lint and JSON/JSONL validation: see `ops/autopilot/STATE.json`.

## Result

M001 remains `WAITING_FOR_EXTERNAL_VALIDATION`. The next real unblocker is still
an operator-supervised WING read-only preflight artifact and manual review.

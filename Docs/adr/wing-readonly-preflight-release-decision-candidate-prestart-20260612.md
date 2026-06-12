# WING Read-Only Preflight Release Decision Candidate Prestart

## Context

The WING read-only preflight path requires an operator-auditable chain from candidate capture through manual review and release summary. Decision review tooling can include candidate prestart contract rows, but the no-WING release summary previously kept only counts and source IDs.

## Options considered

1. Keep candidate prestart fields only in the decision review report.
2. Copy `candidate_reviews` into the release summary as compact JSON.
3. Build a custom markdown table for selected candidate fields.

## Decision

Use option 2. The release summary now serializes and renders `candidate_reviews` from the decision ledger detail.

## Why this won

It is lossless, scoped, and consistent with the existing release summary style. It keeps the prestart contract visible without introducing any backend startup, WING query, OSC/WebSocket write, or live authority change.

## Rejected alternatives

- Review-only visibility was rejected because release handoff remains incomplete without candidate prestart evidence.
- A custom table was deferred because it would either duplicate selected fields or risk hiding future required fields.

## Implementation plan

- Extend `wing_readonly_preflight_review_decision_summary`.
- Render `candidate_reviews` in the markdown `WING Read-Only Preflight Review Decision Ledger` section.
- Extend the existing no-WING summary test with backend-port prestart candidate fields.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py -k "summary_embeds_wing_preflight_review_decision"`
- Affected WING/no-WING tests.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`

## Risks and rollback

The main risk is a noisier markdown summary. Rollback is safe: remove the `candidate_reviews` copy/rendering and the added test assertions. No live behavior or console authority is affected.

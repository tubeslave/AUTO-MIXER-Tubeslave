# WING Read-Only Decision Template Autodiscovery

## Context

The WING read-only preflight review tool writes
`artifacts/wing_readonly_preflight_review_decisions/pending_review_decision_template.json`.
That file is intentionally not a completed operator decision. In a dirty
workspace, automatic decision-root discovery loaded the pending template as a
decision record and turned otherwise isolated read-surface checks into
replacement-required failures.

## Options considered

- Keep loading every JSON file under the decision root.
- Stop generating the pending decision template.
- Ignore the generated pending template during automatic discovery while still
  validating it when explicitly passed as a decision path.

## Decision

Ignore the generated pending decision template filename during automatic
decision-root discovery. Explicit `decision_paths` still load and validate the
template, and an unedited template still fails ledger acceptance.

## Why this won

This preserves fail-closed behavior for real operator decisions while avoiding
test and release-check pollution from the generated pending template. The
operator must still replace placeholders before any decision is accepted.

## Rejected alternatives

Loading every JSON file was rejected because generated scaffolding should not be
treated as a submitted decision. Removing the template was rejected because it is
useful operator guidance and records the exact fields required for review.

## Implementation plan

- Add a shared ignored filename tuple based on
  `WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT`.
- Filter that filename from automatic root discovery only.
- Add a regression test proving auto-discovery ignores the template.
- Add a regression assertion proving explicit template paths still fail until
  the operator edits them.

## Test plan

Run:

```bash
PYTHONPATH=backend python3 -m pytest tests/test_wing_readonly_preflight_review_decision.py -q
PYTHONPATH=backend python3 -m pytest tests/test_wing_readonly_preflight_runner.py -q
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

## Risks and rollback

Risk: a real operator decision accidentally named
`pending_review_decision_template.json` would be ignored by automatic discovery.
Rollback is to rename the real decision file or remove the discovery filter.
Explicit `decision_paths` remains available for forensic review of any file.

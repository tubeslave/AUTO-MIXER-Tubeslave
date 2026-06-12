# ADR: WING MVP Rows Structured Gate

## Context

The WING read-only runbook, handoff packet, and no-WING release summaries now
use a structured `supervised_prestart_gate` contract. The WING MVP blocker rows
for supervised fader and gain proof still mentioned only accepted decision JSON
and the final operator-sequence step.

## Options considered

1. Leave blocker rows unchanged.
2. Add the structured gate equality requirement to supervised fader/gain rows.
3. Replace detailed row wording with a compact gate label.

## Decision

Use option 2. The supervised fader and gain rows now require
`supervised_prestart_gate` to exactly match
`expected_supervised_prestart_gate`.

## Why this won

The blocker table feeds generated progress reports and Project Director
behavior. It must expose the same proof gate as the runbook and release
artifacts.

## Rejected alternatives

Leaving the rows unchanged was rejected because it kept a weaker status surface.
A compact label was rejected because the current safety contract is clearest as
explicit field names.

## Implementation plan

- Patch `.ai/project-layer/wing_mvp_blockers.md`.
- Extend WING MVP rails tests.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Record verification in a report.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py -k "blocker_table or progress_tool"`
- Affected WING/no-WING tests.
- `python3 tools/memory/memory_lint.py`
- Full pytest suite.

## Risks and rollback

The main risk is verbose row text. Roll back by introducing a named structured
gate reference only after a canonical schema document exists.

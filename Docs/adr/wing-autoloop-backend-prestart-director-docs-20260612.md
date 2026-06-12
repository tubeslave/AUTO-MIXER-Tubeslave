# ADR: WING Autoloop Backend Prestart Director Docs

## Context

The supervised WING MVP now has two machine-checkable prestart boundaries before
supervised fader/gain proof: a first backend-port prestart gate and a final
supervised-proof gate. The Codex cycle prompt and blocker table carried both,
but the autoloop README only named the final gate. Project Director JSON reports
also include `forbidden_scope` policy strings; direction guard should not score
those negative policy fields as executed drift work.

## Options considered

1. Keep the README unchanged because the prompt is already stricter.
2. Update the README and tests to require the backend prestart gate.
3. Modify Project Director code to synthesize the backend gate independently.
4. Ignore Project Director JSON policy fields only during drift scoring.

## Decision

Use options 2 and 4.

## Why this won

The blocker table remains the source of truth for Project Director selected
actions, while the README is the operator-facing contract for recurring Codex
cycles. Both should describe the same safety boundary. Filtering JSON policy
fields keeps the guard strict for real drift while avoiding a false stop caused
by negative `forbidden_scope` text.

## Rejected alternatives

Option 1 leaves stale operator-facing automation docs. Option 3 risks creating
another source of truth instead of reading the blocker table. Ignoring entire
Project Director reports was rejected because they carry useful WING target
evidence.

## Implementation plan

- Add the backend-port prestart gate wording to `tools/codex_autoloop/README.md`.
- Extend Project Director tests to assert the selected action, prompt, and
  README all carry the backend gate.
- Remove JSON `forbidden_scope` and `forbidden_next_actions` policy fields only
  from direction guard drift scoring.
- Add a regression test for Project Director JSON.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py tests/test_codex_direction_guard.py`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: documentation may be overread as hardware readiness. Another risk is
over-filtering drift evidence; the filter is limited to JSON policy fields and
keeps real report bodies scoreable. Rollback is a simple revert of
README/test/report/guard changes; no live authority is changed.

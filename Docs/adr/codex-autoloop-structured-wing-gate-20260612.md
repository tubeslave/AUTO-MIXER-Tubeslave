# ADR: Codex Autoloop Structured WING Gate

## Context

The WING MVP proof boundary now requires the no-WING handoff summary to show
`supervised_prestart_gate` exactly matching `expected_supervised_prestart_gate`
and `supervised_prestart_gate.grants_live_authority: false`. The Codex autoloop
prompt and README still described the WING-connected sequence without this
structured gate.

## Options considered

1. Leave autoloop guidance unchanged.
2. Add structured gate checks to the prompt and README.
3. Replace the existing read-only preflight wording with only structured gate
   wording.

## Decision

Use option 2. Autoloop guidance now requires the structured handoff gate before
any supervised fader/gain proof while preserving the existing read-only
preflight and manual-review wording.

## Why this won

Autonomous cycles read prompt guidance directly. The guidance must match the
same release-summary evidence required by the runbook, blocker table, MVP doc,
and capability matrix.

## Rejected alternatives

Leaving the prompt unchanged was rejected because it could preserve a
trace-only proof boundary. Removing read-only preflight wording was rejected
because the structured gate does not replace physical evidence.

## Implementation plan

- Patch `tools/codex_autoloop/codex_cycle_prompt.md`.
- Patch `tools/codex_autoloop/README.md`.
- Extend prompt/readme tests.
- Record validation in a report.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py -k codex_prompt`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py -k prompt_and_readme`
- Affected WING/autoloop tests.
- `python3 tools/memory/memory_lint.py`
- Full pytest suite.

## Risks and rollback

The main risk is prompt verbosity. Roll back by introducing a named structured
gate section only if all current field names remain covered by tests.

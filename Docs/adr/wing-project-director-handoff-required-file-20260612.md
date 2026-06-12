# ADR: WING Project Director Handoff Required File

## Context

The supervised WING MVP currently depends on a no-WING handoff summary that
contains structured backend and supervised prestart gates. Project Director
selects the first non-closed blocker and tells future cycles which files must be
read before planning. It already listed the runner, review tool, no-WING release
check, and runbooks, but omitted the handoff packet builder that emits the gate
contract.

## Options considered

1. Keep the builder implicit through the no-WING release checker.
2. Add `tools/build_wing_readonly_preflight_handoff_packet.py` to Project
   Director required files.
3. Add a broader WING docs bundle.

## Decision

Use option 2.

## Why this won

The selected action explicitly depends on handoff summary fields. Listing the
builder keeps automation and reviews pointed at the direct source of the
structured gate without widening live authority.

## Rejected alternatives

Option 1 hides a critical contract behind another tool. Option 3 adds noise and
does not target the first blocker as directly.

## Implementation plan

- Add the builder path to `required_files`.
- Add Project Director test coverage.
- Record this planning-only change in a report.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: required files become too broad. Rollback is to remove the one path and
its test assertion. No WING transport or live authority is changed.

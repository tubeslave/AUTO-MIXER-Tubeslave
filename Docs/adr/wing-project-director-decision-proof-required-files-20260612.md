# ADR: WING Project Director Decision Proof Required Files

## Context

The WING MVP selected action requires accepted review source ids and a valid
accepted-candidate prestart trace before supervised proof planning. Those
contracts are implemented by the read-only preflight decision ledger/review
tool and by `backend/supervised_wing_proof_plan.py`. Project Director should
name those files explicitly for future cycles.

## Options considered

1. Leave the files implicit through no-WING release validation.
2. Add only the decision review tool.
3. Add the decision ledger backend, decision review tool, and supervised proof
   plan backend.

## Decision

Use option 3.

## Why this won

The selected action names both accepted review source ids and the proof-plan
blocker. Future cycles need the direct contract files available during planning,
while the change remains audit/planning only.

## Rejected alternatives

Option 1 leaves the chain hidden behind a release check. Option 2 covers
operator decisions but not the proof-plan blocker that prevents supervised
proof when accepted-candidate trace is invalid.

## Implementation plan

- Add the three direct gate files to Project Director `required_files`.
- Add unit assertions.
- Record the planning-only boundary in a report.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: required files become too broad. Rollback is to remove the three paths and
their test assertions. No WING transport or live authority is changed.

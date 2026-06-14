# ADR: WING MVP Autoloop Proof-Plan Summary Fields

## Context

The WING read-only connection proof remains the first open MVP blocker. The
handoff packet now declares the required supervised proof-plan summary fields,
and the no-WING release gate fails closed if the list is missing or stale.
However, the Codex autoloop prompt and README still described only the older
supervised prestart marker checks.

## Options considered

1. Leave the autoloop prompt unchanged and rely on no-WING validation.
2. Update only operator-facing docs.
3. Update the autoloop prompt and README, then protect the wording with static
   tests.

## Decision

Update `tools/codex_autoloop/codex_cycle_prompt.md` and
`tools/codex_autoloop/README.md` so autonomous cycles must verify the handoff
summary exposes `required_supervised_proof_plan_summary_fields`,
`expected_required_supervised_proof_plan_summary_fields`, and
`required_supervised_proof_plan_summary_fields_match_contract: true` before any
supervised proof. Also require the list to include the expected accepted-
candidate source ID and violations map summary fields.

## Why this won

The autoloop runner uses `codex_cycle_prompt.md` as its operating contract. If
that contract lags behind the no-WING gate, later cycles can waste work by
checking weaker readiness conditions. Static tests make the prompt/README drift
visible.

## Rejected alternatives

Relying only on no-WING validation catches stale artifacts late but does not
guide the next cycle. Documentation-only changes outside the prompt would not
affect autonomous execution.

## Implementation plan

- Add the proof-plan summary field checks to `codex_cycle_prompt.md`.
- Mirror the same requirements in `tools/codex_autoloop/README.md`.
- Extend static Project Director and WING rail tests.
- Record the report in the first WING blocker evidence row and docs index.

## Test plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_codex_cycle_prompt.py tests/test_codex_autoloop_contract.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk is limited to stricter autonomous-cycle instructions and static tests.
Rollback removes the prompt/README bullets and corresponding assertions. No
backend startup, WING connection, console query, console write, readiness
closure, or live authority is introduced.

# ADR: WING MVP Handoff Proof-Plan Summary Fields

## Context

The WING read-only connection proof remains the first open MVP blocker. Before
any supervised proof, the operator needs both an accepted read-only preflight
decision and a valid accepted-candidate prestart trace. The supervised proof
plan now exposes canonical expected source IDs and violations maps, but the
handoff packet did not explicitly bind itself to those proof-plan summary
fields.

## Options considered

1. Keep the handoff packet focused only on operator sequence and gates.
2. Mention required proof-plan fields only in docs.
3. Add a shared contract list and validate it in the handoff packet release
   gate.

## Decision

Add `WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_REQUIRED_PROOF_PLAN_SUMMARY_FIELDS`
to the shared WING read-only preflight contract, emit it from the handoff
packet builder, render it in Markdown, and require it in no-WING validation.

## Why this won

It makes the operator handoff packet name the exact proof-plan diagnostics that
must be present before supervised proof planning. A stale packet now fails
closed instead of silently omitting the expected diagnostics contract.

## Rejected alternatives

Keeping this implicit leaves a review gap between the no-WING proof-plan
summary and the handoff packet. Documentation-only wording is not enforceable.

## Implementation plan

- Add the shared required-summary-field tuple to
  `backend/wing_readonly_preflight_contract.py`.
- Include the list in `build_handoff_packet`.
- Render the list in the handoff Markdown.
- Validate and summarize the list in `tools/no_wing_release_check.py`.
- Add positive and stale-packet tests.

## Test plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile backend/wing_readonly_preflight_contract.py tools/build_wing_readonly_preflight_handoff_packet.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_wing_mvp_rails.py tests/test_project_director_agent.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --artifact-root /tmp/automixer_goal_handoff_proof_plan_summary_fields_nowing_20260614`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk is limited to stricter offline handoff packet validation. Rollback removes
the required field list from the packet and validator. No backend startup, WING
connection, console query, console write, readiness closure, or live authority
is introduced.

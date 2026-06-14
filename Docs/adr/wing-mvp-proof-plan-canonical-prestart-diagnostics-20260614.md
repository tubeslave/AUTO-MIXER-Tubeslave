# ADR: WING MVP Proof Plan Canonical Prestart Diagnostics

## Context

The WING read-only connection proof remains the first open MVP blocker. The
next supervised proof plan must not proceed unless an accepted read-only
preflight source has an expected-artifact-bound, no-WING-prestarted candidate
trace.

The no-WING release gate computes the canonical accepted-candidate prestart
verdict through `backend.wing_readonly_preflight_contract`, but prior summaries
only exposed the reported candidate prestart fields. When a report was stale or
incomplete, the mismatch was visible as a violation name but not as the exact
expected source IDs and per-source violations.

## Options considered

1. Keep expected diagnostics internal to the validator.
2. Expose only a canonical expected boolean.
3. Expose canonical expected source IDs and per-source violations in the proof
   plan and no-WING summaries.

## Decision

Expose the canonical expected source IDs and violations map, and require the
reported plan fields to match them.

## Why this won

It gives the operator and Project Director the same handles the release gate
used: which accepted source is expected to qualify, and which prestart contract
fields prevent it from qualifying. This keeps review work tied to the real
WING blocker instead of requiring code inspection.

## Rejected alternatives

Keeping diagnostics internal makes failures harder to review. A boolean is too
coarse because it hides the artifact path and missing contract fields that an
operator needs to correct or recapture.

## Implementation plan

- Add expected prestart diagnostic fields to `build_supervised_wing_proof_plan`.
- In `validate_supervised_proof_report_next_wing_plan`, parse reported
  expected fields, recompute canonical expected fields, and fail on mismatch.
- Return the recomputed expected fields in the no-WING JSON summary.
- Render the recomputed expected fields in the no-WING Markdown summary.
- Add fixture and focused tests for passing and failing expected diagnostics.

## Test plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile backend/supervised_wing_proof_plan.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_mvp_rails.py tests/test_project_director_agent.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --artifact-root /tmp/automixer_goal_proof_plan_canonical_prestart_diagnostics_nowing_20260614`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk is limited to stricter offline report validation and more detailed
summary output. Rollback removes the expected diagnostic fields and mismatch
checks. No backend startup, WING connection, console query, console write,
readiness closure, or live authority is introduced.

# ADR: user-presets-intake-release-sequence-gate-20260605

## Context

User-presets physical readback artifact intake exists as a read-only validator
and summary helper, but the main no-WING release sequence did not execute it.
The generated Markdown summary also omitted the user-presets artifact-intake
section. That let a release report include the user-presets contract while
silently skipping the operator artifact-intake gate.

## Options Considered

1. Keep the helper unused until a physical artifact exists.
2. Run the user-presets artifact-intake validator in the no-WING sequence only.
3. Run the validator and publish the intake block in both JSON and Markdown
   summaries.

## Decision

Use option 3. The no-WING release sequence now runs
`user_presets_physical_readback_artifact_intake_schema` after the user-presets
contract gate and before dLive gates. `summary.json` and `summary.md` both expose
the user-presets artifact-intake status and safety flags.

## Why This Won

The project needs complete, operator-visible read-surface proof evidence before
any scene/preset automation can progress. JSON-only evidence is not enough
because Markdown is used for human review. The validator remains a missing or
manual-review candidate gate; it does not execute preset recall/store and does
not close fixture gaps.

## Rejected Alternatives

- Leaving the helper unused hides missing physical artifacts.
- Adding only the run result still hides the gate from the Markdown operator
  summary.
- Treating a valid artifact as automatic fixture closure was rejected because
  preset recall/store is destructive and still requires explicit operator
  review.

## Implementation Plan

1. Insert the user-presets artifact-intake validator after the user-presets
   contract validator in `run_release_checks()`.
2. Add the Markdown section for `User Presets Physical Readback Artifact Intake`.
3. Add direct validator, run-order, JSON summary, and Markdown summary tests.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_no_wing_release_check.py -q`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend`
- `PYTHONPATH=backend python3 -m py_compile tools/no_wing_release_check.py`
- `/opt/homebrew/bin/git diff --check`

## Risks And Rollback

Risk: user-presets intake visibility is mistaken for live preset authority.
Mitigation: summaries and tests keep `fixture_closure_allowed_by_intake=false`,
`preset_recall_allowed=false`, `preset_store_allowed=false`,
`live_authority_granted=false`, and `full_autonomous_ready=false`.

Rollback: remove the validator from the no-WING run sequence and remove the
Markdown section. This would return the report to contract-only user-presets
visibility and should be treated as a release evidence regression.

# Proposal: user-presets-intake-release-sequence-gate-20260605

## Short Thesis

The existing user-presets artifact-intake validator should be part of the
mandatory no-WING release evidence sequence, with JSON and Markdown visibility,
but it must remain read-only and operator-review-only.

## What I Understood

User presets are a destructive scene/preset surface. The project now has a
schema contract and an artifact-intake validator, but release evidence can omit
the intake step because `run_release_checks()` does not append it. That makes
the no-WING report weaker than the code surface suggests.

## Proposed Solution

- Append `validate_user_presets_physical_readback_artifact_intake_snapshot()`
  immediately after `validate_user_presets_physical_readback_contract_snapshot()`.
- Add a `summary.md` section for
  `User Presets Physical Readback Artifact Intake`.
- Extend tests for direct validation, run ordering, JSON summary, and Markdown
  summary.

## Likely Files To Touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/memory/project.md`
- `Docs/adr/user-presets-intake-release-sequence-gate-20260605.md`

## Alternatives Considered

- Leave the helper unused until physical artifacts exist. Rejected because the
  release gate should prove missing artifacts explicitly.
- Only add the run sequence result. Rejected because Markdown is the operator
  review surface and would still hide the gate.

## Risks

- The gate can be mistaken for permission to recall or store presets. Mitigate
  by preserving false safety flags and asserting them in tests.
- Future summary edits can drop the Markdown block. Mitigate with explicit
  Markdown assertions.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_no_wing_release_check.py -q`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend`
- `PYTHONPATH=backend python3 -m py_compile tools/no_wing_release_check.py`
- `/opt/homebrew/bin/git diff --check`

## Where The Other Agent May Disagree

Kimi might argue for a broader artifact-review queue before touching release
ordering. I would keep this fix narrower because it repairs a concrete missing
gate for an already implemented validator.

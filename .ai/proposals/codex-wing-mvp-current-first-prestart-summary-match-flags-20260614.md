# Codex Proposal: WING MVP Current First Prestart Summary Match Flags

## Short thesis

Require the no-WING handoff prestart summary `*_matches_plan` flags in
`Current First Command` so Project Director does not accept a packet whose
prestart summary contract drifted from the plan.

## What I understood

The packet validator can now detect drift for summary type, status, connection
policy, no-contact booleans, and backend check status. The active Goal-mode rail
should require those exposed verdicts directly, not only the raw values.

## Proposed solution

- Add the match flags to `CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS`.
- Add the same flags to `.ai/project-layer/wing_mvp_blockers.md` Current First
  Command.
- Sync the operator-facing MVP docs and autoloop prompt/readme.
- Add tests that assert the flags appear in Project Director action, docs, and
  generated progress.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `tools/codex_autoloop/update_mvp_progress.py`
- `Docs/MVP_WING_SUPERVISED.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/reports/MVP_PROGRESS_LATEST.md`

## Alternatives considered

1. Rely on validator tests only. This leaves Goal-mode selected action wording
   weaker than the validator.
2. Add the fields only to docs. This would not protect Project Director baseline
   coverage.

## Risks

The main risk is marker-list churn. The benefit is that Goal-mode cycles now
stop on explicit packet-vs-plan drift instead of accepting raw fields alone.

## Test plan

- Run focused Project Director and WING MVP rail tests.
- Generate Project Director JSON and check baseline coverage.
- Run no-WING quick to preserve fail-closed behavior.
- Run full standard pytest before commit.

## Where the other agent may disagree

Kimi may prefer avoiding more marker checks in docs. The counterpoint is that
these are already validator outputs and should be visible in the selected action
that directs future Goal-mode work.

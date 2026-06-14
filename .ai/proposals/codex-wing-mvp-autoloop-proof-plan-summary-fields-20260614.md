# Codex Proposal: WING MVP Autoloop Proof-Plan Summary Fields

## Short thesis

Teach the autonomous Codex cycle prompt and README to require the same
supervised proof-plan summary fields that the handoff packet and no-WING gate
now validate.

## What I understood

The live blocker is still the real WING read-only connection proof. The latest
handoff packet change made a stale packet fail closed when it omits required
proof-plan summary fields. The next autonomous cycle instructions must name
that same contract, otherwise Goal-mode runs can drift back to checking only
older supervised prestart markers.

## Proposed solution

Add `required_supervised_proof_plan_summary_fields`,
`expected_required_supervised_proof_plan_summary_fields`, and
`required_supervised_proof_plan_summary_fields_match_contract: true` to
`tools/codex_autoloop/codex_cycle_prompt.md` and
`tools/codex_autoloop/README.md`. Also require the list to include the
expected accepted-candidate source IDs and violations map fields. Extend the
static Project Director and WING rail tests so future prompt drift fails.

## Likely files to touch

- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/INDEX.md`
- `Docs/reports/AUTOMIXER_WING_MVP_AUTOLOOP_PROOF_PLAN_SUMMARY_FIELDS_20260614.md`

## Alternatives considered

Leave the autoloop prompt implicit. Rejected because autonomous cycles use the
prompt as their operating contract.

Only update the README. Rejected because `codex_cycle_prompt.md` is the file
used by the runner.

## Risks

The risk is prompt bloat. This is acceptable because the fields are already
part of the first WING blocker gate and are protected by focused static tests.
This change does not grant execution authority.

## Test plan

Run focused prompt/Project Director/WING rail tests, Project Director,
direction guard, memory lint, `git diff --check`, and the standard full pytest
command.

## Where the other agent may disagree

Kimi may prefer deriving prompt requirements from the shared Python contract.
That is a reasonable future improvement, but the scoped fix here closes the
current prompt drift with minimal risk.

# Codex Proposal: WING MVP Director Action Coverage Digest

## Short thesis

Expose the selected-action digest and current-first-command coverage verdict
directly in Project Director output.

## What I understood

The shorter `Current First Command` action is useful only if the autonomous
cycle can prove it still covers the WING safety/proof baseline. Without an
explicit verdict and digest, later handoff steps must infer that from long text.

## Proposed solution

- Add a helper that returns the missing baseline coverage markers.
- Emit `selected_action_sha256` from Project Director.
- Emit coverage required/pass fields, marker count, and missing marker list.
- Render the new fields in the markdown directive.
- Document the fields in the autoloop prompt and README.

## Likely files to touch

- `tools/codex_autoloop/update_mvp_progress.py`
- `tools/codex_autoloop/project_director_agent.py`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/adr/wing-mvp-director-action-coverage-digest-20260614.md`

## Alternatives considered

1. Leave the proof implicit in selected action text.
2. Revert to always appending the blocker-row baseline.
3. Publish compact machine-readable proof fields and keep fallback behavior.

## Risks

The risk is adding metadata that future cycles ignore. Updating prompt/README
and tests makes the fields part of the cycle contract.

## Test plan

- Focused Project Director and WING rail tests.
- Project Director JSON and Direction Guard.
- no-WING quick fail-closed check.
- memory lint, diff whitespace check, and full pytest suite.

## Where the other agent may disagree

Kimi may argue that the selected action digest is not sufficient proof of real
readiness. That is correct; this change is only an audit aid for offline
autoloop handoff and does not close readiness.

# Proposal: Require expected-artifact marker in supervised gate

## Short Thesis

The final WING supervised prestart gate should explicitly require an
expected-artifact-bound accepted source id before pointing operators to the
supervised fader/gain proof runbook.

## What I Understood

The decision ledger now enforces expected-artifact binding in code. The operator
handoff and Project Director selected action should say the same thing, because
the final operator-sequence step is the visible transition from read-only proof
review to supervised proof planning.

## Proposed Solution

- Add `expected-artifact-bound accepted source id` to the shared supervised
  prestart gate required markers.
- Update the final operator-sequence step text emitted by the handoff packet.
- Add a no-WING regression that rejects the old three-marker final step.
- Update Project Director, WING rails, MVP, runbook, and autoloop docs/tests.

## Likely Files To Touch

- `backend/wing_readonly_preflight_contract.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`

## Alternatives Considered

Leaving the binding implied by `valid accepted-candidate prestart trace` was
rejected because operators and future agents can miss what made the trace valid.
Adding a separate downstream check in the supervised proof runner was rejected
for this slice because no supervised proof should be reached until the handoff
gate is explicit and fail-closed.

## Risks

Older handoff packets will fail validation. That is intended; they predate the
expected-artifact binding requirement and should be regenerated.

## Test Plan

- Compile touched Python.
- Focused handoff/no-WING/Project Director/WING rail tests.
- no-WING quick release check.
- Full pytest before commit.

## Where The Other Agent May Disagree

Kimi may argue that the marker is redundant. I recommend the explicit marker
because the final operator sequence is a human-facing safety boundary, not only
a code path.

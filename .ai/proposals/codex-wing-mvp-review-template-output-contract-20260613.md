# Codex Proposal: WING MVP Review Template Output Contract

## Short Thesis

Centralize the pending WING read-only review decision template path so every
operator handoff command points to the same JSON artifact.

## What I Understood

The project is still on the supervised WING Rack MVP. The current first blocker
is real `WING read-only connection proof`, and the generated decision template
is a manual-review handoff artifact. If the path differs between the plan,
handoff packet, review CLI, and no-WING validator, the operator can edit one
file while the validator expects another.

## Proposed Solution

Add shared constants for `WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOT` and
`WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT` in
`backend/wing_readonly_preflight_contract.py`. Make
`tools/wing_readonly_preflight_runner.py`,
`tools/build_wing_readonly_preflight_handoff_packet.py`,
`tools/review_wing_readonly_preflight_artifacts.py`, and
`tools/no_wing_release_check.py` consume those constants. Add focused tests that
compare generated commands and defaults against the shared contract.

## Likely Files To Touch

- `backend/wing_readonly_preflight_contract.py`
- `tools/wing_readonly_preflight_runner.py`
- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/review_wing_readonly_preflight_artifacts.py`
- `tools/no_wing_release_check.py`
- WING readonly/no-WING focused tests
- WING MVP report, ADR, and documentation index

## Alternatives Considered

Leaving literals in place is simple but keeps a real handoff drift point.
Centralizing only the filename would not protect the decision root. Moving the
constant into one tool would make other tools depend on implementation details,
so the shared WING preflight contract is the right owner.

## Risks

The main risk is accidentally changing a public CLI default. The change should
preserve the exact path string and only move ownership. Focused tests cover the
runner post-run command, handoff packet commands, review tool default, and
no-WING command construction.

## Test Plan

- Compile touched Python modules.
- Run focused WING readonly runner, handoff, review, and no-WING tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard.
- Run the full pytest suite.

## Where The Other Agent May Disagree

Kimi may argue this is maintenance. The counterpoint is that this exact path is
the operator bridge from real read-only evidence to the decision ledger; drift
here can block or misroute the first real WING proof.

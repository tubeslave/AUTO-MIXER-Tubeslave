# Proposal: WING MVP Handoff Operator Sequence Contract

## Short thesis

The exact WING read-only handoff checklist should be a shared contract, not a
builder-local list protected only by its own digest.

## What I understood

The active MVP still targets supervised WING Rack proof. The first blocker
remains WING read-only connection proof. Offline tooling may strengthen the
handoff rails, but it must not claim live readiness or close the blocker.

## Proposed solution

Add `expected_handoff_operator_sequence()` to the shared read-only preflight
contract. The handoff builder emits the actual sequence plus
`expected_operator_sequence_sha256`, `operator_sequence_matches_contract`, and
`operator_sequence_sha256_matches_contract`. The no-WING handoff validator
recomputes the expected sequence and fails closed if the actual sequence, actual
SHA, expected SHA, or declared match flags drift from the contract.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- no-WING, handoff packet, Project Director, and WING rails tests
- WING MVP docs, runbook, autoloop docs, index, and progress report

## Alternatives considered

Keep only digest validation. This detects post-generation changes but not a
builder-generated checklist that has drifted from the canonical plan.

Duplicate the expected list in validator tests. That is brittle and preserves
two sources of truth.

## Risks

The main risk is stricter offline schema validation. That is acceptable because
a drifted operator checklist should stop before any WING session.

## Test plan

Run focused no-WING and handoff packet tests, WING MVP rails tests, no-WING
quick, Project Director, direction guard, memory lint, `git diff --check`, and
the standard full pytest command.

## Where the other agent may disagree

Kimi may push for hardware proof next. Until real hardware is available in the
thread, the safest useful progress is to reduce packet drift risk in the exact
operator handoff that precedes the hardware run.

# Proposal: WING MVP Handoff Operator Sequence SHA256

## Short thesis

The handoff packet should prove the exact ordered operator sequence, not just
the number of steps and selected marker positions.

## What I understood

The active MVP remains the supervised WING Rack proof. The first blocker is
still WING read-only connection proof, and offline packets cannot close it. The
next useful movement is to make the operator handoff artifact harder to drift
before a real WING session.

## Proposed solution

Compute a canonical SHA256 over the full `operator_sequence` list in the packet
builder. In no-WING validation, recompute the digest from the packet's actual
sequence and fail closed if the declared digest is missing or mismatched.
Expose `operator_sequence_sha256`, `actual_operator_sequence_sha256`, and
`operator_sequence_sha256_matches_actual` in JSON and Markdown summaries.

## Likely files to touch

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- WING MVP docs, runbook, autoloop docs, Project Director rails, and report docs

## Alternatives considered

Keep marker-only validation. This keeps existing tests green but does not
detect drift in non-marker operator steps.

Add more individual step marker checks. This grows ad hoc checks and still does
not bind the entire ordered checklist.

## Risks

The main risk is making the offline handoff schema stricter. That is acceptable
because this is a pre-hardware operator gate and a mismatch should stop proof
planning until the packet is regenerated.

## Test plan

Run focused handoff/no-WING tests, WING MVP rails tests, py_compile, no-WING
quick, Project Director, direction guard, memory lint, `git diff --check`, and
the standard full pytest command.

## Where the other agent may disagree

Kimi may prefer moving directly to hardware execution. Without a WING session
available, binding the exact operator checklist is the safer non-live progress
that reduces operator handoff drift before that session.

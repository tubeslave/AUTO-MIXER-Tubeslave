# Codex Proposal: WING Handoff Prestart Gate Validator

## Short thesis

The no-WING release validator should reject read-only handoff packets that tell
operators to enter supervised proof after `accepted_decision_ready` without also
requiring a valid accepted-candidate prestart trace.

## What I understood

The generated handoff packet text has been corrected, but release validation is
the durable safety gate for generated and manually supplied artifacts. Without a
content check, stale packet artifacts can still pass with outdated transition
wording.

## Proposed solution

- Add required supervised-proof transition markers to
  `tools/no_wing_release_check.py`.
- Validate that `operator_sequence` includes `accepted_decision_ready`, `valid
  accepted-candidate prestart trace`, and `WING_SUPERVISED_FADER_GAIN_PROOF`.
- Expose the boolean and marker list in validation detail.
- Add a failing regression for stale operator sequence wording.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`

## Alternatives considered

- Only test the generator. Rejected because release validation must protect
  stale or manually edited packet artifacts.
- Add a new JSON field to the packet schema. Rejected for this slice because the
  existing operator sequence already carries the handoff instruction and can be
  validated without schema churn.

## Risks

Older handoff packets without the prestart trace wording now fail release
validation. That is intentional fail-closed behavior before physical proof.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py -k "handoff_packet"`
- Affected WING/no-WING tests.
- Full pytest suite.
- `git diff --check`.

## Where the other agent may disagree

Kimi may prefer schema-level fields over text markers. I would add schema fields
later if the packet format changes, but this slice directly closes the stale
artifact acceptance hole.

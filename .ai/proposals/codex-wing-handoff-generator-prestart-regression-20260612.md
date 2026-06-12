# Codex Proposal: WING Handoff Generator Prestart Regression

## Short thesis

The handoff packet generator test should directly prove the generated operator
sequence and Markdown include the supervised prestart gate.

## What I understood

Release validation already rejects stale artifacts, but generator regressions
should be caught at the generator unit-test layer before they reach release
validation.

## Proposed solution

- Extend `tests/test_wing_readonly_preflight_handoff_packet.py` to assert the
  final operator sequence step includes `accepted_decision_ready`, `valid
  accepted-candidate prestart trace`, and `WING_SUPERVISED_FADER_GAIN_PROOF`.
- Assert rendered Markdown includes the same operator-facing transition.

## Likely files to touch

- `tests/test_wing_readonly_preflight_handoff_packet.py`

## Alternatives considered

- Rely only on no-WING release validation. Rejected because that catches stale
  artifacts later, not the generator contract at source.

## Risks

Low. This is test-only coverage for existing behavior.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py`
- Affected handoff/no-WING tests.
- Full pytest suite.
- `git diff --check`.

## Where the other agent may disagree

Kimi may prefer exporting marker constants from the generator. I would keep this
as test coverage for now; release validation already owns the canonical marker
set.

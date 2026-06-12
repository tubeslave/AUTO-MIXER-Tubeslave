# Codex Proposal: WING Supervised Runbook Prestart Trace Gate

## Short thesis

The supervised WING proof runbook and read-only handoff packet should require
the same accepted-candidate prestart trace gate as the proof-plan builder before
operators move from accepted decision review into fader/gain proof.

## What I understood

The current MVP still waits for real WING read-only proof. Recent gates prevent
supervised proof planning when accepted source ids lack backend-port prestart
contract evidence, but the operator runbook and handoff sequence could still be
read as accepted decision alone being sufficient.

## Proposed solution

- Update `Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md` preconditions and
  fail-closed list to include valid accepted-candidate prestart trace and the
  specific proof-plan blocker string.
- Update `tools/build_wing_readonly_preflight_handoff_packet.py` operator
  sequence so transition to `WING_SUPERVISED_FADER_GAIN_PROOF` requires both
  `accepted_decision_ready` and valid accepted-candidate prestart trace.
- Add focused assertions in MVP rails/no-WING tests.
- Add a report and documentation-index row.

## Likely files to touch

- `Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`
- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tests/test_wing_mvp_rails.py`
- `tests/test_no_wing_release_check.py`
- `Docs/INDEX.md`

## Alternatives considered

- Leave the runbook unchanged and rely on release validation. Rejected because
  operator-facing instructions should not lag behind safety gates.
- Add a separate supervised proof runbook. Rejected because the existing runbook
  is the canonical path and should stay current.

## Risks

The only risk is longer operator text. It is acceptable because this is a
fail-closed handoff boundary before physical writes.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py tests/test_no_wing_release_check.py -k "runbooks_keep_readonly or handoff"`
- Affected WING/no-WING tests.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`

## Where the other agent may disagree

Kimi may prefer encoding this as a new packet field instead of text in
`operator_sequence`. That can be a later schema change; this slice fixes the
current misleading transition wording without changing packet compatibility.

# Codex Proposal: WING MVP Handoff Required-False Detail Contract

## Short thesis

Bind the backend-port required-false detail fields through plan, packet, and
summary so stale or trimmed no-WING evidence cannot look ready for a real WING
read-only preflight.

## What I understood

The current target is still a real read-only WING connection proof. Offline
checks can only prepare that run. The new backend-port no-authority fields need
to remain visible after the prestart summary is compressed into handoff and
progress artifacts.

## Proposed solution

- Reuse
  `WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS`
  inside `tools/no_wing_release_check.py`.
- Fail the offline plan artifact if any required field is missing or not
  exactly `false`.
- Put those fields into the handoff packet and validate packet-vs-plan match
  verdicts.
- Render the packet value, plan value, and match verdicts in the no-WING summary
  Markdown.
- Add focused tests for valid, missing, unsafe, and drifted cases.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- Council trail and report/index files.

## Alternatives considered

- Keep the checks only in `wing_readonly_preflight_runner.py`. Rejected because
  the handoff packet is the operator-facing bridge to the real run.
- Duplicate the field list locally. Rejected because the shared contract already
  owns the required false-field set.

## Risks

Older plan or handoff artifacts without these fields will now fail validation.
That is intended; operators should regenerate no-WING prestart evidence before
hardware work.

## Test plan

- Focused no-WING plan/handoff pytest.
- Broader no-WING and WING rail pytest.
- no-WING quick checks, memory lint, diff check, and standard full pytest.

## Where the other agent may disagree

Kimi may argue for fewer duplicated visible fields. I prefer explicit visibility
because these fields are safety-critical no-authority evidence.

# Codex Proposal: WING Read-Only Validate Inputs Handoff Step

## Short thesis

Make `validate_readonly_preflight_inputs` an explicit operator handoff step between offline request-plan preview and real WING read-only capture.

## What I understood

The next useful movement toward the WING blocker is reducing operator error before the real WING websocket capture. The runner already has a validation-only mode; the official packet and no-WING summary should expose it.

## Proposed solution

Add a shared operator-sequence step and command binding for `--validate-run-inputs-only`, then require it in no-WING validation. The command must use the same URL, channels, value types, artifact root, prestart summary, and expected plan as capture, while explicitly not granting live authority.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- WING/no-WING tests
- ADR and report

## Alternatives considered

Leave the step documented only in the runner tests. Rejected because operators need the command in the packet and release summary.

## Risks

Operator sequence count changes from 10 to 11, so stale fixtures must be updated to use the shared contract where possible.

## Test plan

Run focused WING handoff/no-WING suites, no-WING quick validation, memory lint, and full pytest.

## Where the other agent may disagree

Kimi may prefer avoiding sequence-count churn until hardware is attached. I think the churn is justified because the new step reduces risk before the real capture and stays transport-free.

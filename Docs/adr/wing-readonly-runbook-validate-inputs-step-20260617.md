# WING Read-Only Runbook Validate Inputs Step

## Context

The WING read-only handoff packet now includes `validate_readonly_preflight_inputs` before real capture. The operator runbook still documented the old 10-step flow and old index expectations.

## Options considered

1. Leave the runbook unchanged and rely on generated handoff packets.
2. Update the runbook to match the 11-step contract.

## Decision

Use option 2. The runbook must explicitly include the validation-only command before `capture_readonly_preflight`.

## Why this won

The runbook is an operator-facing safety surface. It must not describe a shorter flow than the packet and no-WING contract, because that can lead to skipped pre-transport validation.

## Rejected alternatives

Packet-only guidance was rejected because it leaves stale manual instructions in the hardware preflight path.

## Implementation plan

- Add a `--validate-run-inputs-only` command block between preview and capture.
- Update operator sequence count and step indices.
- Add validate-inputs command binding expectations.
- Record the package report with validation evidence.

## Test plan

- Search for stale runbook values.
- Run focused WING handoff/no-WING tests.
- Run no-WING quick.
- Run full pytest.

## Risks and rollback

The risk is documentation drift in other non-runbook surfaces. Rollback is to restore the previous runbook text if the handoff contract is reverted.

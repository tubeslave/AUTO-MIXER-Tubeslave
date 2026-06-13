# Codex Proposal: WING MVP Handoff Prestart Summary Placeholder

## Short thesis

Bind every operator-facing read-only preflight command to a passed no-WING
prestart `summary.json` before WING read-only proof planning can continue.

## What I understood

The runner already requires `--prestart-summary`, but the generated handoff
packet and no-WING validators could still present stale commands without that
argument. That makes the operator packet drift from the actual fail-closed
runner contract.

## Proposed solution

Add a shared prestart summary placeholder constant, pass it through handoff
packet plan/capture commands, validate the same path in no-WING evidence and
handoff artifact checks, and let the no-WING release harness write a local
prestart-summary-shaped artifact for its offline plan command after the backend
port gate passes.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- focused tests for handoff packet and no-WING release checks
- WING MVP docs and reports

## Alternatives considered

Keep the requirement only in the runbook. Rejected because the generated
handoff packet is the operator sequence the release gate validates.

Use the final no-WING release `summary.json` during the same release run.
Rejected because the final file is written after command checks; a dedicated
prestart summary artifact avoids hash drift.

## Risks

The path is still a placeholder until a real operator no-WING run produces a
passed summary. Validators must keep treating this as a gate, not readiness.

## Test plan

Run focused handoff/no-WING tests, WING MVP rails, Project Director, direction
guard, diff check, memory lint, and the standard backend pytest command.

## Where the other agent may disagree

Kimi may prefer only documenting the command sequence. The stronger contract is
to make no-WING release checks fail if generated commands lose the prestart
summary binding.

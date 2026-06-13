# Brief: WING MVP Runner Defaults Contract

## Context

The active MVP blocker remains `WING read-only connection proof`. The handoff
packet and no-WING validator now use shared target defaults for the read-only
preflight, but the physical runner CLI must also inherit those defaults from
the same contract.

## Task

Make `tools/wing_readonly_preflight_runner.py` use the shared WING read-only
preflight target defaults for its argparse defaults and make no-WING validation
fail closed if those runner defaults drift.

## Constraints

- Do not contact WING hardware.
- Do not start, stop, or reconfigure backend/WING listeners.
- Do not authorize OSC/WING writes.
- Do not close readiness from runner, handoff, or no-WING evidence.
- Preserve the current operator preflight target contract.

## Validation

- Compile touched tools.
- Run focused runner and no-WING tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard checks.
- Run `git diff --check`, memory lint, and full backend pytest before commit.

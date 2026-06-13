# Codex Proposal: WING MVP Handoff Gate Contract Source

## Short Thesis

Make the WING read-only handoff gate contracts a shared preflight contract so
the packet builder and no-WING validator cannot drift before the first real WING
run.

## What I Understood

The current MVP is still blocked on real WING read-only proof. Offline tooling
can only reduce operator error and fail closed. The structured backend-port
prestart gate and supervised prestart gate are part of the first blocker rail,
but their literals lived in multiple executable modules.

## Proposed Solution

Add shared constants and expected-gate builder functions to
`backend/wing_readonly_preflight_contract.py` for:

- backend-port prestart command name, step text, index, and required markers;
- supervised prestart step text, index, required markers, runbook, and required
  decision status;
- expected backend-port and supervised gate dictionaries.

Then update `tools/build_wing_readonly_preflight_handoff_packet.py` and
`tools/no_wing_release_check.py` to consume the shared contract.

## Likely Files To Touch

- `backend/wing_readonly_preflight_contract.py`
- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- focused handoff/no-WING tests
- WING MVP report/ADR/index/evidence docs

## Alternatives Considered

Leaving the literals duplicated would preserve today's behavior but keep the
same drift risk that already affected preflight fail-closed conditions. Moving
only marker lists would be smaller, but the expected gate dictionaries are the
actual validator contract.

## Risks

The main risk is accidentally changing packet JSON shape or operator sequence
semantics. Tests should compare generated packets and no-WING validation details
against the shared contract while preserving the current public fields.

## Test Plan

- Compile touched Python modules.
- Run focused handoff/no-WING tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard.
- Run the full pytest suite.

## Where The Other Agent May Disagree

Kimi may argue this is another contract refactor instead of hardware progress.
The counterpoint is that this gate is directly in the operator path before the
first WING preflight run, and removing drift risk improves that first real run
without widening live authority.

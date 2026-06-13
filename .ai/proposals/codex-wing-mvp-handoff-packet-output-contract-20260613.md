# Codex Proposal: WING MVP Handoff Packet Output Contract

## Short Thesis

The WING read-only handoff packet output path should be a shared contract value
and no-WING should validate the actual command output path.

## What I Understood

The current MVP is blocked on real WING read-only proof. The handoff packet is
the operator-visible starting point for the physical proof run, so output-path
drift can send operators to the wrong packet without changing any runtime mixer
state.

## Proposed Solution

- Add `WING_READONLY_PREFLIGHT_HANDOFF_PACKET_OUTPUT` to the shared preflight
  contract.
- Make handoff builder and no-WING consume that constant.
- Parse the handoff command `--output` value in no-WING evidence validation.
- Add a fail-closed mismatch violation and expose the match field in summaries.
- Cover valid and stale-output cases in tests.

## Likely Files To Touch

- `backend/wing_readonly_preflight_contract.py`
- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/wing-mvp-handoff-packet-output-contract-20260613.md`
- `.ai/proposals/codex-wing-mvp-handoff-packet-output-contract-20260613.md`
- `Docs/adr/wing-mvp-handoff-packet-output-contract-20260613.md`
- `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PACKET_OUTPUT_CONTRACT_20260613.md`

## Alternatives Considered

- Keep literals in both tools. Rejected because this is exactly the drift being
  removed from the operator proof chain.
- Centralize without validation. Rejected because stale commands or monkeypatches
  would still pass.
- Change the output path. Rejected because existing operator-facing paths should
  remain stable.

## Risks

The main risk is stale fixtures. Runtime risk is low because the change only
updates offline command construction and release-gate validation.

## Test Plan

- Compile touched modules.
- Run focused handoff, no-WING, Project Director, direction guard, and rails tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard.
- Run `git diff --check`, memory lint, and the full pytest suite.

## Where The Other Agent May Disagree

Kimi may prefer centralizing more runner defaults in one pass. This proposal
keeps the diff focused on the operator-facing handoff packet output path.

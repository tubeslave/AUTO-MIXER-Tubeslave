# Codex Proposal: WING Supervised Proof Physical Readback Only - 2026-06-07

## Short thesis

Physical WING proof must require `verified_from_physical_readback` for pre-read,
apply, and rollback. Cache can remain useful for dry-run planning, but not for
physical promotion evidence.

## What I understood

The project is still blocked on real WING proof. Current live scope is only the
supervised fader/gain path through Product Layer proposal apply and rollback.
Any proof artifact used later for readiness must show physical console evidence.

## Proposed solution

Change `tools/wing_operator_proposal_validation_runner.py` so `--execute` forces
`query_mixer_parameter` for the current value, rejects non-physical pre-read,
and accepts apply/rollback only when readback status is
`verified_from_physical_readback`.

Add `validate_wing_operator_proposal_runner_physical_readback_policy()` to the
no-WING release check so this contract remains guarded even without hardware.

## Likely files to touch

- `tools/wing_operator_proposal_validation_runner.py`
- `tools/no_wing_release_check.py`
- focused runner and no-WING tests
- WING supervised proof runbook
- report/index/ADR/brief artifacts

## Alternatives considered

- Keep accepting cache and rely on manual review. Rejected because the runner can
  enforce physical evidence before an artifact is ever considered.
- Require physical readback for dry-run. Rejected because dry-run is explicitly
  non-mutating and useful without hardware.

## Risks

Existing fake physical tests need to emit physical readback status. This is the
correct pressure: fake proof should model the hardware contract accurately.

## Test plan

- Compile changed Python modules.
- Run focused runner/no-WING tests.
- Run quick no-WING release check.
- Run direction guard, memory lint, diff check, and full pytest before commit.

## Where the other agent may disagree

Kimi may ask for a separate artifact schema version bump. The runner already
records explicit physical-readback flags; a separate schema bump can be added
later if external consumers require it.

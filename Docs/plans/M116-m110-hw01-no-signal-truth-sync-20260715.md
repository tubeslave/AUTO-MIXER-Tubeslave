# M116 — M110 HW01 No-Signal Truth Sync

## Boundary

Truth/control-plane and tests only. No unattended WING, backend, OSC or
CoreAudio contact. No change to normal allowlist, safety policy, reset executor
or correction logic.

| Checkpoint | Goal | Evidence | Status |
|---|---|---|---|
| M116-CP01 | Rebaseline diagnostic | runtime summary/log hashes, current HEAD and teardown | complete |
| M116-CP02 | Council decision | brief, proposals, reciprocal critiques and ADR | complete |
| M116-CP03 | Canonical truth sync | STATE, append-only ledger and all current mirrors agree | complete |
| M116-CP04 | Validation | JSON/JSONL, focused, rails, full suite, frontend, no-WING, lint and review | complete |
| M116-CP05 | Source checkpoint | non-marker commit on released lease | complete |
| M116-CP06 | CP05 rebind | marker-only commit and clean final verification | complete |

## Acceptance

- HW01 is incomplete with reason `master_channel_signal_missing`.
- Physical WING/CoreAudio contact is true; route artifact, accepted HW01 proof
  and serial-bound USB identity remain false.
- HW02-HW05 remain pending.
- Mutation attempts, reset, corrections and live writes remain zero.
- Strict score remains `88/100`, target not achieved, allowlist zero and live
  authority false.
- Next operator action requires a confirmed non-silent source before retry.

## Rollback

Revert the non-marker M116 commit and its marker-only CP05 rebind together.

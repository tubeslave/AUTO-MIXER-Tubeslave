# M117: Operator Diagnostic Truth Contract

## Boundary

Truth/control-plane and tests only. No backend, CoreAudio, OSC or WING contact.
No production code, normal allowlist, safety policy, reset/correction executor
or standing authority change.

| Checkpoint | Goal | Evidence | Status |
|---|---|---|---|
| M117-CP01 | Rebaseline | three independent read-only audits and exact contradictions | complete |
| M117-CP02 | Council decision | brief, proposals, reciprocal critiques and ADR | complete with documented Kimi 402 fallback |
| M117-CP03 | Truth implementation | STATE, append-only ledger, all current mirrors and M116 plan agree | complete |
| M117-CP04 | Regression rails | historical M116 event and current provenance wording pinned | complete |
| M117-CP05 | Validation | pre-CP05 2 focused/137+1 deselected/7215+3 skipped+1 deselected; baseline tree 138 rails and 7216+3 skipped | complete |
| M117-CP06 | Source checkpoint | non-marker commit `7267b74ae` after verified diff | complete |
| M117-CP07 | CP05 rebind | completion is derived from the current marker-only commit and dynamic gate; `551fa6b4f` is the clean validation baseline, not the final binding | dynamic: complete iff current gate verifies |

## Acceptance

- M116 history keeps real hardware/backend/WING contact true and live writes,
  readiness and standing authority false.
- The latest M117 row is offline/no-contact and does not erase M116.
- All current zero-signal prose says operator-observed and not independently
  machine-verified.
- M116 remains blocked at HW01; HW02-HW05, reset and corrections stay pending/0.
- Strict score remains 88/100 and normal allowlist remains 0.
- Completed M116 CP04-CP06 and the fail-closed M113 identity gate are described
  truthfully.

## Rollback

Revert the M117 source commit and its marker-only CP05 rebind together. Never
edit the historical M116 ledger row.

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
| M117-CP05 | Validation | 2 focused; 137 rails + 1 expected CP05 deselection; 7215 passed, 3 skipped + same deselection; no-WING/lint/JSON/diff passed | complete |
| M117-CP06 | Source checkpoint | non-marker commit after verified diff | pending |
| M117-CP07 | CP05 rebind | released lease, marker-only commit and clean verification | pending |

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

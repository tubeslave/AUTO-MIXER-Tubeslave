# proof-evidence-settings-visibility-20260605

## Context

The Product Layer now exposes `supervised_proof_evidence` and sanitized
`autonomy_evidence_overlay` rows. Adapter contract tests cover the row mapping,
but the browser-level Settings path still uses an older processing snapshot
fixture and does not prove that proof freshness is visible to an operator.

## Goal

Make supervised proof evidence freshness visible in the existing Settings module
status surface and cover it in the backend DTO e2e path.

## Non-goals

- Do not add apply buttons, queue mutations, or new live console commands.
- Do not promote any autonomy target based on artifact display alone.
- Do not create a separate page for proof evidence in this slice.

## Safety constraints

- All proof evidence rows must remain `canWriteLive=false`.
- Stale proof evidence must remain review-only and must not imply promotion
  readiness.
- The UI must keep source availability separate from target-level autonomy
  evidence rows.

## Expected evidence

- Settings inspector summarizes proof artifact status and freshness.
- Settings module table can filter proof evidence rows.
- E2E backend DTO fixture includes supervised proof evidence and autonomy
  overlay rows.
- E2E assertions prove the rows and freshness text render without sending live
  write commands.

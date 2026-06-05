# Brief: fx-send-schema-gate-20260605

## Context

The long-term Automixer goal requires autonomous FX management and effect-send
tuning. Current live FX writes remain quarantined, but the Product Layer now
exposes read-only FX parameter schema hints for cached FX slots.

`fx_send_proposal` already blocks proposal promotion on stale current FX state
and untrusted source identity. It should also avoid creating proposal-only FX
send recommendations when the current FX parameter schema is unknown, generic,
or scoped to a different FX family.

## Task

Add a read-only FX parameter schema gate to `fx_send_proposal`.

## Constraints

- Do not import mixer clients or transports.
- Do not mutate mixer state or the operator queue.
- Do not allow FX/send apply.
- Keep absent schema payloads as legacy-compatible evidence, but block explicit
  stale, generic, missing, unscoped, or mismatched schema payloads.
- Preserve `can_apply_live=False` and `can_import_to_queue=False`.

## Acceptance

- `FXSendAnalysisInput` can carry current FX parameter schema evidence.
- Generic or mismatched schema evidence produces `analysis_only`.
- Proposal evidence records schema status, expected schema family, and the
  normalized schema payload.
- Proposal-only adapter contract requires FX parameter schema hints as promotion
  evidence.
- Focused and full tests pass.

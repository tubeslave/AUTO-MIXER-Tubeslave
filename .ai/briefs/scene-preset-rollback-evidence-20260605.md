# scene-preset-rollback-evidence-20260605

## Context

Scene/preset proposals require rollback evidence, but the current contract is a
boolean plus rollback ID. For broad console-state surfaces, a future autonomous
system needs structured rollback evidence that can be checked before any
operator-facing recommendation is treated as proposal-ready.

## Goal

Add transport-free rollback evidence normalization to
`scene_preset_management_proposal`. The analyzer should be able to derive
rollback availability, rollback ID, rollback confidence, and target coverage from
a read-only rollback payload.

## Non-goals

- Do not execute rollback.
- Do not load, save, recall, or store scenes/presets.
- Do not query a console or add live write authority.

## Safety Constraints

- Rollback evidence must be read-only and non-mutating.
- Evidence that claims live writes or console queries must block proposal-only
  promotion.
- Rollback evidence must cover the target scope when that field is provided.

## Expected Evidence

- Read-only rollback evidence can support a proposal-only recommendation when
  other gates pass.
- Unsafe rollback evidence is `analysis_only`.
- Rollback target mismatch is `analysis_only`.
- Registry and proposal contracts name rollback evidence as required safety
  evidence.

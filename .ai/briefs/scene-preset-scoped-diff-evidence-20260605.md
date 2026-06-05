# scene-preset-scoped-diff-evidence-20260605

## Context

Scene/preset recommendations currently require `scoped_diff_available`,
affected-surface counts, destructive-scope score, preview flags, and rollback
flags as separate caller-provided fields. This keeps scene/preset control
proposal-only, but it does not give the Product Layer a structured diff evidence
contract.

## Goal

Add a transport-free scoped-diff evidence payload for scene/preset proposals so
the analyzer can derive preview, scoped diff, rollback, affected-surface counts,
and destructive-scope score from one read-only evidence object.

## Non-goals

- Do not build scene diffs by querying a mixer.
- Do not load, save, recall, or store scenes/presets.
- Do not make scene/preset proposals queueable or live-applicable.

## Safety Constraints

- Scoped-diff evidence must be read-only.
- Evidence that claims live writes, console queries, or state mutation must block
  proposal-only promotion.
- Broad diffs must still trigger the existing scope blockers.

## Expected Evidence

- A read-only scoped diff can make a scene/preset recommendation proposal-only
  when the other gates pass.
- A live/query/mutating diff payload is `analysis_only`.
- A broad diff payload blocks through existing destructive-scope limits.
- Registry and proposal contracts name the scoped-diff evidence contract.

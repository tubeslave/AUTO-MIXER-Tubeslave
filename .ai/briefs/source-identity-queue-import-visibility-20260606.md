# Brief: source-identity-queue-import-visibility-20260606

## Context

Source identity queue import can now place ordered proposals into the native
operator queue after explicit acknowledgement. The decision queue exposes the
pending proposals, but its compact summary does not tell the operator which
entries came from source identity ranking, which are still analysis-only, and
which remain eligible only for the existing supervised apply path.

## Objective

Expose source identity queue-import visibility in `build_decision_queue` so the
Product Layer can report imported proposal provenance, applyability counts, and
the no-live-write safety envelope without changing the executor boundary.

## Safety Constraints

- Do not mutate queue contents while building the decision queue.
- Do not call mixer clients, supervised console apply, or source identity trust
  stores.
- Do not widen live writes beyond the existing `apply_operator_proposal`
  supervised manual gate.
- Keep source identity queue import safety flags explicitly false for direct
  live apply.

## Expected Evidence

- Decision queue summary reports source-identity import pending/history counts.
- Imported source-identity proposals are split into applyable and analysis-only
  pending counts.
- Summary states that source identity queue import itself does not write live
  and that later console apply remains governed by the existing proposal gate.
- Tests cover imported queue proposals without calling mixer write functions.

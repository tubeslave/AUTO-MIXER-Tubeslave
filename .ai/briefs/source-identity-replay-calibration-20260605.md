# Brief: source-identity-replay-calibration-20260605

## Context

The project goal requires the Automixer to improve from accumulated outcomes,
logs, tests, and previous shows. The source identity confidence layer now emits
operator review records for weak classifier outputs, but downstream proposal
families still need replay evidence before they can trust accepted labels or
hold repeatedly rejected labels.

The worktree is dirty from earlier proposal-only safety and freshness cycles.
This task must not revert unrelated local changes.

## Task

Add a transport-free `source_identity_replay_calibration` layer. It should
summarize accepted, corrected, rejected, and dismissed source identity replay
outcomes into operator-review calibration records. The records should expose
trust states for downstream proposal priority without mutating the classifier,
memory store, proposal queue, or live console.

## Safety constraints

- Do not add live OSC, WING, dLive, or Mixing Station writes.
- Do not create queueable live actions.
- Do not mutate model thresholds or memory automatically.
- Block calibration output when provided replay/outcome context is stale.
- Require operator review before any calibration record changes downstream
  source-specific proposal priority.

## Acceptance criteria

- Accepted replay outcomes can produce a `trusted_for_downstream_priority`
  calibration record.
- Corrected/rejected outcomes can produce a `hold_for_operator_review` record.
- Safety regressions produce a blocked downstream-trust state.
- Stale replay context blocks calibration output.
- Registry, proposal-only contract, and capability matrix reference the new
  layer without widening live writes.
- Focused tests and the standard test command pass.

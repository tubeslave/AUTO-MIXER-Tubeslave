# Brief: decision-queue-apply-preflight-visibility-20260606

## Context

The Product Layer now has a dry-run handler for a single proposal apply
preflight. The decision queue still lacks a compact summary that tells the
operator which pending proposals are candidates for that preflight and which
are analysis-only.

## Objective

Expose queue-level apply-preflight visibility in `build_decision_queue` without
running preflight, reserving idempotency keys, mutating proposals, or writing
to the mixer.

## Safety Constraints

- This is read-only queue visibility.
- Do not run `preflight_operator_proposal_apply` from `build_decision_queue`.
- Do not reserve idempotency keys or call `_apply_manual_console_write`.
- Do not grant source-identity imported proposals extra live authority.
- Preserve the existing live apply entrypoint: `apply_operator_proposal`.

## Expected Evidence

- Decision queue summary reports pending preflight candidate IDs.
- Analysis-only proposal IDs remain visible separately.
- Summary advertises the preflight handler endpoint and no-live-write safety
  envelope.
- Tests prove imported source-identity proposals are split into preflight
  candidates and analysis-only entries without mixer writes.

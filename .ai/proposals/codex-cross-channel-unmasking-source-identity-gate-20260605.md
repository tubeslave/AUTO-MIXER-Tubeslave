# Proposal: cross-channel-unmasking-source-identity-gate-20260605

## Short thesis

Gate cross-channel unmasking proposals with replay-calibrated source identity
trust before linked mirror-EQ actions can reach operator review.

## What I understood

The Automixer goal requires cross-channel unmasking, but current EQ writes are
not inside the proven live executor. `cross_adaptive_eq` already produces
proposal-only linked cut/boost actions behind a freshness gate. The remaining
gap is that source identity and source role trust can still be stale or held
while the wrapper uses channel priority to decide which source is protected and
which source is cut.

## Proposed solution

Keep `calculate_corrections` unchanged for legacy offline/shared callers. Extend
`analyze_unmasking_proposals` with optional `channel_source_identity`. For each
channel in the masking context, evaluate the shared
`source_identity_downstream_gate`. If every decision is usable, attach trust
evidence to result metadata and to each linked action's safety evidence. If any
channel is held, blocked, unknown, or based on stale replay context, return no
proposal actions and record the gate blockers.

## Likely files to touch

- `backend/cross_adaptive_eq.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_cross_adaptive_eq.py`
- `.ai/briefs/cross-channel-unmasking-source-identity-gate-20260605.md`
- `Docs/adr/cross-channel-unmasking-source-identity-gate-20260605.md`

## Alternatives considered

- Require source identity metadata for every legacy unmasking call immediately.
  Rejected because upstream callers do not all provide replay calibration yet.
- Gate only the frequency-separation proposal layer. Rejected because
  `cross_adaptive_eq` is a separate unmasking path and can emit linked EQ
  proposals directly.
- Add direct live EQ writes after source identity trust. Rejected because EQ-band
  readback, sum-bus verification, rollback, and hardware proof remain incomplete.

## Risks

- Callers that pass held or stale source identity evidence will produce fewer
  unmasking proposals.
- Missing identity evidence remains legacy-compatible until callers pass
  replay-calibrated source identity metadata.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/cross_adaptive_eq.py backend/source_identity_downstream_gate.py backend/processing_registry.py backend/processing_proposal_adapters.py backend/project_goal_capability_matrix.py tests/test_cross_adaptive_eq.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_cross_adaptive_eq.py tests/test_source_identity_downstream_gate.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`
- Runtime smoke that verifies held source identity yields zero adjustments,
  zero suggested actions, and no live, queue, model, or memory mutation flags.

## Where the other agent may disagree

Kimi may argue that missing source identity calibration should block all
unmasking proposals. I would keep legacy missing calibration allowed for now
because this wrapper is proposal-only and upstream callers are still being
upgraded. Promotion to any live EQ authority should require fresh trusted source
identity.

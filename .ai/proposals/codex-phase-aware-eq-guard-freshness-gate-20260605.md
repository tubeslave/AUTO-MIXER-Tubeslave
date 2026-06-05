# Proposal: phase-aware-eq-guard-freshness-gate-20260605

## Short thesis

Use current phase/audio context freshness as a dry-run gate before emitting
phase-aware EQ guard candidates.

## What I understood

The Automixer should eventually coordinate phase correction and EQ, but delay,
polarity, and EQ writes are not live-approved. The phase-aware guard is useful
because it can stop unsafe EQ boosts and surface candidate phase fixes. If the
audio window or group phase measurements are stale, the correct behavior is to
hold EQ and request fresh evidence rather than emit candidate moves.

## Proposed solution

Extend `PhaseAwareEQGuardConfig` with optional `current_state_cache_freshness`.
Normalize that payload into result metadata. If it is provided and the status is
not `fresh`, return `PHASE_UNRELIABLE`, emit no suggested actions, and include
`phase_aware_eq_context_cache_not_fresh` in metadata blockers. Keep the module
transport-free and dry-run.

## Likely files to touch

- `backend/phase_aware_eq_guard.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_phase_aware_eq_guard.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`

## Alternatives considered

- Require freshness for every phase-aware EQ guard result immediately. Rejected
  to avoid breaking legacy analysis before all callers pass snapshot freshness.
- Ignore freshness until hardware delay/polarity proof exists. Rejected because
  stale phase evidence can already be identified and should block candidates.
- Promote phase/delay/polarity writes. Rejected because physical readback,
  rollback, cooldown, and emergency-stop proof are not complete.

## Risks

- Callers that provide stale freshness will get `PHASE_UNRELIABLE` with no
  suggested phase candidates.
- Missing freshness remains a legacy gap until upstream integration is complete.

## Test plan

- Focused phase-aware EQ guard tests.
- Registry/capability contract tests.
- Full suite and `git diff --check`.

## Where the other agent may disagree

Kimi may argue missing freshness should block all phase guard candidates. I
agree for promotion, but this cycle keeps legacy analysis compatible while
blocking known stale evidence.

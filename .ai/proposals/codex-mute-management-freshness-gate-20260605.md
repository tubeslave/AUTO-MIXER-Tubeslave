# Proposal: mute-management-freshness-gate-20260605

## Short thesis

Use current mute-state cache freshness as a proposal-only gate before showing
mute or unmute recommendations.

## What I understood

The Automixer should eventually manage mute state automatically, but mute
automation can create dropouts or override operator intent. It cannot safely
promote mute decisions while the current mute state or mute ownership is stale
or unknown. The next useful step is to consume cached readback freshness as
evidence, not to add live writes.

## Proposed solution

Extend `MuteManagementAnalysisInput` with optional
`current_state_cache_freshness`. Normalize that payload into recommendation
evidence. If it is provided and the status is not `fresh`, add
`mute_current_state_cache_not_fresh` and demote the recommendation to
`analysis_only`. Keep the module transport-free and keep operator proposals
without `requested_change` or `requested_changes`.

## Likely files to touch

- `backend/mute_management_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_mute_management_proposal.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`

## Alternatives considered

- Require freshness for every mute proposal immediately. Rejected to avoid
  breaking legacy audio-only analysis before all callers pass product snapshot
  freshness.
- Ignore freshness until physical mute query fixtures exist. Rejected because
  cached freshness is already available and can block known-bad state.
- Add mute live writes. Rejected because intent readback, rollback, manual
  authority, and physical proof are incomplete.

## Risks

- Callers that provide stale freshness will produce fewer mute proposals.
- Missing freshness remains a legacy gap until upstream integration is complete.

## Test plan

- Focused mute management proposal tests.
- Registry/capability contract tests.
- Full suite and `git diff --check`.

## Where the other agent may disagree

Kimi may argue missing freshness should block all mute proposals. I agree for
promotion, but this cycle keeps legacy analysis compatible while blocking known
stale evidence.

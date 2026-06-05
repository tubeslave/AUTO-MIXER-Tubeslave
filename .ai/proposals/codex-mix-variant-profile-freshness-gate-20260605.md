# Proposal: mix-variant-profile-freshness-gate-20260605

## Short thesis

Use current mix-variant context freshness as a proposal-only gate before showing
genre/task profile recommendations.

## What I understood

The Automixer should form different mix variants for genres and tasks, but a
variant profile can influence several downstream proposal layers. Stale
reference metrics or stale arrangement context can produce a coherent-looking
profile that points the rest of the system in the wrong direction. The next
useful step is to consume cached freshness evidence while keeping the module
transport-free and non-queueable.

## Proposed solution

Extend `MixVariantProfileAnalysisInput` with optional
`current_state_cache_freshness`. Normalize that payload into recommendation
evidence. If it is provided and the status is not `fresh`, add
`mix_variant_context_cache_not_fresh` and demote the recommendation to
`analysis_only`. Keep operator proposals without `requested_change` or
`requested_changes`.

## Likely files to touch

- `backend/mix_variant_profile_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_mix_variant_profile_proposal.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`

## Alternatives considered

- Require freshness for every mix variant proposal immediately. Rejected to
  avoid breaking legacy analysis before all callers pass product snapshot
  freshness.
- Ignore freshness until follow-up fan-out orchestration exists. Rejected
  because stale context can already be identified and blocked at the profile
  proposal boundary.
- Add profile apply or preset/scene write support. Rejected because variant
  profiles still require operator review and downstream proposal proof.

## Risks

- Callers that provide stale freshness will produce fewer mix variant
  proposals.
- Missing freshness remains a legacy gap until upstream integration is complete.

## Test plan

- Focused mix variant profile proposal tests.
- Registry/capability contract tests.
- Full suite and `git diff --check`.

## Where the other agent may disagree

Kimi may argue missing freshness should block all mix variant proposals. I agree
for promotion, but this cycle keeps legacy analysis compatible while blocking
known stale evidence.

# Codex Proposal: Cross-channel Unmasking Arrangement Priority Gate

## Short thesis

Cross-channel unmasking should use fresh arrangement-role evidence before it
decides which source to protect and which source to cut.

## What I understood

The current unmasking layer is already proposal-only and freshness gated for
masking context and source identity. The remaining gap is priority provenance:
when `channel_priorities` are missing, the algorithm falls back to equal default
priority and emits nothing, even if fresh arrangement evidence clearly marks one
source as lead and the other as support.

## Proposed solution

Add optional `channel_arrangement_context` and
`arrangement_context_cache_freshness` to
`CrossAdaptiveEQ.analyze_unmasking_proposals`. If the arrangement context is
fresh and trusted enough, derive effective channel priorities from arrangement
roles such as `lead`, `low_anchor`, `rhythmic_anchor`,
`harmonic_support`, `backing`, and `effect`. Keep explicit `channel_priorities`
authoritative when they are supplied. Record priority source and arrangement
evidence in result metadata and per-action safety evidence.

## Likely files to touch

- `backend/cross_adaptive_eq.py`
- `tests/test_cross_adaptive_eq.py`
- `backend/processing_proposal_adapters.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `.ai/project-layer/correction_methods_backlog.md`
- `Docs/adr/`
- `Docs/reports/`

## Alternatives considered

Keeping priority as an opaque caller-provided integer was rejected because it
does not advance arrangement-aware mixing. Promoting unmasking to queueable or
live EQ writes was rejected because physical EQ-band readback, sum-bus proof,
rollback, and hardware proof remain incomplete.

## Risks

Arrangement-role hints may be wrong or stale. The mitigation is to require fresh
context when it is supplied, block low-confidence or unknown roles, and preserve
source-identity trust plus rollback/readback requirements.

## Test plan

Run focused unmasking/registry/capability tests, related proposal tests, then
the full test suite.

## Where the other agent may disagree

Kimi may argue the first WING read-only proof is still the higher operational
blocker. That is true, but hardware is unavailable in this environment and this
work stays explicitly offline/proposal-only.

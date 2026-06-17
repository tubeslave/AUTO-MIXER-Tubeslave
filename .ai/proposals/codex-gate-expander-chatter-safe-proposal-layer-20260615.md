# Codex Proposal: Gate/Expander Chatter-safe Proposal Layer

## Short thesis

Gate/expander recommendations should become non-queueable operator proposals
only when fresh telemetry and source identity support them, and hard gates
should be avoided when chatter risk is high.

## What I understood

The current `ContextAwareGateAgent` is transport-free and already blocks stale
context or untrusted source identity. It can emit dry-run recommendations, but
the Product Layer has no dedicated gate proposal wrapper comparable to the
de-esser, panner, compressor, and frequency-separation layers. Also, transient
drum sources can still choose hard gate mode even when chatter risk is high.

## Proposed solution

Add `ContextGateRecommendation.to_operator_proposal` and
`build_context_gate_proposals`. The proposal remains non-queueable and
non-live-applicable, carries old/rollback gate readback requirements, and
preserves all no-mutation flags. Add a chatter guard to
`ContextAwareGateAgent`: hard gate mode is allowed only below a configured
chatter-risk threshold; otherwise the recommendation falls back to a gentler
expander.

## Likely files to touch

- `backend/context_aware_gate.py`
- `tests/test_agentic_method_foundation.py`
- `backend/processing_proposal_adapters.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `.ai/project-layer/correction_methods_backlog.md`
- `Docs/adr/`
- `Docs/reports/`

## Alternatives considered

Leaving recommendations as dry-run traces only was rejected because operator
review needs a consistent Product Layer proposal shape. Promoting gate writes
was rejected because physical gate readback, rollback, and replay proof are not
complete.

## Risks

High-chatter drum proposals may be less aggressive than expected because they
fall back to expander mode. That is safer for live sound and can be tuned after
replayed operator decisions.

## Test plan

Run focused context-aware gate, registry, capability, and CAIG tests, then the
full suite.

## Where the other agent may disagree

Kimi may argue hard-gate behavior should be held entirely instead of converted
to an expander when chatter is high. The safer compromise is a non-queueable
operator proposal with explicit chatter evidence and no live authority.

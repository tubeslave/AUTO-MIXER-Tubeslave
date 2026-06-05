# Proposal: context-aware-gate-source-identity-freshness-20260605

## Short thesis

Gate/expander recommendations should only become operator-facing gate proposals
when the bleed context is fresh and the source identity is trusted for
downstream decisions.

## What I understood

The Automixer needs automatic gating, but current gate writes remain outside the
proven supervised live executor. The safe path is to strengthen the proposal-only
gate agent so it can explain why a gate/expander action is held when its input
context is stale or source identity is not trusted.

## Proposed solution

Extend `BleedTelemetry` with optional current-state freshness and source identity
trust fields. In `ContextAwareGateAgent.recommend`, evaluate freshness and the
shared `source_identity_downstream_gate` before suggesting gate/expander params.
If either gate blocks, return an `analysis_only` hold recommendation with no
suggested gate params and explicit no-mutation evidence. Keep legacy callers
compatible by allowing missing freshness and missing calibration as unverified
proposal evidence.

## Likely files to touch

- `backend/context_aware_gate.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_agentic_method_foundation.py`
- `tests/test_processing_registry.py`
- `tests/test_project_goal_capability_matrix.py`
- `.ai/briefs/context-aware-gate-source-identity-freshness-20260605.md`
- `Docs/adr/context-aware-gate-source-identity-freshness-20260605.md`

## Alternatives considered

- Drop blocked recommendations entirely. Rejected because analysis-only hold
  records improve operator explainability and future replay.
- Require source identity calibration for every gate recommendation immediately.
  Rejected because upstream callers are not all upgraded yet.
- Promote trusted gate recommendations to live gate writes. Rejected because
  gate readback, rollback, replay validation, and hardware proof are incomplete.

## Risks

- Callers that pass stale context or held identity will get analysis-only holds
  instead of gate/expander parameters.
- Missing freshness and missing calibration remain legacy-compatible until the
  Product Layer can pass those fields everywhere.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/context_aware_gate.py backend/source_identity_downstream_gate.py backend/processing_registry.py backend/processing_proposal_adapters.py backend/project_goal_capability_matrix.py tests/test_agentic_method_foundation.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_agentic_method_foundation.py tests/test_source_identity_downstream_gate.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`
- Runtime smoke that verifies held source identity yields `analysis_only`, no
  suggested gate params, and no live, queue, model, or memory mutation.

## Where the other agent may disagree

Kimi may argue missing source identity calibration should block gate proposals
now. I would defer that to promotion: this patch keeps proposal-only legacy
analysis working while blocking known stale or untrusted evidence.

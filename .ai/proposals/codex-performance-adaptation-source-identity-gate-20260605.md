# Proposal: performance-adaptation-source-identity-gate-20260605

## Short thesis

Use replay-calibrated source identity trust as a performance-adaptation blocker
so source-specific level rides cannot rely on explicitly untrusted source
labels.

## What I understood

The performance-adaptation layer already checks baseline level, current level
readback, rollback, operator intent, cooldown, persistence, latency budget,
true-peak headroom, feedback risk, bleed risk, and current context freshness.
It still trusts `source_role` directly when categorizing the target. After
source identity replay calibration was added, adaptation decisions should
consume that trust evidence before making recommendations.

## Proposed solution

Add optional source identity trust fields to
`PerformanceAdaptationAnalysisInput` and evaluate them through
`source_identity_downstream_gate`. Add the resulting blockers and evidence to
`PerformanceAdaptationRecommendation`. Keep legacy missing calibration allowed
so existing callers remain compatible, but block explicit `hold`, `blocked`,
stale, or unknown trust states.

Update the proposal-only contract and capability matrix support paths, then add
focused tests.

## Likely files to touch

- `backend/performance_adaptation_proposal.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_performance_adaptation_proposal.py`
- `.ai/briefs/performance-adaptation-source-identity-gate-20260605.md`
- `Docs/adr/performance-adaptation-source-identity-gate-20260605.md`

## Alternatives considered

- Require source identity calibration for all adaptation inputs immediately.
  Rejected because legacy offline callers do not yet pass replay artifacts.
- Ignore replay trust for adaptation. Rejected because a wrong label can steer
  ride severity and high-bleed or critical-source safety handling.
- Duplicate trust-state parsing inside performance adaptation. Rejected because
  the shared gate keeps blocker names and evidence consistent across downstream
  modules.

## Risks

- Conservative blocking may hold valid adaptation proposals until replay
  calibration is consistently available.
- The gate currently relies on upstream trust-state strings; future DTO wiring
  should pass structured calibration records.

## Test plan

```bash
PYTHONPATH=backend python3 -m py_compile \
  backend/performance_adaptation_proposal.py \
  backend/source_identity_downstream_gate.py \
  backend/processing_proposal_adapters.py \
  backend/project_goal_capability_matrix.py \
  tests/test_performance_adaptation_proposal.py
PYTHONPATH=backend python3 -m pytest \
  tests/test_performance_adaptation_proposal.py \
  tests/test_source_identity_downstream_gate.py \
  tests/test_processing_registry.py \
  tests/test_project_goal_capability_matrix.py -q
PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q
/opt/homebrew/bin/git diff --check
```

## Where the other agent may disagree

Kimi may argue that adaptation should still emit low-priority suggestions when
source identity is held. I would block explicitly untrusted source labels first:
performance adaptation can later become a real-time fader/gain candidate, and
wrong source classification during a show is not a harmless recommendation.

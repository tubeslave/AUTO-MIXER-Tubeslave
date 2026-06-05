# Proposal: mix-variant-source-identity-gate-20260605

## Short thesis

Use replay-calibrated source identity trust as a mix-variant blocker so
genre/task profiles cannot fan out from explicitly untrusted source-role
context.

## What I understood

The mix-variant layer already checks genre and task support, operator intent,
safe windows, arrangement context, reference metrics, style-clone safety,
confidence thresholds, headroom, feedback risk, duplicate profiles, and
current-context freshness. It is explicitly non-queueable and transport-free.
Because a profile can fan out into bus/group balance, FX send, master-bus, and
scene/preset proposals, source identity trust should become part of the safety
evidence before profile fan-out is proposed.

## Proposed solution

Add optional source identity trust fields to `MixVariantProfileAnalysisInput`
and evaluate them through `source_identity_downstream_gate`. Add the resulting
blockers and evidence to `MixVariantProfileRecommendation`. Keep legacy missing
calibration allowed so existing callers remain compatible, but block explicit
`hold`, `blocked`, stale, or unrecognized trust states.

Update the proposal-only contract and capability matrix support path, then add
focused tests for legacy compatibility, held trust, stale replay context, and
contract/matrix evidence.

## Likely files to touch

- `backend/mix_variant_profile_proposal.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_mix_variant_profile_proposal.py`
- `.ai/briefs/mix-variant-source-identity-gate-20260605.md`
- `Docs/adr/mix-variant-source-identity-gate-20260605.md`

## Alternatives considered

- Require source identity calibration for all mix-variant inputs immediately.
  Rejected because legacy callers do not yet pass replay artifacts.
- Ignore replay trust for mix variants. Rejected because a wrong source or
  role context can mis-shape the entire downstream profile fan-out.
- Duplicate trust-state parsing inside mix-variant profiles. Rejected because
  the shared gate keeps blocker names and no-mutation evidence consistent
  across downstream modules.

## Risks

- Conservative blocking may hold useful variant profiles until replay
  calibration is consistently available.
- The gate currently relies on upstream trust-state strings; future DTO wiring
  should pass structured calibration records.

## Test plan

```bash
PYTHONPATH=backend python3 -m py_compile \
  backend/mix_variant_profile_proposal.py \
  backend/source_identity_downstream_gate.py \
  backend/processing_proposal_adapters.py \
  backend/project_goal_capability_matrix.py \
  tests/test_mix_variant_profile_proposal.py
PYTHONPATH=backend python3 -m pytest \
  tests/test_mix_variant_profile_proposal.py \
  tests/test_source_identity_downstream_gate.py \
  tests/test_processing_registry.py \
  tests/test_project_goal_capability_matrix.py -q
PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q
/opt/homebrew/bin/git diff --check
```

## Where the other agent may disagree

Kimi may argue that mix variants can rely on arrangement confidence alone. I
would keep the source identity gate because arrangement confidence measures
context quality, while replay trust decides whether that identity context is
allowed to influence downstream proposal fan-out.

# Proposal: source-identity-confidence-proposal-20260605

## Short thesis

Add a proposal-only source identity confidence layer that converts weak or
ambiguous classifier output into explainable operator review records without
creating live console authority.

## What I understood

The Automixer cannot safely trust source-specific EQ, dynamics, FX, arrangement,
or frequency-lane decisions when the source type classifier is uncertain. The
current matrix names the gap, but there is no Product Layer contract that says
why a channel identity should be reviewed or why downstream proposals should be
held.

## Proposed solution

Create `backend/source_identity_confidence_proposal.py` with transport-free
dataclasses and `analyze_source_identity_confidence`. The analyzer should:

- accept already-produced classifier/name-recognizer observations;
- normalize optional current-state freshness;
- block stale provided freshness;
- detect unknown, low-confidence, ambiguous, or signal-unreliable identities;
- emit non-queueable source identity review proposals with downstream rejection
  reasons and priority multipliers;
- provide a helper for `ChannelClassification` outputs without importing mixer
  transports.

Register the method in `processing_registry`, expose a `source_identity`
proposal-only contract, add a preview adapter, and update the source type
capability row.

## Likely files to touch

- `backend/source_identity_confidence_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_source_identity_confidence_proposal.py`
- `.ai/briefs/source-identity-confidence-proposal-20260605.md`
- `Docs/adr/source-identity-confidence-proposal-20260605.md`

## Alternatives considered

- Put the logic inside `channel_recognizer`. Rejected because the recognizer
  should keep producing classifications; proposal priority and downstream
  rejection are Product Layer concerns.
- Run ML inference in the new layer. Rejected because this layer should review
  existing classifier artifacts without audio or hardware side effects.
- Auto-correct labels. Rejected because wrong labels can steer many downstream
  processing families; operator acceptance and replay should come first.

## Risks

- Confidence thresholds are conservative defaults and need tuning from real
  accepted/rejected labels.
- Downstream proposal families still need explicit integration to consume the
  rejection reasons.

## Test plan

```bash
PYTHONPATH=backend python -m py_compile \
  backend/source_identity_confidence_proposal.py \
  backend/processing_registry.py \
  backend/processing_proposal_adapters.py \
  backend/project_goal_capability_matrix.py \
  tests/test_source_identity_confidence_proposal.py
PYTHONPATH=backend python -m pytest \
  tests/test_source_identity_confidence_proposal.py \
  tests/test_processing_registry.py \
  tests/test_project_goal_capability_matrix.py -q
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
/opt/homebrew/bin/git diff --check
```

## Where the other agent may disagree

Kimi may argue that downstream modules should consume classifier confidence
directly. I would keep this first layer explicit: a shared review contract makes
operator acceptance, replay, and rejection reasons visible before confidence
starts steering many processing families.

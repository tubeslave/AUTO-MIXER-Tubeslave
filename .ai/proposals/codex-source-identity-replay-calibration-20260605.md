# Proposal: source-identity-replay-calibration-20260605

## Short thesis

Add a proposal-only source identity replay calibration layer so accepted and
rejected classifier-label outcomes can guide future downstream proposal priority
without mutating the model or granting live authority.

## What I understood

Source identity is foundational: a bad label can mislead arrangement, EQ,
dynamics, FX, frequency separation, and balance decisions. The system now has
source identity review proposals, but it still needs a replay-derived trust
surface before downstream families can safely use accepted/rejected labels.

## Proposed solution

Create `backend/source_identity_replay_calibration.py` with transport-free
dataclasses and `analyze_source_identity_replay_calibration`. The analyzer
should:

- accept existing source identity replay outcome events;
- normalize optional current-state freshness;
- block stale provided freshness;
- aggregate accepted, corrected, rejected, dismissed, replay-match, override,
  and safety-regression counts by predicted source type;
- emit non-queueable calibration proposals with recommended trust states:
  `trusted_for_downstream_priority`, `hold_for_operator_review`,
  `blocked_from_downstream_trust`, or `needs_more_replay_outcomes`;
- keep model, memory, queue, and console mutation disabled.

Register the method in `processing_registry`, expose a
`source_identity_calibration` proposal-only contract, and update the source type
capability row.

## Likely files to touch

- `backend/source_identity_replay_calibration.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_source_identity_replay_calibration.py`
- `.ai/briefs/source-identity-replay-calibration-20260605.md`
- `Docs/adr/source-identity-replay-calibration-20260605.md`

## Alternatives considered

- Let downstream modules interpret replay labels independently. Rejected because
  source identity trust should be auditable and consistent.
- Auto-update classifier thresholds from replay outcomes. Rejected because
  threshold changes need operator review and offline validation.
- Fold the logic into `long_term_memory_learning_proposal`. Rejected because
  that module tracks generic learning readiness; this layer needs a specific
  per-source identity trust state for downstream proposal priority.

## Risks

- Early thresholds may over-hold labels until enough real replay outcomes exist.
- Downstream modules still need explicit wiring to consume accepted calibration
  records.

## Test plan

```bash
PYTHONPATH=backend python -m py_compile \
  backend/source_identity_replay_calibration.py \
  backend/processing_registry.py \
  backend/processing_proposal_adapters.py \
  backend/project_goal_capability_matrix.py \
  tests/test_source_identity_replay_calibration.py
PYTHONPATH=backend python -m pytest \
  tests/test_source_identity_replay_calibration.py \
  tests/test_processing_registry.py \
  tests/test_project_goal_capability_matrix.py -q
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
/opt/homebrew/bin/git diff --check
```

## Where the other agent may disagree

Kimi may argue that replay calibration should immediately feed model threshold
updates. I would keep this as explicit operator-review evidence first because
source-label mistakes fan out into many processing families.

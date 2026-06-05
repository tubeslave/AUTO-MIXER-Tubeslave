# Proposal: source-identity-replay-preview-20260606

## Short Thesis

Wire source identity replay calibration into `main_pipeline_preview` as a
review-only surface so trusted/held/blocked source labels are visible before
they affect downstream proposal priority.

## What I Understood

The project goal depends on reliable automatic source typing. Current replay
calibration can produce trust recommendations, and downstream modules already
consume source-identity trust states, but the main operator preview needs to
show calibration candidates without mutating queues, models, memory, or live
console state.

## Proposed Solution

- Add a source-identity replay calibration adapter in
  `processing_proposal_adapters.py`.
- Let `processing_pipeline_preview.py` read optional
  `server.source_identity_replay_events` or configured replay events.
- Convert valid replay event mappings into `SourceIdentityReplayEvent` records.
- Analyze calibration with existing replay logic and adapt recommendations as
  preview-only proposals.
- Publish counts and metadata in `main_pipeline_preview.summary` and
  `source_identity_replay_calibration`.

## Likely Files To Touch

- `backend/processing_pipeline_preview.py`
- `backend/processing_proposal_adapters.py`
- `tests/test_source_identity_replay_calibration.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/source-identity-replay-preview-20260606.md`

## Alternatives Considered

- Import replay calibration directly into the operator proposal queue. Rejected
  because calibration affects future decision priority, not an immediate
  supervised fader/gain apply.
- Mutate model or memory trust state from replay results. Rejected because this
  requires explicit operator acceptance and a separate persistence contract.
- Keep replay calibration only as a standalone helper. Rejected because
  operators need to see trust candidates before downstream source-specific
  decisions depend on them.

## Risks

- Preview users may assume calibration already changed downstream behavior.
  Mitigate with explicit `downstream_priority_mutation=false` evidence and
  non-queueable blockers.
- Stale replay context could create misleading recommendations. Mitigate by
  reusing the existing freshness gate in `source_identity_replay_calibration`.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_replay_calibration.py tests/test_operator_product_state.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/processing_pipeline_preview.py backend/processing_proposal_adapters.py backend/source_identity_replay_calibration.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where The Other Agent May Disagree

Kimi might argue that replay calibration should wait until memory persistence is
implemented. I would expose the review-only preview first because it gives the
operator visibility without writing memory or changing downstream priority.

# Proposal: operator-proposal-apply-sequence-policy-memory-writer-20260606

## Short Thesis

Add an acknowledged append-only memory writer for verified sequence policy
drafts, following the existing source-identity memory writer pattern.

## What I Understood

The system now has a supervised fader/gain execution evidence chain through
plan, dry-run, gate, executor, audit, policy gate, and memory draft. To progress
toward a learning Automixer, the next step is an explicit memory write path that
records safe outcomes without touching live mixer control.

## Proposed Solution

- Extend sequence policy memory draft records with:
  - `target_memory_path=.ai/memory/project.md`;
  - `target_section=Project Memory Lessons`;
  - `append_only=true`;
  - `memory_write_allowed_by_draft=false`;
  - deterministic `draft_text`.
- Add `write_operator_proposal_apply_sequence_policy_memory`.
- Validate the draft envelope and records.
- Return dry-run ready when records are valid but `apply/operator_ack` are not
  both present.
- Require `apply=true` and `operator_ack=true` before appending.
- Restrict file writes to the allowed relative target path under caller/root.
- Keep queue/model/policy/live authority flags false.

## Likely Files To Touch

- `backend/handlers/product_state_handlers.py`
- `backend/operator_product_state.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-writer-20260606.md`

## Alternatives Considered

- Keep memory draft only. Rejected because the long-term goal needs a real
  learning write path.
- Write runtime memory instead of `.ai/memory/project.md`. Rejected because the
  existing repo-local durable memory pattern is append-only Markdown.
- Allow arbitrary target paths. Rejected because memory writes need a narrow,
  auditable target.

## Risks

The writer can be mistaken for live autonomy. Mitigation: the writer is memory
only, requires operator acknowledgement, rejects live authority claims, and
never calls mixer/apply handlers.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_write_operator_proposal_apply_sequence_policy_memory_appends_with_operator_ack -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_write_operator_proposal_apply_sequence_policy_memory_requires_operator_ack -q`
- `PYTHONPATH=backend python3 -m py_compile backend/handlers/product_state_handlers.py backend/operator_product_state.py`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`

## Where The Other Agent May Disagree

Kimi may prefer keeping all product-layer memory writes dry-run. I would land
the acknowledged writer because it mirrors an existing repository pattern and
keeps the write path narrow, append-only, and testable.

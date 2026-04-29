# Source-Grounded Learning Proposal

## Short Thesis

Build a small, auditable source-knowledge layer that stores paraphrased rules
with provenance and logs every attempted use as JSONL. Keep it off the live path
until enough human feedback has accumulated.

## What I Understood

The current mixes converge because different algorithms still use similar local
DSP instincts. We need a learning layer grounded in books, papers, standards,
and vetted videos, then later compare its decisions with MERT/Codex.

## Proposed Solution

- Add `backend/source_knowledge/`.
- Seed `sources.yaml` and `rules.jsonl`.
- Add a retriever with deterministic keyword scoring and filters.
- Add a non-blocking JSONL logger for decisions and human feedback.
- Add config defaults with `enabled: false`.
- Add tests and docs.

## Likely Files To Touch

- `backend/source_knowledge/*`
- `backend/config_manager.py`
- `config/automixer.yaml`
- `config/source_knowledge.yaml`
- `tests/test_source_knowledge.py`
- `Docs/source_grounded_learning.md`
- `Docs/adr/source-grounded-learning.md`

## Alternatives Considered

- Put rules directly into `backend/ai/knowledge/`: simpler, but weaker schema and
  weaker auditability for decisions.
- Integrate immediately with `RuleEngine`: too risky for live behavior.

## Risks

- Seed rules may look authoritative while still being paraphrased. Mitigation:
  store source IDs, confidence, status, and no long quotes.
- Users may expect automatic improvement immediately. Mitigation: first step is
  data and logging only.

## Test Plan

Run focused pytest for source knowledge and config defaults.

## Where The Other Agent May Disagree

Kimi may prefer direct RAG over structured JSONL. I would keep structured rules
first, because training data needs stable fields.

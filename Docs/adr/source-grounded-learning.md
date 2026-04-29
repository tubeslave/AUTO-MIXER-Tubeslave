# Source-Grounded Learning

## Context

Different offline mixing passes produced similar results despite different
methods. The project needs a way to learn from authoritative engineering
sources without letting an LLM, MERT model, or project-local heuristic become an
unattributed source of truth.

## Options Considered

- Add more rules to the current `RuleEngine`.
- Put study notes into the existing AI knowledge base.
- Add a separate source-grounded layer with provenance and decision logs.

## Decision

Add a separate `source_knowledge` layer. It stores source metadata, paraphrased
atomic rules, retrieval helpers, and JSONL decision/feedback logging. It is
disabled by default and has no live OSC effect.

## Why This Won

It keeps provenance explicit, makes human feedback trainable, and preserves live
sound safety while allowing later MERT/Codex evaluation.

## Rejected Alternatives

- Direct `RuleEngine` integration: too much live behavior risk for the first
  step.
- Free-form RAG only: hard to audit and hard to train on.
- Copying source text: copyright and maintainability risk.

## Implementation Plan

- Seed source registry and rule JSONL.
- Add deterministic retriever and validation.
- Add non-blocking logger.
- Add tests and documentation.

## Test Plan

Run focused pytest for source loading, retrieval, logging, and config defaults.

## Risks And Rollback

If the layer causes issues, disable `source_knowledge.enabled` or remove calls to
the wrapper. Because it is not wired into OSC by default, rollback is low risk.

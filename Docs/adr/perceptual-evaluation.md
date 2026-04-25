# ADR: Perceptual Evaluation Shadow Layer

## Context

AUTO-MIXER-Tubeslave currently evaluates live sound corrections mostly through
engineering signals: level, spectrum, dynamics, source role, safety limits, and
runtime policies. The next step is to collect perceptual feedback signals based
on audio embeddings, while preserving live safety and existing OSC behavior.

## Options Considered

1. Run MERT-like embeddings inline with live analysis.
2. Add a sidecar shadow evaluator with lightweight fallback.
3. Block OSC writes when the perceptual score is worse.

## Decision

Use a sidecar shadow/offline evaluator. It captures before/after windows around
already-safe actions, computes embedding metrics outside the audio callback, and
logs the result for later analysis. Default config keeps it disabled.

## Why This Won

This keeps live behavior stable, avoids heavy model work in the real-time path,
and starts collecting reward-shaped data without risking console control.

## Rejected Alternatives

- Inline MERT was rejected because model inference can exceed live latency
  budgets and may load unavailable dependencies.
- OSC blocking was rejected because the first stage has not been validated
  against real post-console captures.
- Adding mandatory `torch`/`transformers` dependencies was rejected because the
  project must run safely without them.

## Implementation Plan

- Add `backend/perceptual/` with backends, metrics, reference store, evaluator,
  JSONL logging, and `RewardSignal`.
- Add `perceptual` config defaults with `enabled: false`.
- Hook `AutoSoundcheckEngine` shadow evaluation after safe action execution and
  delayed action evaluation.
- Document setup and log interpretation.

## Test Plan

- Unit tests for lightweight embeddings, MERT fallback, MSE/cosine, JSONL
  logging, reward combination, and unchanged OSC behavior in shadow mode.
- Run `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`.

## Risks And Rollback

If the evaluator causes errors, initialization and scoring failures are caught
and logged, while the main engine continues. Rollback is to set
`perceptual.enabled: false` or remove the shadow hook; no mixer-state migration
is involved.

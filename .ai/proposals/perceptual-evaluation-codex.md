# Codex Proposal: Perceptual Evaluation

## Short Thesis

Build the perceptual evaluator as a sidecar shadow module. It observes
before/after audio snapshots and action metadata, logs embedding-based scores,
and never changes OSC behavior in the first release.

## What I Understood

The current system is safety-critical live sound software. The new perceptual
layer should help answer whether a correction sounded better, but it must not
sit in the real-time audio path or overrule the existing safety controller.

## Proposed Solution

- Add `backend/perceptual/` with a stable `PerceptualEvaluator` API.
- Provide a numpy-only lightweight backend for immediate use.
- Provide an optional MERT backend that lazy-loads `torch`/`transformers` and
  falls back to lightweight on failure.
- Log shadow decisions to `logs/perceptual_decisions.jsonl`.
- Add `RewardSignal` so future agents can combine engineering, perceptual, and
  safety scores.
- Integrate from `AutoSoundcheckEngine` after the normal safe action path.

## Likely Files To Touch

- `backend/perceptual/*`
- `backend/auto_soundcheck_engine.py`
- `backend/config_manager.py`
- `config/automixer.yaml`
- `config/perceptual.yaml`
- `tests/test_perceptual_evaluation.py`
- `Docs/perceptual_evaluation.md`

## Alternatives Considered

- Running MERT in the audio callback: rejected for latency and reliability.
- Blocking OSC when perceptual score worsens: rejected for the first stage.
- Adding hard production dependencies: rejected; fallback must work with numpy.

## Risks

- Without a post-console capture, channel snapshots may not reflect the actual
  audience mix.
- Lightweight embeddings are useful for data collection, not a final truth
  metric.
- MERT model downloads can be expensive and should be explicitly enabled.

## Test Plan

- Lightweight evaluator extraction and scoring.
- MERT absence fallback.
- Metric stability for MSE/cosine.
- JSONL logging.
- Shadow mode leaves mixer calls unchanged.
- Reward combination bounds.
- Full repository pytest command.

## Where Another Agent May Disagree

Another reviewer may ask for a stricter reference-store design before logging
any reward data. I would keep the first version smaller and gather data before
locking the reward policy.

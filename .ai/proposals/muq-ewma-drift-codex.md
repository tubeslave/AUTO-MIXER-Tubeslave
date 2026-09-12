# MuQ EWMA Drift - Codex Proposal

## 1. Short thesis

Add a small stateful EWMA drift monitor in `backend/ewma_metrics.py`, wire it
into `MuQEvalService` as an optional per-stem batch update path, and let
`AutoSoundcheckEngine` use the results for logging, UI visualization, and
stem-level ML correction freeze decisions.

## 2. What I understood

MuQ scores are useful as a feedback layer, but live safety needs temporal
smoothing and debounced drift detection before reacting. A single bad window
should not cause rollback; a persistent quality drop should trigger restore and
freeze for the affected stem.

## 3. Proposed solution

- Implement `EwmaDrift` with `update`, `set_baseline`, `snapshot_params`, and
  `restore_last_good`.
- Add `StemEwmaDriftMonitor` for per-stem/per-mask instances, group defaults,
  optional band masks, freeze recovery, and concise summaries.
- Keep the module disabled by default through config.
- Add `MuQEvalService.update_stem_score_batch(...)` as the integration point for
  incoming stem MuQ batches.
- Add an engine proxy method for logs, OSC-style UI payloads, and freeze checks
  before auto corrections.

## 4. Likely files to touch

- `backend/ewma_metrics.py`
- `backend/evaluation/muq_eval_service.py`
- `backend/evaluation/__init__.py`
- `backend/auto_soundcheck_engine.py`
- `backend/handlers/soundcheck_handlers.py`
- `config/muq_eval.yaml`
- `config/automixer.yaml`
- `config/default_config.json`
- `config/ewma_metrics.yaml`
- `tests/test_ewma_metrics.py`
- `tests/test_muq_eval_service.py`

## 5. Alternatives considered

- Put drift logic directly in `AutoSoundcheckEngine`; rejected because MuQ
  quality scoring belongs in the evaluation layer and future batch sources can
  reuse the service API.
- Roll back through mixer snapshots immediately on every low score; rejected
  because it is too sensitive for live audio.

## 6. Risks

- Stem naming can vary. The monitor should normalize common role names but keep
  unknown stems on safe default thresholds.
- CRIT restore can only return the stored parameter snapshot; applying the
  snapshot remains the caller's responsibility.

## 7. Test plan

- Run focused EWMA and MuQ service tests.
- Run the standard repository pytest command if the working tree state permits.

## 8. Where the other agent may disagree

Kimi might prefer a dedicated OSC server for visualization. This proposal keeps
visualization as an OSC-style payload over the existing soundcheck observation
channel to avoid sending non-console OSC messages to live mixer clients.

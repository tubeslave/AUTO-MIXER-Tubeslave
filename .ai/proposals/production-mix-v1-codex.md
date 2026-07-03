# Codex Proposal: production_mix_v1

## 1. Short Thesis

Implement `production_mix_v1` as a separate, opt-in package under `automixer/`
that reuses existing safety/evaluation concepts but does not alter legacy
offline or live defaults.

## 2. What I Understood

The project needs a production-ready offline candidate pipeline now, and a safe
future bridge to OSC/live mode. The key architectural boundary is:

Analyzer -> Candidate Generator -> Critics/Scoring -> Decision Engine -> Safety
Governor -> OSCCommandQueue/Report.

## 3. Proposed Solution

- Add a dedicated YAML config in `config/pipelines/`.
- Add a compact candidate-based pipeline with baseline candidate,
  deterministic numpy analysis, optional MuQ/MERT/Essentia adapters, weighted
  scoring, explainable candidate selection, final safety validation, dry-run OSC
  queueing, and JSON/Markdown reports.
- Provide a smoke CLI that can run from generated audio without equipment.

## 4. Likely Files To Touch

- New: `automixer/production_mix_v1/*`
- New: `config/pipelines/production_mix_v1.yaml`
- New: `tests/test_production_mix_v1.py`
- New: `docs/production_mix_v1_inventory.md`
- New: `Docs/adr/production-mix-v1.md`

## 5. Alternatives Considered

- Extending `tools/offline_agent_mix.py`: rejected because it is already a
  large legacy script and rollback would be harder.
- Extending `decision_engine_v2`: rejected for now because v2 is action-plan
  oriented, while this request needs staged audio candidates and A/B scoring.
- Importing missing `ai_mixing_pipeline` sources: rejected because only pycache
  files are present.

## 6. Risks

- Optional ML evaluators may be unavailable; the pipeline must warn and use
  fallbacks.
- Fallback/proxy scores can look useful but must not be authoritative.
- Live mode could become unsafe if queueing bypasses final safety.

## 7. Test Plan

- Unit tests for config, candidate generation, scoring/decision, safety, OSC
  queue, and dry-run smoke pipeline.
- Run the production smoke CLI in offline dry-run mode.
- Attempt the repository standard pytest command or report any blocker.

## 8. Where Another Agent May Disagree

Another reviewer may prefer to integrate more deeply with
`tools/offline_agent_mix.py` to reuse its richer DSP. I would keep this first
version smaller and adapter-based, then pull proven pieces across only where
tests show clear value.

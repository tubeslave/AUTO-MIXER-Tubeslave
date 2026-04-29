# Brief: production_mix_v1

## Goal

Add an opt-in, dry-run-first production mixing pipeline for high-quality offline
mixes and future live OSC control.

## Constraints

- Do not modify default behavior or existing pipelines.
- Keep real OSC disabled by default.
- Decision Engine must not send OSC.
- Safety Governor is the final boundary before actions are applied or queued.
- Proxy/fallback models cannot make the final decision alone.
- Every accepted or rejected action needs rationale.

## Deliverables

- `config/pipelines/production_mix_v1.yaml`
- `automixer/production_mix_v1/`
- `docs/production_mix_v1_inventory.md`
- ADR and tests
- Offline dry-run smoke run

## Test Command

`PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

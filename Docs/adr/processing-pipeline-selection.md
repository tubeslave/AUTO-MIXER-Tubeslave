# Processing Pipeline Selection

## Context

AUTO-MIXER contains several generations of processing code: live soundcheck
automation, AutoEQ, fader/balance controllers, compressor and gate modules,
phase tools, FX helpers, offline renderers, ML/evaluation code, and legacy
pipeline skeletons. Some modules can analyze safely, some can render offline,
and some still contain direct mixer write surfaces.

The live WING Rack path must stay safer than the DSP surface around it. Current
MVP work is an operator assistant, not autonomous live mixing.

## Options considered

1. Pick the most advanced DSP module and make it the main pipeline.
2. Keep every module available and rely on UI copy to explain which ones are
   safe.
3. Add a single read-only registry that classifies methods and profiles, then
   route canonical live behavior through Product Layer proposals and the
   supervised fader/gain gate.

## Decision

Use option 3. Add a declarative processing registry and four pipeline profiles:

- `main_live_assist`: canonical live-safe operator pipeline.
- `offline_mix`: offline render and comparison pipeline.
- `shadow_quality`: scoring/evaluation pipeline with no executor.
- `lab_dsp`: proposal/research pool.
- `quarantine`: direct-write, placeholder, empty, or deprecated paths.

For MVP, `main_live_assist` may apply only supervised channel `fader` and
`gain`. EQ, compressor, gate, pan, phase, FX, routing, snapshot, and reset
paths may analyze or propose, but they are not live executors.

Slice 2 adds a read-only `main_live_assist` preview over this decision. The
preview can surface fader/gain proposal candidates and blocked method status,
but it does not create queue entries and does not execute mixer writes.

Slice 3 adds proof-readiness for the existing supervised `fader/gain` gate
before any physical `gain` validation. It does not add a new executor. The gate
must distinguish cache evidence from physical readback evidence, fail closed
when pre-read is unavailable, and mark unverified apply or rollback readback as
partial rather than ready.

## Why this won

The registry gives the Product Layer and UI a single source of truth while
preserving the existing WING deployment boundary. It also lets useful research
and offline code remain visible without quietly becoming live runtime behavior.

## Rejected alternatives

Directly promoting AutoEQ, AutoCompressor, AutoFader v2, routing, snapshot, or
FX writes was rejected because those paths do not yet have the required
supervised live proof. Hiding old methods was rejected because agents need to
see which paths are placeholders, lab-only, or quarantined.

## Implementation plan

- Add `backend/processing_registry.py` with method statuses, domains, flags,
  pipeline profiles, and promotion requirements.
- Expose `get_processing_pipeline_snapshot` as a read-only Product Layer
  websocket handler.
- Add processing runtime status to the dashboard snapshot so UI surfaces can
  show `main_live_assist` and the `supervised_fader_gain_only` policy.
- Add `get_main_pipeline_preview` with proposal adapters for operator analysis,
  SafeGain, and registry blockers. The preview is read-only and keeps apply
  delegated to `apply_operator_proposal` through the supervised manual gate.
- Add manual-write proof metadata (`pre_read_status`, `pre_read_source`,
  `readback_source`, `rollback_readback_status`, `rollback_proof_status`,
  `proof_status`) before physical `gain` proof.
- Add a fake-only gain proof dry-run runner for proposal apply, readback,
  cooldown, rollback, and emergency-stop rehearsal.
- Add `Docs/PROCESSING_METHODS_INVENTORY.md` as the human-readable inventory.
- Add tests proving placeholders/quarantined methods are not main live
  executors and live-capable methods require supervision.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_processing_registry.py tests/test_operator_product_state.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- Frontend build/test if the frontend package is part of the implementation
  slice.

## Risks and rollback

Risk: a registry can drift from actual modules. Mitigation: focused tests and
inventory updates when methods are promoted.

Risk: UI could imply unsupported live capability. Mitigation: statuses include
`proposal_only`, `offline_only`, `shadow_only`, `lab`, `placeholder`, and
`quarantined`, and only the supervised fader/gain gate is marked live-capable.

Rollback is removing the registry, Product Layer handler, tests, and inventory
document. No live write surface is widened by this ADR.

# Automixer Write Surface Hardening - 2026-06-01

## Summary

This slice adds a dedicated, testable inventory for direct or legacy
write-capable paths in Automixer. It does not enable any new live OSC/WING
writes. The purpose is to make every risky surface explicit before future work
can promote, block, migrate, or delete it.

## Result

New source of truth:

- `backend/write_surface_inventory.py`

Product Layer/registry exposure:

- `processing_pipeline_snapshot.write_surface_inventory`

Only one surface is allowed to write live:

- `product_layer_fader_gain_apply`
- status: `canonical_supervised`
- write kinds: `fader`, `gain`
- gate: `apply_operator_proposal -> supervised manual write gate -> physical readback -> rollback evidence`

Every other known write-capable or write-adjacent path is marked as
`proposal_only`, `lab_only`, `quarantined`, or `blocked`.

## Status Classes

| Status | Meaning | Live write authority |
| --- | --- | --- |
| `canonical_supervised` | Approved live path with operator approval, readback and rollback. | yes, only fader/gain |
| `proposal_only` | Can analyze and suggest, cannot write to console. | no |
| `lab_only` | Local/offline/lab tooling, not Product Layer live control. | no |
| `quarantined` | Direct-write or broad-state surface blocked from main pipeline. | no |
| `blocked` | Must not be invoked by automation/autoloop/cockpit. | no |

## Classified Surfaces

### Canonical Supervised

- Product Layer supervised fader/gain apply:
  `backend/operator_proposal_queue.py`,
  `backend/handlers/mixer_handlers.py`,
  `backend/server.py`,
  `tools/wing_operator_proposal_validation_runner.py`.

### Proposal-Only

- Gain/fader proposal generators:
  `backend/auto_soundcheck_engine.py`,
  `backend/lufs_gain_staging.py`,
  `backend/auto_fader.py`,
  `backend/live_shared_mix.py`,
  `backend/soundcheck_recommendation_workflow.py`,
  `backend/processing_proposal_adapters.py`,
  `backend/operator_recommendation_bridge.py`.
- Context-aware gate/bleed recommendations:
  `backend/context_aware_gate.py`,
  `backend/agentic_method_foundation.py`.
- EQ recommendation and audit sources:
  `backend/auto_eq.py`,
  `backend/cross_adaptive_eq.py`,
  `backend/autoeq_audit.py`,
  `backend/dynamic_eq_playbook.py`.
- Compressor/dynamics recommendation sources:
  `backend/auto_compressor.py`,
  `backend/auto_compressor_cf.py`,
  `backend/autofoh_safety.py`.

### Lab-Only

- Offline render local mixer facades:
  `tools/offline_agent_mix.py`, `mix_agent/`,
  `backend/offline_validation_harness.py`.
- Proof and release-check tooling:
  `tools/no_wing_release_check.py`, `tools/gain_proof_dry_run_runner.py`,
  `tools/wing_operator_proposal_validation_runner.py`,
  `tools/wing_method_proof_runner.py`, `tools/wing_supervised_smoke_runner.py`,
  `tools/wing_readonly_preflight_runner.py`,
  `tools/review_wing_readonly_preflight_artifacts.py`, and
  `backend/wing_readonly_preflight_evidence.py`. These may contain forbidden
  write-token strings for audit/event-log validation, but they do not grant
  live authority.
- Latency benchmark tools:
  `tools/latency/audio_e2e.py`,
  `tools/latency/osc_rtt_benchmark.py`,
  `tools/latency/capture_network.sh`.
- Ableton control surface:
  `backend/ableton_client.py`.

### Quarantined

- Routing, snapshot, and scene writes:
  routing handlers, snapshot handlers, and load/route scripts.
- Direct EQ apply/reset:
  EQ handlers, server reset EQ, and AutoEQ direct apply.
- Direct compressor apply:
  compressor handlers/server paths and AutoCompressor direct calls.
- Gate, HPF and send direct apply:
  AutoGate, AutoSoundcheck, AutoFader v2 and AutoFOH safety direct setters.
- Pan, phase, polarity and delay direct apply:
  AutoPanner, phase alignment and AutoFOH safety direct setters.
- FX, insert and send direct apply:
  AutoFX/AutoEffects/AutoReverb plus restore/reset scripts.
- Voice and AI direct command surfaces:
  voice control modules and `backend/ai/agent.py`.

### Blocked

- Manual reset/restore/bulk mutation scripts:
  `backend/reset_all_channels.py`,
  `backend/reset_modules_trim_faders.py`,
  `backend/restore_channels.py`,
  `backend/disable_modules_set_faders.py`.
- Legacy frontend direct websocket helpers:
  `frontend/src/services/websocket.js`, `frontend/src/App.js`.
- Legacy websocket-to-OSC router:
  `backend/routing.py`.
- ML generated OSC command output:
  `backend/ml/style_transfer.py`, `backend/ml/`.

## Tests

Added tests in `tests/test_processing_registry.py`:

- only `product_layer_fader_gain_apply` may declare `allowed_live=true`;
- canonical live write kinds remain exactly `fader` and `gain`;
- known direct-write paths are represented in the inventory;
- blocked/quarantined status counts are exposed in the snapshot;
- `processing_pipeline_snapshot` now includes the write surface inventory.

## Safety Confirmation

- No live OSC/WING writes were executed.
- No new write endpoint was added.
- No direct console apply path was promoted.
- The change is declarative, read-only, and test-backed.

## Next Work

1. Move blocked manual reset/restore scripts under an explicit lab-only/manual
   recovery namespace, or add hard CLI confirmation guards.
2. Convert voice/AI direct commands into Product Layer proposal intents.
3. Split direct compressor/gate/FX executor code from recommendation code.
4. Remove or wrap legacy frontend direct websocket helpers after compatibility
   audit.
5. Add a CI check that scans new write tokens and requires inventory coverage.

## Simple Explanation

Automixer now has a written map of every place in the project that could
potentially change the mixer. Only one road is allowed for real WING changes:
the supervised fader/gain proposal path. Everything else is marked as a
suggestion, lab tool, quarantined path, or blocked path.

This makes the project safer because future code cannot quietly pretend that
old experimental write code is part of the approved live system.

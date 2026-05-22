# Processing Methods Inventory

Date: 2026-05-20
Status: canonical method-selection inventory

This inventory defines which processing methods may be used by each AUTO-MIXER
pipeline. It does not approve new WING/OSC writes.

## Selection rule

The main live pipeline is `main_live_assist`:

```text
Product/Operator State
  -> AutoSoundcheck / analysis
  -> gain/fader and shared-mix proposals
  -> Safety Gate
  -> supervised manual fader/gain executor
```

Every other method starts as analysis, proposal, offline render, shadow quality,
lab, placeholder, quarantined, or deprecated. A method becomes live-capable only
after a separate ADR, tests, physical WING readback, rollback, cooldown, and
emergency-stop proof.

## Pipeline profiles

| Profile | Status | Purpose | Live executor |
| --- | --- | --- | --- |
| `main_live_assist` | `canonical` | Live-safe operator assistant. | supervised channel `fader` / `gain` only |
| `offline_mix` | `offline_only` | Render, compare, and report mixes without OSC. | none |
| `shadow_quality` | `shadow_only` | Score, warn, and prioritize proposals. | none |
| `lab_dsp` | `lab` | Keep DSP experiments visible and isolated. | none |
| `quarantine` | `quarantined` | Block direct-write, placeholder, empty, or deprecated paths. | none |

## Method selection

| Method | Status | Domain | Use now | Not allowed now |
| --- | --- | --- | --- | --- |
| Product / Operator State | `canonical` | `evaluation` | Readiness, mode, proposal queue, dashboard state. | Direct mixer writes |
| AutoSoundcheck Engine | `canonical` guarded | `gain_fader` | Analyze and form recommendations. | Free live executor behavior |
| Supervised Manual Fader/Gain Gate | `canonical` | `gain_fader` | Only MVP live executor. | EQ, compressor, FX, routing, reset |
| Auto Fader v1 | `proposal_only` | `gain_fader` | Balance proposals. | Direct live fader automation |
| LUFS Gain Staging | `proposal_only` | `gain_fader` | Ayaic-style level-plane guidance. | Standalone live writer |
| Live Shared Mix Planner | `proposal_only` | `gain_fader` | Grouped balance proposals. | Routing/name/master-processing writes |
| Auto EQ | `proposal_only` | `eq` | EQ recommendations and blockers. | Direct `set_eq` in main live pipeline |
| Cross-Adaptive EQ | `proposal_only` | `eq` | Masking-aware proposal/offline candidate. | Direct live EQ apply |
| Auto Compressor | `proposal_only` | `dynamics` | Dynamics recommendations. | Direct live compressor apply |
| Crest-Factor Compressor | `lab` | `dynamics` | Research/proposal candidate. | Live executor |
| CAIG Auto Gate | `proposal_only` | `gate` | Gate recommendations. | Direct live gate apply |
| Auto Panner | `lab` | `pan` | Offline/proposal review. | Live pan apply |
| Phase / Delay Alignment | `proposal_only` | `phase` | Phase/delay recommendations. | Live delay/polarity apply |
| FX / Reverb / Sends | `lab` | `fx` | Offline/lab planning. | Live send/FX apply |
| Mix Agent Conservative Renderer | `offline_only` | `offline_render` | Conservative offline WAV/report/JSON path. | OSC or live imports |
| Offline Agent Mix | `offline_only` | `offline_render` | Extended offline render and reference workflow. | OSC or live imports |
| MuQ / Perceptual Shadow Quality | `shadow_only` | `evaluation` | Scores, warnings, confidence. | Executor authority |
| ML Processing Graph | `lab` | `evaluation` | Research and shadow analysis. | Live runtime default |
| Direct AutoEQ Apply | `quarantined` | `eq` | Block and document. | Main live executor |
| Direct Compressor Apply | `quarantined` | `dynamics` | Block and document. | Main live executor |
| Routing / Snapshot / Reset Writes | `quarantined` | `evaluation` | Block and document. | Processing pipeline executor |
| Direct FX / Insert Apply | `quarantined` | `fx` | Block and document. | Main live executor |
| Mix Agent Unsupported Actions | `placeholder` | `dynamics` | Advisory only. | Fake compression/sends/stereo/dynamic EQ apply |
| Production Mix v1 Package | `placeholder` | `offline_render` | Do not select as entrypoint until restored. | Main/offline executor claim |
| AI Mixing Pipeline Legacy Skeleton | `deprecated` | `offline_render` | Historical reference only. | Current executor |

## Promotion ladder

| From | To | Requirements |
| --- | --- | --- |
| `lab` | `proposal_only` | Adapter emits dry-run trace, tests prove no direct mixer writes, operator reason/confidence exist. |
| `proposal_only` | `supervised_live_candidate` | Fake mixer tests cover bounds/blocked cases, readback and rollback schema exists, cooldown and emergency-stop behavior stay intact. |
| `supervised_live_candidate` | `canonical_live` | Physical WING proof with readback, rollback, cooldown, emergency stop, and explicit operator approval. |

## Runtime contract

Backend registry source: `backend/processing_registry.py`.

Product Layer websocket surface: `get_processing_pipeline_snapshot`.

Main pipeline preview surface: `get_main_pipeline_preview`.

Dashboard runtime summary: `dashboard_snapshot.runtime.processing`.

The registry is declarative and must stay free of mixer-client imports. It is a
read-only contract for agents and UI surfaces, not an execution engine.

## Slice 2 proposal adapter contract

Implementation source:

- `backend/processing_proposal_adapters.py`
- `backend/processing_pipeline_preview.py`

Preview websocket output:

- `type`: `main_pipeline_preview`;
- `pipeline_id`: `main_live_assist`;
- `executor_policy.live_apply_policy`: `supervised_fader_gain_only`;
- `executor_policy.queue_mutation`: `false`;
- `executor_policy.live_write`: `false`;
- `proposals`: preview-only fader/gain candidates;
- `blockers`: unavailable EQ/compressor/gate/pan/phase/FX/routing/snapshot/reset methods.

Proposal adapters are the next step after this inventory. They may connect DSP
methods to the Product Layer, but they must not execute mixer writes.

Minimum adapter output:

| Field | Requirement |
| --- | --- |
| `method_id` | Must match `backend/processing_registry.py`. |
| `pipeline_id` | Usually `main_live_assist`, `offline_mix`, `shadow_quality`, or `lab_dsp`. |
| `domain` | One of the registry domains. |
| `status` | `proposal_only`, `offline_only`, `shadow_only`, or `lab`; never `canonical_live` by default. |
| `title` | Operator-readable summary. |
| `reason` | Human-readable evidence for the recommendation. |
| `confidence` | Numeric or textual confidence suitable for UI display. |
| `risk` | `low`, `medium`, `high`, or `blocked`. |
| `requested_change` | Allowed only for supervised `fader`/`gain` proposals. |
| `requested_changes` | Allowed only for linked supervised `fader`/`gain` proposals. |
| `blockers` | Required when the method recommends EQ/compressor/gate/pan/phase/FX/routing/snapshot/reset instead of apply. |
| `evidence` | Metrics, current value, proposed value, rollback value, and readback requirement when applicable. |

Rules:

- Adapters must not import or instantiate `WingClient` or `EnhancedOSCClient`.
- Adapters must not call `mixer_client.set_*`, `send`, routing, snapshot, FX, or
  reset methods.
- EQ, compressor, gate, pan, phase, FX, routing, snapshot, scene, and reset
  recommendations must be emitted as proposals/blockers only.
- `requested_change.value_type` may be only `fader` or `gain` for
  `main_live_assist`.
- Apply remains `apply_operator_proposal` -> supervised manual gate.

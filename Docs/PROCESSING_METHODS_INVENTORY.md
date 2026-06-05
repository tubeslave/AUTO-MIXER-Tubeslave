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
| AutoSoundcheck Engine | `canonical` guarded | `gain_fader` | Analyze and form recommendations; legacy EQ-enable, compressor-threshold, fader-restore, and mute writes are blocked evidence. | Free live executor behavior |
| AutoFOH Safety Typed Actions | `quarantined` | `console_control` | Bound action values and emit blocked evidence for non-channel-fader/gain surfaces. | Direct non-fader/gain console apply |
| Supervised Manual Fader/Gain Gate | `canonical` | `gain_fader` | Only MVP live executor. | EQ, compressor, FX, routing, reset |
| Auto Fader v1 | `proposal_only` | `gain_fader` | Balance proposals. | Direct live fader automation |
| LUFS Gain Staging | `proposal_only` | `gain_fader` | Ayaic-style level-plane guidance and offline input-trim render. | Standalone live writer |
| Live Shared Mix Planner | `proposal_only` | `gain_fader` | Grouped balance proposals. | Routing/name/master-processing writes |
| Auto EQ | `proposal_only` | `eq` | EQ recommendations and blockers; direct apply/reset and AutoFaderV2 masking EQ paths are fail-closed blocked evidence. | Direct `set_eq` in main live pipeline |
| AutoEQ Audit Validation | `shadow_only` | `eq` | Validate EQ units, phase, IIR stability, splits, export and headroom. | Live EQ apply |
| Diff-DEQ Dynamic EQ Playbook Bridge | `shadow_only` | `eq` | Import external PEQ+DRC predictions as dry-run research playbook entries. | Model runtime, live EQ/dynamics apply |
| Cross-Adaptive EQ | `proposal_only` | `eq` | Masking-aware proposal/offline candidate; AutoFaderV2 mirror EQ writes are blocked evidence only. | Direct live EQ apply |
| Auto Compressor | `proposal_only` | `dynamics` | Dynamics recommendations; direct enable/apply paths are fail-closed blocked evidence. | Direct live compressor apply |
| Crest-Factor Compressor | `proposal_only` | `dynamics` | Research/proposal candidate; direct apply path is fail-closed blocked evidence. | Live executor |
| CAIG Auto Gate | `proposal_only` | `gate` | Gate recommendations; CAIG and AutoFaderV2 adaptive gate writes are blocked evidence only. | Direct live gate apply |
| Auto Panner | `proposal_only` | `pan` | Reviewable pan/adaptive-pan plans; direct apply paths are fail-closed blocked evidence. | Live pan apply |
| Phase / Delay Alignment | `proposal_only` | `phase` | Phase/delay recommendations; direct apply/reset paths are fail-closed blocked evidence. | Live delay/polarity apply |
| Phase-Aware EQ Guard | `proposal_only` | `phase` | Group phase-before-EQ guard, delay/polarity proposal records, low-frequency cancellation blocker. | Live delay/polarity/EQ apply |
| FX / Reverb / Sends | `proposal_only` | `fx` | Reviewable FX/reverb plans; `apply_to_mixer` paths are fail-closed blocked evidence. | Live send/FX/reverb apply |
| Mix Agent Conservative Renderer | `offline_only` | `offline_render` | Conservative offline WAV/report/JSON path. | OSC or live imports |
| Offline Agent Mix | `offline_only` | `offline_render` | Extended offline render and reference workflow. | OSC or live imports |
| MuQ / Perceptual Shadow Quality | `shadow_only` | `evaluation` | Scores, warnings, confidence. | Executor authority |
| ML Processing Graph | `lab` | `evaluation` | Research and shadow analysis. | Live runtime default |
| Direct EQ Apply | `quarantined` | `eq` | Block and document AutoEQ, AutoFaderV2 masking EQ, AutoFOH EQ/notch, and AutoSoundcheck EQ-enable paths. | Main live executor |
| Direct Compressor Apply | `quarantined` | `dynamics` | Block and document AutoCompressor/CF, server compressor reset helper, AutoFOH compressor, and AutoSoundcheck compressor-threshold paths. | Main live executor |
| Bus/DCA/Master Fader Apply | `quarantined` | `gain_fader` | Block and document AutoFOH bus, DCA, and master fader actions. | Main live executor |
| Routing / Snapshot / Reset Writes | `quarantined` | `evaluation` | Block and document. | Processing pipeline executor |
| Direct FX / Insert Apply | `quarantined` | `fx` | Block and document; AutoFX/AutoReverb direct apply does not mutate mixer state. | Main live executor |
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

UI module map: `processing_pipeline_snapshot.module_status_map` is rendered in
the frontend Settings surface as a status-only table. It lists each module's
entrypoint, UI surface, live-write scope, blocked reason, and next step. This
table is informational only and does not create a new apply path.

Write surface inventory: `processing_pipeline_snapshot.write_surface_inventory`
is backed by `backend/write_surface_inventory.py`. It classifies every known
write-capable or legacy write-adjacent surface as one of:

| Status | Meaning |
| --- | --- |
| `canonical_supervised` | The only approved live executor class: operator-approved fader/gain via Product Layer and manual gate. |
| `proposal_only` | May analyze and propose, but cannot write to the console. |
| `lab_only` | May run in local/offline/lab contexts only; no WING Product Layer authority. |
| `quarantined` | Known direct-write or broad-state surface; blocked from canonical pipeline. |
| `blocked` | Must not be invoked by automation/autoloop/cockpit. Human-only recovery or historical reference. |

Current policy: only `product_layer_fader_gain_apply` may declare
`allowed_live=true`, and its allowed live write kinds are `fader` and `gain`.

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

## Slice 3 proposal-only correction contracts

Implementation source:

- `backend/processing_proposal_adapters.py`
- `backend/processing_pipeline_preview.py`
- `backend/operator_proposal_queue.py`

Preview websocket output:

- `proposal_only_contracts`: non-queueable correction contracts for EQ,
  compressor, gate, pan, phase/delay, FX/sends, routing, and direct FX/rack
  surfaces;
- `summary.proposal_only_contract_count`: number of visible contracts;
- each contract has `queue_action=not_queueable_until_promoted`,
  `can_import_to_queue=false`, and `can_apply_live=false`;
- each contract declares required future evidence: `target`, `old_value`,
  `proposed_value`, `safety_evidence`, `required_approval`,
  `readback_required`, and `rollback_value`.

Operator queue rule:

- only `fader` and `gain` proposals are applyable;
- any other `requested_change.value_type` remains visible for review but gets an
  explicit `apply_blockers` entry such as `unsupported_value_type:eq`;
- unsupported proposal apply still fails closed in `apply_operator_proposal` and
  does not call the supervised manual write gate.

This slice does not promote EQ, compressor, gate, pan, phase, FX, routing,
snapshot, scene, or reset behavior to live execution.

## Slice 4 automation foundation layer

Implementation sources:

- `backend/automation_trace.py`
- `backend/handlers/product_state_handlers.py`
- `backend/operator_product_state.py`
- `backend/offline_validation_harness.py`
- `frontend/src/services/useOperatorBackend.ts`
- `frontend/src/services/operatorAdapters.ts`
- `frontend/src/components/prototype/TaskQueuePanels.tsx`

Purpose:

- create one append-only evidence shape for proposal, offline replay, shadow,
  blocked, apply, readback, rollback, and safety decisions;
- make the cockpit show decision evidence from backend DTOs instead of new mock
  data;
- support offline replay reports without importing WING/OSC live transport;
- harden supervised apply without widening the live-write surface.

Decision trace contract:

| Field | Requirement |
| --- | --- |
| `decision_id` | Stable row id for the trace. |
| `idempotency_key` | Required for supervised apply requests. Duplicate apply keys fail closed. |
| `source` | Producer, such as Product Layer, offline replay, or operator UI. |
| `mode` | Operator mode or replay/shadow context. |
| `target` | Channel, bus, or subsystem target. |
| `action_type` | Intended action; only `fader` and `gain` may reach the live executor. |
| `current_value` | Current or pre-read value when available. |
| `proposed_value` | Proposed value or null for blocked/advisory decisions. |
| `delta` | Proposed change amount used for bounds checks. |
| `safety_limits` | Per-target bounds and policy evidence. |
| `pre_read` | Required before live supervised apply. |
| `readback` | Readback evidence after apply when available. |
| `rollback` | Rollback target and result evidence when available. |
| `proof_status` | `proposed`, `blocked`, `applied`, `partial`, `verified`, or similar UI-safe status. |
| `blocked_reason` | Required when a decision fails closed. |
| `created_at` | ISO timestamp for audit ordering. |

Product Layer websocket surfaces:

- `get_decision_traces` -> `decision_traces`;
- `get_safety_monitor` -> `safety_monitor`;
- `get_replay_runs` -> `replay_runs`;
- `start_replay_run` -> offline/dry-run replay only;
- `get_channel_intelligence` -> `channel_intelligence`.

Apply hardening:

- `apply_operator_proposal` requires `approved=true`, `approval_id`, `reason`,
  and `idempotency_key`;
- duplicate idempotency keys are rejected before the manual gate;
- fader proposals are bounded to a maximum `1.0 dB` step;
- gain-ride proposals are bounded to a maximum `0.5 dB` step;
- auto-generated upward `fader` or `gain` moves are blocked unless explicitly
  operator-authored;
- any non-`fader`/`gain` action type stays quarantined and must produce a
  blocked/proposal trace, not a WING write.

Replay contract:

- offline replay writes `decision_traces.jsonl`;
- offline replay writes `metrics_before_after.json`;
- offline replay writes `report.md`;
- `start_replay_run` records a summary `offline_replay_v1` decision trace in
  the shared trace store;
- the replay summary trace can feed decision memory as `offline` evidence;
- `replay_runs` rows expose run-level `trace_count`, `offline_only`, and
  `live_write` evidence for cockpit display;
- MuQ/perceptual scores remain advisory-only and never grant live-apply
  authority.

This slice is a foundation for agentic live mixing, not autonomous live mixing.
It improves evidence, replay, memory readiness, and cockpit visibility while
preserving the existing supervised-only live boundary.

## Slice 5 shadow gain rides v0

Implementation sources:

- `backend/shadow_gain_rides.py`
- `backend/operator_analysis.py`
- `tests/test_operator_product_state.py`

Purpose:

- generate conservative vocal/intelligibility ride candidates as evidence;
- record `gain_ride_shadow` decision traces during explicit operator analysis;
- keep all ride output shadow-only until a separate promotion plan, fake tests,
  physical readback, rollback, cooldown, and emergency-stop proof exist.

Rules:

- max step is `0.5 dB`;
- candidates are built only for active vocal channels with usable meter data;
- upward rides require peak headroom at or below `-6.0 dBFS`;
- near-peak channels can only produce shadow-down candidates;
- oscillation risk blocks candidates after alternating recent directions;
- high pumping risk blocks candidates;
- generated traces use `proof_status=shadow_only`;
- generated traces set `live_apply_allowed=false`;
- generated traces are not queue proposals and do not call
  `apply_operator_proposal`.

This slice intentionally does not promote gain rides to live execution. It gives
the Director, Replay Engine, Channel Intelligence, and future memory layer more
evidence to inspect.

## Slice 6 decision memory v1

Implementation sources:

- `backend/decision_memory.py`
- `backend/server.py`
- `backend/operator_product_state.py`
- `tests/test_operator_product_state.py`

Purpose:

- persist append-only operational memory from decision traces;
- include only verified or offline evidence;
- expose derived memory rows through `memory_knowledge_snapshot`;
- keep memory as proposal context, not apply authority.

Rules:

- `proof_status=verified` traces can become `verified` decision memory;
- offline/replay traces can become `offline` decision memory;
- `shadow_only` traces do not become decision memory;
- memory rows do not contain direct OSC/write command keys;
- memory rows do not change operator mode, proposal status, or live readiness;
- memory must never call `apply_operator_proposal` or any direct mixer method.

This slice makes successful/verified/offline evidence durable for future
proposal context while preserving the supervised live boundary.

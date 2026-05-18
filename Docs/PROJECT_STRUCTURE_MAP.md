# AUTO-MIXER Project Structure Map

Date: 2026-05-18
Status: canonical structure baseline
Audience: project owner, Codex, Kimi, Paperclip, GPT Director, engineering agents

This document explains how the repository is organized now and how future work
must be sorted. It does not approve live WING/OSC writes.

## Current Baseline

- Branch state at inventory time: `main...origin/main [ahead 62, behind 26]`.
- Working tree is dirty and already contains Product Layer changes.
- Approximate source/document file count after excluding caches, build outputs,
  `node_modules`, backend venv, and pyc files: 711.
- Existing implementation plan: `Docs/PROJECT_IMPLEMENTATION_PLAN.md`.
- Primary safety rule: live console behavior is read-only or dry-run by default.
  Real writes require explicit supervised approval, readback, rollback,
  cooldown, and emergency-stop proof.

## Status Labels

Use these labels before moving, deleting, or rewriting anything.

| Label | Meaning | Action rule |
| --- | --- | --- |
| `canonical` | Main supported path for the MVP. | Protect, test, document. |
| `optional` | Useful opt-in module or feature. | Keep config-gated and off by default. |
| `lab` | Research, prototype, benchmark, or local experiment. | Keep isolated from live/runtime defaults. |
| `legacy` | Old path that may still hold useful behavior. | Do not delete until replacement is proven. |
| `quarantine` | Risky, write-capable, stale, or misleading path. | Block from canonical flow; require explicit review. |
| `generated` | Build output, cache, report output, runtime artifact. | Do not treat as source; preview cleanup before deletion. |

## Product Goal

The project is being structured around one clear MVP path:

```text
observe state
  -> analyze
  -> create operator proposal
  -> safety gate review
  -> show proposal in Control Center
  -> operator approval
  -> supervised apply
  -> readback
  -> rollback/evidence
```

Anything that cannot fit this path must be marked `optional`, `lab`, `legacy`,
or `quarantine` instead of being mixed into the main flow.

## Canonical Domains

### 1. Live Runtime And WING Safety

Purpose: connect to the real console, observe state, and apply only supervised
operator-approved changes.

| Path | Status | Role |
| --- | --- | --- |
| `backend/wing_client.py` | `canonical` | WING OSC client, write gate, supervised bundle path, telemetry hooks. |
| `backend/server.py` | `canonical` | WebSocket coordinator and current runtime integration point. |
| `backend/handlers/` | `canonical` | Modular WebSocket message handlers. |
| `backend/auto_soundcheck_engine.py` | `canonical` with guarded use | Soundcheck analysis and recommendations; live apply must stay supervised. |
| `backend/replay_write_intent_adapter.py` | `canonical` | Converts risky operations into replay/write-intent evidence. |
| `backend/wing_telemetry_*` | `optional` | WING telemetry capture and comparison support. |
| `backend/tub345_supervised_write_runner.py` | `optional` | Limited real-WING proof runner; requires explicit operator approval. |

Hard boundary:

- No Auto-EQ, compressor, routing, snapshot, FX, mute, or scene write may become
  canonical live behavior unless it is routed through supervised gate tests.
- Partial hardware proof is not "fully live ready".

### 2. Backend Product Layer

Purpose: convert analyzer output into operator-visible state and proposals.

| Path | Status | Role |
| --- | --- | --- |
| `backend/operator_product_state.py` | `canonical` | Product-facing state model: mode, safety, connection, blockers. |
| `backend/operator_proposal_queue.py` | `canonical` | Proposal lifecycle: create, accept, dismiss, apply evidence. |
| `backend/operator_recommendation_bridge.py` | `canonical` | Imports safe recommendations into the operator workflow. |
| `backend/operator_analysis.py` | `canonical` | Product-level analysis summary and readiness blockers. |
| `backend/operator_mode_policy.py` | `canonical` | Observe / Assist / Supervised mode policy. |
| `backend/handlers/product_state_handlers.py` | `canonical` | WebSocket surface for Product Layer state. |

Rule: automation modules should feed proposals first. They should not quietly
become autonomous live action paths.

### 3. Frontend Operator Experience

Purpose: give the operator a clear Control Center, not a marketing page or
hidden developer console.

| Path | Status | Role |
| --- | --- | --- |
| `frontend/src/App.js` | `canonical` | Main React shell. |
| `frontend/src/components/ControlCenterViews.js` | `canonical` | Operator-facing views and product state display. |
| `frontend/src/components/ControlCenterViews.css` | `canonical` | Control Center layout and visual states. |
| `frontend/src/services/websocket.js` | `canonical` | Frontend/backend message contract. |
| Other module tabs under `frontend/src/components/` | `optional` or `legacy` until checked | Existing automation surfaces; must not imply unsupported live control. |

Rule: disabled or unsupported controls must be explicit status-only surfaces.
No fake live buttons.

### 4. Offline Mixing And Render Pipelines

Purpose: render, compare, and improve mixes without touching a live console.

| Path | Status | Role |
| --- | --- | --- |
| `automixer/production_mix_v1/` | `canonical` for offline | Accepted opt-in production mix pipeline. |
| `config/pipelines/production_mix_v1.yaml` | `canonical` if present | Production mix config entrypoint. |
| `mix_agent/` | `optional` / `legacy` | Offline agent flow and rules. Useful but separate from live runtime. |
| `ai_mixing_pipeline/` | `optional` / `legacy` | Older offline decision/replay/critic infrastructure. |
| `production_mix_v1_out/`, `mixes/`, `offline_test_input/` | `generated` or fixture data | Outputs and local render inputs; handle case by case. |

Rule: offline experiments may use critics, references, EQ variants, FX, and
candidate renders. None of that applies to WING live runtime automatically.

### 5. Analyzer / Decision / Safety Architecture

Purpose: preserve the clean architecture boundary:

```text
Analyzer -> Knowledge -> Critics -> Decision Engine -> Safety Gate -> Executor -> Logs
```

| Path | Status | Role |
| --- | --- | --- |
| `automixer/analyzer/` | `canonical` architecture spine | Analysis modules. |
| `automixer/knowledge/` | `canonical` architecture spine | Project/audio knowledge. |
| `automixer/critics/` | `optional` | Critics must stay opt-in and explainable. |
| `automixer/decision/` | `canonical` architecture spine | ActionPlan creation. |
| `automixer/safety/` | `canonical` architecture spine | Safety checks before execution. |
| `automixer/executor/` | `canonical` architecture spine | Execution boundary; live execution must be supervised. |
| `automixer/experiments/` | `lab` | Experiments only. |

Rule: Decision Engine forms plans. Safety Gate decides whether they are allowed.
Executor applies only when the target mode permits it.

### 6. Backend DSP And Automation Modules

Purpose: hold audio algorithms and automation logic, but with strict routing
into Product Layer proposals or supervised live paths.

| Path | Status | Role |
| --- | --- | --- |
| `backend/auto_eq.py`, `backend/cross_adaptive_eq.py` | `optional` until fully proposal-gated | EQ and anti-masking logic. |
| `backend/auto_compressor*.py` | `optional` until fully proposal-gated | Compressor automation. |
| `backend/auto_fader.py`, `backend/auto_fader_v2/` | `canonical` candidate for MVP proposals | Fader/balance automation. |
| `backend/gain_fader_runtime.py`, `backend/lufs_gain_staging.py` | `canonical` candidate for MVP proposals | Gain and fader runtime support. |
| `backend/auto_gate_caig.py`, `backend/auto_panner*.py`, `backend/auto_reverb.py`, `backend/auto_effects.py` | `lab` / `optional` | Useful later, not MVP live defaults. |
| `backend/phase_alignment.py`, `backend/auto_phase_gcc_phat.py` | `optional` | Phase/delay support, live writes must stay supervised. |

Rule: each automation module needs an explicit contract:

- analysis output;
- proposed action;
- risk level;
- dry-run evidence;
- live write target, if any;
- safety gate path.

### 7. Voice Input

Purpose: operator command input, not an autonomous control authority.

| Path | Status | Role |
| --- | --- | --- |
| `backend/voice_runtime.py` | `optional` | Runtime summary and voice status. |
| `backend/voice_control*.py` | `lab` / `optional` | Multiple recognition experiments. |
| `backend/models/sherpa-*` | `lab` | Local voice model asset. |
| `backend/VOICE_CONTROL_README.md` | `legacy` | Useful notes; needs consolidation. |

Rule: voice commands may create proposals or operator requests. They must not
bypass Product Layer and safety policy.

### 8. ML, Research, And External Sources

Purpose: keep useful research available without making it part of the critical
runtime path.

| Path | Status | Role |
| --- | --- | --- |
| `backend/ml/` | `lab` | Training/inference experiments and differentiable console code. |
| `backend/perceptual/`, `backend/evaluation/` | `optional` | Scoring and evaluation helpers. |
| `external/` | `lab` / vendored research | Papers, MuQ, Diff-MST, FxNorm, automix-toolkit, dasp-pytorch. |
| `models/training_datasets/` | `lab` | Datasets and training inputs. |

Rule: research code can inform product decisions, but cannot be imported into
live runtime without a small contract and tests.

### 9. Agent, Paperclip, GPT, Telegram, And Knowledge Surfaces

Purpose: coordinate work and expose project context without mutating live
systems by default.

| Path | Status | Role |
| --- | --- | --- |
| `tools/automixer_operator/` | `canonical` support | Read-only operational audit and safe operator workflow. |
| `tools/gpt_control_bridge/` | `canonical` support | Read-only local GPT context bridge. |
| `tools/paperclip_watchdog/` | `optional` | Paperclip watchdog; dry-run/default guarded behavior. |
| `tools/project_vector_index/` | `optional` | ChromaDB/vector index support for project knowledge. |
| `plugins/telegram/` | `optional` | Telegram bridge/plugin surface. |
| `.paperclip/reports/` | `generated` reports | Durable reports for agents and Paperclip. |

Rule: coordination tools do not grant runtime approval. They may report status,
create dry-run plans, and publish evidence.

### 10. Generated, Cache, And Build Areas

These paths are not product source:

- `__pycache__/`
- `.pytest_cache/`
- `frontend/node_modules/`
- `frontend/build/`
- `frontend/dist/`
- `backend/venv/`
- `backend/build/`
- `backend/dist/`
- `logs/`
- `runtime/`
- selected files under `artifacts/`

Rule: cleanup must start with preview commands and preserve evidence artifacts
for live/safety work.

## Current Dirty Product-Layer Slice

At inventory time, the working tree already contained this relevant slice:

- `backend/handlers/__init__.py`
- `backend/server.py`
- `backend/handlers/product_state_handlers.py`
- `backend/operator_analysis.py`
- `backend/operator_product_state.py`
- `backend/operator_proposal_queue.py`
- `backend/operator_recommendation_bridge.py`
- `frontend/src/App.js`
- `frontend/src/components/ControlCenterViews.css`
- `frontend/src/components/ControlCenterViews.js`
- `frontend/src/services/websocket.js`
- `tests/test_operator_product_state.py`
- `Docs/PROJECT_IMPLEMENTATION_PLAN.md`

Treat this as the current Product Layer implementation slice until reviewed.
Do not mix unrelated cleanup into that slice.

## Working Rules For Future Agents

1. Read `Docs/PROJECT_IMPLEMENTATION_PLAN.md` and this file before project-wide
   changes.
2. Do not delete or move source code until it has a status label.
3. Do not claim live readiness from offline tests.
4. Do not let offline render logic import live console clients.
5. Do not let live runtime import lab/research code without an explicit
   contract and focused tests.
6. Prefer proposals over direct actions.
7. Keep new methods opt-in and config-gated.
8. Preserve old behavior until the replacement path is tested.
9. Publish durable reports when work changes agent-facing context.
10. Keep cleanup batches small enough to review.

## Next Structural Target

The next concrete engineering target is to review the current Product Layer
slice and prove that:

- backend Product State imports cleanly;
- proposal queue has tests;
- WebSocket handler registration is explicit;
- frontend build still passes;
- no new live write path was introduced.

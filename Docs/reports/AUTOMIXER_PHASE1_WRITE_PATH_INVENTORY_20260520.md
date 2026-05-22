# Automixer Phase 1 Write Path Inventory - 2026-05-20

## Purpose

This inventory records the current live-write boundary after the narrow WING
fader proof. It is not a permission document. It exists so future agents do not
accidentally expand live behavior while improving the cockpit.

## Approved MVP Live Path

Only this path is allowed for supervised hardware validation:

1. Product Layer creates or imports a proposal.
2. Operator mode is `supervised`.
3. Operator explicitly approves apply.
4. Frontend/backend sends `apply_operator_proposal`.
5. Backend routes to `_apply_manual_console_write`.
6. Value type is `fader` or `gain`.
7. Backend records old/current value, readback, rollback metadata, and audit
   artifact.
8. Rollback uses `rollback_last_manual_write` through the same gate.

Current live hardware proof covers only fader. Gain remains next-proof, not
proven.

Slice 3 proof-readiness adds explicit proof metadata to this path:

- pre-read must succeed before write;
- evidence source is reported as `verified_from_cache` or
  `verified_from_physical_readback`;
- unavailable pre-read blocks the write;
- unavailable or mismatched apply/rollback readback is `partial`, not ready;
- rollback reports `rollback_readback_status` and `rollback_proof_status`.

## Fail-Closed / Quarantined Surfaces

These are not approved for live WING writes in the MVP cockpit:

- generic websocket `set_eq`;
- generic websocket `set_compressor`;
- routing writes;
- snapshot/scene load or save;
- phase/delay writes;
- FX send/write surfaces;
- voice-command write surfaces;
- training writes;
- rule-library writes;
- autonomous agent writes.

These surfaces must either stay read-only/status-only or receive their own ADR,
tests, readback, rollback, cooldown, and supervised proof.

## Detailed Surface Status

| Surface | Entry point | Status | Required behavior under WING boundary |
| --- | --- | --- | --- |
| `set_fader` | `backend/handlers/mixer_handlers.py` | `supervised` | Route through `_apply_manual_console_write`; require supervised mode and approval. |
| `set_gain` | `backend/handlers/mixer_handlers.py` | `supervised` | Route through `_apply_manual_console_write`; require supervised mode and approval. |
| `rollback_last_manual_write` | `backend/handlers/mixer_handlers.py` | `supervised` | Use the same manual gate and preserve rollback evidence. |
| `set_eq` | `backend/handlers/mixer_handlers.py` | `quarantined` | Return `deployment_write_surface_quarantined`; do not call `set_eq_band`. |
| `set_compressor` | `backend/handlers/mixer_handlers.py` | `quarantined` | Return `deployment_write_surface_quarantined`; do not call `set_compressor`. |
| `load_snap` / `save_snap` | `backend/handlers/snapshot_handlers.py` | `quarantined` | Return `deployment_write_surface_quarantined`; do not load/save scene state. |
| `restore_snapshot` | `backend/server.py` | `quarantined` | Return `deployment_write_surface_quarantined`; do not restore mixer processing. |
| `create_snapshot` | `backend/server.py` | `read_only_backup` | Allowed as backup/read path; no console mutation. |
| `route_output` / `route_multiple_outputs` | `backend/handlers/routing_handlers.py` | `quarantined` | Return `deployment_write_surface_quarantined`; do not call routing writes. |
| `set_channel_input` / `set_channel_alt_input` | `backend/handlers/routing_handlers.py` | `quarantined` | Return `deployment_write_surface_quarantined`; do not repatch inputs. |
| `set_insert` | `backend/handlers/fx_handlers.py` | `quarantined` | Return `deployment_write_surface_quarantined`; do not assign inserts. |
| `set_fx_model` / `set_fx_on` / `set_fx_mix` / `set_fx_parameter` | `backend/handlers/fx_handlers.py` | `quarantined` | Return `deployment_write_surface_quarantined`; do not change FX rack state. |
| AutoEQ apply/reset/all corrections | `backend/server.py` | `quarantined` or `proposal_only` | Analyze only unless routed through a future proposal adapter. |
| AutoCompressor start/live/manual | `backend/server.py` | `quarantined` or `proposal_only` | Analyze only; no live compressor writes. |
| AutoSoundcheck start | `backend/server.py` | `quarantined` for direct live start | Must become proposal/intent first before supervised promotion. |
| Voice fader/gain/snapshot/EQ/compressor commands | `backend/server.py` | `quarantined` | Return `voice_command_blocked`; do not call direct voice write methods. Future voice control must create Product Layer proposals first. |
| Training/rule/memory writes | Product/read-model handlers | `read_only` for cockpit | UI may display state; no live runtime mutation. |

## Proposal-Only Surfaces

These can produce operator-readable recommendations but must not write directly:

- Auto-EQ analysis;
- Auto-Fader and balance analysis;
- Auto-Gain/safe gain suggestions;
- Auto-Compressor analysis;
- Soundcheck recommendations;
- Agent suggestions;
- rules/training/memory recommendations.

`get_main_pipeline_preview` is the Slice 2 read-only preview surface for these
recommendations. It may return fader/gain `requested_change` candidates for
operator review, but it does not import proposals into the queue and does not
call any mixer write method.

Product Layer proposals must contain:

- target;
- value type;
- current/old value;
- proposed value;
- reason;
- safety evidence;
- rollback value;
- readback requirement.

## Emergency Stop Rule

Emergency stop is a latch:

- while active, fader/gain writes are blocked with `emergency_stop_active`;
- `ready_for_live` is false with `manual_write_emergency_stop_active`;
- reset requires explicit operator-approved
  `reset_manual_write_emergency_stop`;
- reset writes an audit event;
- reset does not silently unfreeze automation;
- after emergency proof, the system stays not ready until freeze/readiness is
  reviewed.

## Remaining Phase 1 Work

- Current cockpit/package surfaces are covered for this handoff. Treat Phase 1
  as still open only for newly touched runtime surfaces and the separate
  physical `gain` proof.
- Full local verification is green for this handoff: backend suite
  `1126 passed, 1 skipped`, frontend e2e `8 passed`, and focused safety/product
  tests `99 passed`.
- Continue deeper grep/code review as new runtime surfaces are touched. The
  current top-level WING write surfaces for Auto-EQ, Auto-Fader,
  Auto-Compressor, phase, routing, FX, snapshots, soundcheck, and voice commands
  are either supervised fader/gain or fail-closed under the deployment boundary.
- Keep frontend direct-write methods out of cockpit workflows. Current cockpit
  e2e coverage explicitly rejects fader/gain, EQ/compressor, routing, FX,
  phase, snapshot, soundcheck, and auto-compressor write message types.
- Extend tests to any new cockpit/runtime surfaces that could send `set_fader`,
  `set_gain`, `set_eq`, `set_compressor`, routing, FX, phase, snapshot, or
  autonomous apply messages.
- Treat any rollback/readback failure as `partial/blocked`, never ready.
- Run the physical gain proof only after separate operator approval, safe-state
  reset, and artifact plan review.

## Added Guard Coverage

This slice adds focused tests for:

- quarantined snapshot `save_snap` and `restore_snapshot`;
- quarantined server write surfaces for bypass, EQ reset/apply, phase, fader
  live apply, auto-balance apply, auto-soundcheck, and auto-compressor;
- quarantined voice-command writes before they can call mixer methods;
- frontend and dry-run runner denial lists for direct unsafe message types.

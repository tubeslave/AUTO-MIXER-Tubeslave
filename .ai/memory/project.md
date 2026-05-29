# Project Memory

## Stable Conventions

- Read `AGENTS.md` and `CLAUDE.md` before planning or editing.
- Use `Docs/adr/` for accepted architecture decisions.
- Default test gate: `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- Keep diffs small and avoid new production dependencies without written justification.

## Safety-Critical Rules

- Prefer safer output changes over louder ones.
- Preserve true-peak headroom before increasing gain.
- Respect feedback protection and mixer-safety constraints from `CLAUDE.md`.
- Never change secrets or `.env` files as part of council work.

## Architecture Reminders

- `server.py` is a coordinator; keep logic in dedicated handlers or modules.
- `AudioCapture` is the shared audio service; do not duplicate capture pipelines casually.
- Repeated DSP or protocol rules belong in `CLAUDE.md` or `Docs/CONVENTIONS.md`, not only in task-specific notes.
- Live WING shared-mix analysis lives in `backend/live_shared_mix.py`; keep it as a pure planner and send actions through `AutoFOHSafetyController`.

## Council Lessons

- Separate planning, implementation, and review into different phases.
- The writing agent and reviewing agent should use different worktrees.
- Durable lessons should be generic enough to survive beyond a single task.
- Routing and channel-name writes on a live WING require an explicit expected patch/name map; audit/readback can be automatic, but patch writes should stay disabled by default.
- Automation foundation evidence uses `AutomationDecisionTrace` as the canonical row shape for proposal, replay, blocked, apply, readback, rollback, and safety decisions. This is an evidence and audit layer, not a new live executor.
- Supervised apply remains limited to operator-approved `fader` and `gain` proposals. Apply requests require an idempotency key and bounded deltas before the manual gate; EQ, compressor, gate, pan, phase/delay, FX, routing, snapshots, scenes, training, and rules remain proposal-only, shadow-only, offline-only, or quarantined.
- Cockpit replay visibility belongs on evidence surfaces, not live controls. `start_replay_run` is safe only as offline replay/artifact generation; it must not become a WING/OSC write path.
- Shadow gain ride v0 is evidence-only: `backend/shadow_gain_rides.py` emits bounded `gain_ride_shadow` traces during explicit operator analysis, blocks oscillation and pumping risk, and must not create queueable live proposals until separately promoted with hardware proof.
- Decision memory v1 is append-only and derived only from verified/offline decision traces. Shadow-only traces stay out of durable memory, and memory rows provide context only; they do not grant executor authority or live readiness.
- Offline replay runs must leave audit evidence in the shared trace/memory flow: `start_replay_run` records one `offline_replay_v1` `replay_run` trace after artifact generation, with `offline_only=true`, `live_write=false`, and `mixer_transport_allowed=false`.
- Replay DTO rows should carry their own policy evidence for cockpit rendering: `trace_count`, `offline_only`, and `live_write` come from replay metrics so the UI can show proof facts without parsing report files.
- Frontend replay launch should update the cockpit immediately from the `replay_run_started` event and then refresh `get_replay_runs`; this keeps the latest offline run visible while preserving read-only/no-live-write behavior.
- Keep `Docs/FRONTEND_DTO_CONTRACTS.md` in sync with Product Layer evidence DTOs: `decision_traces`, `safety_monitor`, `replay_runs`, and `channel_intelligence` are display-safe read models, not apply authority.
- MuQ-A1 offline rock passes should A/B `large_system_polish` and extra drum-focus boosts instead of treating them as defaults; on SONG REPA, MuQ preferred the full agent/source-knowledge/AutoFOH/phase stack with `large_system_polish` disabled.
- MuQ-A1 FX-led rock passes should treat `0.9` whole-mix score as an aspirational, not fixed, gate unless calibrated against references; on SONG REPA the provided commercial reference scored `0.7959`, while subtle reference-guided FX plus section-selective modulation reached `0.7725` without clipping.
- Operator accepted the `MIX_AYAIC_FX_SOFTPRINT_FINAL` direction for Desktop/MIX: use AYAIC Mix Monolith-inspired level-plane/relative balance, shared filtered FX returns, real MuQ-A1 audit, and light mix-print mastering. Avoid MuQ score chasing through segment EQ, clipping, heavy limiting, or aggressive compression when this direction is requested.
- Operator accepted Ayaic level-plane balancing as the default rule for offline mixes and real-console soundcheck/live work: gain staging and balance must measure both individual sources and the summed stems/groups. Use Ayaic planes as the primary default (`-22` lead/main, `-25` strong sources and Drums/Bass/Guitars/BG Vocal buses, `-27` tucked backing vocals, `-30` keys/synth/ambient support, `-35` FX/deep effects), while live safety limits override target loudness.

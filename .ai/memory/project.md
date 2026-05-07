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
- MuQ-A1 offline rock passes should A/B `large_system_polish` and extra drum-focus boosts instead of treating them as defaults; on SONG REPA, MuQ preferred the full agent/source-knowledge/AutoFOH/phase stack with `large_system_polish` disabled.
- MuQ-A1 FX-led rock passes should treat `0.9` whole-mix score as an aspirational, not fixed, gate unless calibrated against references; on SONG REPA the provided commercial reference scored `0.7959`, while subtle reference-guided FX plus section-selective modulation reached `0.7725` without clipping.
- Operator accepted the `MIX_AYAIC_FX_SOFTPRINT_FINAL` direction for Desktop/MIX: use AYAIC Mix Monolith-inspired level-plane/relative balance, shared filtered FX returns, real MuQ-A1 audit, and light mix-print mastering. Avoid MuQ score chasing through segment EQ, clipping, heavy limiting, or aggressive compression when this direction is requested.
- Operator accepted strict MuQ-A1 control as a successful SONG REPA mixing method with a calibrated `0.78` gate. The closest safe render was `SONG_REPA_MUQ_A1_CLOSEST_TO_078_20260427.wav` from `FX15_SECTION_MOD_SPACE` plus a small `air -0.80 dB` postmaster move: MuQ-A1 `0.778733894`, peak about `-4.98 dBFS`, no clipping. Future MuQ-led SONG REPA passes should use this as a near-threshold baseline, continue with small reversible moves, and stop instead of forcing the gate through aggressive EQ, limiting, or loudness tricks.

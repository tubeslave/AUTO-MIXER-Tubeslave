# ADR: P1 — fader ceiling defence-in-depth at WingClient level

**Date:** 2026-07-19
**Status:** Accepted (operator directive "продолжай", 2026-07-19)
**Authors:** Kimi (implementation), review requested from Codex
**Refs:** `.ai/briefs/p1-fader-ceiling-defence-in-depth.md`, review High #7 (2026-07-19), CLAUDE.md fader-ceiling canon

## 1. Context

`WingClient.set_channel_fader` clamped only the hardware range [-144, +10] dB. The 0 dBFS ceiling required by the project canon lived exclusively in upper layers (auto_fader caps, heuristics, manual-write config). Any raw programmatic caller — a future bug, an unclamped agent, a script using the client directly — could write +10 dB to a live console channel. Review rated this High: protection must exist at the transport boundary, not only above it.

## 2. Options considered

A. Per-call `allow_above_unity` flag threaded from server approval paths.
B. Client-level ceiling: constructor param + env, default 0.0 (chosen).
C. Hard clamp at 0 with no override.

## 3. Decision

Option B. `WingClient.__init__` gains keyword-only `fader_ceiling_db: Optional[float] = None`; resolution order: constructor argument → `AUTOMIXER_FADER_CEILING_DB` env → `0.0`; value itself clamped to the hardware range. `set_channel_fader` clamps writes above `self._fader_ceiling_db` with a WARNING log (visible, not silent); the hardware clamp [-144, +10] still applies after the ceiling.

## 4. Why this won

- Option A required changing the `MixerClientBase` contract, `ObservationMixerClient`, four duck-typed test doubles (including `test_manual_write_wal.py`, inside Codex's active M-thread scope) and five voice-path call sites — wide blast radius and merge risk with the parallel thread.
- Option B changes one production file with zero signature-contract breakage: every existing caller, fake and wrapper keeps working unchanged.
- Option C removes a legitimate operator capability (explicit > 0 dBFS request) without an escape hatch; the env/param opt-in preserves it at the infrastructure level, which is the correct authority boundary for a capability this dangerous.

## 5. Rejected alternatives

See §4. Noted behavior change, accepted intentionally: operator manual-write and voice paths keep their documented `min(10.0, …)` request clamps, but the console write now lands at the client ceiling (default 0 dBFS) with a warning unless the operator explicitly opts in via `AUTOMIXER_FADER_CEILING_DB`. This is the policy hardening the review recommended.

## 6. Implementation plan

Done: wing_client.py (param + env resolution + enforcement + warning), tests, CLAUDE.md fixed-rules entry, this ADR. Follow-up (optional): surface the ceiling in the operator UI so a clamped manual request explains itself; document the env var in README install section.

## 7. Test plan

`tests/test_wing_client.py::TestWingClientFaderCeiling` — 7 tests: default clamp at 0 with warning; below-ceiling passthrough; param opt-in (incl. hardware clamp above +10); env opt-in; param-over-env precedence; invalid env fallback; param clamped to hardware range. Regression: full `test_wing_client.py`, `test_observation_mixer.py`, `test_manual_write_wal.py`, `test_server.py` voice-path subset.

## 8. Risks and rollback

- A legitimate operator workflow that relied on > 0 dBFS writes now requires `AUTOMIXER_FADER_CEILING_DB` — visible via the WARNING log; documented here and in CLAUDE.md.
- Rollback: revert the single code commit; ceiling disappears with no other coupling.

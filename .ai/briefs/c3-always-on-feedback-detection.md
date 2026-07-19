# Brief: C3 — always-on feedback detection service in server.py

**Author:** Kimi (independent architect role, per AGENTS.md)
**Date:** 2026-07-19
**Priority:** P0-remnant (third critical finding of the 2026-07-19 review)
**Depends on:** P0 secret hygiene + fail-loud feedback path (merged as `0b6787310`, `ffd334b51`, `d9c30938f`)

## Context

Review finding **C3**: `server.py:977` declares `self.feedback_detector: Optional[FeedbackDetector] = None` and never instantiates it. The only live feedback-suppression loop exists inside `AutoSoundcheckEngine._monitor_loop`, i.e. protection is active **only while a soundcheck engine run is in progress**. In normal server operation (the mode in which the console is actually used for mixing) there is no continuous feedback detection at all — only a one-shot batch recheck (`server.py:23644`) invoked around corrections. CLAUDE.md canon says "feedback first: feedback_detector has absolute priority", but in practice that priority only holds during soundcheck.

## Scope (Kimi implements)

1. An always-on feedback **detection + operator alerting** loop in the backend, driven by the existing single `AudioCapture` service (per convention: no new PyAudio streams), active whenever audio capture is running — independent of the soundcheck engine.
2. Alerts delivered to operator clients over the existing WebSocket broadcast path as a new additive message type (no frontend change required; additive only).
3. Optional auto-action (bounded fader reduction, −3 dB, floor −144 dB) behind an **explicit config opt-in, default OFF**, routed through the server's existing safe write path. Notch EQ is NOT auto-applied by this service (band management stays with the soundcheck engine / operator).
4. Detector configuration sourced from `config/automixer.yaml` with conservative defaults matching `FeedbackDetectorConfig` (tonality criteria on, persistence ≥ 6).
5. Tests with synthetic signals (per project convention: DSP changes require synthetic-signal tests) + contract tests for the WS message; ADR; focused commits.

## Non-goals

- Auto-notch outside soundcheck; changes to `AutoSoundcheckEngine` (P0 patches already merged).
- Frontend rendering of the new alert (additive backend message; frontend may adopt later).
- Rewiring the batch recheck at `server.py:23644` (leave as is).

## Coordination

Codex continues the M110–M122 release-evidence thread; no file overlap expected (`server.py` feedback region, new service module, config, tests). Proposal follows the 8-point format; Codex review requested after implementation, per AGENTS.md workflow step 8.
